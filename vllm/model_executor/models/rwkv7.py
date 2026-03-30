# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Inference-only RWKV7 model."""

from collections.abc import Iterable
from itertools import islice

import torch
import torch.nn.functional as F
from torch import nn
from transformers.activations import ACT2FN as HF_ACT2FN

from vllm.config import CacheConfig, ModelConfig, VllmConfig, get_current_vllm_config
from vllm.distributed.parallel_state import (
    get_pp_group,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    model_parallel_is_initialized,
)
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.mamba.abstract import MambaBase
from vllm.model_executor.layers.mamba.mamba_utils import (
    MambaStateCopyFunc,
    MambaStateCopyFuncCalculator,
)
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.models.interfaces import (
    HasInnerState,
    IsAttentionFree,
    SupportsPP,
)
from vllm.sequence import IntermediateTensors
from vllm.v1.attention.backends.linear_attn import LinearAttentionMetadata

from .utils import AutoWeightsLoader, PPMissingLayer, make_layers, maybe_prefix

LOG_DECAY_SCALE = -0.6065306597126334
RWKV7_RUNTIME_DTYPE = torch.float32


def get_tp_world_size() -> int:
    return get_tensor_model_parallel_world_size() if model_parallel_is_initialized() else 1


def get_tp_rank() -> int:
    return get_tensor_model_parallel_rank() if model_parallel_is_initialized() else 0


def sqrelu(x: torch.Tensor) -> torch.Tensor:
    return torch.relu(x).square()


def get_activation_fn(name: str):
    if name == "sqrelu":
        return sqrelu
    if name not in HF_ACT2FN:
        raise ValueError(f"Unsupported RWKV7 activation: {name}")
    return HF_ACT2FN[name]


def token_shift_with_cache(
    hidden_states: torch.Tensor, cached_state: torch.Tensor | None
) -> tuple[torch.Tensor, torch.Tensor]:
    delta = torch.empty_like(hidden_states)
    if hidden_states.shape[0] == 0:
        final_state = (
            cached_state if cached_state is not None else hidden_states.new_empty(0)
        )
        return delta, final_state

    if cached_state is None:
        delta[0] = -hidden_states[0]
    else:
        delta[0] = cached_state.to(hidden_states.dtype) - hidden_states[0]
    if hidden_states.shape[0] > 1:
        delta[1:] = hidden_states[:-1] - hidden_states[1:]
    final_state = hidden_states[-1].to(
        cached_state.dtype if cached_state is not None else hidden_states.dtype
    )
    return delta, final_state


class RWKV7LoRA(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        low_rank_dim: int,
        bias: bool,
        activation: str | None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        if activation is None:
            act = nn.Identity()
        elif activation == "sigmoid":
            act = nn.Sigmoid()
        elif activation == "tanh":
            act = nn.Tanh()
        elif activation == "relu":
            act = nn.ReLU()
        else:
            raise ValueError(f"Unsupported RWKV7 LoRA activation: {activation}")

        self.lora = nn.Sequential(
            ReplicatedLinear(
                input_dim,
                low_rank_dim,
                bias=False,
                quant_config=quant_config,
                prefix=f"{prefix}.lora.0",
            ),
            act,
            ColumnParallelLinear(
                low_rank_dim,
                output_dim,
                bias=bias,
                quant_config=quant_config,
                prefix=f"{prefix}.lora.2",
            ),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, _ = self.lora[0](x)
        x = self.lora[1](x)
        x, bias = self.lora[2](x)
        if bias is not None:
            x = x + bias
        return x


class RWKV7GroupNorm(nn.Module):
    def __init__(
        self,
        num_heads: int,
        head_dim: int,
        value_dim: int,
        eps: float,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.value_dim = value_dim
        self.tp_rank = get_tp_rank()
        self.tp_size = get_tp_world_size()
        self.local_num_heads = self.num_heads // self.tp_size
        self.local_value_dim = self.value_dim // self.tp_size
        self.value_start = self.tp_rank * self.local_value_dim
        self.value_end = self.value_start + self.local_value_dim
        self.eps = self.head_dim * eps

        self.weight = nn.Parameter(torch.ones(self.value_dim))
        self.bias = nn.Parameter(torch.zeros(self.value_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = self.weight[self.value_start : self.value_end]
        bias = self.bias[self.value_start : self.value_end]
        x = x.to(torch.float32)
        x = F.group_norm(
            x.unsqueeze(-1),
            num_groups=self.local_num_heads,
            weight=weight.to(torch.float32),
            bias=bias.to(torch.float32),
            eps=self.eps,
        )
        return x.squeeze(-1)


class RWKV7FeedForward(nn.Module):
    def __init__(
        self,
        config,
        layer_idx: int,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        if config.intermediate_size is None:
            hidden_ratio = 4 if config.hidden_ratio is None else config.hidden_ratio
            intermediate_size = int(config.hidden_size * hidden_ratio)
            intermediate_size = 32 * ((intermediate_size + 31) // 32)
        else:
            intermediate_size = config.intermediate_size

        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.act_fn = get_activation_fn(config.hidden_act)
        self.x_k = nn.Parameter(torch.zeros(self.hidden_size))
        self.key = ColumnParallelLinear(
            self.hidden_size,
            intermediate_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.key",
        )
        self.value = RowParallelLinear(
            intermediate_size,
            self.hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.value",
        )

    def forward(
        self, hidden_states: torch.Tensor, cached_state: torch.Tensor | None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        delta, final_state = token_shift_with_cache(hidden_states, cached_state)
        mixed = hidden_states.addcmul(delta, self.x_k)
        hidden, _ = self.key(mixed)
        hidden = self.act_fn(hidden)
        hidden, _ = self.value(hidden)
        return hidden, final_state


class RWKV7Attention(nn.Module):
    def __init__(
        self,
        config,
        layer_idx: int,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_heads
        self.head_dim = config.head_dim
        self.value_dim = config.value_dim[layer_idx]
        self.head_v_dim = self.value_dim // self.num_heads
        self.tp_rank = get_tp_rank()
        self.tp_size = get_tp_world_size()
        self.local_num_heads = self.num_heads // self.tp_size
        self.local_key_dim = self.hidden_size // self.tp_size
        self.local_value_dim = self.value_dim // self.tp_size
        self.key_start = self.tp_rank * self.local_key_dim
        self.key_end = self.key_start + self.local_key_dim
        self.value_start = self.tp_rank * self.local_value_dim
        self.value_end = self.value_start + self.local_value_dim

        self.x_r = nn.Parameter(torch.zeros(1, 1, self.hidden_size))
        self.x_w = nn.Parameter(torch.zeros(1, 1, self.hidden_size))
        self.x_k = nn.Parameter(torch.zeros(1, 1, self.hidden_size))
        self.x_v = nn.Parameter(torch.zeros(1, 1, self.hidden_size))
        self.x_a = nn.Parameter(torch.zeros(1, 1, self.hidden_size))
        self.x_g = nn.Parameter(torch.zeros(1, 1, self.hidden_size))

        self.k_k = nn.Parameter(torch.zeros(self.hidden_size))
        self.k_a = nn.Parameter(torch.zeros(self.hidden_size))
        self.r_k = nn.Parameter(torch.zeros(self.num_heads, self.head_dim))

        self.r_proj = ColumnParallelLinear(
            self.hidden_size,
            self.hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.r_proj",
        )
        self.k_proj = ColumnParallelLinear(
            self.hidden_size,
            self.hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.k_proj",
        )
        self.v_proj = ColumnParallelLinear(
            self.hidden_size,
            self.value_dim,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.v_proj",
        )
        self.o_proj = RowParallelLinear(
            self.value_dim,
            self.hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
        )

        self.w_lora = RWKV7LoRA(
            self.hidden_size,
            self.hidden_size,
            config.decay_low_rank_dim,
            bias=True,
            activation="tanh",
            quant_config=quant_config,
            prefix=f"{prefix}.w_lora",
        )
        self.a_lora = RWKV7LoRA(
            self.hidden_size,
            self.hidden_size,
            config.a_low_rank_dim,
            bias=True,
            activation=None,
            quant_config=quant_config,
            prefix=f"{prefix}.a_lora",
        )
        if self.layer_idx != 0:
            self.v_lora = RWKV7LoRA(
                self.hidden_size,
                self.value_dim,
                config.v_low_rank_dim,
                bias=True,
                activation=None,
                quant_config=quant_config,
                prefix=f"{prefix}.v_lora",
            )
        self.g_lora = RWKV7LoRA(
            self.hidden_size,
            self.value_dim,
            config.gate_low_rank_dim,
            bias=False,
            activation="sigmoid",
            quant_config=quant_config,
            prefix=f"{prefix}.g_lora",
        )
        self.g_norm = RWKV7GroupNorm(
            num_heads=self.num_heads,
            head_dim=self.head_dim,
            value_dim=self.value_dim,
            eps=config.norm_eps,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        cached_shift_state: torch.Tensor | None,
        recurrent_state: torch.Tensor | None,
        v_first: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        delta, final_shift_state = token_shift_with_cache(hidden_states, cached_shift_state)

        x_r = self.x_r.squeeze(0).squeeze(0)
        x_w = self.x_w.squeeze(0).squeeze(0)
        x_k = self.x_k.squeeze(0).squeeze(0)
        x_v = self.x_v.squeeze(0).squeeze(0)
        x_a = self.x_a.squeeze(0).squeeze(0)
        x_g = self.x_g.squeeze(0).squeeze(0)

        xr = hidden_states.addcmul(delta, x_r)
        xw = hidden_states.addcmul(delta, x_w)
        xk = hidden_states.addcmul(delta, x_k)
        xv = hidden_states.addcmul(delta, x_v)
        xa = hidden_states.addcmul(delta, x_a)
        xg = hidden_states.addcmul(delta, x_g)

        r, _ = self.r_proj(xr)
        w = LOG_DECAY_SCALE * self.w_lora(xw).sigmoid()
        k, _ = self.k_proj(xk)
        v, _ = self.v_proj(xv)

        if self.layer_idx == 0:
            v_first_out = v
        else:
            if v_first is None:
                raise ValueError("RWKV7 layers after layer 0 require `v_first`.")
            v = torch.lerp(v, v_first, self.v_lora(xv).sigmoid())
            v_first_out = v_first

        a = self.a_lora(xa).sigmoid()
        g = self.g_lora(xg)

        r = r.view(-1, self.local_num_heads, self.head_dim).to(torch.float32)
        w = w.view(-1, self.local_num_heads, self.head_dim).to(torch.float32)
        k = k.view(-1, self.local_num_heads, self.head_dim).to(torch.float32)
        a = a.view(-1, self.local_num_heads, self.head_dim).to(torch.float32)
        v = v.view(-1, self.local_num_heads, self.head_v_dim).to(torch.float32)

        local_k_k = self.k_k[self.key_start : self.key_end].view(
            1, self.local_num_heads, self.head_dim
        )
        local_k_a = self.k_a[self.key_start : self.key_end].view(
            1, self.local_num_heads, self.head_dim
        )

        kk = F.normalize(k * local_k_k.to(torch.float32), dim=-1, p=2.0)
        k = k * (1 + (a - 1) * local_k_a.to(torch.float32))

        if recurrent_state is None:
            recurrent_state = torch.zeros(
                self.local_num_heads,
                self.head_dim,
                self.head_v_dim,
                device=hidden_states.device,
                dtype=torch.float32,
            )
        else:
            recurrent_state = recurrent_state.to(torch.float32)

        outputs: list[torch.Tensor] = []
        for idx in range(hidden_states.shape[0]):
            sa = (recurrent_state * (-kk[idx]).unsqueeze(-1)).sum(dim=-2)
            recurrent_state = (
                torch.exp(w[idx]).unsqueeze(-1) * recurrent_state
                + (kk[idx] * a[idx]).unsqueeze(-1) * sa.unsqueeze(-2)
                + k[idx].unsqueeze(-1) * v[idx].unsqueeze(-2)
            )
            outputs.append(
                (recurrent_state * r[idx].unsqueeze(-1)).sum(dim=-2)
            )

        output = torch.stack(outputs, dim=0).reshape(-1, self.local_value_dim)
        output = self.g_norm(output)

        local_r_k = self.r_k[
            self.tp_rank * self.local_num_heads : (self.tp_rank + 1)
            * self.local_num_heads
        ].to(torch.float32)
        correction = (
            (r * k * local_r_k.unsqueeze(0)).sum(dim=-1, keepdim=True) * v
        ).reshape(-1, self.local_value_dim)
        output = (output + correction) * g.to(torch.float32)
        output = output.to(hidden_states.dtype)
        output, _ = self.o_proj(output)
        return output, final_shift_state, recurrent_state, v_first_out


class RWKV7Block(nn.Module, MambaBase):
    def __init__(
        self,
        config,
        layer_idx: int,
        model_config: ModelConfig | None = None,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.model_config = model_config
        self.cache_config = cache_config
        self.prefix = prefix
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_heads
        self.head_dim = config.head_dim
        self.value_dim = config.value_dim[layer_idx]
        self.local_value_dim = self.value_dim // get_tp_world_size()

        self.pre_norm = None
        if config.norm_first and layer_idx == 0:
            self.pre_norm = nn.LayerNorm(
                config.hidden_size,
                eps=config.norm_eps,
                elementwise_affine=True,
                bias=config.norm_bias,
            )
        self.attn_norm = nn.LayerNorm(
            config.hidden_size,
            eps=config.norm_eps,
            elementwise_affine=True,
            bias=config.norm_bias,
        )
        self.attn = RWKV7Attention(
            config=config,
            layer_idx=layer_idx,
            quant_config=quant_config,
            prefix=f"{prefix}.attn",
        )
        self.ffn_norm = nn.LayerNorm(
            config.hidden_size,
            eps=config.norm_eps,
            elementwise_affine=True,
            bias=config.norm_bias,
        )
        self.ffn = RWKV7FeedForward(
            config=config,
            layer_idx=layer_idx,
            quant_config=quant_config,
            prefix=f"{prefix}.ffn",
        )

        compilation_config = get_current_vllm_config().compilation_config
        if prefix in compilation_config.static_forward_context:
            raise ValueError(f"Duplicate layer name: {prefix}")
        compilation_config.static_forward_context[prefix] = self

        self.kv_cache = (
            torch.tensor([]),
            torch.tensor([]),
            torch.tensor([]),
        )

    @property
    def mamba_type(self) -> str:
        return "linear_attention"

    def get_state_dtype(self) -> tuple[torch.dtype, ...]:
        return (
            RWKV7_RUNTIME_DTYPE,
            RWKV7_RUNTIME_DTYPE,
            RWKV7_RUNTIME_DTYPE,
        )

    def get_state_shape(self) -> tuple[tuple[int, ...], ...]:
        return (
            (self.hidden_size,),
            (
                self.num_heads // get_tp_world_size(),
                self.head_dim,
                self.value_dim // self.num_heads,
            ),
            (self.hidden_size,),
        )

    def _run_sequence(
        self,
        hidden_states: torch.Tensor,
        v_first: torch.Tensor | None,
        attn_shift_state: torch.Tensor | None,
        recurrent_state: torch.Tensor | None,
        ffn_shift_state: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        residual = hidden_states
        if self.pre_norm is not None:
            residual = self.pre_norm(residual)

        attn_input = self.attn_norm(residual)
        attn_out, attn_shift_state, recurrent_state, v_first_out = self.attn(
            attn_input,
            attn_shift_state,
            recurrent_state,
            v_first,
        )
        hidden_states = residual + attn_out

        ffn_input = self.ffn_norm(hidden_states)
        ffn_out, ffn_shift_state = self.ffn(ffn_input, ffn_shift_state)
        hidden_states = hidden_states + ffn_out
        return hidden_states, v_first_out, attn_shift_state, recurrent_state, ffn_shift_state

    def _get_kv_state(
        self, slot_id: int, use_initial_state: bool
    ) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
        if not use_initial_state:
            return None, None, None
        return (
            self.kv_cache[0][slot_id],
            self.kv_cache[1][slot_id],
            self.kv_cache[2][slot_id],
        )

    def _store_kv_state(
        self,
        slot_id: int,
        attn_shift_state: torch.Tensor,
        recurrent_state: torch.Tensor,
        ffn_shift_state: torch.Tensor,
    ) -> None:
        self.kv_cache[0][slot_id].copy_(
            attn_shift_state.to(self.kv_cache[0][slot_id].dtype)
        )
        self.kv_cache[1][slot_id].copy_(
            recurrent_state.to(self.kv_cache[1][slot_id].dtype)
        )
        self.kv_cache[2][slot_id].copy_(
            ffn_shift_state.to(self.kv_cache[2][slot_id].dtype)
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        v_first: torch.Tensor | None,
        attn_metadata: LinearAttentionMetadata | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if attn_metadata is None:
            output, v_first_out, _, _, _ = self._run_sequence(
                hidden_states, v_first, None, None, None
            )
            return output, v_first_out

        num_actual_tokens = (
            attn_metadata.num_decode_tokens + attn_metadata.num_prefill_tokens
        )
        hidden_states = hidden_states[:num_actual_tokens]
        if v_first is not None:
            v_first = v_first[:num_actual_tokens]

        output = torch.empty_like(hidden_states)
        v_first_out = torch.empty(
            (num_actual_tokens, self.local_value_dim),
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )
        state_indices = attn_metadata.state_indices_tensor

        for idx in range(attn_metadata.num_decode_tokens):
            slot_id = int(state_indices[idx].item())
            states = self._get_kv_state(slot_id, use_initial_state=True)
            out, vf_out, attn_shift, recurrent, ffn_shift = self._run_sequence(
                hidden_states[idx : idx + 1],
                None if v_first is None else v_first[idx : idx + 1],
                *states,
            )
            output[idx : idx + 1] = out
            v_first_out[idx : idx + 1] = vf_out
            self._store_kv_state(slot_id, attn_shift, recurrent, ffn_shift)

        decode_offset = attn_metadata.num_decode_tokens
        for prefill_idx in range(attn_metadata.num_prefills):
            batch_idx = decode_offset + prefill_idx
            start = int(attn_metadata.query_start_loc[batch_idx].item())
            end = int(attn_metadata.query_start_loc[batch_idx + 1].item())
            slot_id = int(state_indices[batch_idx].item())
            query_len = end - start
            context_len = int(attn_metadata.seq_lens[batch_idx].item()) - query_len
            states = self._get_kv_state(slot_id, use_initial_state=context_len > 0)
            out, vf_out, attn_shift, recurrent, ffn_shift = self._run_sequence(
                hidden_states[start:end],
                None if v_first is None else v_first[start:end],
                *states,
            )
            output[start:end] = out
            v_first_out[start:end] = vf_out
            self._store_kv_state(slot_id, attn_shift, recurrent, ffn_shift)

        return output, v_first_out


class RWKV7Model(nn.Module):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        config = vllm_config.model_config.hf_config
        model_config = vllm_config.model_config
        cache_config = vllm_config.cache_config
        quant_config = vllm_config.quant_config
        self.config = config

        if config.attn is not None:
            raise NotImplementedError(
                "Hybrid RWKV7 checkpoints with transformer attention are not supported yet."
            )

        value_dims = config.value_dim
        if len(set(value_dims)) != 1:
            raise NotImplementedError(
                "RWKV7 with per-layer `value_dim` variation is not supported yet."
            )

        self.local_value_dim = value_dims[0] // get_tp_world_size()
        self.vocab_size = config.vocab_size
        self.embed_tokens = (
            VocabParallelEmbedding(config.vocab_size, config.hidden_size)
            if get_pp_group().is_first_rank
            else PPMissingLayer()
        )

        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers,
            lambda prefix: RWKV7Block(
                config=config,
                layer_idx=int(prefix.split(".")[-1]),
                model_config=model_config,
                cache_config=cache_config,
                quant_config=quant_config,
                prefix=prefix,
            ),
            prefix=f"{prefix}.layers",
        )

        self.norm = (
            nn.LayerNorm(
                config.hidden_size,
                eps=config.norm_eps,
                elementwise_affine=True,
                bias=config.norm_bias,
            )
            if get_pp_group().is_last_rank
            else PPMissingLayer()
        )

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def make_empty_intermediate_tensors(
        self, batch_size: int, dtype: torch.dtype, device: torch.device
    ) -> IntermediateTensors:
        return IntermediateTensors(
            {
                "hidden_states": torch.zeros(
                    (batch_size, self.config.hidden_size), dtype=dtype, device=device
                ),
                "v_first": torch.zeros(
                    (batch_size, self.local_value_dim), dtype=dtype, device=device
                ),
            }
        )

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor | IntermediateTensors:
        del positions
        forward_context = get_forward_context()
        attn_metadata = forward_context.attn_metadata
        if attn_metadata is not None:
            assert isinstance(attn_metadata, dict)

        if get_pp_group().is_first_rank:
            hidden_states = (
                inputs_embeds if inputs_embeds is not None else self.embed_input_ids(input_ids)
            )
            v_first = None
        else:
            assert intermediate_tensors is not None
            hidden_states = intermediate_tensors["hidden_states"]
            v_first = intermediate_tensors["v_first"]

        for layer in islice(self.layers, self.start_layer, self.end_layer):
            layer_metadata = (
                None if attn_metadata is None else attn_metadata.get(layer.prefix)
            )
            assert layer_metadata is None or isinstance(
                layer_metadata, LinearAttentionMetadata
            )
            hidden_states, v_first = layer(
                hidden_states=hidden_states,
                v_first=v_first,
                attn_metadata=layer_metadata,
            )

        if not get_pp_group().is_last_rank:
            assert v_first is not None
            return IntermediateTensors(
                {"hidden_states": hidden_states, "v_first": v_first}
            )

        hidden_states = self.norm(hidden_states)
        return hidden_states


class RWKV7ForCausalLM(nn.Module, HasInnerState, IsAttentionFree, SupportsPP):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        config = vllm_config.model_config.hf_config
        self.config = config
        self.model = RWKV7Model(
            vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model")
        )

        if get_pp_group().is_last_rank:
            if config.tie_word_embeddings:
                self.lm_head = self.model.embed_tokens
            else:
                self.lm_head = ParallelLMHead(
                    config.vocab_size,
                    config.hidden_size,
                    prefix=maybe_prefix(prefix, "lm_head"),
                )
            self.logits_processor = LogitsProcessor(config.vocab_size)
        else:
            self.lm_head = PPMissingLayer()

        self.model.to(RWKV7_RUNTIME_DTYPE)
        if get_pp_group().is_last_rank:
            self.lm_head.to(RWKV7_RUNTIME_DTYPE)

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

    def make_empty_intermediate_tensors(
        self, batch_size: int, dtype: torch.dtype, device: torch.device
    ) -> IntermediateTensors:
        return self.model.make_empty_intermediate_tensors(batch_size, dtype, device)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor | IntermediateTensors:
        return self.model(
            input_ids=input_ids,
            positions=positions,
            intermediate_tensors=intermediate_tensors,
            inputs_embeds=inputs_embeds,
            **kwargs,
        )

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        logits_dtype = getattr(self.lm_head, "weight", hidden_states).dtype
        return self.logits_processor(self.lm_head, hidden_states.to(logits_dtype))

    @classmethod
    def get_mamba_state_dtype_from_config(
        cls, vllm_config: VllmConfig
    ) -> tuple[torch.dtype, ...]:
        return (
            RWKV7_RUNTIME_DTYPE,
            RWKV7_RUNTIME_DTYPE,
            RWKV7_RUNTIME_DTYPE,
        )

    @classmethod
    def get_mamba_state_shape_from_config(
        cls, vllm_config: VllmConfig
    ) -> tuple[tuple[int, ...], ...]:
        config = vllm_config.model_config.hf_config
        if len(set(config.value_dim)) != 1:
            raise NotImplementedError(
                "RWKV7 with per-layer `value_dim` variation is not supported yet."
            )
        return (
            (config.hidden_size,),
            (
                config.num_heads // vllm_config.parallel_config.tensor_parallel_size,
                config.head_dim,
                config.value_dim[0] // config.num_heads,
            ),
            (config.hidden_size,),
        )

    @classmethod
    def get_mamba_state_copy_func(cls) -> tuple[MambaStateCopyFunc, ...]:
        conv, temporal = MambaStateCopyFuncCalculator.mamba1_state_copy_func()
        return (conv, temporal, conv)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        def iter_weights():
            for name, tensor in weights:
                if name == "model.embeddings.weight":
                    yield "model.embed_tokens.weight", tensor
                else:
                    yield name, tensor

        loader = AutoWeightsLoader(self)
        return loader.load_weights(iter_weights())

