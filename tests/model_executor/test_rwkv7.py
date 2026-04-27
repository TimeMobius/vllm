# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from vllm.config import (
    CacheConfig,
    DeviceConfig,
    ModelConfig,
    ParallelConfig,
    VllmConfig,
    set_current_vllm_config,
)
from vllm.config.compilation import CUDAGraphMode
from vllm.distributed import cleanup_dist_env_and_memory
from vllm.distributed.parallel_state import (
    ensure_model_parallel_initialized,
    init_distributed_environment,
)
from vllm.forward_context import set_forward_context
from vllm.model_executor.layers.fla.ops import (
    fused_mul_recurrent_rwkv7,
    fused_mul_recurrent_rwkv7_with_checkpoints,
    rwkv7_kk_pre,
    rwkv7_kk_pre_reference,
    rwkv7_mix6,
    rwkv7_mix6_reference,
    rwkv7_recurrent_reference,
    rwkv7_recurrent_reference_with_checkpoints,
)
from vllm.model_executor.layers.mamba.mamba_utils import (
    get_conv_copy_spec,
    get_temporal_copy_spec,
)
from vllm.model_executor.models.config import RWKV7ForCausalLMConfig
from vllm.model_executor.models.rwkv7 import (
    RWKV7Attention,
    RWKV7Block,
    RWKV7FeedForward,
    RWKV7ForCausalLM,
    RWKV7Model,
    _load_rwkv7_perf_flags,
)
from vllm.transformers_utils.configs.rwkv7 import RWKV7Config
from vllm.utils.network_utils import get_open_port
from vllm.v1.attention.backends.linear_attn import LinearAttentionMetadata

try:
    import pytest
except ImportError:
    pytest = None


def _make_config() -> RWKV7Config:
    return RWKV7Config(
        vocab_size=128,
        hidden_size=64,
        hidden_ratio=2,
        num_hidden_layers=2,
        head_dim=16,
        num_heads=4,
        decay_low_rank_dim=16,
        gate_low_rank_dim=16,
        a_low_rank_dim=16,
        v_low_rank_dim=16,
        norm_bias=True,
        value_dim=64,
    )


def _write_rwkv7_config_dir(tmp_path: Path, config: RWKV7Config) -> Path:
    model_path = tmp_path / "rwkv7-native-config"
    model_path.mkdir()
    config_dict = config.to_dict()
    config_dict["architectures"] = ["RWKV7ForCausalLM"]
    (model_path / "config.json").write_text(json.dumps(config_dict), encoding="utf-8")
    return model_path


def _make_native_rwkv7_state_dict(model: RWKV7ForCausalLM) -> dict[str, torch.Tensor]:
    state_dict: dict[str, torch.Tensor] = {
        "emb.weight": model.model.embed_tokens.weight.detach().clone(),
        "ln_out.weight": model.model.norm.weight.detach().clone(),
        "ln_out.bias": model.model.norm.bias.detach().clone(),
        "head.weight": model.lm_head.weight.detach().clone(),
    }

    for layer_idx, layer in enumerate(model.model.layers):
        block_prefix = f"blocks.{layer_idx}"
        if layer.pre_norm is not None:
            state_dict[f"{block_prefix}.ln0.weight"] = (
                layer.pre_norm.weight.detach().clone()
            )
            state_dict[f"{block_prefix}.ln0.bias"] = (
                layer.pre_norm.bias.detach().clone()
            )

        state_dict[f"{block_prefix}.ln1.weight"] = (
            layer.attn_norm.weight.detach().clone()
        )
        state_dict[f"{block_prefix}.ln1.bias"] = layer.attn_norm.bias.detach().clone()
        state_dict[f"{block_prefix}.ln2.weight"] = (
            layer.ffn_norm.weight.detach().clone()
        )
        state_dict[f"{block_prefix}.ln2.bias"] = layer.ffn_norm.bias.detach().clone()

        attn = layer.attn
        for name in ("x_r", "x_w", "x_k", "x_v", "x_a", "x_g"):
            state_dict[f"{block_prefix}.att.{name}"] = (
                getattr(attn, name).detach().clone()
            )
        state_dict[f"{block_prefix}.att.k_k"] = attn.k_k.detach().clone().view(1, 1, -1)
        state_dict[f"{block_prefix}.att.k_a"] = attn.k_a.detach().clone().view(1, 1, -1)
        state_dict[f"{block_prefix}.att.r_k"] = attn.r_k.detach().clone()
        state_dict[f"{block_prefix}.att.receptance.weight"] = (
            attn.r_proj.weight.detach().clone()
        )
        state_dict[f"{block_prefix}.att.key.weight"] = (
            attn.k_proj.weight.detach().clone()
        )
        state_dict[f"{block_prefix}.att.value.weight"] = (
            attn.v_proj.weight.detach().clone()
        )
        state_dict[f"{block_prefix}.att.output.weight"] = (
            attn.o_proj.weight.detach().clone()
        )
        state_dict[f"{block_prefix}.att.ln_x.weight"] = (
            attn.g_norm.weight.detach().clone()
        )
        state_dict[f"{block_prefix}.att.ln_x.bias"] = attn.g_norm.bias.detach().clone()

        state_dict[f"{block_prefix}.att.w1"] = (
            attn.w_lora.lora[0].weight.detach().clone().transpose(0, 1)
        )
        state_dict[f"{block_prefix}.att.w2"] = (
            attn.w_lora.lora[2].weight.detach().clone().transpose(0, 1)
        )
        state_dict[f"{block_prefix}.att.w0"] = (
            attn.w_lora.lora[2].bias.detach().clone().view(1, 1, -1)
        )
        state_dict[f"{block_prefix}.att.a1"] = (
            attn.a_lora.lora[0].weight.detach().clone().transpose(0, 1)
        )
        state_dict[f"{block_prefix}.att.a2"] = (
            attn.a_lora.lora[2].weight.detach().clone().transpose(0, 1)
        )
        state_dict[f"{block_prefix}.att.a0"] = (
            attn.a_lora.lora[2].bias.detach().clone().view(1, 1, -1)
        )
        state_dict[f"{block_prefix}.att.g1"] = (
            attn.g_lora.lora[0].weight.detach().clone().transpose(0, 1)
        )
        state_dict[f"{block_prefix}.att.g2"] = (
            attn.g_lora.lora[2].weight.detach().clone().transpose(0, 1)
        )

        if layer_idx == 0:
            state_dict[f"{block_prefix}.att.v0"] = torch.zeros(
                1,
                1,
                attn.value_dim,
                dtype=attn.x_r.dtype,
            )
            state_dict[f"{block_prefix}.att.v1"] = torch.zeros(
                attn.hidden_size,
                model.config.v_low_rank_dim,
                dtype=attn.x_r.dtype,
            )
            state_dict[f"{block_prefix}.att.v2"] = torch.zeros(
                model.config.v_low_rank_dim,
                attn.value_dim,
                dtype=attn.x_r.dtype,
            )
        else:
            state_dict[f"{block_prefix}.att.v1"] = (
                attn.v_lora.lora[0].weight.detach().clone().transpose(0, 1)
            )
            state_dict[f"{block_prefix}.att.v2"] = (
                attn.v_lora.lora[2].weight.detach().clone().transpose(0, 1)
            )
            state_dict[f"{block_prefix}.att.v0"] = (
                attn.v_lora.lora[2].bias.detach().clone().view(1, 1, -1)
            )

        ffn = layer.ffn
        state_dict[f"{block_prefix}.ffn.x_k"] = ffn.x_k.detach().clone().view(1, 1, -1)
        state_dict[f"{block_prefix}.ffn.key.weight"] = ffn.key.weight.detach().clone()
        state_dict[f"{block_prefix}.ffn.value.weight"] = (
            ffn.value.weight.detach().clone()
        )

    return state_dict


def _initialize_module_parameters(module: torch.nn.Module) -> None:
    generator = torch.Generator().manual_seed(0)
    for name, parameter in module.named_parameters():
        if parameter.ndim == 0 or name.endswith(".bias"):
            parameter.data.zero_()
        elif "g_norm.weight" in name or "k_a" in name:
            parameter.data.fill_(1.0)
        else:
            parameter.data.normal_(mean=0.0, std=0.02, generator=generator)


def _make_prefill_metadata(
    seq_len: int, *, device: torch.device
) -> LinearAttentionMetadata:
    return LinearAttentionMetadata(
        num_prefills=1,
        num_prefill_tokens=seq_len,
        num_decodes=0,
        num_decode_tokens=0,
        query_start_loc=torch.tensor([0, seq_len], dtype=torch.int32, device=device),
        seq_lens=torch.tensor([seq_len], dtype=torch.int32, device=device),
        state_indices_tensor=torch.tensor([0], dtype=torch.long, device=device),
    )


def _make_decode_metadata(
    total_seq_len: int, *, device: torch.device
) -> LinearAttentionMetadata:
    return LinearAttentionMetadata(
        num_prefills=0,
        num_prefill_tokens=0,
        num_decodes=1,
        num_decode_tokens=1,
        query_start_loc=torch.tensor([0, 1], dtype=torch.int32, device=device),
        seq_lens=torch.tensor([total_seq_len], dtype=torch.int32, device=device),
        state_indices_tensor=torch.tensor([0], dtype=torch.long, device=device),
    )


def _make_multi_decode_metadata(
    total_seq_lens: list[int], state_indices: list[int], *, device: torch.device
) -> LinearAttentionMetadata:
    num_decodes = len(total_seq_lens)
    return LinearAttentionMetadata(
        num_prefills=0,
        num_prefill_tokens=0,
        num_decodes=num_decodes,
        num_decode_tokens=num_decodes,
        query_start_loc=torch.arange(
            0, num_decodes + 1, dtype=torch.int32, device=device
        ),
        seq_lens=torch.tensor(total_seq_lens, dtype=torch.int32, device=device),
        state_indices_tensor=torch.tensor(
            state_indices, dtype=torch.long, device=device
        ),
    )


def _make_multi_prefill_metadata(
    query_lens: list[int],
    total_seq_lens: list[int],
    state_indices: list[int],
    *,
    device: torch.device,
) -> LinearAttentionMetadata:
    query_start_loc = [0]
    for query_len in query_lens:
        query_start_loc.append(query_start_loc[-1] + query_len)
    return LinearAttentionMetadata(
        num_prefills=len(query_lens),
        num_prefill_tokens=query_start_loc[-1],
        num_decodes=0,
        num_decode_tokens=0,
        query_start_loc=torch.tensor(
            query_start_loc,
            dtype=torch.int32,
            device=device,
        ),
        seq_lens=torch.tensor(total_seq_lens, dtype=torch.int32, device=device),
        state_indices_tensor=torch.tensor(
            state_indices, dtype=torch.long, device=device
        ),
    )


def _make_cache_all_prefill_metadata(
    *,
    query_len: int,
    total_seq_len: int,
    block_table: list[int],
    num_computed_tokens: int,
    block_size: int,
    device: torch.device,
) -> LinearAttentionMetadata:
    return LinearAttentionMetadata(
        num_prefills=1,
        num_prefill_tokens=query_len,
        num_decodes=0,
        num_decode_tokens=0,
        query_start_loc=torch.tensor([0, query_len], dtype=torch.int32, device=device),
        seq_lens=torch.tensor([total_seq_len], dtype=torch.int32, device=device),
        state_indices_tensor=torch.tensor(
            [block_table], dtype=torch.long, device=device
        ),
        num_computed_tokens=torch.tensor(
            [num_computed_tokens], dtype=torch.int32, device=device
        ),
        block_idx_last_computed_token=torch.tensor(
            [max((num_computed_tokens + block_size - 1) // block_size - 1, 0)],
            dtype=torch.int32,
            device=device,
        ),
        block_idx_first_scheduled_token=torch.tensor(
            [((num_computed_tokens + 1) + block_size - 1) // block_size - 1],
            dtype=torch.int32,
            device=device,
        ),
        block_idx_last_scheduled_token=torch.tensor(
            [(total_seq_len + block_size - 1) // block_size - 1],
            dtype=torch.int32,
            device=device,
        ),
    )


def _make_cache_all_multi_prefill_metadata(
    *,
    query_lens: list[int],
    total_seq_lens: list[int],
    block_tables: list[list[int]],
    num_computed_tokens: list[int],
    block_size: int,
    device: torch.device,
) -> LinearAttentionMetadata:
    query_start_loc = [0]
    for query_len in query_lens:
        query_start_loc.append(query_start_loc[-1] + query_len)
    return LinearAttentionMetadata(
        num_prefills=len(query_lens),
        num_prefill_tokens=query_start_loc[-1],
        num_decodes=0,
        num_decode_tokens=0,
        query_start_loc=torch.tensor(query_start_loc, dtype=torch.int32, device=device),
        seq_lens=torch.tensor(total_seq_lens, dtype=torch.int32, device=device),
        state_indices_tensor=torch.tensor(
            block_tables, dtype=torch.long, device=device
        ),
        num_computed_tokens=torch.tensor(
            num_computed_tokens, dtype=torch.int32, device=device
        ),
        block_idx_last_computed_token=torch.tensor(
            [
                max((tokens + block_size - 1) // block_size - 1, 0)
                for tokens in num_computed_tokens
            ],
            dtype=torch.int32,
            device=device,
        ),
        block_idx_first_scheduled_token=torch.tensor(
            [
                ((tokens + 1) + block_size - 1) // block_size - 1
                for tokens in num_computed_tokens
            ],
            dtype=torch.int32,
            device=device,
        ),
        block_idx_last_scheduled_token=torch.tensor(
            [
                (total_seq_len + block_size - 1) // block_size - 1
                for total_seq_len in total_seq_lens
            ],
            dtype=torch.int32,
            device=device,
        ),
    )


def _make_cache_all_decode_metadata(
    *,
    total_seq_len: int,
    block_table: list[int],
    num_computed_tokens: int,
    block_size: int,
    device: torch.device,
) -> LinearAttentionMetadata:
    return LinearAttentionMetadata(
        num_prefills=0,
        num_prefill_tokens=0,
        num_decodes=1,
        num_decode_tokens=1,
        query_start_loc=torch.tensor([0, 1], dtype=torch.int32, device=device),
        seq_lens=torch.tensor([total_seq_len], dtype=torch.int32, device=device),
        state_indices_tensor=torch.tensor(
            [block_table], dtype=torch.long, device=device
        ),
        num_computed_tokens=torch.tensor(
            [num_computed_tokens], dtype=torch.int32, device=device
        ),
        block_idx_last_computed_token=torch.tensor(
            [max((num_computed_tokens + block_size - 1) // block_size - 1, 0)],
            dtype=torch.int32,
            device=device,
        ),
        block_idx_first_scheduled_token=torch.tensor(
            [((num_computed_tokens + 1) + block_size - 1) // block_size - 1],
            dtype=torch.int32,
            device=device,
        ),
        block_idx_last_scheduled_token=torch.tensor(
            [(total_seq_len + block_size - 1) // block_size - 1],
            dtype=torch.int32,
            device=device,
        ),
    )


def _require_reference_checkpoint() -> tuple[Path, Any]:
    if pytest is None:
        raise RuntimeError("pytest is required to run RWKV7 integration tests.")

    model_path = os.getenv("VLLM_RWKV7_TEST_MODEL_PATH")
    fla_path = os.getenv("VLLM_RWKV7_TEST_FLA_PATH")

    if not model_path:
        pytest.skip("Set VLLM_RWKV7_TEST_MODEL_PATH to run RWKV7 parity tests.")
    if not fla_path:
        pytest.skip("Set VLLM_RWKV7_TEST_FLA_PATH to run RWKV7 parity tests.")
    assert model_path is not None
    assert fla_path is not None

    model_dir = Path(model_path)
    fla_dir = Path(fla_path)
    if not model_dir.exists():
        pytest.skip(f"RWKV7 model path does not exist: {model_dir}")
    if not fla_dir.exists():
        pytest.skip(f"FLA path does not exist: {fla_dir}")

    if str(fla_dir) not in sys.path:
        sys.path.insert(0, str(fla_dir))

    from fla.models.rwkv7 import RWKV7ForCausalLM as ReferenceRWKV7ForCausalLM

    return model_dir, ReferenceRWKV7ForCausalLM


def _make_vllm_config(model_path: Path) -> VllmConfig:
    return VllmConfig(
        model_config=ModelConfig(
            str(model_path),
            trust_remote_code=False,
            dtype="float32",
            runner="generate",
        ),
        parallel_config=ParallelConfig(
            tensor_parallel_size=1,
            pipeline_parallel_size=1,
        ),
        cache_config=CacheConfig(),
        device_config=DeviceConfig("cuda"),
    )


def _allocate_kv_cache(model: RWKV7ForCausalLM, *, device: torch.device) -> None:
    for layer in model.model.layers:
        state_shapes = layer.get_state_shape()
        state_dtypes = layer.get_state_dtype()
        layer.kv_cache = tuple(
            torch.zeros((1, *shape), dtype=dtype, device=device)
            for shape, dtype in zip(state_shapes, state_dtypes)
        )


def test_rwkv7_perf_flags_from_env(monkeypatch):
    monkeypatch.setenv("RWKV7_USE_FUSED_MIX6", "1")
    monkeypatch.setenv("RWKV7_USE_FUSED_KK_PRE", "true")
    monkeypatch.setenv("RWKV7_USE_FUSED_LNX_RKVRES_XG", "on")
    monkeypatch.setenv("RWKV7_USE_FUSED_CMIX", "yes")
    monkeypatch.setenv("RWKV7_USE_ALT_RECURRENT_KERNEL", "1")

    flags = _load_rwkv7_perf_flags()

    assert flags.use_fused_mix6 is True
    assert flags.use_fused_kk_pre is True
    assert flags.use_fused_lnx_rkvres_xg is True
    assert flags.use_fused_cmix is True
    assert flags.use_alt_recurrent_kernel is True


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_rwkv7_mix6_triton_matches_reference(dtype):
    if dtype is torch.bfloat16 and not torch.cuda.is_bf16_supported():
        pytest.skip("bfloat16 is not supported on this CUDA device.")

    hidden_states = torch.randn(17, 64, device="cuda", dtype=dtype)
    delta = torch.randn_like(hidden_states)
    params = [torch.randn(64, device="cuda", dtype=dtype) for _ in range(6)]

    expected = rwkv7_mix6_reference(hidden_states, delta, *params)
    actual = rwkv7_mix6(hidden_states, delta, *params)

    tol = 1e-6 if dtype is torch.float32 else 2e-2
    for got, ref in zip(actual, expected):
        torch.testing.assert_close(got, ref, atol=tol, rtol=tol)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_rwkv7_kk_pre_triton_matches_reference():
    k = torch.randn(19, 8, 64, device="cuda", dtype=torch.float32)
    a = torch.randn_like(k)
    k_k = torch.randn(8, 64, device="cuda", dtype=torch.float32)
    k_a = torch.randn(8, 64, device="cuda", dtype=torch.float32)

    expected_k, expected_kk = rwkv7_kk_pre_reference(k, k_k, a, k_a)
    actual_k, actual_kk = rwkv7_kk_pre(k, k_k, a, k_a)

    torch.testing.assert_close(actual_k, expected_k, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(actual_kk, expected_kk, atol=1e-6, rtol=1e-6)


def test_rwkv7_perf_hooks_match_reference_formulas():
    config = _make_config()
    vllm_config = VllmConfig(device_config=DeviceConfig("cpu"))
    with set_current_vllm_config(vllm_config):
        init_distributed_environment(
            world_size=1,
            rank=0,
            local_rank=0,
            distributed_init_method=f"tcp://127.0.0.1:{get_open_port()}",
            backend="gloo",
        )
        ensure_model_parallel_initialized(1, 1, backend="gloo")
        try:
            attn = RWKV7Attention(
                config=config,
                layer_idx=1,
                prefix="model.layers.1.attn",
            )
            ffn = RWKV7FeedForward(
                config=config,
                layer_idx=1,
                prefix="model.layers.1.ffn",
            )
            _initialize_module_parameters(attn)
            _initialize_module_parameters(ffn)

            hidden_states = torch.randn(5, config.hidden_size, dtype=torch.float32)
            delta = torch.randn_like(hidden_states)

            mixed = ffn._mix_ffn_inputs(hidden_states, delta)
            torch.testing.assert_close(mixed, hidden_states.addcmul(delta, ffn.x_k))

            helper_outputs = attn._mix_recurrent_inputs(hidden_states, delta)
            expected_outputs = (
                hidden_states.addcmul(delta, attn.x_r.squeeze(0).squeeze(0)),
                hidden_states.addcmul(delta, attn.x_w.squeeze(0).squeeze(0)),
                hidden_states.addcmul(delta, attn.x_k.squeeze(0).squeeze(0)),
                hidden_states.addcmul(delta, attn.x_v.squeeze(0).squeeze(0)),
                hidden_states.addcmul(delta, attn.x_a.squeeze(0).squeeze(0)),
                hidden_states.addcmul(delta, attn.x_g.squeeze(0).squeeze(0)),
            )
            for got, expected in zip(helper_outputs, expected_outputs):
                torch.testing.assert_close(got, expected)

            k = torch.randn(
                5,
                attn.local_num_heads,
                attn.head_dim,
                dtype=torch.float32,
            )
            a = torch.randn_like(k)
            prepared_k, kk = attn._prepare_recurrent_key_terms(k, a)
            local_k_k = attn.k_k[attn.key_start : attn.key_end].view(
                1, attn.local_num_heads, attn.head_dim
            )
            local_k_a = attn.k_a[attn.key_start : attn.key_end].view(
                1, attn.local_num_heads, attn.head_dim
            )
            torch.testing.assert_close(
                kk, F.normalize(k * local_k_k.to(torch.float32), dim=-1, p=2.0)
            )
            torch.testing.assert_close(
                prepared_k,
                k * (1 + (a - 1) * local_k_a.to(torch.float32)),
            )

            recurrent_output = torch.randn(
                5,
                attn.local_num_heads,
                attn.head_v_dim,
                dtype=torch.float32,
            )
            r = torch.randn(
                5,
                attn.local_num_heads,
                attn.head_dim,
                dtype=torch.float32,
            )
            v = torch.randn(
                5,
                attn.local_num_heads,
                attn.head_v_dim,
                dtype=torch.float32,
            )
            g = torch.randn(
                5,
                attn.local_value_dim,
                dtype=torch.float32,
            )

            epilogue_out = attn._finalize_attention_output(
                recurrent_output,
                r,
                prepared_k,
                v,
                g,
                torch.float32,
            )
            manual = attn.g_norm(recurrent_output.reshape(-1, attn.local_value_dim))
            local_r_k = attn.r_k[
                attn.tp_rank * attn.local_num_heads : (attn.tp_rank + 1)
                * attn.local_num_heads
            ].to(torch.float32)
            correction = (
                (r * prepared_k * local_r_k.unsqueeze(0)).sum(dim=-1, keepdim=True) * v
            ).reshape(-1, attn.local_value_dim)
            manual = (manual + correction) * g.to(torch.float32)
            manual, _ = attn.o_proj(manual)
            torch.testing.assert_close(epilogue_out, manual)
        finally:
            cleanup_dist_env_and_memory()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_rwkv7_attention_mix6_flag_matches_reference(monkeypatch):
    monkeypatch.setenv("RWKV7_USE_FUSED_MIX6", "1")

    config = _make_config()
    vllm_config = VllmConfig(device_config=DeviceConfig("cuda"))
    with set_current_vllm_config(vllm_config):
        init_distributed_environment(
            world_size=1,
            rank=0,
            local_rank=0,
            distributed_init_method=f"tcp://127.0.0.1:{get_open_port()}",
            backend="gloo",
        )
        ensure_model_parallel_initialized(1, 1, backend="gloo")
        try:
            attn = RWKV7Attention(
                config=config,
                layer_idx=1,
                prefix="model.layers.1.attn.fused_mix6",
            )
            _initialize_module_parameters(attn)
            attn = attn.cuda()

            hidden_states = torch.randn(
                13,
                config.hidden_size,
                device="cuda",
                dtype=torch.bfloat16,
            )
            delta = torch.randn_like(hidden_states)

            actual = attn._mix_recurrent_inputs(hidden_states, delta)
            expected = rwkv7_mix6_reference(
                hidden_states,
                delta,
                attn.x_r.squeeze(0).squeeze(0),
                attn.x_w.squeeze(0).squeeze(0),
                attn.x_k.squeeze(0).squeeze(0),
                attn.x_v.squeeze(0).squeeze(0),
                attn.x_a.squeeze(0).squeeze(0),
                attn.x_g.squeeze(0).squeeze(0),
            )

            for got, ref in zip(actual, expected):
                torch.testing.assert_close(got, ref, atol=2e-2, rtol=2e-2)
        finally:
            cleanup_dist_env_and_memory()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_rwkv7_attention_kk_pre_flag_matches_reference(monkeypatch):
    monkeypatch.setenv("RWKV7_USE_FUSED_KK_PRE", "1")

    config = _make_config()
    vllm_config = VllmConfig(device_config=DeviceConfig("cuda"))
    with set_current_vllm_config(vllm_config):
        init_distributed_environment(
            world_size=1,
            rank=0,
            local_rank=0,
            distributed_init_method=f"tcp://127.0.0.1:{get_open_port()}",
            backend="gloo",
        )
        ensure_model_parallel_initialized(1, 1, backend="gloo")
        try:
            attn = RWKV7Attention(
                config=config,
                layer_idx=1,
                prefix="model.layers.1.attn.fused_kk_pre",
            )
            _initialize_module_parameters(attn)
            attn = attn.cuda()

            k = torch.randn(
                13,
                attn.local_num_heads,
                attn.head_dim,
                device="cuda",
                dtype=torch.float32,
            )
            a = torch.randn_like(k)
            local_k_k = attn.k_k[attn.key_start : attn.key_end].view(
                attn.local_num_heads, attn.head_dim
            )
            local_k_a = attn.k_a[attn.key_start : attn.key_end].view(
                attn.local_num_heads, attn.head_dim
            )

            actual_k, actual_kk = attn._prepare_recurrent_key_terms(k, a)
            expected_k, expected_kk = rwkv7_kk_pre_reference(
                k,
                local_k_k.to(torch.float32),
                a,
                local_k_a.to(torch.float32),
            )

            torch.testing.assert_close(actual_k, expected_k, atol=1e-6, rtol=1e-6)
            torch.testing.assert_close(actual_kk, expected_kk, atol=1e-6, rtol=1e-6)
        finally:
            cleanup_dist_env_and_memory()


def test_rwkv7_block_forward_without_metadata():
    config = _make_config()
    vllm_config = VllmConfig(device_config=DeviceConfig("cpu"))
    with set_current_vllm_config(vllm_config):
        init_distributed_environment(
            world_size=1,
            rank=0,
            local_rank=0,
            distributed_init_method=f"tcp://127.0.0.1:{get_open_port()}",
            backend="gloo",
        )
        ensure_model_parallel_initialized(1, 1, backend="gloo")
        try:
            block0 = RWKV7Block(config=config, layer_idx=0, prefix="model.layers.0")
            block1 = RWKV7Block(config=config, layer_idx=1, prefix="model.layers.1")
            _initialize_module_parameters(block0)
            _initialize_module_parameters(block1)

            hidden_states = torch.randn(5, config.hidden_size)
            hidden_states, v_first = block0(hidden_states, None, None)
            hidden_states, v_first = block1(hidden_states, v_first, None)

            assert hidden_states.shape == (5, config.hidden_size)
            assert v_first.shape == (5, config.hidden_size)
            assert torch.isfinite(hidden_states).all()
            assert torch.isfinite(v_first).all()
        finally:
            cleanup_dist_env_and_memory()


def test_rwkv7_load_weights_supports_native_pth_names(tmp_path):
    config = _make_config()
    model_path = _write_rwkv7_config_dir(tmp_path, config)
    vllm_config = VllmConfig(
        model_config=ModelConfig(
            str(model_path),
            trust_remote_code=False,
            dtype="float32",
            runner="generate",
        ),
        parallel_config=ParallelConfig(
            tensor_parallel_size=1,
            pipeline_parallel_size=1,
        ),
        cache_config=CacheConfig(),
        device_config=DeviceConfig("cpu"),
    )

    with set_current_vllm_config(vllm_config):
        init_distributed_environment(
            world_size=1,
            rank=0,
            local_rank=0,
            distributed_init_method=f"tcp://127.0.0.1:{get_open_port()}",
            backend="gloo",
        )
        ensure_model_parallel_initialized(1, 1, backend="gloo")
        try:
            model = RWKV7ForCausalLM(vllm_config=vllm_config)
            _initialize_module_parameters(model)
            expected_state = {
                name: parameter.detach().clone()
                for name, parameter in model.named_parameters()
            }
            native_state = _make_native_rwkv7_state_dict(model)

            for parameter in model.parameters():
                parameter.data.zero_()

            loaded_weights = model.load_weights(native_state.items())
            expected_names = {name for name, _ in model.named_parameters()}

            assert loaded_weights == expected_names
            for name, parameter in model.named_parameters():
                torch.testing.assert_close(parameter, expected_state[name])
        finally:
            cleanup_dist_env_and_memory()


def test_rwkv7_block_registers_static_forward_context():
    config = _make_config()
    vllm_config = VllmConfig(device_config=DeviceConfig("cpu"))
    with set_current_vllm_config(vllm_config):
        init_distributed_environment(
            world_size=1,
            rank=0,
            local_rank=0,
            distributed_init_method=f"tcp://127.0.0.1:{get_open_port()}",
            backend="gloo",
        )
        ensure_model_parallel_initialized(1, 1, backend="gloo")
        try:
            prefix = "model.layers.0"
            block = RWKV7Block(config=config, layer_idx=0, prefix=prefix)
            assert (
                vllm_config.compilation_config.static_forward_context[prefix] is block
            )
            assert (
                vllm_config.compilation_config.static_forward_context[f"{prefix}.attn"]
                is block.attn
            )
        finally:
            cleanup_dist_env_and_memory()


def test_rwkv7_attention_custom_op_matches_direct_forward():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required to exercise the RWKV7 attention custom op.")

    config = _make_config()
    vllm_config = VllmConfig(device_config=DeviceConfig("cuda"))
    with set_current_vllm_config(vllm_config):
        init_distributed_environment(
            world_size=1,
            rank=0,
            local_rank=0,
            distributed_init_method=f"tcp://127.0.0.1:{get_open_port()}",
            backend="nccl",
        )
        ensure_model_parallel_initialized(1, 1, backend="nccl")
        try:
            block = RWKV7Block(config=config, layer_idx=0, prefix="model.layers.0")
            _initialize_module_parameters(block)
            block = block.to("cuda", torch.float32)
            hidden_states = torch.randn(4, config.hidden_size, device="cuda")

            direct = block.attn._forward(hidden_states, None, None, None)
            with set_forward_context(None, vllm_config):
                wrapped = block.attn(hidden_states, None, None, None)

            for wrapped_tensor, direct_tensor in zip(wrapped, direct):
                torch.testing.assert_close(wrapped_tensor, direct_tensor)
        finally:
            cleanup_dist_env_and_memory()


def test_rwkv7_fused_recurrent_matches_reference():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required to exercise the RWKV7 fused recurrent op.")

    torch.manual_seed(0)
    device = torch.device("cuda")
    batch_size = 1
    seq_len = 17
    num_heads = 4
    head_dim = 16
    head_v_dim = 16

    r = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device)
    w = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device)
    k = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device)
    v = torch.randn(batch_size, seq_len, num_heads, head_v_dim, device=device)
    kk = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device)
    a = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device)
    initial_state = torch.randn(
        batch_size, num_heads, head_dim, head_v_dim, device=device
    )

    out_ref, state_ref = rwkv7_recurrent_reference(
        r=r,
        w=w,
        k=k,
        v=v,
        kk=kk,
        a=a,
        initial_state=initial_state,
        output_final_state=True,
    )
    out_fused, state_fused = fused_mul_recurrent_rwkv7(
        r=r,
        w=w,
        k=k,
        v=v,
        kk=kk,
        a=a,
        initial_state=initial_state,
        output_final_state=True,
    )

    torch.testing.assert_close(out_fused, out_ref, rtol=2e-4, atol=1e-3)
    torch.testing.assert_close(state_fused, state_ref, rtol=2e-4, atol=1e-3)


def test_rwkv7_fused_recurrent_checkpoint_states_match_reference():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required to exercise RWKV7 checkpoint emission.")

    torch.manual_seed(1)
    device = torch.device("cuda")
    batch_size = 1
    seq_len = 17
    num_heads = 4
    head_dim = 16
    head_v_dim = 16

    r = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device)
    w = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device)
    k = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device)
    v = torch.randn(batch_size, seq_len, num_heads, head_v_dim, device=device)
    kk = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device)
    a = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device)
    initial_state = torch.randn(
        batch_size, num_heads, head_dim, head_v_dim, device=device
    )
    checkpoint_positions = torch.tensor([7, 15], device=device, dtype=torch.long)
    checkpoint_offsets = torch.tensor([0, 2], device=device, dtype=torch.long)

    out_ref, state_ref, checkpoint_ref = rwkv7_recurrent_reference_with_checkpoints(
        r=r,
        w=w,
        k=k,
        v=v,
        kk=kk,
        a=a,
        initial_state=initial_state,
        output_final_state=True,
        checkpoint_positions=checkpoint_positions,
        checkpoint_offsets=checkpoint_offsets,
        output_checkpoint_states=True,
    )
    out_fused, state_fused, checkpoint_fused = (
        fused_mul_recurrent_rwkv7_with_checkpoints(
            r=r,
            w=w,
            k=k,
            v=v,
            kk=kk,
            a=a,
            checkpoint_positions=checkpoint_positions,
            checkpoint_offsets=checkpoint_offsets,
            initial_state=initial_state,
            output_final_state=True,
        )
    )

    torch.testing.assert_close(out_fused, out_ref, rtol=2e-3, atol=1e-1)
    torch.testing.assert_close(state_fused, state_ref, rtol=2e-3, atol=1e-1)
    torch.testing.assert_close(checkpoint_fused, checkpoint_ref, rtol=2e-3, atol=1e-1)


def test_rwkv7_block_updates_cached_states():
    config = _make_config()
    vllm_config = VllmConfig(device_config=DeviceConfig("cpu"))
    with set_current_vllm_config(vllm_config):
        init_distributed_environment(
            world_size=1,
            rank=0,
            local_rank=0,
            distributed_init_method=f"tcp://127.0.0.1:{get_open_port()}",
            backend="gloo",
        )
        ensure_model_parallel_initialized(1, 1, backend="gloo")
        try:
            block = RWKV7Block(config=config, layer_idx=0, prefix="model.layers.0")
            _initialize_module_parameters(block)

            block.kv_cache = (
                torch.zeros(1, config.hidden_size),
                torch.zeros(1, config.num_heads, config.head_dim, config.head_dim),
                torch.zeros(1, config.hidden_size),
            )

            prefill_metadata = _make_prefill_metadata(3, device=torch.device("cpu"))
            hidden_states = torch.randn(3, config.hidden_size)
            output, v_first = block(hidden_states, None, prefill_metadata)

            assert output.shape == hidden_states.shape
            assert v_first.shape == hidden_states.shape
            assert torch.isfinite(output).all()
            assert block.kv_cache[0][0].abs().sum() > 0
            assert block.kv_cache[1][0].abs().sum() > 0
            assert block.kv_cache[2][0].abs().sum() > 0

            decode_metadata = _make_decode_metadata(4, device=torch.device("cpu"))
            decode_hidden = torch.randn(1, config.hidden_size)
            decode_output, decode_v_first = block(
                decode_hidden, v_first[:1].clone(), decode_metadata
            )

            assert decode_output.shape == decode_hidden.shape
            assert decode_v_first.shape == decode_hidden.shape
            assert torch.isfinite(decode_output).all()
            assert torch.isfinite(decode_v_first).all()
        finally:
            cleanup_dist_env_and_memory()


def test_rwkv7_block_batches_decode_tokens_without_changing_results():
    config = _make_config()
    vllm_config = VllmConfig(device_config=DeviceConfig("cpu"))
    with set_current_vllm_config(vllm_config):
        init_distributed_environment(
            world_size=1,
            rank=0,
            local_rank=0,
            distributed_init_method=f"tcp://127.0.0.1:{get_open_port()}",
            backend="gloo",
        )
        ensure_model_parallel_initialized(1, 1, backend="gloo")
        try:
            block_batched = RWKV7Block(
                config=config, layer_idx=0, prefix="model.layers.0"
            )
            _initialize_module_parameters(block_batched)
            block_ref = RWKV7Block(config=config, layer_idx=0, prefix="model.layers.1")
            block_ref.load_state_dict(block_batched.state_dict())

            generator = torch.Generator().manual_seed(123)
            state_shapes = block_batched.get_state_shape()
            state_dtypes = block_batched.get_state_dtype()

            def make_cache() -> tuple[torch.Tensor, ...]:
                return tuple(
                    torch.randn(
                        (2, *shape),
                        generator=generator,
                        dtype=dtype,
                    )
                    for shape, dtype in zip(state_shapes, state_dtypes)
                )

            block_batched.kv_cache = make_cache()
            block_ref.kv_cache = tuple(
                cache.clone() for cache in block_batched.kv_cache
            )

            hidden_states = torch.randn(
                2, config.hidden_size, generator=generator, dtype=torch.float32
            )
            metadata = _make_multi_decode_metadata(
                [5, 7], [0, 1], device=torch.device("cpu")
            )

            output_batched, v_first_batched = block_batched(
                hidden_states, None, metadata
            )

            output_ref = torch.empty_like(hidden_states)
            v_first_ref = torch.empty_like(hidden_states)
            for idx, slot_id in enumerate([0, 1]):
                states = block_ref._get_kv_state(slot_id, use_initial_state=True)
                out, v_first_out, attn_shift, recurrent, ffn_shift = (
                    block_ref._run_sequence(
                        hidden_states[idx : idx + 1],
                        None,
                        *states,
                    )
                )
                output_ref[idx : idx + 1] = out
                v_first_ref[idx : idx + 1] = v_first_out
                block_ref._store_kv_state(slot_id, attn_shift, recurrent, ffn_shift)

            torch.testing.assert_close(output_batched, output_ref)
            torch.testing.assert_close(v_first_batched, v_first_ref)
            for batched_state, ref_state in zip(
                block_batched.kv_cache, block_ref.kv_cache
            ):
                torch.testing.assert_close(batched_state, ref_state)
        finally:
            cleanup_dist_env_and_memory()


def test_rwkv7_block_batches_decode_tokens_without_changing_results_cuda():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required to exercise fused RWKV7 decode batching.")

    config = _make_config()
    vllm_config = VllmConfig(device_config=DeviceConfig("cuda"))
    with set_current_vllm_config(vllm_config):
        init_distributed_environment(
            world_size=1,
            rank=0,
            local_rank=0,
            distributed_init_method=f"tcp://127.0.0.1:{get_open_port()}",
            backend="nccl",
        )
        ensure_model_parallel_initialized(1, 1, backend="nccl")
        try:
            block_batched = RWKV7Block(
                config=config, layer_idx=0, prefix="model.layers.0"
            )
            _initialize_module_parameters(block_batched)
            block_batched = block_batched.to("cuda", torch.float32)

            block_ref = RWKV7Block(config=config, layer_idx=0, prefix="model.layers.1")
            block_ref.load_state_dict(block_batched.state_dict())
            block_ref = block_ref.to("cuda", torch.float32)

            torch.manual_seed(123)
            state_shapes = block_batched.get_state_shape()
            state_dtypes = block_batched.get_state_dtype()

            def make_cache() -> tuple[torch.Tensor, ...]:
                return tuple(
                    torch.randn(
                        (2, *shape),
                        dtype=dtype,
                        device="cuda",
                    )
                    for shape, dtype in zip(state_shapes, state_dtypes)
                )

            block_batched.kv_cache = make_cache()
            block_ref.kv_cache = tuple(
                cache.clone() for cache in block_batched.kv_cache
            )

            hidden_states = torch.randn(
                2,
                config.hidden_size,
                device="cuda",
                dtype=torch.float32,
            )
            metadata = _make_multi_decode_metadata(
                [5, 7], [0, 1], device=torch.device("cuda")
            )

            output_batched, v_first_batched = block_batched(
                hidden_states, None, metadata
            )

            output_ref = torch.empty_like(hidden_states)
            v_first_ref = torch.empty_like(hidden_states)
            for idx, slot_id in enumerate([0, 1]):
                states = block_ref._get_kv_state(slot_id, use_initial_state=True)
                out, v_first_out, attn_shift, recurrent, ffn_shift = (
                    block_ref._run_sequence(
                        hidden_states[idx : idx + 1],
                        None,
                        *states,
                    )
                )
                output_ref[idx : idx + 1] = out
                v_first_ref[idx : idx + 1] = v_first_out
                block_ref._store_kv_state(slot_id, attn_shift, recurrent, ffn_shift)

            torch.testing.assert_close(output_batched, output_ref, rtol=2e-4, atol=1e-3)
            torch.testing.assert_close(
                v_first_batched, v_first_ref, rtol=2e-4, atol=1e-3
            )
            for batched_state, ref_state in zip(
                block_batched.kv_cache, block_ref.kv_cache
            ):
                torch.testing.assert_close(
                    batched_state, ref_state, rtol=2e-4, atol=1e-3
                )
        finally:
            cleanup_dist_env_and_memory()


def test_rwkv7_block_batches_prefill_tokens_without_changing_results():
    config = _make_config()
    vllm_config = VllmConfig(device_config=DeviceConfig("cpu"))
    with set_current_vllm_config(vllm_config):
        init_distributed_environment(
            world_size=1,
            rank=0,
            local_rank=0,
            distributed_init_method=f"tcp://127.0.0.1:{get_open_port()}",
            backend="gloo",
        )
        ensure_model_parallel_initialized(1, 1, backend="gloo")
        try:
            block_batched = RWKV7Block(
                config=config, layer_idx=0, prefix="model.layers.0"
            )
            _initialize_module_parameters(block_batched)
            block_ref = RWKV7Block(config=config, layer_idx=0, prefix="model.layers.1")
            block_ref.load_state_dict(block_batched.state_dict())

            generator = torch.Generator().manual_seed(321)
            state_shapes = block_batched.get_state_shape()
            state_dtypes = block_batched.get_state_dtype()

            def make_cache() -> tuple[torch.Tensor, ...]:
                return tuple(
                    torch.randn(
                        (2, *shape),
                        generator=generator,
                        dtype=dtype,
                    )
                    for shape, dtype in zip(state_shapes, state_dtypes)
                )

            block_batched.kv_cache = make_cache()
            block_ref.kv_cache = tuple(
                cache.clone() for cache in block_batched.kv_cache
            )

            query_lens = [2, 3]
            total_seq_lens = [2, 5]
            state_indices = [0, 1]
            hidden_states = torch.randn(
                sum(query_lens),
                config.hidden_size,
                generator=generator,
                dtype=torch.float32,
            )
            metadata = _make_multi_prefill_metadata(
                query_lens,
                total_seq_lens,
                state_indices,
                device=torch.device("cpu"),
            )

            output_batched, v_first_batched = block_batched(
                hidden_states, None, metadata
            )

            output_ref = torch.empty_like(hidden_states)
            v_first_ref = torch.empty_like(hidden_states)
            start = 0
            for slot_id, query_len, total_seq_len in zip(
                state_indices,
                query_lens,
                total_seq_lens,
            ):
                end = start + query_len
                states = block_ref._get_kv_state(
                    slot_id,
                    use_initial_state=total_seq_len > query_len,
                )
                out, v_first_out, attn_shift, recurrent, ffn_shift = (
                    block_ref._run_sequence(
                        hidden_states[start:end],
                        None,
                        *states,
                    )
                )
                output_ref[start:end] = out
                v_first_ref[start:end] = v_first_out
                block_ref._store_kv_state(slot_id, attn_shift, recurrent, ffn_shift)
                start = end

            torch.testing.assert_close(output_batched, output_ref)
            torch.testing.assert_close(v_first_batched, v_first_ref)
            for batched_state, ref_state in zip(
                block_batched.kv_cache, block_ref.kv_cache
            ):
                torch.testing.assert_close(batched_state, ref_state)
        finally:
            cleanup_dist_env_and_memory()


def test_rwkv7_mamba_state_copy_function_types():
    copy_funcs = RWKV7ForCausalLM.get_mamba_state_copy_func()
    assert copy_funcs == (
        get_conv_copy_spec,
        get_temporal_copy_spec,
        get_conv_copy_spec,
    )


def test_rwkv7_declares_mamba_prefix_caching_support():
    assert getattr(RWKV7ForCausalLM, "supports_mamba_prefix_caching", False) is True


def test_rwkv7_pp_runtime_uses_effective_vllm_dtype():
    dummy_model = SimpleNamespace(
        model_config=SimpleNamespace(dtype=torch.bfloat16),
        config=SimpleNamespace(torch_dtype=torch.float32),
    )

    assert RWKV7Model._get_effective_model_dtype(dummy_model) == torch.bfloat16


def test_rwkv7_pp_runtime_falls_back_to_hf_dtype_without_model_config():
    dummy_model = SimpleNamespace(
        model_config=None,
        config=SimpleNamespace(torch_dtype=torch.float32),
    )

    assert RWKV7Model._get_effective_model_dtype(dummy_model) == torch.float32


def test_rwkv7_config_allows_non_eager_when_cudagraphs_are_enabled():
    vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(
            enforce_eager=False,
            supports_mamba_prefix_caching=False,
            architecture="RWKV7ForCausalLM",
            max_model_len=2048,
        ),
        cache_config=SimpleNamespace(
            enable_prefix_caching=False,
            mamba_cache_mode="none",
            mamba_block_size=None,
            block_size=16,
        ),
        compilation_config=SimpleNamespace(cudagraph_mode=CUDAGraphMode.PIECEWISE),
        scheduler_config=SimpleNamespace(enable_chunked_prefill=True),
    )

    RWKV7ForCausalLMConfig.verify_and_update_config(vllm_config)

    assert vllm_config.model_config.enforce_eager is False


def test_rwkv7_config_allows_non_eager_when_cudagraphs_are_disabled():
    vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(
            enforce_eager=False,
            supports_mamba_prefix_caching=False,
            architecture="RWKV7ForCausalLM",
            max_model_len=2048,
        ),
        cache_config=SimpleNamespace(
            enable_prefix_caching=False,
            mamba_cache_mode="none",
            mamba_block_size=None,
            block_size=16,
        ),
        compilation_config=SimpleNamespace(cudagraph_mode=CUDAGraphMode.NONE),
        scheduler_config=SimpleNamespace(enable_chunked_prefill=True),
    )

    RWKV7ForCausalLMConfig.verify_and_update_config(vllm_config)

    assert vllm_config.model_config.enforce_eager is False


def test_rwkv7_config_defaults_mamba_cache_align_when_prefix_caching_is_enabled():
    vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(
            enforce_eager=False,
            supports_mamba_prefix_caching=True,
            architecture="RWKV7ForCausalLM",
            max_model_len=2048,
        ),
        cache_config=SimpleNamespace(
            enable_prefix_caching=True,
            mamba_cache_mode="none",
            mamba_block_size=None,
            block_size=16,
        ),
        compilation_config=SimpleNamespace(cudagraph_mode=CUDAGraphMode.PIECEWISE),
        scheduler_config=SimpleNamespace(enable_chunked_prefill=True),
    )

    RWKV7ForCausalLMConfig.verify_and_update_config(vllm_config)

    assert vllm_config.cache_config.mamba_cache_mode == "align"
    assert vllm_config.cache_config.mamba_block_size == 16


def test_rwkv7_config_preserves_explicit_mamba_cache_all():
    vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(
            enforce_eager=False,
            supports_mamba_prefix_caching=True,
            architecture="RWKV7ForCausalLM",
            max_model_len=2048,
        ),
        cache_config=SimpleNamespace(
            enable_prefix_caching=True,
            mamba_cache_mode="all",
            mamba_block_size=None,
            block_size=16,
        ),
        compilation_config=SimpleNamespace(cudagraph_mode=CUDAGraphMode.PIECEWISE),
        scheduler_config=SimpleNamespace(enable_chunked_prefill=True),
    )

    RWKV7ForCausalLMConfig.verify_and_update_config(vllm_config)

    assert vllm_config.cache_config.mamba_cache_mode == "all"
    assert vllm_config.cache_config.mamba_block_size == 16


def test_rwkv7_post_optimization_defaults_choose_piecewise():
    vllm_config = SimpleNamespace(
        compilation_config=SimpleNamespace(
            cudagraph_mode=CUDAGraphMode.FULL_AND_PIECEWISE
        )
    )

    RWKV7ForCausalLMConfig.apply_post_optimization_level_defaults(vllm_config)

    assert vllm_config.compilation_config.cudagraph_mode == CUDAGraphMode.PIECEWISE


def test_rwkv7_block_uses_fp32_runtime_state_dtype():
    config = _make_config()
    vllm_config = VllmConfig(device_config=DeviceConfig("cpu"))
    with set_current_vllm_config(vllm_config):
        init_distributed_environment(
            world_size=1,
            rank=0,
            local_rank=0,
            distributed_init_method=f"tcp://127.0.0.1:{get_open_port()}",
            backend="gloo",
        )
        ensure_model_parallel_initialized(1, 1, backend="gloo")
        try:
            block = RWKV7Block(config=config, layer_idx=0, prefix="model.layers.0")
            assert block.get_state_dtype() == (
                torch.float32,
                torch.float32,
                torch.float32,
            )
        finally:
            cleanup_dist_env_and_memory()


def test_rwkv7_block_cache_all_prefill_writes_aligned_states():
    config = _make_config()
    block_size = 8
    cache_config = CacheConfig(
        enable_prefix_caching=True,
        mamba_cache_mode="all",
        block_size=block_size,
        mamba_block_size=block_size,
    )
    vllm_config = VllmConfig(
        cache_config=cache_config,
        device_config=DeviceConfig("cpu"),
    )
    with set_current_vllm_config(vllm_config):
        init_distributed_environment(
            world_size=1,
            rank=0,
            local_rank=0,
            distributed_init_method=f"tcp://127.0.0.1:{get_open_port()}",
            backend="gloo",
        )
        ensure_model_parallel_initialized(1, 1, backend="gloo")
        try:
            block_all = RWKV7Block(
                config=config,
                layer_idx=0,
                cache_config=cache_config,
                prefix="model.layers.0",
            )
            _initialize_module_parameters(block_all)

            block_ref = RWKV7Block(
                config=config,
                layer_idx=0,
                cache_config=cache_config,
                prefix="model.layers.1",
            )
            block_ref.load_state_dict(block_all.state_dict())

            state_shapes = block_all.get_state_shape()
            state_dtypes = block_all.get_state_dtype()
            block_all.kv_cache = tuple(
                torch.zeros((3, *shape), dtype=dtype)
                for shape, dtype in zip(state_shapes, state_dtypes)
            )
            block_ref.kv_cache = tuple(cache.clone() for cache in block_all.kv_cache)

            hidden_states = torch.randn(17, config.hidden_size, dtype=torch.float32)
            metadata = _make_cache_all_prefill_metadata(
                query_len=17,
                total_seq_len=17,
                block_table=[0, 1, 2],
                num_computed_tokens=0,
                block_size=block_size,
                device=torch.device("cpu"),
            )

            output_all, v_first_all = block_all(hidden_states, None, metadata)

            output_ref = torch.empty_like(hidden_states)
            v_first_ref = torch.empty_like(hidden_states)
            state = (None, None, None)
            boundaries = [(0, 8, 0), (8, 16, 1), (16, 17, 2)]
            for start, end, slot_id in boundaries:
                out, vf_out, attn_shift, recurrent, ffn_shift = block_ref._run_sequence(
                    hidden_states[start:end],
                    None,
                    *state,
                )
                output_ref[start:end] = out
                v_first_ref[start:end] = vf_out
                block_ref._store_kv_state(slot_id, attn_shift, recurrent, ffn_shift)
                state = (attn_shift, recurrent, ffn_shift)

            torch.testing.assert_close(output_all, output_ref, rtol=2e-4, atol=1e-4)
            torch.testing.assert_close(v_first_all, v_first_ref, rtol=2e-4, atol=1e-4)
            for cached_all, cached_ref in zip(block_all.kv_cache, block_ref.kv_cache):
                torch.testing.assert_close(cached_all, cached_ref, rtol=2e-4, atol=1e-4)
        finally:
            cleanup_dist_env_and_memory()


def test_rwkv7_block_cache_all_prefill_batches_multiple_sequences():
    config = _make_config()
    block_size = 8
    cache_config = CacheConfig(
        enable_prefix_caching=True,
        mamba_cache_mode="all",
        block_size=block_size,
        mamba_block_size=block_size,
    )
    vllm_config = VllmConfig(
        cache_config=cache_config,
        device_config=DeviceConfig("cpu"),
    )
    with set_current_vllm_config(vllm_config):
        init_distributed_environment(
            world_size=1,
            rank=0,
            local_rank=0,
            distributed_init_method=f"tcp://127.0.0.1:{get_open_port()}",
            backend="gloo",
        )
        ensure_model_parallel_initialized(1, 1, backend="gloo")
        try:
            block_all = RWKV7Block(
                config=config,
                layer_idx=0,
                cache_config=cache_config,
                prefix="model.layers.0",
            )
            _initialize_module_parameters(block_all)

            block_ref = RWKV7Block(
                config=config,
                layer_idx=0,
                cache_config=cache_config,
                prefix="model.layers.1",
            )
            block_ref.load_state_dict(block_all.state_dict())

            state_shapes = block_all.get_state_shape()
            state_dtypes = block_all.get_state_dtype()
            block_all.kv_cache = tuple(
                torch.zeros((5, *shape), dtype=dtype)
                for shape, dtype in zip(state_shapes, state_dtypes)
            )
            block_ref.kv_cache = tuple(cache.clone() for cache in block_all.kv_cache)

            query_lens = [17, 10]
            total_seq_lens = [17, 10]
            block_tables = [[0, 1, 2], [3, 4, 4]]
            hidden_states = torch.randn(
                sum(query_lens), config.hidden_size, dtype=torch.float32
            )
            metadata = _make_cache_all_multi_prefill_metadata(
                query_lens=query_lens,
                total_seq_lens=total_seq_lens,
                block_tables=block_tables,
                num_computed_tokens=[0, 0],
                block_size=block_size,
                device=torch.device("cpu"),
            )

            output_all, v_first_all = block_all(hidden_states, None, metadata)

            output_ref = torch.empty_like(hidden_states)
            v_first_ref = torch.empty_like(hidden_states)
            start = 0
            ref_boundaries = [
                [(0, 8, 0), (8, 16, 1), (16, 17, 2)],
                [(0, 8, 3), (8, 10, 4)],
            ]
            for query_len, boundaries in zip(query_lens, ref_boundaries, strict=True):
                seq_hidden = hidden_states[start : start + query_len]
                state = (None, None, None)
                seq_output = torch.empty_like(seq_hidden)
                seq_v_first = torch.empty_like(seq_hidden)
                for boundary_start, boundary_end, slot_id in boundaries:
                    out, vf_out, attn_shift, recurrent, ffn_shift = (
                        block_ref._run_sequence(
                            seq_hidden[boundary_start:boundary_end],
                            None,
                            *state,
                        )
                    )
                    seq_output[boundary_start:boundary_end] = out
                    seq_v_first[boundary_start:boundary_end] = vf_out
                    block_ref._store_kv_state(slot_id, attn_shift, recurrent, ffn_shift)
                    state = (attn_shift, recurrent, ffn_shift)
                output_ref[start : start + query_len] = seq_output
                v_first_ref[start : start + query_len] = seq_v_first
                start += query_len

            torch.testing.assert_close(output_all, output_ref)
            torch.testing.assert_close(v_first_all, v_first_ref)
            for cached_all, cached_ref in zip(block_all.kv_cache, block_ref.kv_cache):
                torch.testing.assert_close(cached_all, cached_ref)
        finally:
            cleanup_dist_env_and_memory()


def test_rwkv7_block_cache_all_decode_writes_next_block_slot():
    config = _make_config()
    block_size = 8
    cache_config = CacheConfig(
        enable_prefix_caching=True,
        mamba_cache_mode="all",
        block_size=block_size,
        mamba_block_size=block_size,
    )
    vllm_config = VllmConfig(
        cache_config=cache_config,
        device_config=DeviceConfig("cpu"),
    )
    with set_current_vllm_config(vllm_config):
        init_distributed_environment(
            world_size=1,
            rank=0,
            local_rank=0,
            distributed_init_method=f"tcp://127.0.0.1:{get_open_port()}",
            backend="gloo",
        )
        ensure_model_parallel_initialized(1, 1, backend="gloo")
        try:
            block_all = RWKV7Block(
                config=config,
                layer_idx=0,
                cache_config=cache_config,
                prefix="model.layers.0",
            )
            _initialize_module_parameters(block_all)

            block_ref = RWKV7Block(
                config=config,
                layer_idx=0,
                cache_config=cache_config,
                prefix="model.layers.1",
            )
            block_ref.load_state_dict(block_all.state_dict())

            generator = torch.Generator().manual_seed(321)
            state_shapes = block_all.get_state_shape()
            state_dtypes = block_all.get_state_dtype()
            block_all.kv_cache = tuple(
                torch.randn((2, *shape), generator=generator, dtype=dtype)
                for shape, dtype in zip(state_shapes, state_dtypes)
            )
            block_ref.kv_cache = tuple(cache.clone() for cache in block_all.kv_cache)

            decode_hidden = torch.randn(1, config.hidden_size, generator=generator)
            metadata = _make_cache_all_decode_metadata(
                total_seq_len=9,
                block_table=[0, 1],
                num_computed_tokens=8,
                block_size=block_size,
                device=torch.device("cpu"),
            )

            output_all, v_first_all = block_all(decode_hidden, None, metadata)

            states = block_ref._get_kv_state(0, use_initial_state=True)
            output_ref, v_first_ref, attn_shift, recurrent, ffn_shift = (
                block_ref._run_sequence(decode_hidden, None, *states)
            )
            block_ref._store_kv_state(1, attn_shift, recurrent, ffn_shift)

            torch.testing.assert_close(output_all, output_ref)
            torch.testing.assert_close(v_first_all, v_first_ref)
            for cached_all, cached_ref in zip(block_all.kv_cache, block_ref.kv_cache):
                torch.testing.assert_close(cached_all, cached_ref)
        finally:
            cleanup_dist_env_and_memory()


def test_rwkv7_reference_parity_full_forward():
    if pytest is None:
        raise RuntimeError("pytest is required to run RWKV7 integration tests.")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for RWKV7 reference parity tests.")

    model_path, reference_cls = _require_reference_checkpoint()
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    inputs = tokenizer("Hello RWKV7, this is a parity check.", return_tensors="pt")[
        "input_ids"
    ].to("cuda")
    flat_input_ids = inputs[0]
    positions = torch.arange(flat_input_ids.numel(), device="cuda", dtype=torch.long)

    reference_model = (
        reference_cls.from_pretrained(model_path, dtype=torch.float32).eval().to("cuda")
    )
    vllm_config = _make_vllm_config(model_path)

    with set_current_vllm_config(vllm_config):
        init_distributed_environment(
            world_size=1,
            rank=0,
            local_rank=0,
            distributed_init_method=f"tcp://127.0.0.1:{get_open_port()}",
            backend="nccl",
        )
        ensure_model_parallel_initialized(1, 1, backend="nccl")
        try:
            vllm_model = RWKV7ForCausalLM(vllm_config=vllm_config)
            vllm_model.load_weights(reference_model.state_dict().items())
            vllm_model = vllm_model.eval().to("cuda", torch.float32)

            with torch.no_grad():
                reference_outputs = reference_model(
                    input_ids=inputs,
                    use_cache=False,
                )
                reference_hidden = reference_model.model(
                    input_ids=inputs,
                    use_cache=False,
                )[0][0]

            with torch.no_grad(), set_forward_context(None, vllm_config):
                hidden_states = vllm_model(
                    input_ids=flat_input_ids,
                    positions=positions,
                )
                logits = vllm_model.compute_logits(hidden_states)

            hidden_diff = (hidden_states - reference_hidden).abs()
            logits_diff = (logits - reference_outputs.logits[0]).abs()
            assert hidden_diff.max().item() < 5e-5
            assert logits_diff.max().item() < 5e-5
        finally:
            cleanup_dist_env_and_memory()


def test_rwkv7_reference_parity_prefill_decode():
    if pytest is None:
        raise RuntimeError("pytest is required to run RWKV7 integration tests.")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for RWKV7 reference parity tests.")

    model_path, reference_cls = _require_reference_checkpoint()
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    prompt_ids = tokenizer("The capital of France is", return_tensors="pt")[
        "input_ids"
    ].to("cuda")

    reference_model = (
        reference_cls.from_pretrained(model_path, dtype=torch.float32).eval().to("cuda")
    )
    vllm_config = _make_vllm_config(model_path)

    with set_current_vllm_config(vllm_config):
        init_distributed_environment(
            world_size=1,
            rank=0,
            local_rank=0,
            distributed_init_method=f"tcp://127.0.0.1:{get_open_port()}",
            backend="nccl",
        )
        ensure_model_parallel_initialized(1, 1, backend="nccl")
        try:
            vllm_model = RWKV7ForCausalLM(vllm_config=vllm_config)
            vllm_model.load_weights(reference_model.state_dict().items())
            vllm_model = vllm_model.eval().to("cuda", torch.float32)
            _allocate_kv_cache(vllm_model, device=torch.device("cuda"))

            prompt_flat = prompt_ids[0]
            prompt_positions = torch.arange(
                prompt_flat.numel(), device="cuda", dtype=torch.long
            )
            prompt_metadata = {
                layer.prefix: _make_prefill_metadata(
                    prompt_ids.shape[1], device=torch.device("cuda")
                )
                for layer in vllm_model.model.layers
            }

            with torch.no_grad():
                reference_prompt_logits = reference_model(
                    input_ids=prompt_ids, use_cache=False
                ).logits[0]

            with torch.no_grad(), set_forward_context(prompt_metadata, vllm_config):
                hidden_states = vllm_model(
                    input_ids=prompt_flat,
                    positions=prompt_positions,
                )
                logits = vllm_model.compute_logits(hidden_states)

            assert (logits - reference_prompt_logits).abs().max().item() < 5e-5

            next_token = logits[-1].argmax().view(1)
            reference_first_token = reference_prompt_logits[-1].argmax().view(1)
            assert int(next_token.item()) == int(reference_first_token.item())

            generated_vllm = [int(next_token.item())]
            generated_ref = [int(reference_first_token.item())]
            current_ids = prompt_ids.clone()

            for _ in range(3):
                total_seq_len = current_ids.shape[1] + 1
                decode_metadata = {
                    layer.prefix: _make_decode_metadata(
                        total_seq_len, device=torch.device("cuda")
                    )
                    for layer in vllm_model.model.layers
                }
                position = torch.tensor(
                    [current_ids.shape[1]], device="cuda", dtype=torch.long
                )

                with torch.no_grad(), set_forward_context(decode_metadata, vllm_config):
                    hidden_states = vllm_model(
                        input_ids=next_token,
                        positions=position,
                    )
                    logits = vllm_model.compute_logits(hidden_states)

                full_ids = torch.cat([current_ids, next_token.view(1, 1)], dim=1)
                with torch.no_grad():
                    reference_last_logits = reference_model(
                        input_ids=full_ids,
                        use_cache=False,
                    ).logits[0, -1]

                assert (logits[-1] - reference_last_logits).abs().max().item() < 5e-5

                next_token = logits[-1].argmax().view(1)
                reference_next_token = reference_last_logits.argmax().view(1)
                assert int(next_token.item()) == int(reference_next_token.item())

                generated_vllm.append(int(next_token.item()))
                generated_ref.append(int(reference_next_token.item()))
                current_ids = full_ids

            assert generated_vllm == generated_ref
            assert tokenizer.decode(generated_vllm) == tokenizer.decode(generated_ref)
        finally:
            cleanup_dist_env_and_memory()
