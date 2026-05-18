# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import argparse
import gc
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from vllm import LLM, SamplingParams
from vllm.tokenizers import get_tokenizer

LOG_DECAY_SCALE = -0.6065306597126334
DEFAULT_PROMPTS = [
    "The capital of France is",
    "北京是中国的首都。下面一句继续：",
    "User: hello\n\nAssistant:",
    "1 + 1 =",
]


def accelerator_available() -> bool:
    return hasattr(torch, "accelerator") and torch.accelerator.is_available()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Standalone native RWKV7 .pt/.pth greedy generation without the "
            "vLLM engine. Useful for isolating model-vs-runtime issues."
        )
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument(
        "--device", default="cuda" if accelerator_available() else "cpu"
    )
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--topk", type=int, default=8)
    parser.add_argument("--compare-vllm-tokenizer", action="store_true")
    parser.add_argument("--vllm-tokenizer-mode", default="auto")
    parser.add_argument("--compare-vllm-engine", action="store_true")
    parser.add_argument("--max-model-len", type=int, default=2048)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.6)
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--hf-config-path")
    parser.add_argument("--hf-overrides", default="{}")
    parser.add_argument(
        "--apply-chat-template",
        action="store_true",
        help=(
            "Render each --prompt as a chat message with the tokenizer's chat "
            "template before tokenization."
        ),
    )
    parser.add_argument(
        "--message-role",
        default="user",
        choices=("system", "user", "assistant", "tool"),
        help=("Role assigned to each --prompt when --apply-chat-template is enabled."),
    )
    parser.add_argument(
        "--system-prompt",
        help=(
            "Optional system message to prepend when --apply-chat-template is enabled."
        ),
    )
    parser.add_argument(
        "--no-add-generation-prompt",
        action="store_true",
        help=(
            "Disable add_generation_prompt when applying the tokenizer chat template."
        ),
    )
    parser.add_argument(
        "--text-report",
        action="store_true",
        help="Print a human-readable report in addition to the JSON lines.",
    )
    parser.add_argument(
        "--prompt",
        action="append",
        dest="prompts",
        help="Add a prompt. Can be specified multiple times.",
    )
    return parser.parse_args()


def print_json(tag: str, payload: dict[str, Any]) -> None:
    print(f"{tag}=" + json.dumps(payload, ensure_ascii=False))


def print_section(title: str, body: str) -> None:
    print(f"{title}_BEGIN")
    print(body)
    print(f"{title}_END")


def format_id_preview(token_ids: list[int], *, limit: int = 64) -> str:
    if len(token_ids) <= limit:
        return json.dumps(token_ids, ensure_ascii=False)
    preview = json.dumps(token_ids[:limit], ensure_ascii=False)
    return f"{preview} ... (+{len(token_ids) - limit} more)"


def print_text_report(result: dict[str, Any], *, index: int) -> None:
    print(f"=== Prompt {index} ===")
    print(f"raw_prompt={result['prompt']!r}")
    print(f"used_chat_template={result['used_chat_template']}")
    if result.get("messages") is not None:
        print(
            "messages_json="
            + json.dumps(result["messages"], ensure_ascii=False, separators=(",", ":"))
        )
    print_section("RENDERED_PROMPT", str(result["rendered_prompt"]))
    print(f"prompt_token_count={len(result['prompt_ids'])}")
    print("prompt_ids=" + format_id_preview(result["prompt_ids"]))
    print("first_step_topk:")
    for rank, row in enumerate(result["first_step_topk"], start=1):
        print(
            f"  {rank}. id={row['token_id']} token={row['token']!r} "
            f"decoded={row['decoded']!r} logit={row['logit']:.4f}"
        )
    print_section("GENERATED_TEXT", str(result["generated_text"]))
    print("generated_ids=" + format_id_preview(result["generated_ids"]))


def parse_json_dict(raw: str) -> dict[str, Any]:
    parsed = json.loads(raw)
    if not isinstance(parsed, dict):
        raise ValueError(f"Expected JSON object, got {type(parsed).__name__}")
    return parsed


def resolve_dtype(dtype_name: str) -> torch.dtype:
    lowered = dtype_name.lower()
    if lowered in {"fp32", "float32"}:
        return torch.float32
    if lowered in {"fp16", "float16", "half"}:
        return torch.float16
    if lowered in {"bf16", "bfloat16"}:
        return torch.bfloat16
    raise ValueError(f"Unsupported dtype: {dtype_name}")


def load_checkpoint(path: Path) -> dict[str, torch.Tensor]:
    loaded = torch.load(path, map_location="cpu", weights_only=True)
    if isinstance(loaded, dict):
        for key in ("state_dict", "model", "weights"):
            maybe_nested = loaded.get(key)
            if isinstance(maybe_nested, dict):
                loaded = maybe_nested
                break
    if not isinstance(loaded, dict):
        raise TypeError(f"Unexpected checkpoint type: {type(loaded).__name__}")

    state_dict: dict[str, torch.Tensor] = {}
    for key, value in loaded.items():
        if isinstance(value, torch.Tensor):
            state_dict[key] = value

    required = {
        "emb.weight",
        "ln_out.weight",
        "head.weight",
        "blocks.0.att.receptance.weight",
        "blocks.0.ffn.key.weight",
    }
    missing = sorted(required - state_dict.keys())
    if missing:
        raise KeyError(
            f"Checkpoint does not look like native RWKV7. Missing: {missing}"
        )
    return state_dict


def maybe_to_dtype(tensor: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    if torch.is_floating_point(tensor):
        return tensor.to(dtype)
    return tensor


def layer_norm_1d(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    eps: float = 1e-5,
) -> torch.Tensor:
    x32 = x.to(torch.float32)
    y = F.layer_norm(
        x32,
        normalized_shape=(x32.shape[-1],),
        weight=weight.to(torch.float32),
        bias=None if bias is None else bias.to(torch.float32),
        eps=eps,
    )
    return y.to(x.dtype)


def sqrelu(x: torch.Tensor) -> torch.Tensor:
    return torch.relu(x).square()


def act_fn(name: str):
    if name == "sqrelu":
        return sqrelu
    raise ValueError(f"Unsupported activation: {name}")


def lora_forward(
    x: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    bias: torch.Tensor | None,
    activation: str | None,
) -> torch.Tensor:
    hidden = F.linear(x, w1.transpose(0, 1))
    if activation == "tanh":
        hidden = torch.tanh(hidden)
    elif activation == "sigmoid":
        hidden = torch.sigmoid(hidden)
    elif activation is None:
        pass
    else:
        raise ValueError(f"Unsupported LoRA activation: {activation}")
    return F.linear(hidden, w2.transpose(0, 1), bias)


def group_norm_token(
    x: torch.Tensor,
    num_heads: int,
    head_dim: int,
    weight: torch.Tensor,
    bias: torch.Tensor,
    eps: float = 1e-5,
) -> torch.Tensor:
    x32 = x.to(torch.float32).view(1, -1, 1)
    y = F.group_norm(
        x32,
        num_groups=num_heads,
        weight=weight.to(torch.float32),
        bias=bias.to(torch.float32),
        eps=head_dim * eps,
    )
    return y.view(-1).to(x.dtype)


@dataclass
class LayerState:
    att_shift: torch.Tensor | None
    recurrent: torch.Tensor | None
    ffn_shift: torch.Tensor | None


@dataclass
class LayerConfig:
    layer_idx: int
    hidden_size: int
    num_heads: int
    head_dim: int
    value_dim: int
    head_v_dim: int
    norm_eps: float = 1e-5


class NativeRWKV7:
    def __init__(
        self,
        state_dict: dict[str, torch.Tensor],
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        self.device = device
        self.dtype = dtype
        self.state_dict = {
            key: maybe_to_dtype(value.to(device), dtype)
            for key, value in state_dict.items()
        }
        self.hidden_size = int(self.state_dict["emb.weight"].shape[1])
        self.vocab_size = int(self.state_dict["emb.weight"].shape[0])
        self.num_layers = (
            max(
                int(key.split(".")[1])
                for key in self.state_dict
                if key.startswith("blocks.")
            )
            + 1
        )
        self.layers = [self._infer_layer_config(i) for i in range(self.num_layers)]

    def _infer_layer_config(self, layer_idx: int) -> LayerConfig:
        prefix = f"blocks.{layer_idx}.att."
        r_k = self.state_dict[f"{prefix}r_k"]
        v_weight = self.state_dict[f"{prefix}value.weight"]
        num_heads = int(r_k.shape[0])
        head_dim = int(r_k.shape[1])
        value_dim = int(v_weight.shape[0])
        if value_dim % num_heads != 0:
            raise ValueError(
                "Layer "
                f"{layer_idx} value_dim={value_dim} not divisible by "
                f"num_heads={num_heads}"
            )
        return LayerConfig(
            layer_idx=layer_idx,
            hidden_size=self.hidden_size,
            num_heads=num_heads,
            head_dim=head_dim,
            value_dim=value_dim,
            head_v_dim=value_dim // num_heads,
        )

    def init_states(self) -> list[LayerState]:
        return [LayerState(None, None, None) for _ in range(self.num_layers)]

    def _block_prefix(self, layer_idx: int) -> str:
        return f"blocks.{layer_idx}"

    def _att(
        self,
        layer: LayerConfig,
        x: torch.Tensor,
        state: LayerState,
        v_first: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, LayerState]:
        prefix = self._block_prefix(layer.layer_idx)

        att_shift = state.att_shift
        delta = -x if att_shift is None else att_shift.to(x.dtype) - x
        final_att_shift = x

        x_r = self.state_dict[f"{prefix}.att.x_r"].view(-1)
        x_w = self.state_dict[f"{prefix}.att.x_w"].view(-1)
        x_k = self.state_dict[f"{prefix}.att.x_k"].view(-1)
        x_v = self.state_dict[f"{prefix}.att.x_v"].view(-1)
        x_a = self.state_dict[f"{prefix}.att.x_a"].view(-1)
        x_g = self.state_dict[f"{prefix}.att.x_g"].view(-1)

        xr = x + delta * x_r
        xw = x + delta * x_w
        xk = x + delta * x_k
        xv = x + delta * x_v
        xa = x + delta * x_a
        xg = x + delta * x_g

        r = F.linear(xr, self.state_dict[f"{prefix}.att.receptance.weight"])
        w = LOG_DECAY_SCALE * torch.sigmoid(
            lora_forward(
                xw,
                self.state_dict[f"{prefix}.att.w1"],
                self.state_dict[f"{prefix}.att.w2"],
                self.state_dict[f"{prefix}.att.w0"].reshape(-1),
                "tanh",
            )
        )
        k = F.linear(xk, self.state_dict[f"{prefix}.att.key.weight"])
        v = F.linear(xv, self.state_dict[f"{prefix}.att.value.weight"])

        if layer.layer_idx == 0:
            v_first_out = v
        else:
            if v_first is None:
                raise ValueError(f"Layer {layer.layer_idx} requires v_first.")
            v_mix = torch.sigmoid(
                lora_forward(
                    xv,
                    self.state_dict[f"{prefix}.att.v1"],
                    self.state_dict[f"{prefix}.att.v2"],
                    self.state_dict[f"{prefix}.att.v0"].reshape(-1),
                    None,
                )
            )
            v = torch.lerp(v, v_first, v_mix)
            v_first_out = v_first

        a = torch.sigmoid(
            lora_forward(
                xa,
                self.state_dict[f"{prefix}.att.a1"],
                self.state_dict[f"{prefix}.att.a2"],
                self.state_dict[f"{prefix}.att.a0"].reshape(-1),
                None,
            )
        )
        g = lora_forward(
            xg,
            self.state_dict[f"{prefix}.att.g1"],
            self.state_dict[f"{prefix}.att.g2"],
            None,
            "sigmoid",
        )

        r = r.view(layer.num_heads, layer.head_dim).to(torch.float32)
        w = w.view(layer.num_heads, layer.head_dim).to(torch.float32)
        k = k.view(layer.num_heads, layer.head_dim).to(torch.float32)
        a = a.view(layer.num_heads, layer.head_dim).to(torch.float32)
        v = v.view(layer.num_heads, layer.head_v_dim).to(torch.float32)

        k_k = self.state_dict[f"{prefix}.att.k_k"].view(layer.num_heads, layer.head_dim)
        k_a = self.state_dict[f"{prefix}.att.k_a"].view(layer.num_heads, layer.head_dim)
        kk = F.normalize(k * k_k.to(torch.float32), dim=-1, p=2.0)
        k = k * (1 + (a - 1) * k_a.to(torch.float32))

        recurrent = state.recurrent
        if recurrent is None:
            recurrent = torch.zeros(
                layer.num_heads,
                layer.head_dim,
                layer.head_v_dim,
                device=self.device,
                dtype=torch.float32,
            )
        else:
            recurrent = recurrent.to(torch.float32)

        sa = (recurrent * (-kk).unsqueeze(-1)).sum(dim=-2)
        recurrent = (
            torch.exp(w).unsqueeze(-1) * recurrent
            + (kk * a).unsqueeze(-1) * sa.unsqueeze(-2)
            + k.unsqueeze(-1) * v.unsqueeze(-2)
        )
        recurrent_output = (recurrent * r.unsqueeze(-1)).sum(dim=-2)
        output = recurrent_output.reshape(-1).to(self.dtype)

        output = group_norm_token(
            output,
            num_heads=layer.num_heads,
            head_dim=layer.head_dim,
            weight=self.state_dict[f"{prefix}.att.ln_x.weight"],
            bias=self.state_dict[f"{prefix}.att.ln_x.bias"],
            eps=layer.norm_eps,
        )
        r_k = self.state_dict[f"{prefix}.att.r_k"].to(torch.float32)
        correction = ((r * k * r_k).sum(dim=-1, keepdim=True) * v).reshape(-1)
        output = (output.to(torch.float32) + correction) * g.to(torch.float32)
        output = output.to(self.dtype)
        output = F.linear(output, self.state_dict[f"{prefix}.att.output.weight"])
        next_state = LayerState(
            att_shift=final_att_shift,
            recurrent=recurrent,
            ffn_shift=state.ffn_shift,
        )
        return output, v_first_out, next_state

    def _ffn(
        self, layer_idx: int, x: torch.Tensor, state: LayerState
    ) -> tuple[torch.Tensor, LayerState]:
        prefix = self._block_prefix(layer_idx)
        ffn_shift = state.ffn_shift
        delta = -x if ffn_shift is None else ffn_shift.to(x.dtype) - x
        mixed = x + delta * self.state_dict[f"{prefix}.ffn.x_k"]
        hidden = F.linear(mixed, self.state_dict[f"{prefix}.ffn.key.weight"])
        hidden = act_fn("sqrelu")(hidden)
        out = F.linear(hidden, self.state_dict[f"{prefix}.ffn.value.weight"])
        next_state = LayerState(
            att_shift=state.att_shift,
            recurrent=state.recurrent,
            ffn_shift=x,
        )
        return out, next_state

    def forward_token(
        self,
        token_id: int,
        states: list[LayerState],
    ) -> tuple[torch.Tensor, list[LayerState]]:
        hidden = self.state_dict["emb.weight"][token_id].to(self.dtype)
        v_first: torch.Tensor | None = None
        next_states: list[LayerState] = []

        for layer in self.layers:
            prefix = self._block_prefix(layer.layer_idx)
            residual = hidden
            if layer.layer_idx == 0 and f"{prefix}.ln0.weight" in self.state_dict:
                residual = layer_norm_1d(
                    residual,
                    self.state_dict[f"{prefix}.ln0.weight"],
                    self.state_dict.get(f"{prefix}.ln0.bias"),
                    eps=layer.norm_eps,
                )

            attn_input = layer_norm_1d(
                residual,
                self.state_dict[f"{prefix}.ln1.weight"],
                self.state_dict.get(f"{prefix}.ln1.bias"),
                eps=layer.norm_eps,
            )
            attn_out, v_first, state_after_attn = self._att(
                layer,
                attn_input,
                states[layer.layer_idx],
                v_first,
            )
            hidden = residual + attn_out

            ffn_input = layer_norm_1d(
                hidden,
                self.state_dict[f"{prefix}.ln2.weight"],
                self.state_dict.get(f"{prefix}.ln2.bias"),
                eps=layer.norm_eps,
            )
            ffn_out, next_state = self._ffn(
                layer.layer_idx,
                ffn_input,
                state_after_attn,
            )
            hidden = hidden + ffn_out
            next_states.append(next_state)

        hidden = layer_norm_1d(
            hidden,
            self.state_dict["ln_out.weight"],
            self.state_dict.get("ln_out.bias"),
            eps=1e-5,
        )
        logits = F.linear(hidden, self.state_dict["head.weight"].to(hidden.dtype))
        return logits.reshape(-1).to(torch.float32), next_states


def topk_tokens(
    tokenizer: Any,
    logits: torch.Tensor,
    topk: int,
) -> list[dict[str, Any]]:
    logits = logits.reshape(-1)
    values, indices = torch.topk(logits, k=min(topk, logits.shape[-1]))
    rows = []
    for score, token_id in zip(values.tolist(), indices.tolist(), strict=True):
        rows.append(
            {
                "token_id": int(token_id),
                "token": tokenizer.convert_ids_to_tokens(int(token_id)),
                "decoded": tokenizer.decode([int(token_id)], skip_special_tokens=False),
                "logit": float(score),
            }
        )
    return rows


def eos_id_set(tokenizer: Any) -> set[int]:
    eos = getattr(tokenizer, "eos_token_id", None)
    if eos is None:
        return set()
    if isinstance(eos, int):
        return {eos}
    return {int(x) for x in eos}


def prepare_prompt_inputs(
    tokenizer: Any,
    prompt: str,
    *,
    apply_chat_template: bool,
    message_role: str,
    system_prompt: str | None,
    add_generation_prompt: bool,
) -> dict[str, Any]:
    if not apply_chat_template:
        return {
            "prompt": prompt,
            "rendered_prompt": prompt,
            "prompt_ids": list(tokenizer.encode(prompt, add_special_tokens=False)),
            "messages": None,
            "used_chat_template": False,
        }

    messages: list[dict[str, str]] = []
    if system_prompt is not None:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": message_role, "content": prompt})

    if not hasattr(tokenizer, "apply_chat_template"):
        raise ValueError(
            "Tokenizer "
            f"{type(tokenizer).__name__} does not support apply_chat_template()."
        )

    try:
        rendered_prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
        )
        prompt_ids = list(
            tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=add_generation_prompt,
            )
        )
    except Exception as exc:
        raise ValueError(
            "Failed to apply chat template with tokenizer "
            f"{type(tokenizer).__name__}: {exc}"
        ) from exc

    return {
        "prompt": prompt,
        "rendered_prompt": rendered_prompt,
        "prompt_ids": prompt_ids,
        "messages": messages,
        "used_chat_template": True,
    }


def compare_with_vllm_tokenizer(
    tokenizer_path: str,
    prompts: list[str],
    *,
    tokenizer_mode: str,
    apply_chat_template: bool,
    message_role: str,
    system_prompt: str | None,
    add_generation_prompt: bool,
) -> dict[str, Any]:
    hf_tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path,
        trust_remote_code=True,
    )
    vllm_tokenizer = get_tokenizer(
        tokenizer_path,
        tokenizer_mode=tokenizer_mode,
        trust_remote_code=True,
    )

    rows = []
    all_same_ids = True
    for prompt in prompts:
        hf_prepared = prepare_prompt_inputs(
            hf_tokenizer,
            prompt,
            apply_chat_template=apply_chat_template,
            message_role=message_role,
            system_prompt=system_prompt,
            add_generation_prompt=add_generation_prompt,
        )
        vllm_prepared = prepare_prompt_inputs(
            vllm_tokenizer,
            prompt,
            apply_chat_template=apply_chat_template,
            message_role=message_role,
            system_prompt=system_prompt,
            add_generation_prompt=add_generation_prompt,
        )
        hf_ids = list(hf_prepared["prompt_ids"])
        vllm_ids = list(vllm_prepared["prompt_ids"])
        same_ids = hf_ids == vllm_ids
        all_same_ids = all_same_ids and same_ids
        rows.append(
            {
                "prompt": prompt,
                "used_chat_template": apply_chat_template,
                "same_ids": same_ids,
                "same_rendered_prompt": (
                    hf_prepared["rendered_prompt"] == vllm_prepared["rendered_prompt"]
                ),
                "hf_ids": hf_ids,
                "vllm_ids": vllm_ids,
                "hf_rendered_prompt": hf_prepared["rendered_prompt"],
                "vllm_rendered_prompt": vllm_prepared["rendered_prompt"],
                "hf_decoded": hf_tokenizer.decode(
                    hf_ids,
                    skip_special_tokens=False,
                ),
                "vllm_decoded": vllm_tokenizer.decode(
                    vllm_ids,
                    skip_special_tokens=False,
                ),
            }
        )

    return {
        "tokenizer_path": tokenizer_path,
        "tokenizer_mode": tokenizer_mode,
        "all_same_ids": all_same_ids,
        "vllm_is_fast": getattr(vllm_tokenizer, "is_fast", None),
        "rows": rows,
    }


def run_prompt(
    model: NativeRWKV7,
    tokenizer: Any,
    prompt: str,
    *,
    max_new_tokens: int,
    topk: int,
    apply_chat_template: bool,
    message_role: str,
    system_prompt: str | None,
    add_generation_prompt: bool,
) -> dict[str, Any]:
    prepared = prepare_prompt_inputs(
        tokenizer,
        prompt,
        apply_chat_template=apply_chat_template,
        message_role=message_role,
        system_prompt=system_prompt,
        add_generation_prompt=add_generation_prompt,
    )
    prompt_ids = list(prepared["prompt_ids"])
    if not prompt_ids:
        raise ValueError("Empty prompt after tokenization is not supported.")

    states = model.init_states()
    logits = None
    with torch.inference_mode():
        for token_id in prompt_ids:
            logits, states = model.forward_token(token_id, states)
        assert logits is not None

        first_step_topk = topk_tokens(tokenizer, logits, topk)
        generated_ids: list[int] = []
        stop_ids = eos_id_set(tokenizer)

        for _ in range(max_new_tokens):
            next_token = int(torch.argmax(logits.reshape(-1)).item())
            generated_ids.append(next_token)
            if next_token in stop_ids:
                break
            logits, states = model.forward_token(next_token, states)

    return {
        **prepared,
        "prompt_ids": prompt_ids,
        "first_step_topk": first_step_topk,
        "generated_ids": generated_ids,
        "generated_tokens": tokenizer.convert_ids_to_tokens(generated_ids),
        "generated_text": tokenizer.decode(generated_ids, skip_special_tokens=False),
    }


def run_vllm_generation(
    *,
    model: str,
    tokenizer: str,
    tokenizer_mode: str,
    prompts: list[str],
    max_new_tokens: int,
    max_model_len: int,
    gpu_memory_utilization: float,
    enforce_eager: bool,
    hf_config_path: str | None,
    hf_overrides: dict[str, Any],
) -> list[dict[str, Any]]:
    llm = LLM(
        model=model,
        tokenizer=tokenizer,
        tokenizer_mode=tokenizer_mode,
        trust_remote_code=True,
        max_model_len=max_model_len,
        gpu_memory_utilization=gpu_memory_utilization,
        enforce_eager=enforce_eager,
        hf_config_path=hf_config_path,
        hf_overrides=hf_overrides,
        max_num_seqs=len(prompts),
        disable_log_stats=True,
    )
    sampling = SamplingParams(
        temperature=0.0,
        top_p=1.0,
        max_tokens=max_new_tokens,
        seed=0,
        detokenize=True,
        skip_special_tokens=False,
        spaces_between_special_tokens=False,
    )
    outputs = llm.generate(prompts, sampling, use_tqdm=False)
    rows = []
    for prompt, output in zip(prompts, outputs, strict=True):
        generated = output.outputs[0]
        rows.append(
            {
                "prompt": prompt,
                "generated_ids": list(generated.token_ids),
                "generated_text": generated.text,
            }
        )

    del llm
    gc.collect()
    if accelerator_available():
        torch.accelerator.empty_cache()
    return rows


def diff_token_sequences(
    pure_rows: list[dict[str, Any]],
    vllm_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    by_prompt = {row["prompt"]: row for row in vllm_rows}
    rows = []
    all_exact_match = True
    for pure_row in pure_rows:
        prompt = pure_row["prompt"]
        vllm_row = by_prompt[prompt]
        pure_ids = pure_row["generated_ids"]
        vllm_ids = vllm_row["generated_ids"]
        divergence_index = None
        shared_len = min(len(pure_ids), len(vllm_ids))
        for idx in range(shared_len):
            if pure_ids[idx] != vllm_ids[idx]:
                divergence_index = idx
                break
        if divergence_index is None and len(pure_ids) != len(vllm_ids):
            divergence_index = shared_len
        exact_match = divergence_index is None
        all_exact_match = all_exact_match and exact_match
        rows.append(
            {
                "prompt": prompt,
                "exact_match": exact_match,
                "divergence_index": divergence_index,
                "pure_ids": pure_ids,
                "vllm_ids": vllm_ids,
                "pure_text": pure_row["generated_text"],
                "vllm_text": vllm_row["generated_text"],
            }
        )
    return {
        "all_exact_match": all_exact_match,
        "rows": rows,
    }


def main() -> None:
    args = parse_args()
    prompts = args.prompts or DEFAULT_PROMPTS
    dtype = resolve_dtype(args.dtype)
    device = torch.device(args.device)
    hf_overrides = parse_json_dict(args.hf_overrides)
    add_generation_prompt = not args.no_add_generation_prompt

    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer,
        trust_remote_code=True,
    )
    if args.compare_vllm_tokenizer:
        tokenizer_compare = compare_with_vllm_tokenizer(
            args.tokenizer,
            prompts,
            tokenizer_mode=args.vllm_tokenizer_mode,
            apply_chat_template=args.apply_chat_template,
            message_role=args.message_role,
            system_prompt=args.system_prompt,
            add_generation_prompt=add_generation_prompt,
        )
        print_json("TOKENIZER_COMPARE_JSON", tokenizer_compare)
    checkpoint = load_checkpoint(Path(args.model))
    model = NativeRWKV7(
        checkpoint,
        device=device,
        dtype=dtype,
    )

    summary = {
        "model": args.model,
        "tokenizer": args.tokenizer,
        "device": str(device),
        "dtype": str(dtype),
        "vocab_size": model.vocab_size,
        "hidden_size": model.hidden_size,
        "num_layers": model.num_layers,
        "tokenizer_full_vocab_size": len(tokenizer),
        "special_ids": {
            "bos_token_id": getattr(tokenizer, "bos_token_id", None),
            "eos_token_id": getattr(tokenizer, "eos_token_id", None),
            "pad_token_id": getattr(tokenizer, "pad_token_id", None),
        },
        "apply_chat_template": args.apply_chat_template,
        "message_role": args.message_role,
        "system_prompt": args.system_prompt,
        "add_generation_prompt": add_generation_prompt,
    }
    print_json("STANDALONE_MODEL_SUMMARY_JSON", summary)

    pure_results = []
    for index, prompt in enumerate(prompts, start=1):
        result = run_prompt(
            model,
            tokenizer,
            prompt,
            max_new_tokens=args.max_new_tokens,
            topk=args.topk,
            apply_chat_template=args.apply_chat_template,
            message_role=args.message_role,
            system_prompt=args.system_prompt,
            add_generation_prompt=add_generation_prompt,
        )
        pure_results.append(result)
        print_json("STANDALONE_GENERATE_JSON", result)
        if args.text_report:
            print_text_report(result, index=index)

    if not args.compare_vllm_engine:
        return

    del model
    del checkpoint
    gc.collect()
    if accelerator_available():
        torch.accelerator.empty_cache()

    vllm_results = run_vllm_generation(
        model=args.model,
        tokenizer=args.tokenizer,
        tokenizer_mode=args.vllm_tokenizer_mode,
        prompts=[str(result["rendered_prompt"]) for result in pure_results],
        max_new_tokens=args.max_new_tokens,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enforce_eager=args.enforce_eager,
        hf_config_path=args.hf_config_path,
        hf_overrides=hf_overrides,
    )
    print_json(
        "VLLM_GENERATE_JSON",
        {
            "tokenizer_mode": args.vllm_tokenizer_mode,
            "rows": vllm_results,
        },
    )
    print_json(
        "TORCH_VS_VLLM_DIFF_JSON",
        diff_token_sequences(pure_results, vllm_results),
    )


if __name__ == "__main__":
    main()
