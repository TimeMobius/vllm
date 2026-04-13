import argparse
import json
from pathlib import Path

import torch
from safetensors.torch import load_file
from transformers import AutoTokenizer

from vllm.config import (
    CacheConfig,
    CompilationConfig,
    CUDAGraphMode,
    DeviceConfig,
    ModelConfig,
    ParallelConfig,
    VllmConfig,
    set_current_vllm_config,
)
from vllm.config.compilation import CompilationMode
from vllm.distributed import cleanup_dist_env_and_memory
from vllm.distributed.parallel_state import (
    ensure_model_parallel_initialized,
    init_distributed_environment,
)
from vllm.forward_context import set_forward_context
from vllm.model_executor.models.rwkv7 import RWKV7ForCausalLM
from vllm.utils.network_utils import get_open_port
from vllm.v1.attention.backends.linear_attn import LinearAttentionMetadata


def make_prefill_metadata(
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


def allocate_kv_cache(model: RWKV7ForCausalLM, *, device: torch.device) -> None:
    for layer in model.model.layers:
        state_shapes = layer.get_state_shape()
        state_dtypes = layer.get_state_dtype()
        layer.kv_cache = tuple(
            torch.zeros((1, *shape), dtype=dtype, device=device)
            for shape, dtype in zip(state_shapes, state_dtypes)
        )


def make_vllm_config(
    model_path: str,
    *,
    dtype: str,
    compiled: bool,
) -> VllmConfig:
    vllm_config = VllmConfig(
        model_config=ModelConfig(
            model_path,
            trust_remote_code=True,
            dtype=dtype,
            runner="generate",
            enforce_eager=not compiled,
        ),
        parallel_config=ParallelConfig(
            tensor_parallel_size=1,
            pipeline_parallel_size=1,
        ),
        cache_config=CacheConfig(),
        device_config=DeviceConfig("cuda"),
        compilation_config=CompilationConfig(
            mode=(
                CompilationMode.VLLM_COMPILE
                if compiled
                else CompilationMode.NONE
            ),
            cudagraph_mode=CUDAGraphMode.NONE,
            cudagraph_copy_inputs=False,
        ),
    )
    vllm_config.model_config.enforce_eager = not compiled
    vllm_config.compilation_config.mode = (
        CompilationMode.VLLM_COMPILE if compiled else CompilationMode.NONE
    )
    vllm_config.compilation_config.cudagraph_mode = CUDAGraphMode.NONE
    vllm_config.compilation_config.cudagraph_copy_inputs = False
    return vllm_config


def max_abs_diff(a: torch.Tensor, b: torch.Tensor) -> float:
    return (a - b).abs().max().item()


def run_prefill(
    *,
    model_path: str,
    prompt: str,
    dtype: str,
    compiled: bool,
    weights: dict[str, torch.Tensor],
) -> dict:
    device = torch.device("cuda")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"][0].to(device)
    positions = torch.arange(input_ids.numel(), device=device, dtype=torch.long)
    vllm_config = make_vllm_config(model_path, dtype=dtype, compiled=compiled)

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
            model = RWKV7ForCausalLM(vllm_config=vllm_config)
            model.load_weights(weights.items())
            model = model.eval().to(device, vllm_config.model_config.dtype)
            allocate_kv_cache(model, device=device)

            prompt_metadata = {
                layer.prefix: make_prefill_metadata(input_ids.numel(), device=device)
                for layer in model.model.layers
            }
            with torch.no_grad(), set_forward_context(
                prompt_metadata,
                vllm_config,
                skip_compiled=not compiled,
            ):
                hidden_states = model(input_ids=input_ids, positions=positions)
                logits = model.compute_logits(hidden_states)

            state_diffs = []
            for layer_idx, layer in enumerate(model.model.layers):
                state_diffs.append(
                    {
                        "layer_idx": layer_idx,
                        "states": [cache[0].detach().cpu() for cache in layer.kv_cache],
                    }
                )

            return {
                "hidden_last": hidden_states[-1].detach().cpu(),
                "logits_last": logits[-1].detach().cpu(),
                "next_token": int(torch.argmax(logits[-1]).item()),
                "states": state_diffs,
                "dtype": str(vllm_config.model_config.dtype),
            }
        finally:
            cleanup_dist_env_and_memory()
            torch.cuda.empty_cache()


def summarize_diffs(eager: dict, compiled: dict) -> dict:
    layer_rows = []
    first_layer_with_diff = None
    for eager_layer, compiled_layer in zip(eager["states"], compiled["states"]):
        state_rows = []
        state_has_diff = False
        for state_idx, (e_state, c_state) in enumerate(
            zip(eager_layer["states"], compiled_layer["states"])
        ):
            diff = max_abs_diff(e_state, c_state)
            state_rows.append({"state_idx": state_idx, "max_abs_diff": diff})
            state_has_diff = state_has_diff or diff > 0
        if first_layer_with_diff is None and state_has_diff:
            first_layer_with_diff = eager_layer["layer_idx"]
        layer_rows.append(
            {
                "layer_idx": eager_layer["layer_idx"],
                "states": state_rows,
            }
        )
    return {
        "hidden_last_max_abs_diff": max_abs_diff(
            eager["hidden_last"], compiled["hidden_last"]
        ),
        "logits_last_max_abs_diff": max_abs_diff(
            eager["logits_last"], compiled["logits_last"]
        ),
        "eager_next_token": eager["next_token"],
        "compiled_next_token": compiled["next_token"],
        "next_token_match": eager["next_token"] == compiled["next_token"],
        "first_layer_with_state_diff": first_layer_with_diff,
        "layers": layer_rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--prompt", default="北京是")
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--out", default="/tmp/rwkv7_compile_prefill_diff.json")
    args = parser.parse_args()

    weights = load_file(str(Path(args.model) / "model.safetensors"))
    eager = run_prefill(
        model_path=args.model,
        prompt=args.prompt,
        dtype=args.dtype,
        compiled=False,
        weights=weights,
    )
    compiled = run_prefill(
        model_path=args.model,
        prompt=args.prompt,
        dtype=args.dtype,
        compiled=True,
        weights=weights,
    )
    summary = summarize_diffs(eager, compiled)
    payload = {
        "model": args.model,
        "prompt": args.prompt,
        "eager_dtype": eager["dtype"],
        "compiled_dtype": compiled["dtype"],
        "summary": summary,
    }
    out_path = Path(args.out)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
