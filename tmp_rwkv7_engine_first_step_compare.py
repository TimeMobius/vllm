import os
import argparse
import gc
import hashlib
import json
from pathlib import Path

os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"

import torch

from vllm.config.compilation import CUDAGraphMode, CompilationMode
from vllm.distributed.parallel_state import cleanup_dist_env_and_memory
from vllm.engine.arg_utils import EngineArgs
from vllm.model_executor.models.config import MambaModelConfig, RWKV7ForCausalLMConfig
from vllm.sampling_params import SamplingParams
from vllm.v1.engine.llm_engine import LLMEngine


def patch_rwkv7_compile_config():
    orig_verify = RWKV7ForCausalLMConfig.verify_and_update_config
    orig_post = getattr(
        RWKV7ForCausalLMConfig,
        "apply_post_optimization_level_defaults",
        None,
    )

    @classmethod
    def patched_verify(cls, vllm_config):
        MambaModelConfig.verify_and_update_config(vllm_config)
        vllm_config.model_config.enforce_eager = False
        vllm_config.compilation_config.mode = CompilationMode.VLLM_COMPILE
        vllm_config.compilation_config.cudagraph_mode = CUDAGraphMode.NONE
        vllm_config.compilation_config.cudagraph_copy_inputs = False

    @classmethod
    def patched_post(cls, vllm_config):
        vllm_config.model_config.enforce_eager = False
        vllm_config.compilation_config.mode = CompilationMode.VLLM_COMPILE
        vllm_config.compilation_config.cudagraph_mode = CUDAGraphMode.NONE
        vllm_config.compilation_config.cudagraph_copy_inputs = False

    RWKV7ForCausalLMConfig.verify_and_update_config = patched_verify
    RWKV7ForCausalLMConfig.apply_post_optimization_level_defaults = patched_post
    return orig_verify, orig_post


def restore_rwkv7_compile_config(orig_verify, orig_post) -> None:
    RWKV7ForCausalLMConfig.verify_and_update_config = orig_verify
    if orig_post is not None:
        RWKV7ForCausalLMConfig.apply_post_optimization_level_defaults = orig_post


def clone_layer_states(model) -> list[dict]:
    rows = []
    for layer_idx, layer in enumerate(model.model.layers):
        rows.append(
            {
                "layer_idx": layer_idx,
                "states": [
                    summarize_state_tensor(cache) for cache in layer.kv_cache
                ],
            }
        )
    return rows


def max_abs_diff(a, b) -> float:
    a_t = torch.as_tensor(a)
    b_t = torch.as_tensor(b)
    return (a_t - b_t).abs().max().item()


def summarize_state_tensor(tensor: torch.Tensor) -> dict:
    cpu = tensor.detach().cpu().contiguous().to(torch.float32)
    return {
        "shape": list(cpu.shape),
        "dtype": str(cpu.dtype),
        "absmax": float(cpu.abs().max().item()),
        "mean": float(cpu.mean().item()),
        "l2": float(cpu.norm().item()),
        "sha256": hashlib.sha256(cpu.numpy().tobytes()).hexdigest(),
    }


def get_model_from_engine(engine: LLMEngine):
    model_executor = getattr(engine, "model_executor", None)
    if model_executor is None:
        raise RuntimeError(
            "LLMEngine is not running in-process; `model_executor` is unavailable. "
            "Check VLLM_ENABLE_V1_MULTIPROCESSING."
        )
    return model_executor.driver_worker.worker.model_runner.model


def run_first_step(
    *,
    model: str,
    prompt: str,
    max_tokens: int,
    async_scheduling: bool,
) -> dict:
    engine_args = EngineArgs(
        model=model,
        trust_remote_code=True,
        gpu_memory_utilization=0.8,
        max_model_len=2048,
        disable_log_stats=True,
        async_scheduling=async_scheduling,
        compilation_config={"cudagraph_mode": "none"},
    )
    engine = LLMEngine.from_engine_args(engine_args, enable_multiprocessing=False)
    model_obj = None
    try:
        model_obj = get_model_from_engine(engine)
        params = SamplingParams(
            temperature=0.0,
            top_p=1.0,
            max_tokens=max_tokens,
            seed=0,
            detokenize=True,
            skip_special_tokens=False,
            spaces_between_special_tokens=False,
        )
        engine.add_request("rwkv7-debug", prompt, params)
        outputs = engine.step()
        if not outputs:
            raise RuntimeError("engine.step() returned no outputs")
        req_out = outputs[0]
        choice = req_out.outputs[0]
        return {
            "max_tokens": max_tokens,
            "token_ids": list(choice.token_ids),
            "text": choice.text,
            "num_cached_tokens": req_out.num_cached_tokens,
            "states": clone_layer_states(model_obj),
            "engine_core_client": type(engine.engine_core).__name__,
        }
    finally:
        engine.engine_core.shutdown()
        del model_obj
        del engine
        cleanup_dist_env_and_memory()
        gc.collect()


def summarize(run_a: dict, run_b: dict) -> dict:
    first_layer_with_diff = None
    layers = []
    for layer_a, layer_b in zip(run_a["states"], run_b["states"]):
        state_rows = []
        has_diff = False
        for state_idx, (state_a, state_b) in enumerate(
            zip(layer_a["states"], layer_b["states"])
        ):
            match = state_a["sha256"] == state_b["sha256"]
            state_rows.append(
                {
                    "state_idx": state_idx,
                    "match": match,
                    "absmax_a": state_a["absmax"],
                    "absmax_b": state_b["absmax"],
                    "mean_a": state_a["mean"],
                    "mean_b": state_b["mean"],
                    "l2_a": state_a["l2"],
                    "l2_b": state_b["l2"],
                }
            )
            has_diff = has_diff or not match
        if first_layer_with_diff is None and has_diff:
            first_layer_with_diff = layer_a["layer_idx"]
        layers.append({"layer_idx": layer_a["layer_idx"], "states": state_rows})
    return {
        "token_ids_a": run_a["token_ids"],
        "token_ids_b": run_b["token_ids"],
        "text_a": run_a["text"],
        "text_b": run_b["text"],
        "token_match": run_a["token_ids"] == run_b["token_ids"],
        "num_cached_tokens_a": run_a["num_cached_tokens"],
        "num_cached_tokens_b": run_b["num_cached_tokens"],
        "engine_core_client_a": run_a["engine_core_client"],
        "engine_core_client_b": run_b["engine_core_client"],
        "first_layer_with_state_diff": first_layer_with_diff,
        "layers": layers,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--prompt", default="北京是")
    parser.add_argument("--async-scheduling", action="store_true")
    parser.add_argument("--max-tokens", type=int, default=None)
    parser.add_argument("--run-json-a", default=None)
    parser.add_argument("--run-json-b", default=None)
    parser.add_argument(
        "--out", default="/tmp/rwkv7_engine_first_step_compare.json"
    )
    args = parser.parse_args()

    if args.run_json_a is not None or args.run_json_b is not None:
        if args.run_json_a is None or args.run_json_b is None:
            raise ValueError("Both --run-json-a and --run-json-b must be provided.")
        run_a = json.loads(Path(args.run_json_a).read_text(encoding="utf-8"))["run"]
        run_b = json.loads(Path(args.run_json_b).read_text(encoding="utf-8"))["run"]
        payload = {
            "model": args.model,
            "prompt": args.prompt,
            "async_scheduling": args.async_scheduling,
            "summary": summarize(run_a, run_b),
        }
        out_path = Path(args.out)
        out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        print(f"wrote {out_path}")
        return 0

    orig_verify, orig_post = patch_rwkv7_compile_config()
    try:
        if args.max_tokens is not None:
            payload = {
                "model": args.model,
                "prompt": args.prompt,
                "async_scheduling": args.async_scheduling,
                "run": run_first_step(
                    model=args.model,
                    prompt=args.prompt,
                    max_tokens=args.max_tokens,
                    async_scheduling=args.async_scheduling,
                ),
            }
        else:
            run_1 = run_first_step(
                model=args.model,
                prompt=args.prompt,
                max_tokens=1,
                async_scheduling=args.async_scheduling,
            )
            run_8 = run_first_step(
                model=args.model,
                prompt=args.prompt,
                max_tokens=8,
                async_scheduling=args.async_scheduling,
            )
            payload = {
                "model": args.model,
                "prompt": args.prompt,
                "async_scheduling": args.async_scheduling,
                "summary": summarize(run_1, run_8),
            }
    finally:
        restore_rwkv7_compile_config(orig_verify, orig_post)

    out_path = Path(args.out)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
    if args.max_tokens is None:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        print(
            json.dumps(
                {
                    "model": args.model,
                    "prompt": args.prompt,
                    "async_scheduling": args.async_scheduling,
                    "max_tokens": args.max_tokens,
                    "token_ids": payload["run"]["token_ids"],
                    "text": payload["run"]["text"],
                    "num_cached_tokens": payload["run"]["num_cached_tokens"],
                    "engine_core_client": payload["run"]["engine_core_client"],
                },
                ensure_ascii=False,
                indent=2,
            )
        )
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
