import argparse
import gc
import hashlib
import json
import os
import re
import time
from pathlib import Path

os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"

import torch

from vllm.distributed.parallel_state import cleanup_dist_env_and_memory
from vllm.engine.arg_utils import EngineArgs
from vllm.sampling_params import SamplingParams
from vllm.v1.engine import EngineCoreRequest
from vllm.v1.engine.llm_engine import LLMEngine


def clone_layer_states(model) -> list[dict]:
    rows = []
    for layer_idx, layer in enumerate(model.model.layers):
        rows.append(
            {
                "layer_idx": layer_idx,
                "layer_object_id": id(layer),
                "debug_last_forward_summary": getattr(
                    layer, "debug_last_forward_summary", None
                ),
                "debug_last_store_stats": getattr(
                    layer, "debug_last_store_stats", None
                ),
                "attn_debug_last_runtime_metadata_summary": getattr(
                    layer.attn, "debug_last_runtime_metadata_summary", None
                ),
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
    return get_model_runner_from_engine(engine).model


def get_model_runner_from_engine(engine: LLMEngine):
    model_executor = getattr(engine, "model_executor", None)
    if model_executor is None:
        raise RuntimeError(
            "LLMEngine is not running in-process; `model_executor` is unavailable. "
            "Check VLLM_ENABLE_V1_MULTIPROCESSING."
        )
    return model_executor.driver_worker.worker.model_runner


def clone_runner_kv_caches(model_runner) -> list[dict]:
    rows = []
    for layer_idx, layer_states in enumerate(model_runner.kv_caches):
        rows.append(
            {
                "layer_idx": layer_idx,
                "states": [
                    summarize_state_tensor(state_tensor) for state_tensor in layer_states
                ],
            }
        )
    return rows


def layer_name_to_index(layer_name: str) -> int:
    matches = re.findall(r"\.(\d+)(?:\.|$)", layer_name)
    if not matches:
        raise ValueError(f"Unable to extract layer index from {layer_name!r}.")
    return int(matches[-1])


def clone_static_forward_context(model_runner) -> list[dict]:
    rows = []
    for layer_name, layer in sorted(
        (
            item
            for item in model_runner.compilation_config.static_forward_context.items()
            if hasattr(item[1], "debug_last_store_stats")
        ),
        key=lambda item: layer_name_to_index(item[0]),
    ):
        rows.append(
            {
                "layer_name": layer_name,
                "layer_idx": layer_name_to_index(layer_name),
                "layer_object_id": id(layer),
                "debug_last_forward_summary": getattr(
                    layer, "debug_last_forward_summary", None
                ),
                "debug_last_store_stats": getattr(
                    layer, "debug_last_store_stats", None
                ),
                "states": [
                    summarize_state_tensor(cache) for cache in layer.kv_cache
                ],
            }
        )
    return rows


def make_sampling_params(max_tokens: int) -> SamplingParams:
    return SamplingParams(
        temperature=0.0,
        top_p=1.0,
        max_tokens=max_tokens,
        seed=0,
        detokenize=True,
        skip_special_tokens=False,
        spaces_between_special_tokens=False,
    )


def make_prompt_request(
    engine: LLMEngine,
    *,
    request_id: str,
    prompt: str,
    max_tokens: int,
) -> EngineCoreRequest:
    request = engine.input_processor.process_inputs(
        request_id,
        prompt,
        make_sampling_params(max_tokens),
        supported_tasks=engine.get_supported_tasks(),
        arrival_time=time.time(),
    )
    if request.prompt_token_ids is None:
        raise RuntimeError("Prompt preprocessing did not produce prompt_token_ids.")
    return request


def make_prompt_token_id_request(
    *,
    request_id: str,
    prompt_token_ids: list[int],
    max_tokens: int,
) -> EngineCoreRequest:
    return EngineCoreRequest(
        request_id=request_id,
        prompt_token_ids=list(prompt_token_ids),
        mm_features=None,
        sampling_params=make_sampling_params(max_tokens),
        pooling_params=None,
        arrival_time=time.time(),
        lora_request=None,
        cache_salt=None,
        data_parallel_rank=None,
    )


def parse_prompt_token_ids_arg(raw: str) -> list[int]:
    value = json.loads(raw)
    if not isinstance(value, list) or not all(isinstance(token, int) for token in value):
        raise ValueError("--prompt-token-ids must be a JSON list of integers.")
    return value


def load_run_payload(path: str) -> dict:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if "run" not in payload:
        raise ValueError(f"{path} does not contain a top-level 'run' payload.")
    return payload


def compare_state_snapshots(run_a: dict, run_b: dict) -> dict:
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
        "first_layer_with_state_diff": first_layer_with_diff,
        "layers": layers,
    }


def compare_runner_cache_snapshots(run_a: dict, run_b: dict) -> dict | None:
    caches_a = run_a.get("runner_kv_caches")
    caches_b = run_b.get("runner_kv_caches")
    if caches_a is None or caches_b is None:
        return None

    first_layer_with_diff = None
    layers = []
    for layer_a, layer_b in zip(caches_a, caches_b):
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
        "first_layer_with_runner_cache_diff": first_layer_with_diff,
        "layers": layers,
    }


def run_capture(
    *,
    model: str,
    prompt: str | None,
    prompt_token_ids: list[int] | None,
    max_tokens: int,
    capture_generated_tokens: int,
    async_scheduling: bool,
    request_source: str,
    compilation_config: dict | None,
) -> dict:
    if capture_generated_tokens <= 0:
        raise ValueError("--capture-generated-tokens must be positive.")
    if capture_generated_tokens > max_tokens:
        raise ValueError(
            "--capture-generated-tokens cannot exceed --max-tokens."
        )

    engine_args_kwargs = dict(
        model=model,
        trust_remote_code=True,
        gpu_memory_utilization=0.8,
        max_model_len=2048,
        disable_log_stats=True,
        async_scheduling=async_scheduling,
    )
    if compilation_config is not None:
        engine_args_kwargs["compilation_config"] = compilation_config
    engine_args = EngineArgs(**engine_args_kwargs)
    engine = LLMEngine.from_engine_args(engine_args, enable_multiprocessing=False)
    model_obj = None
    model_runner = None
    request = None
    try:
        model_runner = get_model_runner_from_engine(engine)
        model_obj = model_runner.model
        if prompt_token_ids is None:
            if prompt is None:
                raise ValueError(
                    "A text prompt is required unless prompt_token_ids are supplied."
                )
            request = make_prompt_request(
                engine,
                request_id="rwkv7-debug",
                prompt=prompt,
                max_tokens=max_tokens,
            )
        else:
            request = make_prompt_token_id_request(
                request_id="rwkv7-debug",
                prompt_token_ids=prompt_token_ids,
                max_tokens=max_tokens,
            )

        effective_prompt_token_ids = list(request.prompt_token_ids or [])
        engine.add_request(
            request.request_id,
            request,
            request.params,
            prompt_text=prompt,
        )

        capture = None
        trace = []
        step_calls = 0
        while engine.has_unfinished_requests():
            step_calls += 1
            outputs = engine.step()
            if not outputs:
                continue

            req_out = outputs[0]
            if not req_out.outputs:
                continue
            choice = req_out.outputs[0]
            trace.append(
                {
                    "step_call": step_calls,
                    "finished": req_out.finished,
                    "num_cached_tokens": req_out.num_cached_tokens,
                    "token_ids": list(choice.token_ids),
                    "text": choice.text,
                }
            )
            if len(choice.token_ids) >= capture_generated_tokens:
                capture = req_out
                break
            if req_out.finished:
                capture = req_out
                break

        if capture is None:
            raise RuntimeError("Request produced no capturable outputs.")

        choice = capture.outputs[0]
        if len(choice.token_ids) < capture_generated_tokens:
            raise RuntimeError(
                "Request finished before reaching "
                f"{capture_generated_tokens} generated tokens."
            )
        return {
            "request_source": request_source,
            "prompt": prompt,
            "prompt_token_ids": effective_prompt_token_ids,
            "max_tokens": max_tokens,
            "capture_generated_tokens": capture_generated_tokens,
            "step_calls": step_calls,
            "trace": trace,
            "token_ids": list(choice.token_ids),
            "text": choice.text,
            "num_cached_tokens": capture.num_cached_tokens,
            "states": clone_layer_states(model_obj),
            "runner_kv_caches": clone_runner_kv_caches(model_runner),
            "static_forward_context": clone_static_forward_context(model_runner),
            "engine_core_client": type(engine.engine_core).__name__,
        }
    finally:
        engine.engine_core.shutdown()
        del request
        del model_runner
        del model_obj
        del engine
        cleanup_dist_env_and_memory()
        gc.collect()


def summarize(run_a: dict, run_b: dict) -> dict:
    state_summary = compare_state_snapshots(run_a, run_b)
    runner_cache_summary = compare_runner_cache_snapshots(run_a, run_b)
    return {
        "prompt_token_ids_a": run_a.get("prompt_token_ids"),
        "prompt_token_ids_b": run_b.get("prompt_token_ids"),
        "token_ids_a": run_a["token_ids"],
        "token_ids_b": run_b["token_ids"],
        "text_a": run_a["text"],
        "text_b": run_b["text"],
        "token_match": run_a["token_ids"] == run_b["token_ids"],
        "num_cached_tokens_a": run_a["num_cached_tokens"],
        "num_cached_tokens_b": run_b["num_cached_tokens"],
        "engine_core_client_a": run_a["engine_core_client"],
        "engine_core_client_b": run_b["engine_core_client"],
        "first_layer_with_state_diff": state_summary["first_layer_with_state_diff"],
        "layers": state_summary["layers"],
        "first_layer_with_runner_cache_diff": (
            None
            if runner_cache_summary is None
            else runner_cache_summary["first_layer_with_runner_cache_diff"]
        ),
        "runner_kv_caches": (
            None if runner_cache_summary is None else runner_cache_summary["layers"]
        ),
    }


def summarize_second_step_replay(base_run: dict, replay_run: dict) -> dict:
    if len(base_run["token_ids"]) < 2:
        raise ValueError(
            "The base run must capture at least two generated tokens for "
            "second-step replay comparison."
        )
    if len(replay_run["token_ids"]) < 1:
        raise ValueError(
            "The replay run must capture at least one generated token."
        )

    state_summary = compare_state_snapshots(base_run, replay_run)
    runner_cache_summary = compare_runner_cache_snapshots(base_run, replay_run)
    expected_replay_prompt = list(base_run["prompt_token_ids"]) + [
        base_run["token_ids"][0]
    ]
    return {
        "base_prompt_token_ids": base_run["prompt_token_ids"],
        "replay_prompt_token_ids": replay_run["prompt_token_ids"],
        "expected_replay_prompt_token_ids": expected_replay_prompt,
        "replay_prompt_matches_expected": (
            replay_run["prompt_token_ids"] == expected_replay_prompt
        ),
        "base_first_generated_token_id": base_run["token_ids"][0],
        "base_second_generated_token_id": base_run["token_ids"][1],
        "replay_generated_token_id": replay_run["token_ids"][0],
        "base_token_ids": base_run["token_ids"],
        "replay_token_ids": replay_run["token_ids"],
        "base_text": base_run["text"],
        "replay_text": replay_run["text"],
        "second_token_match": (
            base_run["token_ids"][1] == replay_run["token_ids"][0]
        ),
        "base_step_calls": base_run["step_calls"],
        "replay_step_calls": replay_run["step_calls"],
        "base_num_cached_tokens": base_run["num_cached_tokens"],
        "replay_num_cached_tokens": replay_run["num_cached_tokens"],
        "engine_core_client_base": base_run["engine_core_client"],
        "engine_core_client_replay": replay_run["engine_core_client"],
        "first_layer_with_state_diff": state_summary["first_layer_with_state_diff"],
        "layers": state_summary["layers"],
        "first_layer_with_runner_cache_diff": (
            None
            if runner_cache_summary is None
            else runner_cache_summary["first_layer_with_runner_cache_diff"]
        ),
        "runner_kv_caches": (
            None if runner_cache_summary is None else runner_cache_summary["layers"]
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--prompt", default=None)
    parser.add_argument("--prompt-token-ids", default=None)
    parser.add_argument("--append-generated-prefix-from-run-json", default=None)
    parser.add_argument("--generated-prefix-len", type=int, default=1)
    parser.add_argument("--async-scheduling", action="store_true")
    parser.add_argument(
        "--cudagraph-mode",
        choices=["default", "none"],
        default="none",
    )
    parser.add_argument("--cudagraph-copy-inputs", action="store_true")
    parser.add_argument("--disable-compile-cache", action="store_true")
    parser.add_argument("--max-tokens", type=int, default=None)
    parser.add_argument("--capture-generated-tokens", type=int, default=1)
    parser.add_argument("--run-json-a", default=None)
    parser.add_argument("--run-json-b", default=None)
    parser.add_argument("--compare-second-step", action="store_true")
    parser.add_argument(
        "--out", default="/tmp/rwkv7_engine_first_step_compare.json"
    )
    args = parser.parse_args()

    if args.run_json_a is not None or args.run_json_b is not None:
        if args.run_json_a is None or args.run_json_b is None:
            raise ValueError("Both --run-json-a and --run-json-b must be provided.")
        payload_a = load_run_payload(args.run_json_a)
        payload_b = load_run_payload(args.run_json_b)
        run_a = payload_a["run"]
        run_b = payload_b["run"]
        payload = {
            "model": args.model,
            "prompt": args.prompt or payload_a.get("prompt") or payload_b.get("prompt"),
            "async_scheduling": args.async_scheduling,
            "summary": (
                summarize_second_step_replay(run_a, run_b)
                if args.compare_second_step
                else summarize(run_a, run_b)
            ),
        }
        out_path = Path(args.out)
        out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        print(f"wrote {out_path}")
        return 0

    prompt_token_ids = None
    request_source = "prompt"
    prompt = args.prompt
    if args.prompt_token_ids is not None and args.append_generated_prefix_from_run_json is not None:
        raise ValueError(
            "--prompt-token-ids and --append-generated-prefix-from-run-json "
            "cannot be used together."
        )

    if args.prompt_token_ids is not None:
        prompt_token_ids = parse_prompt_token_ids_arg(args.prompt_token_ids)
        request_source = "prompt_token_ids"
    elif args.append_generated_prefix_from_run_json is not None:
        replay_payload = load_run_payload(args.append_generated_prefix_from_run_json)
        replay_run = replay_payload["run"]
        if args.generated_prefix_len < 0:
            raise ValueError("--generated-prefix-len must be non-negative.")
        if args.generated_prefix_len > len(replay_run["token_ids"]):
            raise ValueError(
                "--generated-prefix-len exceeds the number of generated tokens "
                "stored in the source run JSON."
            )
        prompt_token_ids = list(replay_run["prompt_token_ids"]) + list(
            replay_run["token_ids"][: args.generated_prefix_len]
        )
        prompt = prompt or replay_payload.get("prompt")
        request_source = "append_generated_prefix_from_run_json"

    if args.max_tokens is None:
        raise ValueError(
            "--max-tokens is required for single-run capture. "
            "Use separate run JSONs plus --run-json-a/--run-json-b for comparison."
        )

    if prompt_token_ids is None and prompt is None:
        raise ValueError(
            "A text --prompt is required unless prompt token ids are provided."
        )

    if args.disable_compile_cache:
        os.environ["VLLM_DISABLE_COMPILE_CACHE"] = "1"

    compilation_config = None
    if args.cudagraph_mode == "none" or args.cudagraph_copy_inputs:
        compilation_config = {}
        if args.cudagraph_mode == "none":
            compilation_config["cudagraph_mode"] = "none"
        if args.cudagraph_copy_inputs:
            compilation_config["cudagraph_copy_inputs"] = True

    payload = {
        "model": args.model,
        "prompt": prompt,
        "async_scheduling": args.async_scheduling,
        "run": run_capture(
            model=args.model,
            prompt=prompt,
            prompt_token_ids=prompt_token_ids,
            max_tokens=args.max_tokens,
            capture_generated_tokens=args.capture_generated_tokens,
            async_scheduling=args.async_scheduling,
            request_source=request_source,
            compilation_config=compilation_config,
        ),
    }

    out_path = Path(args.out)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
    print(
        json.dumps(
            {
                "model": args.model,
                "prompt": prompt,
                "async_scheduling": args.async_scheduling,
                "request_source": payload["run"]["request_source"],
                "max_tokens": args.max_tokens,
                "capture_generated_tokens": args.capture_generated_tokens,
                "prompt_token_ids": payload["run"]["prompt_token_ids"],
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
