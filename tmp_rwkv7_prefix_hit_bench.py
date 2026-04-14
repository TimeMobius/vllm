# SPDX-License-Identifier: Apache-2.0

import argparse
import json
import os
import subprocess
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


def post(base: str, path: str, payload: dict) -> dict:
    req = urllib.request.Request(
        base + path,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=600) as resp:
        return json.loads(resp.read().decode("utf-8"))


def is_healthy(base: str) -> bool:
    try:
        with urllib.request.urlopen(f"{base}/health", timeout=2) as resp:
            return resp.status == 200
    except (urllib.error.URLError, TimeoutError):
        return False


def extract_server_log_signals(log_path: Path) -> list[str]:
    if not log_path.exists():
        return []

    patterns = (
        "Mamba cache mode",
        "Prefix caching in Mamba cache",
        "falling back to 'align' mode",
    )
    signals: list[str] = []
    for line in log_path.read_text(encoding="utf-8",
                                   errors="replace").splitlines():
        if any(pattern in line for pattern in patterns):
            signals.append(line.strip())
    return signals


def tokenize(base: str, model: str, prompt: str) -> list[int]:
    return post(base, "/tokenize", {"model": model, "prompt": prompt})["tokens"]


def build_token_buffer(
    base: str,
    model: str,
    seed_text: str,
    min_tokens: int,
) -> list[int]:
    prompt = seed_text
    while True:
        token_ids = tokenize(base, model, prompt)
        if len(token_ids) >= min_tokens:
            return token_ids
        prompt = f"{prompt} {seed_text}"


def complete_with_ids(
    base: str,
    model: str,
    prompt_name: str,
    prompt_ids: list[int],
    max_tokens: int,
    seed: int,
) -> dict:
    started = time.perf_counter()
    obj = post(
        base,
        "/v1/completions",
        {
            "model": model,
            "prompt": prompt_ids,
            "max_tokens": max_tokens,
            "temperature": 0.0,
            "top_p": 1.0,
            "return_token_ids": True,
            "seed": seed,
        },
    )
    elapsed = time.perf_counter() - started
    choice = obj["choices"][0]
    token_ids = choice["token_ids"]
    return {
        "prompt_name": prompt_name,
        "token_ids": token_ids,
        "text": choice["text"],
        "elapsed_sec": elapsed,
        "output_tokens": len(token_ids),
        "request_tps": (len(token_ids) / elapsed) if elapsed > 0 else None,
    }


def run_serial_baseline(
    base: str,
    model: str,
    prompt_specs: list[dict],
    max_tokens: int,
    seed: int,
) -> dict[str, dict]:
    baseline: dict[str, dict] = {}
    for prompt_spec in prompt_specs:
        baseline[prompt_spec["prompt_name"]] = complete_with_ids(
            base,
            model,
            prompt_spec["prompt_name"],
            prompt_spec["prompt_ids"],
            max_tokens,
            seed,
        )
    return baseline


def run_concurrent_batch(
    base: str,
    model: str,
    prompt_specs: list[dict],
    max_tokens: int,
    seed: int,
) -> dict:
    started = time.perf_counter()
    results = []
    with ThreadPoolExecutor(max_workers=len(prompt_specs)) as pool:
        futures = [
            pool.submit(
                complete_with_ids,
                base,
                model,
                prompt_spec["prompt_name"],
                prompt_spec["prompt_ids"],
                max_tokens,
                seed,
            )
            for prompt_spec in prompt_specs
        ]
        for future in as_completed(futures):
            results.append(future.result())

    prompt_order = {
        prompt_spec["prompt_name"]: idx for idx, prompt_spec in enumerate(prompt_specs)
    }
    results.sort(key=lambda row: prompt_order[row["prompt_name"]])
    wall_time = time.perf_counter() - started
    total_output_tokens = sum(row["output_tokens"] for row in results)
    return {
        "wall_time_sec": wall_time,
        "total_output_tokens": total_output_tokens,
        "aggregate_tps": (total_output_tokens / wall_time) if wall_time > 0 else None,
        "requests": results,
    }


def build_prompt_specs(
    *,
    shared_prefixes: list[list[int]],
    cold_prefix_buffer: list[int],
    tail_buffer: list[int],
    shared_prefix_len: int,
    tail_len: int,
    hit_ratio: float,
    concurrency: int,
    round_idx: int,
    workload_offset: int,
    cold_prefix_stride: int,
    tail_stride: int,
) -> list[dict]:
    num_hits = min(concurrency, max(0, int(round(hit_ratio * concurrency))))
    prompt_specs: list[dict] = []
    for req_idx in range(concurrency):
        tail_offset = (workload_offset + req_idx) * tail_stride
        tail_ids = tail_buffer[tail_offset : tail_offset + tail_len]
        if len(tail_ids) != tail_len:
            raise ValueError("Tail buffer is too short for requested workload.")

        if req_idx < num_hits:
            prefix_group = req_idx % len(shared_prefixes)
            prefix_ids = shared_prefixes[prefix_group]
            prompt_role = "hit"
        else:
            miss_idx = workload_offset + (req_idx - num_hits)
            cold_offset = miss_idx * cold_prefix_stride
            prefix_ids = cold_prefix_buffer[
                cold_offset : cold_offset + shared_prefix_len
            ]
            if len(prefix_ids) != shared_prefix_len:
                raise ValueError("Cold prefix buffer is too short for requested workload.")
            prefix_group = -1
            prompt_role = "miss"

        prompt_specs.append(
            {
                "prompt_name": f"round{round_idx}_{prompt_role}_req{req_idx}",
                "prompt_role": prompt_role,
                "prefix_group": prefix_group,
                "prompt_ids": prefix_ids + tail_ids,
                "prompt_len": shared_prefix_len + tail_len,
            }
        )
    return prompt_specs


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--gpu-memory-utilization", default="0.8")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--warmup-prompt-len", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--no-async-scheduling", action="store_true")
    parser.add_argument("--enable-prefix-caching", action="store_true")
    parser.add_argument(
        "--mamba-cache-mode",
        choices=["default", "none", "align", "all"],
        default="default",
    )
    parser.add_argument("--no-enable-chunked-prefill", action="store_true")
    parser.add_argument(
        "--cudagraph-mode",
        choices=["default", "none", "piecewise", "full", "full_and_piecewise"],
        default="default",
    )
    parser.add_argument("--compile-no-cg", action="store_true")
    parser.add_argument("--cudagraph-copy-inputs", action="store_true")
    parser.add_argument("--disable-compile-cache", action="store_true")
    parser.add_argument("--compilation-config", default=None)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument(
        "--hit-ratios",
        nargs="+",
        type=float,
        default=[0.0, 0.5, 1.0],
    )
    parser.add_argument("--shared-prefix-len", type=int, default=1024)
    parser.add_argument("--tail-len", type=int, default=128)
    parser.add_argument("--shared-prefix-count", type=int, default=2)
    parser.add_argument("--cold-prefix-stride", type=int, default=128)
    parser.add_argument("--tail-stride", type=int, default=128)
    parser.add_argument(
        "--shared-seed-text",
        default=(
            "The capital of France is Paris. "
            "Beijing is the capital of China. "
            "RWKV7 prefix caching should reward repeated long contexts."
        ),
    )
    parser.add_argument(
        "--cold-seed-text",
        default=(
            "Recurrent sequence models can carry hidden state across steps. "
            "Distinct prompt prefixes should stay cold until explicitly warmed."
        ),
    )
    parser.add_argument(
        "--tail-seed-text",
        default=(
            "Request specific suffix tokens keep each completion unique across rounds."
        ),
    )
    parser.add_argument(
        "--log",
        default="/tmp/vllm_rwkv7_prefix_hit_bench.log",
        help="server log path",
    )
    args = parser.parse_args()

    if args.shared_prefix_count <= 0:
        raise ValueError("--shared-prefix-count must be positive.")

    log_path = Path(args.log)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    if log_path.exists():
        log_path.unlink()

    base = f"http://127.0.0.1:{args.port}"
    cmd = [
        "vllm",
        "serve",
        args.model,
        "--trust-remote-code",
        "--gpu-memory-utilization",
        str(args.gpu_memory_utilization),
        "--dtype",
        args.dtype,
        "--host",
        "127.0.0.1",
        "--port",
        str(args.port),
    ]
    if args.enforce_eager:
        cmd.append("--enforce-eager")
    if args.compile_no_cg:
        cmd.append("-cc.cudagraph_mode=none")
    elif args.cudagraph_mode != "default":
        cmd.append(f"-cc.cudagraph_mode={args.cudagraph_mode}")
    if args.cudagraph_copy_inputs:
        cmd.append("-cc.cudagraph_copy_inputs=true")
    if args.compilation_config is not None:
        cmd.extend(["-cc", args.compilation_config])
    if args.no_async_scheduling:
        cmd.append("--no-async-scheduling")
    if args.enable_prefix_caching:
        cmd.append("--enable-prefix-caching")
    if args.mamba_cache_mode != "default":
        cmd.extend(["--mamba-cache-mode", args.mamba_cache_mode])
    if args.no_enable_chunked_prefill:
        cmd.append("--no-enable-chunked-prefill")

    with log_path.open("w", encoding="utf-8") as logf:
        env = None
        if args.disable_compile_cache:
            env = os.environ.copy()
            env["VLLM_DISABLE_COMPILE_CACHE"] = "1"

        proc = subprocess.Popen(
            cmd,
            stdout=logf,
            stderr=subprocess.STDOUT,
            env=env,
        )
        try:
            deadline = time.time() + 900
            while time.time() < deadline:
                if is_healthy(base):
                    break
                if proc.poll() is not None:
                    print("server exited early")
                    return 2
                time.sleep(2)
            else:
                print("server not ready before timeout")
                return 3

            warmup_buffer = build_token_buffer(
                base,
                args.model,
                args.shared_seed_text,
                max(args.shared_prefix_len, args.warmup_prompt_len),
            )
            warmup_prompt_ids = warmup_buffer[: args.warmup_prompt_len]
            for _ in range(args.warmup):
                complete_with_ids(
                    base,
                    args.model,
                    "warmup",
                    warmup_prompt_ids,
                    min(args.max_tokens, 16),
                    args.seed,
                )

            total_request_slots = max(
                1, len(args.hit_ratios) * args.rounds * args.concurrency
            )
            shared_needed = (
                args.shared_prefix_len
                + (args.shared_prefix_count - 1) * args.cold_prefix_stride
            )
            cold_needed = args.shared_prefix_len + (
                total_request_slots * args.cold_prefix_stride
            )
            tail_needed = args.tail_len + (
                total_request_slots * args.tail_stride
            )

            shared_buffer = build_token_buffer(
                base, args.model, args.shared_seed_text, shared_needed
            )
            cold_buffer = build_token_buffer(
                base, args.model, args.cold_seed_text, cold_needed
            )
            tail_buffer = build_token_buffer(
                base, args.model, args.tail_seed_text, tail_needed
            )

            shared_prefixes = [
                shared_buffer[
                    idx * args.cold_prefix_stride : idx * args.cold_prefix_stride
                    + args.shared_prefix_len
                ]
                for idx in range(args.shared_prefix_count)
            ]

            for prefix_idx, prefix_ids in enumerate(shared_prefixes):
                complete_with_ids(
                    base,
                    args.model,
                    f"warm_prefix_{prefix_idx}",
                    prefix_ids,
                    1,
                    args.seed,
                )

            scenarios = []
            for scenario_idx, hit_ratio in enumerate(args.hit_ratios):
                rounds = []
                for round_idx in range(args.rounds):
                    workload_offset = (
                        scenario_idx * args.rounds * args.concurrency
                        + round_idx * args.concurrency
                    )
                    prompt_specs = build_prompt_specs(
                        shared_prefixes=shared_prefixes,
                        cold_prefix_buffer=cold_buffer,
                        tail_buffer=tail_buffer,
                        shared_prefix_len=args.shared_prefix_len,
                        tail_len=args.tail_len,
                        hit_ratio=hit_ratio,
                        concurrency=args.concurrency,
                        round_idx=round_idx,
                        workload_offset=workload_offset,
                        cold_prefix_stride=args.cold_prefix_stride,
                        tail_stride=args.tail_stride,
                    )
                    batch = run_concurrent_batch(
                        base,
                        args.model,
                        prompt_specs,
                        args.max_tokens,
                        args.seed,
                    )
                    baseline = run_serial_baseline(
                        base,
                        args.model,
                        prompt_specs,
                        args.max_tokens,
                        args.seed,
                    )
                    requests = []
                    for req, prompt_spec in zip(batch["requests"], prompt_specs, strict=True):
                        baseline_req = baseline[prompt_spec["prompt_name"]]
                        requests.append(
                            {
                                **req,
                                "prompt_role": prompt_spec["prompt_role"],
                                "prefix_group": prompt_spec["prefix_group"],
                                "prompt_len": prompt_spec["prompt_len"],
                                "matches_serial_baseline": (
                                    req["token_ids"] == baseline_req["token_ids"]
                                ),
                            }
                        )
                    rounds.append(
                        {
                            "round": round_idx,
                            "wall_time_sec": batch["wall_time_sec"],
                            "total_output_tokens": batch["total_output_tokens"],
                            "aggregate_tps": batch["aggregate_tps"],
                            "all_match_serial_baseline": all(
                                req["matches_serial_baseline"] for req in requests
                            ),
                            "requests": requests,
                        }
                    )

                scenarios.append(
                    {
                        "hit_ratio": hit_ratio,
                        "concurrency": args.concurrency,
                        "shared_prefix_len": args.shared_prefix_len,
                        "tail_len": args.tail_len,
                        "num_hit_requests": int(round(hit_ratio * args.concurrency)),
                        "num_miss_requests": args.concurrency
                        - int(round(hit_ratio * args.concurrency)),
                        "rounds": rounds,
                    }
                )

            print(
                json.dumps(
                    {
                        "model": args.model,
                        "dtype": args.dtype,
                        "enforce_eager": args.enforce_eager,
                        "enable_prefix_caching": args.enable_prefix_caching,
                        "mamba_cache_mode": args.mamba_cache_mode,
                        "cudagraph_mode": args.cudagraph_mode,
                        "compile_no_cg": args.compile_no_cg,
                        "cudagraph_copy_inputs": args.cudagraph_copy_inputs,
                        "disable_compile_cache": args.disable_compile_cache,
                        "max_tokens": args.max_tokens,
                        "rounds": args.rounds,
                        "warmup": args.warmup,
                        "concurrency": args.concurrency,
                        "hit_ratios": args.hit_ratios,
                        "shared_prefix_len": args.shared_prefix_len,
                        "tail_len": args.tail_len,
                        "shared_prefix_count": args.shared_prefix_count,
                        "server_log": str(log_path),
                        "server_log_signals": extract_server_log_signals(log_path),
                        "scenarios": scenarios,
                    },
                    ensure_ascii=False,
                    indent=2,
                )
            )
            return 0
        finally:
            proc.terminate()
            try:
                proc.wait(timeout=20)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=10)


if __name__ == "__main__":
    raise SystemExit(main())
