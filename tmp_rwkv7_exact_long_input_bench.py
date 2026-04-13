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
        "token_ids": token_ids,
        "text": choice["text"],
        "elapsed_sec": elapsed,
        "output_tokens": len(token_ids),
        "request_tps": (len(token_ids) / elapsed) if elapsed > 0 else None,
    }


def run_serial_baseline(
    base: str,
    model: str,
    prompt_ids: list[int],
    max_tokens: int,
    seed: int,
) -> dict:
    return complete_with_ids(base, model, prompt_ids, max_tokens, seed)


def run_concurrent_batch(
    base: str,
    model: str,
    prompt_ids: list[int],
    concurrency: int,
    max_tokens: int,
    seed: int,
) -> dict:
    started = time.perf_counter()
    results = []
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = [
            pool.submit(
                complete_with_ids,
                base,
                model,
                prompt_ids,
                max_tokens,
                seed,
            )
            for _ in range(concurrency)
        ]
        for future in as_completed(futures):
            results.append(future.result())
    wall_time = time.perf_counter() - started
    total_output_tokens = sum(row["output_tokens"] for row in results)
    return {
        "wall_time_sec": wall_time,
        "total_output_tokens": total_output_tokens,
        "aggregate_tps": (total_output_tokens / wall_time) if wall_time > 0 else None,
        "requests": results,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--gpu-memory-utilization", default="0.8")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--rounds", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--warmup-prompt-len", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--no-async-scheduling", action="store_true")
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
    parser.add_argument(
        "--prompt-lengths",
        nargs="+",
        type=int,
        default=[1024, 1984],
    )
    parser.add_argument(
        "--concurrency-levels",
        nargs="+",
        type=int,
        default=[1, 4, 8],
    )
    parser.add_argument(
        "--seed-text",
        default=(
            "The capital of France is Paris. "
            "Beijing is the capital of China. "
            "RWKV7 is a recurrent world model for language generation."
        ),
    )
    parser.add_argument(
        "--log",
        default="/tmp/vllm_rwkv7_exact_long_input_bench.log",
        help="server log path",
    )
    args = parser.parse_args()

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

            token_buffer = build_token_buffer(
                base,
                args.model,
                args.seed_text,
                max(max(args.prompt_lengths), args.warmup_prompt_len),
            )

            warmup_prompt_ids = token_buffer[: args.warmup_prompt_len]
            for _ in range(args.warmup):
                complete_with_ids(
                    base,
                    args.model,
                    warmup_prompt_ids,
                    min(args.max_tokens, 16),
                    args.seed,
                )

            scenarios = []
            for prompt_len in args.prompt_lengths:
                prompt_ids = token_buffer[:prompt_len]
                for concurrency in args.concurrency_levels:
                    baseline = run_serial_baseline(
                        base,
                        args.model,
                        prompt_ids,
                        args.max_tokens,
                        args.seed,
                    )
                    rounds = []
                    for round_idx in range(args.rounds):
                        batch = run_concurrent_batch(
                            base,
                            args.model,
                            prompt_ids,
                            concurrency,
                            args.max_tokens,
                            args.seed,
                        )
                        for req in batch["requests"]:
                            req["matches_serial_baseline"] = (
                                req["token_ids"] == baseline["token_ids"]
                            )
                        batch["round"] = round_idx
                        batch["all_match_serial_baseline"] = all(
                            req["matches_serial_baseline"] for req in batch["requests"]
                        )
                        rounds.append(batch)

                    scenarios.append(
                        {
                            "prompt_len": prompt_len,
                            "concurrency": concurrency,
                            "max_tokens": args.max_tokens,
                            "baseline": baseline,
                            "rounds": rounds,
                        }
                    )

            print(
                json.dumps(
                    {
                        "model": args.model,
                        "dtype": args.dtype,
                        "enforce_eager": args.enforce_eager,
                        "cudagraph_mode": args.cudagraph_mode,
                        "compile_no_cg": args.compile_no_cg,
                        "cudagraph_copy_inputs": args.cudagraph_copy_inputs,
                        "disable_compile_cache": args.disable_compile_cache,
                        "max_tokens": args.max_tokens,
                        "rounds": args.rounds,
                        "warmup": args.warmup,
                        "prompt_lengths": args.prompt_lengths,
                        "concurrency_levels": args.concurrency_levels,
                        "server_log": str(log_path),
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
