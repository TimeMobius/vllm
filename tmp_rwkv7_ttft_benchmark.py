import argparse
import asyncio
import json
import os
import statistics
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path

import aiohttp


AIOHTTP_TIMEOUT = aiohttp.ClientTimeout(total=60 * 60)


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


def mean_or_none(values: list[float]) -> float | None:
    return statistics.mean(values) if values else None


def median_or_none(values: list[float]) -> float | None:
    return statistics.median(values) if values else None


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


async def stream_completion(
    session: aiohttp.ClientSession,
    base: str,
    model: str,
    prompt_ids: list[int],
    max_tokens: int,
    seed: int,
) -> dict:
    payload = {
        "model": model,
        "prompt": prompt_ids,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "top_p": 1.0,
        "stream": True,
        "stream_options": {
            "include_usage": True,
        },
        "seed": seed,
    }
    started = time.perf_counter()
    last_timestamp = started
    ttft = None
    itl: list[float] = []
    generated_text = ""
    output_tokens = None

    try:
        async with session.post(f"{base}/v1/completions", json=payload) as response:
            if response.status != 200:
                return {
                    "success": False,
                    "error": response.reason or f"HTTP {response.status}",
                }

            async for chunk_bytes in response.content:
                chunk_bytes = chunk_bytes.strip()
                if not chunk_bytes:
                    continue

                chunk = chunk_bytes.decode("utf-8").removeprefix("data: ")
                if chunk == "[DONE]":
                    continue

                data = json.loads(chunk)
                if choices := data.get("choices"):
                    timestamp = time.perf_counter()
                    if ttft is None:
                        ttft = timestamp - started
                    else:
                        itl.append(timestamp - last_timestamp)
                    last_timestamp = timestamp
                    generated_text += choices[0].get("text") or ""

                if usage := data.get("usage"):
                    output_tokens = usage.get("completion_tokens")

        if ttft is None:
            return {
                "success": False,
                "error": "Never received a valid token chunk for TTFT.",
            }

        if output_tokens is None:
            output_tokens = 1 + len(itl)

        latency = last_timestamp - started
        return {
            "success": True,
            "latency_sec": latency,
            "ttft_sec": ttft,
            "itl_sec": itl,
            "tpot_sec": mean_or_none(itl),
            "output_tokens": output_tokens,
            "generated_text": generated_text,
        }
    except Exception as exc:
        return {
            "success": False,
            "error": repr(exc),
        }


def summarize_latency_rounds(rows: list[dict]) -> dict:
    success_rows = [row for row in rows if row["success"]]
    ttfts = [row["ttft_sec"] * 1000.0 for row in success_rows]
    latencies = [row["latency_sec"] * 1000.0 for row in success_rows]
    tpots = [
        row["tpot_sec"] * 1000.0
        for row in success_rows
        if row.get("tpot_sec") is not None
    ]
    all_itls_ms = [
        itl_sec * 1000.0
        for row in success_rows
        for itl_sec in row.get("itl_sec", [])
    ]
    return {
        "successful_rounds": len(success_rows),
        "failed_rounds": len(rows) - len(success_rows),
        "avg_ttft_ms": mean_or_none(ttfts),
        "median_ttft_ms": median_or_none(ttfts),
        "avg_latency_ms": mean_or_none(latencies),
        "median_latency_ms": median_or_none(latencies),
        "avg_tpot_ms": mean_or_none(tpots),
        "avg_itl_ms": mean_or_none(all_itls_ms),
    }


async def main_async(args: argparse.Namespace) -> int:
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

        launched_at = time.perf_counter()
        proc = subprocess.Popen(
            cmd,
            stdout=logf,
            stderr=subprocess.STDOUT,
            env=env,
        )
        try:
            deadline = time.time() + args.server_ready_timeout
            while time.time() < deadline:
                if is_healthy(base):
                    break
                if proc.poll() is not None:
                    print("server exited early")
                    return 2
                await asyncio.sleep(2)
            else:
                print("server not ready before timeout")
                return 3

            server_ready_sec = time.perf_counter() - launched_at

            max_prompt_len = max(max(args.prompt_lengths), args.decode_prompt_len)
            token_buffer = build_token_buffer(
                base,
                args.model,
                args.seed_text,
                max_prompt_len,
            )

            async with aiohttp.ClientSession(timeout=AIOHTTP_TIMEOUT) as session:
                if args.warmup > 0:
                    warmup_prompt = token_buffer[: min(args.warmup_prompt_len, max_prompt_len)]
                    for _ in range(args.warmup):
                        await stream_completion(
                            session,
                            base,
                            args.model,
                            warmup_prompt,
                            args.prefill_proxy_output_tokens,
                            args.seed,
                        )

                prefill_proxy_rows = []
                for prompt_len in args.prompt_lengths:
                    prompt_ids = token_buffer[:prompt_len]
                    rounds = []
                    for round_idx in range(args.rounds):
                        result = await stream_completion(
                            session,
                            base,
                            args.model,
                            prompt_ids,
                            args.prefill_proxy_output_tokens,
                            args.seed,
                        )
                        rounds.append(
                            {
                                "round": round_idx,
                                "prompt_len": prompt_len,
                                "max_tokens": args.prefill_proxy_output_tokens,
                                **result,
                            }
                        )
                    prefill_proxy_rows.append(
                        {
                            "prompt_len": prompt_len,
                            "max_tokens": args.prefill_proxy_output_tokens,
                            "summary": summarize_latency_rounds(rounds),
                            "rounds": rounds,
                        }
                    )

                decode_prompt = token_buffer[: args.decode_prompt_len]
                decode_profile_rows = []
                for output_len in args.decode_output_lengths:
                    rounds = []
                    for round_idx in range(args.rounds):
                        result = await stream_completion(
                            session,
                            base,
                            args.model,
                            decode_prompt,
                            output_len,
                            args.seed,
                        )
                        rounds.append(
                            {
                                "round": round_idx,
                                "prompt_len": args.decode_prompt_len,
                                "max_tokens": output_len,
                                **result,
                            }
                        )
                    decode_profile_rows.append(
                        {
                            "prompt_len": args.decode_prompt_len,
                            "max_tokens": output_len,
                            "summary": summarize_latency_rounds(rounds),
                            "rounds": rounds,
                        }
                    )

            result = {
                "model": args.model,
                "dtype": args.dtype,
                "enforce_eager": args.enforce_eager,
                "cudagraph_mode": args.cudagraph_mode,
                "compile_no_cg": args.compile_no_cg,
                "cudagraph_copy_inputs": args.cudagraph_copy_inputs,
                "disable_compile_cache": args.disable_compile_cache,
                "rounds": args.rounds,
                "warmup": args.warmup,
                "server_log": str(log_path),
                "server_ready_sec": server_ready_sec,
                "prompt_lengths": args.prompt_lengths,
                "prefill_proxy_note": (
                    "vLLM requires max_tokens >= 1, so this phase uses streaming "
                    "max_tokens=1 TTFT as a prefill-heavy proxy instead of true "
                    "prefill-only latency."
                ),
                "prefill_proxy": prefill_proxy_rows,
                "decode_profile": decode_profile_rows,
            }
            print(json.dumps(result, ensure_ascii=False, indent=2))
            return 0
        finally:
            proc.terminate()
            try:
                proc.wait(timeout=20)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=10)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--gpu-memory-utilization", default="0.8")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--prompt-lengths",
        type=int,
        nargs="+",
        default=[64, 256, 1024, 1984],
    )
    parser.add_argument(
        "--decode-output-lengths",
        type=int,
        nargs="+",
        default=[32, 64],
    )
    parser.add_argument("--decode-prompt-len", type=int, default=64)
    parser.add_argument("--warmup-prompt-len", type=int, default=64)
    parser.add_argument("--prefill-proxy-output-tokens", type=int, default=1)
    parser.add_argument("--server-ready-timeout", type=int, default=600)
    parser.add_argument(
        "--seed-text",
        default=(
            "The capital of France is Paris. "
            "北京是中国的首都。 "
            "RWKV7 is a recurrent world model for language generation."
        ),
    )
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
        "--log",
        default="/tmp/vllm_rwkv7_ttft_bench.log",
        help="server log path",
    )
    args = parser.parse_args()
    return asyncio.run(main_async(args))


if __name__ == "__main__":
    raise SystemExit(main())
