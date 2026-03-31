import argparse
import json
import os
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path


BASE = "http://127.0.0.1:8000"
DEFAULT_MODEL = "RWKV/RWKV7-Goose-World2.8-0.1B-HF"
PROMPTS = ["i am", "北京是", "The capital of France is"]


def post(path: str, payload: dict) -> dict:
    req = urllib.request.Request(
        BASE + path,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=300) as resp:
        return json.loads(resp.read().decode("utf-8"))


def is_healthy() -> bool:
    try:
        with urllib.request.urlopen(f"{BASE}/health", timeout=2) as resp:
            return resp.status == 200
    except (urllib.error.URLError, TimeoutError):
        return False


def tokenize(model: str, prompt: str) -> list[int]:
    return post("/tokenize", {"model": model, "prompt": prompt})["tokens"]


def complete_with_ids(model: str, prompt_ids: list[int], max_tokens: int) -> list[int]:
    obj = post(
        "/v1/completions",
        {
            "model": model,
            "prompt": prompt_ids,
            "max_tokens": max_tokens,
            "temperature": 0.0,
            "top_p": 1.0,
            "return_token_ids": True,
            "seed": 0,
        },
    )
    return obj["choices"][0]["token_ids"]


def run_compare(model: str) -> list[tuple[str, list[int], list[int], bool]]:
    rows: list[tuple[str, list[int], list[int], bool]] = []
    for prompt in PROMPTS:
        prompt_ids = tokenize(model, prompt)
        one_shot_ids = complete_with_ids(model, prompt_ids, 8)

        cur_ids = list(prompt_ids)
        step_ids: list[int] = []
        for _ in range(8):
            next_ids = complete_with_ids(model, cur_ids, 1)
            step_ids.extend(next_ids)
            cur_ids.extend(next_ids)

        rows.append((prompt, one_shot_ids, step_ids, one_shot_ids == step_ids))
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--no-async-scheduling", action="store_true")
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--max-num-batched-tokens", type=int, default=None)
    parser.add_argument("--enable-prefix-caching", action="store_true")
    parser.add_argument("--no-enable-chunked-prefill", action="store_true")
    parser.add_argument("--compile-no-cg", action="store_true")
    parser.add_argument("--cudagraph-copy-inputs", action="store_true")
    parser.add_argument("--disable-compile-cache", action="store_true")
    parser.add_argument("--compilation-config", default=None)
    parser.add_argument(
        "--log", default="/tmp/vllm_rwkv7_compare.log", help="server log path"
    )
    args = parser.parse_args()

    log_path = Path(args.log)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    if log_path.exists():
        log_path.unlink()

    cmd = [
        "vllm",
        "serve",
        args.model,
        "--trust-remote-code",
        "--gpu-memory-utilization",
        "0.8",
        "--dtype",
        args.dtype,
        "--host",
        "127.0.0.1",
        "--port",
        "8000",
    ]
    if args.compile_no_cg:
        cmd.append("-cc.cudagraph_mode=none")
    if args.cudagraph_copy_inputs:
        cmd.append("-cc.cudagraph_copy_inputs=true")
    if args.compilation_config is not None:
        cmd.extend(["-cc", args.compilation_config])
    if args.no_async_scheduling:
        cmd.append("--no-async-scheduling")
    if args.no_enable_chunked_prefill:
        cmd.append("--no-enable-chunked-prefill")
    if args.max_num_batched_tokens is not None:
        cmd.extend(["--max-num-batched-tokens", str(args.max_num_batched_tokens)])
    if args.enable_prefix_caching:
        cmd.append("--enable-prefix-caching")

    with log_path.open("w", encoding="utf-8") as logf:
        env = None
        if args.disable_compile_cache:
            env = os.environ.copy()
            if args.disable_compile_cache:
                env["VLLM_DISABLE_COMPILE_CACHE"] = "1"

        proc = subprocess.Popen(
            cmd,
            stdout=logf,
            stderr=subprocess.STDOUT,
            env=env,
        )
        try:
            deadline = time.time() + 360
            while time.time() < deadline:
                if is_healthy():
                    break
                if proc.poll() is not None:
                    print("server exited early")
                    return 2
                time.sleep(2)
            else:
                print("server not ready before timeout")
                return 3

            rows = run_compare(args.model)
            for prompt, one_ids, step_ids, ok in rows:
                print(f"prompt={prompt!r}")
                print(f"  one_shot={one_ids}")
                print(f"  step_by_step={step_ids}")
                print(f"  match={ok}")
            return 0
        finally:
            proc.terminate()
            try:
                proc.wait(timeout=20)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=10)
            print(f"log={log_path}")


if __name__ == "__main__":
    raise SystemExit(main())
