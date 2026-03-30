import argparse
import json
import subprocess
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


def post(base: str, path: str, payload: dict) -> dict:
    req = urllib.request.Request(
        base + path,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=300) as resp:
        return json.loads(resp.read().decode("utf-8"))


def is_healthy(base: str) -> bool:
    try:
        with urllib.request.urlopen(f"{base}/health", timeout=2) as resp:
            return resp.status == 200
    except (urllib.error.URLError, TimeoutError):
        return False


def tokenize(base: str, model: str, prompt: str) -> list[int]:
    return post(base, "/tokenize", {"model": model, "prompt": prompt})["tokens"]


def complete_with_ids(
    base: str, model: str, prompt_ids: list[int], max_tokens: int, seed: int
) -> list[int]:
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
    return obj["choices"][0]["token_ids"]


def run_round(
    base: str,
    model: str,
    prompt_ids_by_prompt: dict[str, list[int]],
    max_tokens: int,
    prompts: list[str],
    seed: int,
) -> dict[str, list[int]]:
    with ThreadPoolExecutor(max_workers=len(prompts)) as pool:
        futures = {
            prompt: pool.submit(
                complete_with_ids,
                base,
                model,
                prompt_ids_by_prompt[prompt],
                max_tokens,
                seed,
            )
            for prompt in prompts
        }
    return {prompt: futures[prompt].result() for prompt in prompts}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--gpu-memory-utilization", default="0.8")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--max-tokens", type=int, default=16)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--no-async-scheduling", action="store_true")
    parser.add_argument(
        "--prompts",
        nargs="+",
        default=["i am", "北京是", "The capital of France is"],
    )
    parser.add_argument(
        "--log", default="/tmp/vllm_rwkv7_concurrency.log", help="server log path"
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
        args.gpu_memory_utilization,
        "--dtype",
        args.dtype,
        "--host",
        "127.0.0.1",
        "--port",
        str(args.port),
    ]
    if args.no_async_scheduling:
        cmd.append("--no-async-scheduling")

    with log_path.open("w", encoding="utf-8") as logf:
        proc = subprocess.Popen(cmd, stdout=logf, stderr=subprocess.STDOUT)
        try:
            deadline = time.time() + 480
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

            prompt_ids_by_prompt = {
                prompt: tokenize(base, args.model, prompt) for prompt in args.prompts
            }
            baseline = run_round(
                base,
                args.model,
                prompt_ids_by_prompt,
                args.max_tokens,
                args.prompts,
                seed=0,
            )

            rounds = []
            for round_idx in range(args.rounds):
                outputs = run_round(
                    base,
                    args.model,
                    prompt_ids_by_prompt,
                    args.max_tokens,
                    args.prompts,
                    seed=0,
                )
                rounds.append(
                    {
                        "round": round_idx,
                        "matches_baseline": {
                            prompt: outputs[prompt] == baseline[prompt]
                            for prompt in args.prompts
                        },
                        "outputs": outputs,
                    }
                )

            print(
                json.dumps(
                    {
                        "model": args.model,
                        "max_tokens": args.max_tokens,
                        "prompts": args.prompts,
                        "baseline": baseline,
                        "rounds": rounds,
                        "log": str(log_path),
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
