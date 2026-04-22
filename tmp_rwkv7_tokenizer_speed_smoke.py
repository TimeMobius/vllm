# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import argparse
import json
import statistics
import time

from vllm import LLM, SamplingParams


def build_prompts(repeat: int) -> list[str]:
    base_prompts = [
        "The capital of France is",
        "User: Say hello in one short sentence.\n\nAssistant:",
        "\u5317\u4eac\u662f\u4e2d\u56fd\u7684\u9996\u90fd\u3002",
        (
            "Today is a beautiful day. "
            "\u4eca\u5929\u662f\u7f8e\u597d\u7684\u4e00\u5929\u3002"
        ),
        "<|rwkv_tokenizer_end_of_text|>User: hi\n\nAssistant:",
        "RWKV7 tokenizer speed check: " + "abc123 test " * 16,
    ]
    return base_prompts * repeat


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        default="/mnt/d/codes/RWKV7-Goose-World2.9-0.4B-HF",
    )
    parser.add_argument("--tokenizer-mode", default="auto")
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--rounds", type=int, default=1)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--warmup-tokens", type=int, default=8)
    parser.add_argument("--max-model-len", type=int, default=2048)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.55)
    parser.add_argument("--no-enforce-eager", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    prompts = build_prompts(args.repeat)
    sampling_params = SamplingParams(temperature=0.0, max_tokens=args.max_tokens)

    init_start = time.perf_counter()
    llm = LLM(
        model=args.model,
        trust_remote_code=True,
        tokenizer_mode=args.tokenizer_mode,
        enforce_eager=not args.no_enforce_eager,
        max_model_len=args.max_model_len,
        max_num_seqs=len(prompts),
        gpu_memory_utilization=args.gpu_memory_utilization,
    )
    init_sec = time.perf_counter() - init_start

    if args.warmup_tokens > 0:
        llm.generate(
            prompts[:1],
            SamplingParams(temperature=0.0, max_tokens=args.warmup_tokens),
        )

    round_results = []
    first_output = ""
    for round_index in range(args.rounds):
        run_start = time.perf_counter()
        outputs = llm.generate(prompts, sampling_params)
        wall_sec = time.perf_counter() - run_start

        output_tokens = sum(len(output.outputs[0].token_ids) for output in outputs)
        input_tokens = sum(len(output.prompt_token_ids) for output in outputs)
        if not first_output:
            first_output = outputs[0].outputs[0].text[:160]
        round_result = {
            "round": round_index,
            "wall_sec": wall_sec,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "output_tps": output_tokens / wall_sec,
        }
        round_results.append(round_result)
        print("ROUND_JSON=" + json.dumps(round_result, ensure_ascii=True))

    output_tps_values = [result["output_tps"] for result in round_results]

    print(
        "RESULT_JSON="
        + json.dumps(
            {
                "mode": args.tokenizer_mode,
                "init_sec": init_sec,
                "rounds": args.rounds,
                "num_prompts": len(prompts),
                "input_tokens": round_results[0]["input_tokens"],
                "output_tokens_per_round": round_results[0]["output_tokens"],
                "output_tps_avg": statistics.fmean(output_tps_values),
                "output_tps_median": statistics.median(output_tps_values),
                "output_tps_min": min(output_tps_values),
                "output_tps_max": max(output_tps_values),
                "round_results": round_results,
                "first_output": first_output,
            },
            ensure_ascii=True,
        )
    )


if __name__ == "__main__":
    main()
