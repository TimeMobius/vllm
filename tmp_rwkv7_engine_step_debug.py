import argparse
import json

from vllm.engine.arg_utils import EngineArgs
from vllm.sampling_params import SamplingParams
from vllm.v1.engine.llm_engine import LLMEngine


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--max-tokens", type=int, default=16)
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.8)
    parser.add_argument("--max-num-seqs", type=int, default=1)
    parser.add_argument("--no-async-scheduling", action="store_true")
    args = parser.parse_args()

    engine_args = EngineArgs(
        model=args.model,
        trust_remote_code=True,
        dtype=args.dtype,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_num_seqs=args.max_num_seqs,
        disable_log_stats=True,
        enable_prefix_caching=True,
        async_scheduling=not args.no_async_scheduling,
    )
    engine = LLMEngine.from_engine_args(engine_args)

    request_id = "rwkv7-debug"
    params = SamplingParams(
        temperature=0.0,
        top_p=1.0,
        max_tokens=args.max_tokens,
        seed=0,
        detokenize=True,
        skip_special_tokens=False,
        spaces_between_special_tokens=False,
    )
    engine.add_request(request_id, args.prompt, params)

    step_rows = []
    while engine.has_unfinished_requests():
        outputs = engine.step()
        row = {
            "num_outputs": len(outputs),
            "outputs": [],
        }
        for out in outputs:
            row["outputs"].append(
                {
                    "request_id": out.request_id,
                    "finished": out.finished,
                    "num_cached_tokens": out.num_cached_tokens,
                    "token_ids": list(out.outputs[0].token_ids),
                    "text": out.outputs[0].text,
                    "finish_reason": out.outputs[0].finish_reason,
                }
            )
        step_rows.append(row)

    print(
        json.dumps(
            {
                "model": args.model,
                "prompt": args.prompt,
                "max_tokens": args.max_tokens,
                "no_async_scheduling": args.no_async_scheduling,
                "steps": step_rows,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
