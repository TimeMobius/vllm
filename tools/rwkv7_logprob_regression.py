#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Compare RWKV7 service log-probability traces.

The OpenAI-compatible API does not expose the complete vocabulary logits.  Its
``logprobs`` response does expose the selected-token log-probability and the
Top-K distribution for every generated step; this tool compares those values
instead of relying only on generated token IDs.  It is intended for eager vs
CUDA-Graph A/B checks and stores both raw responses and a machine-readable
error report.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

DEFAULT_PROMPTS = [
    "The capital of France is",
    "北京是",
    "RWKV7 is a recurrent world model for",
    "人工智能的未来",
]


def _request_json(url: str, payload: dict[str, Any], timeout: float) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code} from {url}: {body[:500]}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"request failed for {url}: {exc.reason}") from exc
    result = json.loads(body)
    if not isinstance(result, dict):
        raise RuntimeError(f"unexpected response from {url}: {type(result)!r}")
    if "error" in result:
        raise RuntimeError(f"API error from {url}: {result['error']}")
    return result


def _normalise_url(base_url: str) -> str:
    base_url = base_url.rstrip("/")
    if base_url.endswith("/v1"):
        base_url = base_url[:-3]
    return f"{base_url}/v1/completions"


def _load_prompts(path: str | None, inline: list[str]) -> list[dict[str, str]]:
    if path is None:
        return [{"name": f"prompt_{i}", "prompt": p} for i, p in enumerate(inline)]
    source = Path(path)
    if source.suffix == ".txt":
        prompts = [
            line.strip()
            for line in source.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        return [{"name": f"prompt_{i}", "prompt": p} for i, p in enumerate(prompts)]
    payload = json.loads(source.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        payload = payload.get("prompts")
    if not isinstance(payload, list):
        raise ValueError("prompt file must be a list or an object with a prompts list")
    cases: list[dict[str, str]] = []
    for i, item in enumerate(payload):
        if isinstance(item, str):
            cases.append({"name": f"prompt_{i}", "prompt": item})
        elif isinstance(item, dict) and isinstance(item.get("prompt"), str):
            cases.append(
                {
                    "name": str(item.get("name", f"prompt_{i}")),
                    "prompt": item["prompt"],
                }
            )
        else:
            raise ValueError(f"invalid prompt case at index {i}")
    return cases


def _extract_trace(response: dict[str, Any]) -> dict[str, Any]:
    choices = response.get("choices") or []
    if not choices:
        raise RuntimeError("completion response has no choices")
    choice = choices[0]
    logprobs = choice.get("logprobs") or {}
    tokens = logprobs.get("tokens") or []
    selected = logprobs.get("token_logprobs") or []
    top = logprobs.get("top_logprobs") or []
    if len(tokens) != len(selected) or len(tokens) != len(top):
        raise RuntimeError(
            "incomplete logprobs response: "
            f"tokens={len(tokens)} selected={len(selected)} top={len(top)}"
        )
    return {
        "text": choice.get("text", ""),
        "finish_reason": choice.get("finish_reason"),
        "tokens": tokens,
        "selected_logprobs": selected,
        "top_logprobs": top,
        "usage": response.get("usage"),
    }


def collect(
    base_url: str,
    model: str,
    prompts: list[dict[str, str]],
    *,
    max_tokens: int,
    top_k: int,
    seed: int,
    timeout: float,
) -> list[dict[str, Any]]:
    url = _normalise_url(base_url)
    rows = []
    for case in prompts:
        payload = {
            "model": model,
            "prompt": case["prompt"],
            "max_tokens": max_tokens,
            "temperature": 0,
            "top_p": 1,
            "seed": seed,
            "logprobs": top_k,
        }
        started = time.perf_counter()
        response = _request_json(url, payload, timeout)
        trace = _extract_trace(response)
        rows.append(
            {
                "name": case["name"],
                "prompt": case["prompt"],
                "latency_sec": time.perf_counter() - started,
                "request": payload,
                "trace": trace,
            }
        )
    return rows


def _finite_float(value: Any) -> float | None:
    if value is None:
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def compare_traces(
    reference: list[dict[str, Any]], candidate: list[dict[str, Any]]
) -> dict[str, Any]:
    candidate_by_name = {row["name"]: row for row in candidate}
    errors: list[float] = []
    selected_errors: list[float] = []
    top1_disagreements = 0
    top1_total = 0
    text_mismatches = 0
    step_count = 0
    prompt_reports = []

    for reference_row in reference:
        candidate_row = candidate_by_name.get(reference_row["name"])
        if candidate_row is None:
            prompt_reports.append({"name": reference_row["name"], "missing": True})
            continue
        reference_trace = reference_row["trace"]
        candidate_trace = candidate_row["trace"]
        if reference_trace["text"] != candidate_trace["text"]:
            text_mismatches += 1
        ref_steps = min(len(reference_trace["tokens"]), len(candidate_trace["tokens"]))
        row_errors: list[float] = []
        row_selected_errors: list[float] = []
        row_top1_disagreements = 0
        for index in range(ref_steps):
            ref_top = {
                str(token): _finite_float(value)
                for token, value in reference_trace["top_logprobs"][index].items()
            }
            cand_top = {
                str(token): _finite_float(value)
                for token, value in candidate_trace["top_logprobs"][index].items()
            }
            common = [
                abs(ref_top[token] - cand_top[token])
                for token in ref_top.keys() & cand_top.keys()
                if ref_top[token] is not None and cand_top[token] is not None
            ]
            row_errors.extend(common)
            errors.extend(common)
            ref_selected = _finite_float(reference_trace["selected_logprobs"][index])
            cand_selected = _finite_float(candidate_trace["selected_logprobs"][index])
            if ref_selected is not None and cand_selected is not None:
                delta = abs(ref_selected - cand_selected)
                row_selected_errors.append(delta)
                selected_errors.append(delta)
            ref_top1 = (
                max(ref_top, key=lambda token: ref_top[token]) if ref_top else None
            )
            cand_top1 = (
                max(cand_top, key=lambda token: cand_top[token]) if cand_top else None
            )
            if ref_top1 is not None and cand_top1 is not None:
                top1_total += 1
                if ref_top1 != cand_top1:
                    top1_disagreements += 1
                    row_top1_disagreements += 1
            step_count += 1
        prompt_reports.append(
            {
                "name": reference_row["name"],
                "reference_steps": len(reference_trace["tokens"]),
                "candidate_steps": len(candidate_trace["tokens"]),
                "common_steps": ref_steps,
                "text_equal": reference_trace["text"] == candidate_trace["text"],
                "max_common_topk_abs_error": max(row_errors, default=0.0),
                "mean_common_topk_abs_error": statistics.fmean(row_errors)
                if row_errors
                else 0.0,
                "max_selected_logprob_abs_error": max(row_selected_errors, default=0.0),
                "top1_disagreements": row_top1_disagreements,
            }
        )

    return {
        "prompt_count": len(reference),
        "text_mismatch_count": text_mismatches,
        "step_count": step_count,
        "top1_disagreement_count": top1_disagreements,
        "top1_disagreement_rate": (
            top1_disagreements / top1_total if top1_total else 0.0
        ),
        "common_topk_value_count": len(errors),
        "max_common_topk_abs_error": max(errors, default=0.0),
        "mean_common_topk_abs_error": statistics.fmean(errors) if errors else 0.0,
        "max_selected_logprob_abs_error": max(selected_errors, default=0.0),
        "mean_selected_logprob_abs_error": (
            statistics.fmean(selected_errors) if selected_errors else 0.0
        ),
        "prompts": prompt_reports,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-url", default=None)
    parser.add_argument("--candidate-url", default=None)
    parser.add_argument(
        "--collect-url",
        default=None,
        help="Collect one raw trace set instead of comparing two endpoints.",
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--prompts", nargs="*", default=DEFAULT_PROMPTS)
    parser.add_argument("--prompt-file", default=None)
    parser.add_argument("--max-tokens", type=int, default=16)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--timeout-sec", type=float, default=300.0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.top_k < 1:
        raise ValueError("--top-k must be positive")
    prompts = _load_prompts(args.prompt_file, args.prompts)
    if args.collect_url is not None:
        samples = collect(
            args.collect_url,
            args.model,
            prompts,
            max_tokens=args.max_tokens,
            top_k=args.top_k,
            seed=args.seed,
            timeout=args.timeout_sec,
        )
        result = {
            "url": args.collect_url,
            "model": args.model,
            "max_tokens": args.max_tokens,
            "top_k": args.top_k,
            "seed": args.seed,
            "samples": samples,
        }
    else:
        if args.reference_url is None or args.candidate_url is None:
            parser.error(
                "provide --collect-url or both --reference-url and --candidate-url"
            )
        reference = collect(
            args.reference_url,
            args.model,
            prompts,
            max_tokens=args.max_tokens,
            top_k=args.top_k,
            seed=args.seed,
            timeout=args.timeout_sec,
        )
        candidate = collect(
            args.candidate_url,
            args.model,
            prompts,
            max_tokens=args.max_tokens,
            top_k=args.top_k,
            seed=args.seed,
            timeout=args.timeout_sec,
        )
        result = {
            "reference_url": args.reference_url,
            "candidate_url": args.candidate_url,
            "model": args.model,
            "max_tokens": args.max_tokens,
            "top_k": args.top_k,
            "seed": args.seed,
            "reference": reference,
            "candidate": candidate,
            "comparison": compare_traces(reference, candidate),
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    if "comparison" in result:
        print(json.dumps(result["comparison"], ensure_ascii=False, indent=2))
    else:
        print(json.dumps({"sample_count": len(result["samples"])}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
