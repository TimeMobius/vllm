# SPDX-License-Identifier: Apache-2.0

import argparse
import json
import math
import random
import statistics
import time
import urllib.error
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any


DEFAULT_PROMPTS = [
    "i am",
    "北京是",
    "The capital of France is",
    "Once upon a time",
    "In a shocking finding, scientists discovered",
    "人工智能的未来",
    "Write a short haiku about the sea",
    "The theory of relativity says",
]


def normalize_base_url(base_url: str) -> str:
    base_url = base_url.rstrip("/")
    if base_url.endswith("/v1"):
        return base_url[:-3]
    return base_url


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    sorted_values = sorted(values)
    rank = max(0, min(len(sorted_values) - 1, int(round(q * (len(sorted_values) - 1)))))
    return sorted_values[rank]


def safe_float_mean(values: list[float]) -> float | None:
    if not values:
        return None
    return statistics.fmean(values)


def safe_rate(count: int, window_sec: float) -> float | None:
    if window_sec <= 0:
        return None
    return count / window_sec


def json_dump(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8")


def jsonl_dump(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def parse_json_if_possible(body: str) -> dict[str, Any] | None:
    if not body:
        return None
    try:
        parsed = json.loads(body)
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else {"_raw": parsed}


def build_headers(api_key: str | None) -> dict[str, str]:
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    return headers


def request_json(
    url: str,
    payload: dict[str, Any] | None,
    headers: dict[str, str],
    timeout: float,
) -> tuple[int, dict[str, Any] | None, str | None]:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read().decode("utf-8")
            parsed = parse_json_if_possible(body)
            return resp.status, parsed, None
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        parsed = parse_json_if_possible(body)
        return exc.code, parsed, body[:500]
    except urllib.error.URLError as exc:
        return 0, None, str(exc.reason)
    except TimeoutError:
        return 0, None, "request timed out"


def health_check(base_url: str, headers: dict[str, str],
                 timeout: float) -> dict[str, Any]:
    health_url = f"{normalize_base_url(base_url)}/health"
    status, payload, error = request_json(health_url, None, headers, timeout)
    return {
        "url": health_url,
        "status_code": status,
        "ok": status == 200,
        "payload": payload,
        "error": error,
    }


def load_prompt_cases(prompt_file: str | None,
                      inline_prompts: list[str]) -> list[dict[str, Any]]:
    if prompt_file is None:
        return [{
            "name": f"prompt_{idx}",
            "prompt": prompt,
        } for idx, prompt in enumerate(inline_prompts)]

    path = Path(prompt_file)
    suffix = path.suffix.lower()
    if suffix == ".txt":
        prompts = [
            line.strip() for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        return [{
            "name": f"prompt_{idx}",
            "prompt": prompt,
        } for idx, prompt in enumerate(prompts)]

    if suffix == ".jsonl":
        rows = []
        for idx, line in enumerate(path.read_text(encoding="utf-8").splitlines()):
            if not line.strip():
                continue
            obj = json.loads(line)
            rows.append(normalize_prompt_case(obj, idx))
        return rows

    if suffix == ".json":
        obj = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(obj, dict) and "prompts" in obj:
            obj = obj["prompts"]
        if not isinstance(obj, list):
            raise ValueError("JSON prompt file must contain a list or a {\"prompts\": [...]} object.")
        return [normalize_prompt_case(item, idx) for idx, item in enumerate(obj)]

    raise ValueError(f"Unsupported prompt file suffix: {suffix}")


def normalize_prompt_case(item: Any, idx: int) -> dict[str, Any]:
    if isinstance(item, str):
        return {"name": f"prompt_{idx}", "prompt": item}
    if not isinstance(item, dict):
        raise ValueError("Prompt entries must be strings or objects.")
    name = item.get("name", f"prompt_{idx}")
    prompt = item.get("prompt")
    messages = item.get("messages")
    if prompt is None and messages is None:
        raise ValueError("Prompt object must contain `prompt` or `messages`.")
    return {
        "name": name,
        "prompt": prompt,
        "messages": messages,
    }


def build_payload(
    *,
    endpoint: str,
    model: str,
    case: dict[str, Any],
    max_tokens: int,
    temperature: float,
    top_p: float,
    seed: int | None,
    extra_body: dict[str, Any],
    return_token_ids: bool,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": model,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
    }
    if seed is not None:
        payload["seed"] = seed

    if endpoint == "completions":
        prompt = case.get("prompt")
        if prompt is None:
            raise ValueError(
                f"Prompt case {case['name']!r} only has messages; use --endpoint chat.")
        payload["prompt"] = prompt
        if return_token_ids:
            payload["return_token_ids"] = True
    else:
        messages = case.get("messages")
        if messages is None:
            prompt = case.get("prompt")
            if prompt is None:
                raise ValueError(f"Prompt case {case['name']!r} is missing prompt/messages.")
            messages = [{"role": "user", "content": prompt}]
        payload["messages"] = messages

    payload.update(extra_body)
    return payload


def extract_choice_text(endpoint: str, response: dict[str, Any] | None) -> str | None:
    if not response:
        return None
    choices = response.get("choices") or []
    if not choices:
        return None
    choice = choices[0]
    if endpoint == "completions":
        text = choice.get("text")
        if text is None:
            return None
        return text if isinstance(text, str) else json.dumps(text, ensure_ascii=False)
    message = choice.get("message") or {}
    content = message.get("content")
    if content is None:
        return None
    return content if isinstance(content, str) else json.dumps(content, ensure_ascii=False)


def extract_finish_reason(response: dict[str, Any] | None) -> str | None:
    if not response:
        return None
    choices = response.get("choices") or []
    if not choices:
        return None
    return choices[0].get("finish_reason")


def extract_output_tokens(response: dict[str, Any] | None) -> int | None:
    if not response:
        return None
    usage = response.get("usage")
    if isinstance(usage, dict) and usage.get("completion_tokens") is not None:
        return int(usage["completion_tokens"])
    choices = response.get("choices") or []
    if choices and isinstance(choices[0], dict) and choices[0].get("token_ids") is not None:
        return len(choices[0]["token_ids"])
    return None


def summarize_inflight(records: list[dict[str, Any]]) -> dict[str, float | int | None]:
    if not records:
        return {
            "peak_inflight_requests": 0,
            "avg_inflight_requests": None,
        }

    events: list[tuple[float, int]] = []
    for row in records:
        events.append((float(row["started_at"]), 1))
        events.append((float(row["finished_at"]), -1))

    # Process start events before finish events when timestamps tie so
    # "all-at-once" burst launches do not undercount peak in-flight requests.
    events.sort(key=lambda item: (item[0], -item[1]))

    current = 0
    peak = 0
    area = 0.0
    window_start = events[0][0]
    window_end = events[-1][0]
    last_ts = window_start

    for ts, delta in events:
        if ts > last_ts:
            area += current * (ts - last_ts)
        current += delta
        peak = max(peak, current)
        last_ts = ts

    active_window_sec = max(window_end - window_start, 0.0)
    avg_inflight = (area / active_window_sec) if active_window_sec > 0 else float(peak)

    return {
        "peak_inflight_requests": peak,
        "avg_inflight_requests": avg_inflight,
    }


def summarize_token_tps_buckets(
    records: list[dict[str, Any]],
    *,
    window_start: float,
    window_end: float,
    bucket_sec: float = 1.0,
) -> dict[str, float | None]:
    if window_end <= window_start or bucket_sec <= 0:
        return {
            "bucket_sec": bucket_sec,
            "avg": None,
            "min": None,
            "max": None,
        }

    token_records: list[tuple[float, float, int]] = []
    for row in records:
        if not row["success"]:
            continue
        token_count = row["completion_tokens"]
        if token_count is None:
            token_count = row["output_tokens"]
        if token_count is None:
            continue
        started_at = float(row["started_at"])
        finished_at = float(row["finished_at"])
        if finished_at <= started_at:
            continue
        token_records.append((started_at, finished_at, int(token_count)))

    if not token_records:
        return {
            "bucket_sec": bucket_sec,
            "avg": None,
            "min": None,
            "max": None,
        }

    num_buckets = max(1, math.ceil((window_end - window_start) / bucket_sec))
    bucket_tokens = [0.0] * num_buckets

    for started_at, finished_at, token_count in token_records:
        duration = finished_at - started_at
        start_idx = max(0, int((started_at - window_start) // bucket_sec))
        end_idx = min(
            num_buckets - 1,
            max(0, int(math.ceil((finished_at - window_start) / bucket_sec)) - 1),
        )
        for bucket_idx in range(start_idx, end_idx + 1):
            bucket_start = window_start + bucket_idx * bucket_sec
            bucket_end = min(bucket_start + bucket_sec, window_end)
            overlap = max(
                0.0,
                min(finished_at, bucket_end) - max(started_at, bucket_start),
            )
            if overlap <= 0:
                continue
            bucket_tokens[bucket_idx] += token_count * (overlap / duration)

    bucket_tps_values = []
    for bucket_idx, token_count in enumerate(bucket_tokens):
        bucket_start = window_start + bucket_idx * bucket_sec
        bucket_end = min(bucket_start + bucket_sec, window_end)
        bucket_duration = bucket_end - bucket_start
        if bucket_duration > 0:
            bucket_tps_values.append(token_count / bucket_duration)

    return {
        "bucket_sec": bucket_sec,
        "avg": safe_float_mean(bucket_tps_values),
        "min": min(bucket_tps_values) if bucket_tps_values else None,
        "max": max(bucket_tps_values) if bucket_tps_values else None,
    }


def issue_request(
    *,
    endpoint: str,
    base_url: str,
    headers: dict[str, str],
    payload: dict[str, Any],
    case_name: str,
    request_idx: int,
    release_at: float,
    timeout: float,
) -> dict[str, Any]:
    now = time.perf_counter()
    if release_at > now:
        time.sleep(release_at - now)

    started_at = time.perf_counter()
    path = "/v1/completions" if endpoint == "completions" else "/v1/chat/completions"
    status_code, response, error_text = request_json(
        f"{normalize_base_url(base_url)}{path}",
        payload,
        headers,
        timeout,
    )
    finished_at = time.perf_counter()

    response_text = extract_choice_text(endpoint, response)
    output_tokens = extract_output_tokens(response)
    success = status_code == 200 and response is not None and "error" not in response
    error_message = None
    if not success:
        if response and isinstance(response.get("error"), dict):
            error_message = response["error"].get("message")
        elif error_text:
            error_message = error_text

    usage = response.get("usage") if isinstance(response, dict) else None

    return {
        "request_idx": request_idx,
        "case_name": case_name,
        "release_at": release_at,
        "started_at": started_at,
        "finished_at": finished_at,
        "start_delay_sec": started_at - release_at,
        "latency_sec": finished_at - started_at,
        "status_code": status_code,
        "success": success,
        "finish_reason": extract_finish_reason(response),
        "output_tokens": output_tokens,
        "prompt_tokens": None if not isinstance(usage, dict) else usage.get("prompt_tokens"),
        "completion_tokens": None if not isinstance(usage, dict) else usage.get("completion_tokens"),
        "total_tokens": None if not isinstance(usage, dict) else usage.get("total_tokens"),
        "response_chars": None if response_text is None else len(response_text),
        "response_preview": None if response_text is None else response_text[:160],
        "error": error_message,
    }


def summarize(
    records: list[dict[str, Any]],
    configured_concurrency: int,
    arrival_rate: float | None,
    dispatch_mode: str,
    worker_count: int,
) -> dict[str, Any]:
    success_records = [row for row in records if row["success"]]
    latency_values = [float(row["latency_sec"]) for row in records]
    start_delay_values = [float(row["start_delay_sec"]) for row in records]
    completion_tokens = [
        int(row["completion_tokens"]) for row in success_records
        if row["completion_tokens"] is not None
    ]
    output_tokens = [
        int(row["output_tokens"]) for row in success_records
        if row["output_tokens"] is not None
    ]
    if records:
        wall_start = min(float(row["release_at"]) for row in records)
        active_start = min(float(row["started_at"]) for row in records)
        wall_end = max(float(row["finished_at"]) for row in records)
        wall_time_sec = wall_end - wall_start
        active_window_sec = wall_end - active_start
        client_queue_delay_sec = active_start - wall_start
    else:
        wall_time_sec = 0.0
        active_window_sec = 0.0
        client_queue_delay_sec = 0.0

    status_counts = Counter(str(row["status_code"]) for row in records)
    error_counts = Counter(row["error"] for row in records if row["error"])
    inflight = summarize_inflight(records)
    token_tps_stats = summarize_token_tps_buckets(
        records,
        window_start=active_start if records else 0.0,
        window_end=wall_end if records else 0.0,
    )

    known_output_tokens = completion_tokens if completion_tokens else output_tokens
    aggregate_output_tps = (
        sum(known_output_tokens) / wall_time_sec
        if wall_time_sec > 0 and known_output_tokens else None
    )
    active_output_tps = (
        sum(known_output_tokens) / active_window_sec
        if active_window_sec > 0 and known_output_tokens else None
    )

    return {
        "request_count": len(records),
        "success_count": len(success_records),
        "error_count": len(records) - len(success_records),
        "success_rate": (len(success_records) / len(records)) if records else None,
        "requested_concurrency_arg": configured_concurrency,
        "configured_concurrency": (
            configured_concurrency if dispatch_mode == "closed_loop" else None
        ),
        "client_concurrency_limit": (
            configured_concurrency if dispatch_mode == "closed_loop" else None
        ),
        "dispatch_mode": dispatch_mode,
        "worker_count": worker_count,
        "arrival_rate_rps": arrival_rate,
        "wall_time_sec": wall_time_sec,
        "active_window_sec": active_window_sec,
        "client_queue_delay_before_first_start_sec": client_queue_delay_sec,
        "request_throughput_rps": safe_rate(len(success_records), wall_time_sec),
        "active_request_throughput_rps": safe_rate(len(success_records),
                                                    active_window_sec),
        "aggregate_output_tps": aggregate_output_tps,
        "token_throughput_tps": aggregate_output_tps,
        "active_output_tps": active_output_tps,
        "latency_sec": {
            "avg": safe_float_mean(latency_values),
            "p50": percentile(latency_values, 0.50),
            "p95": percentile(latency_values, 0.95),
            "p99": percentile(latency_values, 0.99),
            "max": max(latency_values) if latency_values else None,
        },
        "start_delay_sec": {
            "avg": safe_float_mean(start_delay_values),
            "p50": percentile(start_delay_values, 0.50),
            "p95": percentile(start_delay_values, 0.95),
            "p99": percentile(start_delay_values, 0.99),
            "max": max(start_delay_values) if start_delay_values else None,
        },
        "status_counts": dict(status_counts),
        "top_errors": [{"error": key, "count": value}
                        for key, value in error_counts.most_common(5)],
        "known_completion_token_requests": len(known_output_tokens),
        "known_completion_tokens": sum(known_output_tokens) if known_output_tokens else None,
        "token_throughput_tps_stats": token_tps_stats,
        **inflight,
    }


def render_markdown(
    *,
    run_name: str,
    args: argparse.Namespace,
    health: dict[str, Any],
    summary: dict[str, Any],
    result_paths: dict[str, str],
) -> str:
    lines = [
        f"# {run_name}",
        "",
        "## Config",
        "",
        f"- base_url: `{args.base_url}`",
        f"- endpoint: `{args.endpoint}`",
        f"- model: `{args.model}`",
        f"- dispatch_mode: `{summary['dispatch_mode']}`",
        f"- request_count: `{args.num_requests}`",
        f"- requested_concurrency_arg: `{summary['requested_concurrency_arg']}`",
        f"- client_concurrency_limit: `{summary['client_concurrency_limit']}`",
        f"- worker_count: `{summary['worker_count']}`",
        f"- arrival_rate_rps: `{args.arrival_rate}`",
        f"- max_tokens: `{args.max_tokens}`",
        f"- prompt_file: `{args.prompt_file}`",
        "",
        "## Health",
        "",
        f"- checked: `{not args.skip_health_check}`",
        f"- ok: `{health['ok']}`",
        f"- status_code: `{health['status_code']}`",
        "",
        "## Summary",
        "",
        f"- success_count: `{summary['success_count']}` / `{summary['request_count']}`",
        f"- success_rate: `{summary['success_rate']}`",
        f"- wall_time_sec: `{summary['wall_time_sec']}`",
        f"- active_window_sec: `{summary['active_window_sec']}`",
        f"- request_throughput_rps: `{summary['request_throughput_rps']}`",
        f"- active_request_throughput_rps: `{summary['active_request_throughput_rps']}`",
        f"- token_throughput_tps: `{summary['token_throughput_tps']}`",
        f"- active_output_tps: `{summary['active_output_tps']}`",
        f"- token_tps_avg_1s: `{summary['token_throughput_tps_stats']['avg']}`",
        f"- token_tps_min_1s: `{summary['token_throughput_tps_stats']['min']}`",
        f"- token_tps_max_1s: `{summary['token_throughput_tps_stats']['max']}`",
        "",
        "| metric | value |",
        "|---|---:|",
        f"| latency_avg_sec | `{summary['latency_sec']['avg']}` |",
        f"| latency_p50_sec | `{summary['latency_sec']['p50']}` |",
        f"| latency_p95_sec | `{summary['latency_sec']['p95']}` |",
        f"| latency_p99_sec | `{summary['latency_sec']['p99']}` |",
        f"| peak_inflight_requests | `{summary['peak_inflight_requests']}` |",
        f"| avg_inflight_requests | `{summary['avg_inflight_requests']}` |",
        f"| client_queue_before_first_start_sec | `{summary['client_queue_delay_before_first_start_sec']}` |",
        f"| token_tps_bucket_sec | `{summary['token_throughput_tps_stats']['bucket_sec']}` |",
        f"| start_delay_avg_sec | `{summary['start_delay_sec']['avg']}` |",
        f"| start_delay_p95_sec | `{summary['start_delay_sec']['p95']}` |",
        "",
        "## Output Files",
        "",
        f"- summary_json: `{result_paths['summary_json']}`",
        f"- requests_jsonl: `{result_paths['requests_jsonl']}`",
        f"- config_json: `{result_paths['config_json']}`",
    ]
    if summary["top_errors"]:
        lines.extend([
            "",
            "## Top Errors",
            "",
        ])
        for row in summary["top_errors"]:
            lines.append(f"- `{row['count']}` x `{row['error']}`")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Benchmark a remote OpenAI-compatible vLLM endpoint under concurrent load."
    )
    parser.add_argument("--base-url", required=True,
                        help="Remote vLLM base URL, e.g. http://host:8000 or http://host:8000/v1")
    parser.add_argument("--model", required=True)
    parser.add_argument("--endpoint", choices=["completions", "chat"], default="completions")
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--prompt-file", default=None,
                        help="Optional .txt/.json/.jsonl prompt file.")
    parser.add_argument("--prompts", nargs="+", default=DEFAULT_PROMPTS)
    parser.add_argument("--num-requests", type=int, default=128)
    parser.add_argument("--concurrency", type=int, default=16)
    parser.add_argument(
        "--dispatch-mode",
        choices=["closed_loop", "burst"],
        default="closed_loop",
        help=(
            "closed_loop keeps at most --concurrency client requests in flight. "
            "burst launches all requests immediately so queueing happens mainly inside the server."
        ),
    )
    parser.add_argument(
        "--burst-workers",
        type=int,
        default=None,
        help=(
            "Optional thread count override for --dispatch-mode burst. "
            "Defaults to --num-requests."
        ),
    )
    parser.add_argument("--arrival-rate", type=float, default=None,
                        help="If set, stagger request releases at this requests/sec rate.")
    parser.add_argument("--arrival-jitter-sec", type=float, default=0.0,
                        help="Uniform +- jitter added to each release time.")
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--timeout-sec", type=float, default=300.0)
    parser.add_argument("--return-token-ids", action="store_true",
                        help="Request token IDs for /v1/completions so output token counts can be recovered without usage.")
    parser.add_argument("--extra-body-json", default=None,
                        help="Extra JSON object merged into every request body.")
    parser.add_argument("--skip-health-check", action="store_true")
    parser.add_argument("--output-dir", default="/home/liu/vllm/tmp_rwkv7_remote_bench_runs")
    parser.add_argument("--run-name", default=None)
    args = parser.parse_args()

    if args.num_requests <= 0:
        raise ValueError("--num-requests must be positive.")
    if args.concurrency <= 0:
        raise ValueError("--concurrency must be positive.")
    if args.burst_workers is not None and args.burst_workers <= 0:
        raise ValueError("--burst-workers must be positive when provided.")
    if args.arrival_rate is not None and args.arrival_rate <= 0:
        raise ValueError("--arrival-rate must be positive when provided.")
    if args.arrival_jitter_sec < 0:
        raise ValueError("--arrival-jitter-sec must be non-negative.")

    prompt_cases = load_prompt_cases(args.prompt_file, args.prompts)
    if not prompt_cases:
        raise ValueError("No prompts were loaded.")

    extra_body = {}
    if args.extra_body_json is not None:
        extra_body = json.loads(args.extra_body_json)
        if not isinstance(extra_body, dict):
            raise ValueError("--extra-body-json must parse to a JSON object.")

    run_name = args.run_name or time.strftime("%Y%m%d_%H%M%S_remote_concurrency")
    output_dir = Path(args.output_dir) / run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    headers = build_headers(args.api_key)
    health = {
        "url": f"{normalize_base_url(args.base_url)}/health",
        "status_code": None,
        "ok": None,
        "payload": None,
        "error": None,
    }
    if not args.skip_health_check:
        health = health_check(args.base_url, headers, timeout=min(args.timeout_sec, 10.0))

    worker_count = args.concurrency
    if args.dispatch_mode == "burst":
        worker_count = args.burst_workers or args.num_requests

    bench_start = time.perf_counter() + 0.2
    tasks = []
    for request_idx in range(args.num_requests):
        case = prompt_cases[request_idx % len(prompt_cases)]
        release_at = bench_start
        if args.arrival_rate is not None:
            release_at += request_idx / args.arrival_rate
        if args.arrival_jitter_sec > 0:
            release_at += random.uniform(-args.arrival_jitter_sec,
                                         args.arrival_jitter_sec)
            release_at = max(release_at, bench_start)
        payload = build_payload(
            endpoint=args.endpoint,
            model=args.model,
            case=case,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            seed=args.seed,
            extra_body=extra_body,
            return_token_ids=args.return_token_ids,
        )
        tasks.append({
            "request_idx": request_idx,
            "case_name": case["name"],
            "release_at": release_at,
            "payload": payload,
        })

    records = []
    with ThreadPoolExecutor(max_workers=worker_count) as pool:
        futures = [
            pool.submit(
                issue_request,
                endpoint=args.endpoint,
                base_url=args.base_url,
                headers=headers,
                payload=task["payload"],
                case_name=task["case_name"],
                request_idx=task["request_idx"],
                release_at=task["release_at"],
                timeout=args.timeout_sec,
            )
            for task in tasks
        ]
        for future in as_completed(futures):
            records.append(future.result())

    records.sort(key=lambda row: row["request_idx"])
    summary = summarize(records, args.concurrency, args.arrival_rate,
                        args.dispatch_mode, worker_count)

    config_payload = {
        "run_name": run_name,
        "base_url": args.base_url,
        "endpoint": args.endpoint,
        "model": args.model,
        "num_requests": args.num_requests,
        "concurrency": args.concurrency,
        "dispatch_mode": args.dispatch_mode,
        "worker_count": worker_count,
        "burst_workers": args.burst_workers,
        "arrival_rate": args.arrival_rate,
        "arrival_jitter_sec": args.arrival_jitter_sec,
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "seed": args.seed,
        "timeout_sec": args.timeout_sec,
        "prompt_file": args.prompt_file,
        "prompt_case_count": len(prompt_cases),
        "return_token_ids": args.return_token_ids,
        "extra_body_keys": sorted(extra_body.keys()),
        "has_api_key": bool(args.api_key),
    }

    config_json_path = output_dir / "config.json"
    summary_json_path = output_dir / "summary.json"
    requests_jsonl_path = output_dir / "requests.jsonl"
    summary_md_path = output_dir / "summary.md"

    json_dump(config_json_path, config_payload)
    json_dump(summary_json_path, {
        "config": config_payload,
        "health": health,
        "summary": summary,
    })
    jsonl_dump(requests_jsonl_path, records)

    result_paths = {
        "config_json": str(config_json_path),
        "summary_json": str(summary_json_path),
        "requests_jsonl": str(requests_jsonl_path),
        "summary_md": str(summary_md_path),
    }
    summary_md_path.write_text(
        render_markdown(
            run_name=run_name,
            args=args,
            health=health,
            summary=summary,
            result_paths=result_paths,
        ),
        encoding="utf-8",
    )

    print(
        json.dumps(
            {
                "run_name": run_name,
                "output_dir": str(output_dir),
                "health": health,
                "summary": summary,
                "paths": result_paths,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
