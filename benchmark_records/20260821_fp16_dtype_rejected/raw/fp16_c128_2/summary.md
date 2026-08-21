# fp16_c128_2

## Config

- base_url: `http://127.0.0.1:8032`
- endpoint: `completions`
- model: `xiaoke-5-fp16`
- dispatch_mode: `closed_loop`
- request_count: `128`
- requested_concurrency_arg: `128`
- client_concurrency_limit: `128`
- worker_count: `128`
- arrival_rate_rps: `None`
- max_tokens: `32`
- prompt_file: `None`

## Health

- checked: `True`
- ok: `True`
- status_code: `200`

## Summary

- success_count: `128` / `128`
- success_rate: `1.0`
- wall_time_sec: `3.4099648585543036`
- active_window_sec: `3.4098902009427547`
- request_throughput_rps: `37.53704372609493`
- active_request_throughput_rps: `37.53786557837287`
- token_throughput_tps: `1201.1853992350377`
- active_output_tps: `1201.211698507932`
- token_throughput_tps_avg: `1200.1280395445986`
- token_throughput_tps_min: `1192.3526221411992`
- token_throughput_tps_max: `1207.1466554826627`
- request_token_tps_avg: `9.430833245958299`
- request_token_tps_p50: `9.431827273315703`
- request_token_tps_p95: `9.46115828781334`
- request_token_tps_weighted_avg: `9.430715726005475`

| metric | value |
|---|---:|
| latency_avg_sec | `3.3931677011278225` |
| latency_p50_sec | `3.3929012147709727` |
| latency_p95_sec | `3.4065795969218016` |
| latency_p99_sec | `3.4084963258355856` |
| peak_inflight_requests | `128` |
| avg_inflight_requests | `127.3722730498127` |
| client_queue_before_first_start_sec | `7.46576115489006e-05` |
| token_tps_bucket_sec | `1.0` |
| request_token_tps_min | `9.386981206731221` |
| request_token_tps_max | `9.729054139041585` |
| start_delay_avg_sec | `0.012318040673562791` |
| start_delay_p95_sec | `0.023953200317919254` |

## Output Files

- summary_json: `/tmp/rwkv7_fp16_experimental_perf_20260821/fp16_c128_2/summary.json`
- requests_jsonl: `/tmp/rwkv7_fp16_experimental_perf_20260821/fp16_c128_2/requests.jsonl`
- config_json: `/tmp/rwkv7_fp16_experimental_perf_20260821/fp16_c128_2/config.json`
