# fp8_c1_3

## Config

- base_url: `http://127.0.0.1:8031`
- endpoint: `completions`
- model: `xiaoke-5-fp8`
- dispatch_mode: `closed_loop`
- request_count: `1`
- requested_concurrency_arg: `1`
- client_concurrency_limit: `1`
- worker_count: `1`
- arrival_rate_rps: `None`
- max_tokens: `32`
- prompt_file: `None`

## Health

- checked: `True`
- ok: `True`
- status_code: `200`

## Summary

- success_count: `1` / `1`
- success_rate: `1.0`
- wall_time_sec: `1.0407785410061479`
- active_window_sec: `1.0406282069161534`
- request_throughput_rps: `0.9608191950549574`
- active_request_throughput_rps: `0.960957999556294`
- token_throughput_tps: `30.746214241758636`
- active_output_tps: `30.75065598580141`
- token_throughput_tps_avg: `30.75065598580141`
- token_throughput_tps_min: `30.75065598580141`
- token_throughput_tps_max: `30.75065598580141`
- request_token_tps_avg: `30.75065598580141`
- request_token_tps_p50: `30.75065598580141`
- request_token_tps_p95: `30.75065598580141`
- request_token_tps_weighted_avg: `30.75065598580141`

| metric | value |
|---|---:|
| latency_avg_sec | `1.0406282069161534` |
| latency_p50_sec | `1.0406282069161534` |
| latency_p95_sec | `1.0406282069161534` |
| latency_p99_sec | `1.0406282069161534` |
| peak_inflight_requests | `1` |
| avg_inflight_requests | `1.0` |
| client_queue_before_first_start_sec | `0.00015033408999443054` |
| token_tps_bucket_sec | `1.0` |
| request_token_tps_min | `30.75065598580141` |
| request_token_tps_max | `30.75065598580141` |
| start_delay_avg_sec | `0.00015033408999443054` |
| start_delay_p95_sec | `0.00015033408999443054` |

## Output Files

- summary_json: `/tmp/rwkv7_fp8_experimental_perf_20260821/fp8_c1_3/summary.json`
- requests_jsonl: `/tmp/rwkv7_fp8_experimental_perf_20260821/fp8_c1_3/requests.jsonl`
- config_json: `/tmp/rwkv7_fp8_experimental_perf_20260821/fp8_c1_3/config.json`
