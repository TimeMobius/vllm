# fp16_c1_1

## Config

- base_url: `http://127.0.0.1:8032`
- endpoint: `completions`
- model: `xiaoke-5-fp16`
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
- wall_time_sec: `1.2302960474044085`
- active_window_sec: `1.2302006287500262`
- request_throughput_rps: `0.8128124950980126`
- active_request_throughput_rps: `0.8128755396719908`
- token_throughput_tps: `26.009999843136402`
- active_output_tps: `26.012017269503705`
- token_throughput_tps_avg: `26.012017269503705`
- token_throughput_tps_min: `26.012017269503705`
- token_throughput_tps_max: `26.012017269503705`
- request_token_tps_avg: `26.012017269503705`
- request_token_tps_p50: `26.012017269503705`
- request_token_tps_p95: `26.012017269503705`
- request_token_tps_weighted_avg: `26.012017269503705`

| metric | value |
|---|---:|
| latency_avg_sec | `1.2302006287500262` |
| latency_p50_sec | `1.2302006287500262` |
| latency_p95_sec | `1.2302006287500262` |
| latency_p99_sec | `1.2302006287500262` |
| peak_inflight_requests | `1` |
| avg_inflight_requests | `1.0` |
| client_queue_before_first_start_sec | `9.541865438222885e-05` |
| token_tps_bucket_sec | `1.0` |
| request_token_tps_min | `26.012017269503705` |
| request_token_tps_max | `26.012017269503705` |
| start_delay_avg_sec | `9.541865438222885e-05` |
| start_delay_p95_sec | `9.541865438222885e-05` |

## Output Files

- summary_json: `/tmp/rwkv7_fp16_experimental_perf_20260821/fp16_c1_1/summary.json`
- requests_jsonl: `/tmp/rwkv7_fp16_experimental_perf_20260821/fp16_c1_1/requests.jsonl`
- config_json: `/tmp/rwkv7_fp16_experimental_perf_20260821/fp16_c1_1/config.json`
