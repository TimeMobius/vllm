# 20260820_mix6_baseline_eager_c8_mt32_r3

## Config

- base_url: `http://127.0.0.1:8030`
- endpoint: `completions`
- model: `xiaoke-5`
- dispatch_mode: `closed_loop`
- request_count: `16`
- requested_concurrency_arg: `8`
- client_concurrency_limit: `8`
- worker_count: `8`
- arrival_rate_rps: `None`
- max_tokens: `32`
- prompt_file: `None`

## Health

- checked: `True`
- ok: `True`
- status_code: `200`

## Summary

- success_count: `16` / `16`
- success_rate: `1.0`
- wall_time_sec: `4.093820350244641`
- active_window_sec: `4.09372407477349`
- request_throughput_rps: `3.908329782728219`
- active_request_throughput_rps: `3.9084216981295437`
- token_throughput_tps: `122.86811754451838`
- active_output_tps: `122.87100713494752`
- token_throughput_tps_avg: `117.80816071610786`
- token_throughput_tps_min: `94.93886043484467`
- token_throughput_tps_max: `125.43437797424944`
- request_token_tps_avg: `15.585179272936099`
- request_token_tps_p50: `15.544415931901012`
- request_token_tps_p95: `15.962978860694708`
- request_token_tps_weighted_avg: `15.582149881407869`

| metric | value |
|---|---:|
| latency_avg_sec | `2.017532897531055` |
| latency_p50_sec | `2.033729936927557` |
| latency_p95_sec | `2.0879502072930336` |
| latency_p99_sec | `2.12628741748631` |
| peak_inflight_requests | `8` |
| avg_inflight_requests | `7.885369353400544` |
| client_queue_before_first_start_sec | `9.627547115087509e-05` |
| token_tps_bucket_sec | `1.0` |
| request_token_tps_min | `15.049705762652867` |
| request_token_tps_max | `15.96348350060437` |
| start_delay_avg_sec | `1.0162275960901752` |
| start_delay_p95_sec | `2.088710674084723` |

## Output Files

- summary_json: `benchmark_records/20260820_mix6_baseline_eager_c8_mt32_r3/summary.json`
- requests_jsonl: `benchmark_records/20260820_mix6_baseline_eager_c8_mt32_r3/requests.jsonl`
- config_json: `benchmark_records/20260820_mix6_baseline_eager_c8_mt32_r3/config.json`
