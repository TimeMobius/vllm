# 20260820_mix6_baseline_eager_c8_mt32_r1

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
- wall_time_sec: `4.187987416982651`
- active_window_sec: `4.1879035122692585`
- request_throughput_rps: `3.820450829226138`
- active_request_throughput_rps: `3.8205273720191886`
- token_throughput_tps: `117.00130664505048`
- active_output_tps: `117.00365076808765`
- token_throughput_tps_avg: `113.99910232308741`
- token_throughput_tps_min: `98.50493484951551`
- token_throughput_tps_max: `125.29077605078213`
- request_token_tps_avg: `15.20039685265766`
- request_token_tps_p50: `14.77051212960788`
- request_token_tps_p95: `16.43825075336022`
- request_token_tps_weighted_avg: `15.167678011347013`

| metric | value |
|---|---:|
| latency_avg_sec | `2.019096131727565` |
| latency_p50_sec | `2.075649374164641` |
| latency_p95_sec | `2.2401816491037607` |
| latency_p99_sec | `2.2403639098629355` |
| peak_inflight_requests | `8` |
| avg_inflight_requests | `7.714012038003224` |
| client_queue_before_first_start_sec | `8.390471339225769e-05` |
| token_tps_bucket_sec | `1.0` |
| request_token_tps_min | `14.283393808980678` |
| request_token_tps_max | `16.438415496853793` |
| start_delay_avg_sec | `1.0951364093925804` |
| start_delay_p95_sec | `2.24103179667145` |

## Output Files

- summary_json: `benchmark_records/20260820_mix6_baseline_eager_c8_mt32_r1/summary.json`
- requests_jsonl: `benchmark_records/20260820_mix6_baseline_eager_c8_mt32_r1/requests.jsonl`
- config_json: `benchmark_records/20260820_mix6_baseline_eager_c8_mt32_r1/config.json`
