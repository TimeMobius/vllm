# fp8_c128_1

## Config

- base_url: `http://127.0.0.1:8031`
- endpoint: `completions`
- model: `xiaoke-5-fp8`
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
- wall_time_sec: `3.4134317394346`
- active_window_sec: `3.413351945579052`
- request_throughput_rps: `37.49891890944973`
- active_request_throughput_rps: `37.499795520876376`
- token_throughput_tps: `1199.9654051023913`
- active_output_tps: `1199.993456668044`
- token_throughput_tps_avg: `1198.7623649402053`
- token_throughput_tps_min: `1191.5993831272792`
- token_throughput_tps_max: `1204.8400978012053`
- request_token_tps_avg: `9.412813264071913`
- request_token_tps_p50: `9.409817935755594`
- request_token_tps_p95: `9.437507813830496`
- request_token_tps_weighted_avg: `9.412712595870863`

| metric | value |
|---|---:|
| latency_avg_sec | `3.3996576092249597` |
| latency_p50_sec | `3.4007154274731874` |
| latency_p95_sec | `3.4101286632940173` |
| latency_p99_sec | `3.4113643998280168` |
| peak_inflight_requests | `128` |
| avg_inflight_requests | `127.48646518692743` |
| client_queue_before_first_start_sec | `7.979385554790497e-05` |
| token_tps_bucket_sec | `1.0` |
| request_token_tps_min | `9.379460435862145` |
| request_token_tps_max | `9.709742004544331` |
| start_delay_avg_sec | `0.009261334904294927` |
| start_delay_p95_sec | `0.017274940386414528` |

## Output Files

- summary_json: `/tmp/rwkv7_fp8_experimental_perf_20260821/fp8_c128_1/summary.json`
- requests_jsonl: `/tmp/rwkv7_fp8_experimental_perf_20260821/fp8_c128_1/requests.jsonl`
- config_json: `/tmp/rwkv7_fp8_experimental_perf_20260821/fp8_c128_1/config.json`
