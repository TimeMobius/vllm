# fp16_c128_3

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
- wall_time_sec: `3.4243427542969584`
- active_window_sec: `3.424258626997471`
- request_throughput_rps: `37.37943575869621`
- active_request_throughput_rps: `37.38035409791333`
- token_throughput_tps: `1196.1419442782787`
- active_output_tps: `1196.1713311332267`
- token_throughput_tps_avg: `1194.8470269532736`
- token_throughput_tps_min: `1186.9706431712214`
- token_throughput_tps_max: `1202.5018969813668`
- request_token_tps_avg: `9.394546070166928`
- request_token_tps_p50: `9.390107963487276`
- request_token_tps_p95: `9.423033722026297`
- request_token_tps_weighted_avg: `9.39435413681977`

| metric | value |
|---|---:|
| latency_avg_sec | `3.4063012245387654` |
| latency_p50_sec | `3.4081794936209917` |
| latency_p95_sec | `3.419700068421662` |
| latency_p99_sec | `3.421452476643026` |
| peak_inflight_requests | `128` |
| avg_inflight_requests | `127.32874593741485` |
| client_queue_before_first_start_sec | `8.412729948759079e-05` |
| token_tps_bucket_sec | `1.0` |
| request_token_tps_min | `9.351033980099086` |
| request_token_tps_max | `9.701306631788212` |
| start_delay_avg_sec | `0.012623137023183517` |
| start_delay_p95_sec | `0.02317140717059374` |

## Output Files

- summary_json: `/tmp/rwkv7_fp16_experimental_perf_20260821/fp16_c128_3/summary.json`
- requests_jsonl: `/tmp/rwkv7_fp16_experimental_perf_20260821/fp16_c128_3/requests.jsonl`
- config_json: `/tmp/rwkv7_fp16_experimental_perf_20260821/fp16_c128_3/config.json`
