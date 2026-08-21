# fp8_c128_3

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
- wall_time_sec: `3.3721893401816487`
- active_window_sec: `3.37210610602051`
- request_throughput_rps: `37.95753651042176`
- active_request_throughput_rps: `37.95847342154229`
- token_throughput_tps: `1214.6411683334964`
- active_output_tps: `1214.6711494893532`
- token_throughput_tps_avg: `1213.32251326379`
- token_throughput_tps_min: `1205.318794475375`
- token_throughput_tps_max: `1220.9458001751536`
- request_token_tps_avg: `9.538639063868386`
- request_token_tps_p50: `9.537254269683187`
- request_token_tps_p95: `9.570343421269797`
- request_token_tps_weighted_avg: `9.538525118884209`

| metric | value |
|---|---:|
| latency_avg_sec | `3.354816347513406` |
| latency_p50_sec | `3.355518531985581` |
| latency_p95_sec | `3.3678815709427` |
| latency_p99_sec | `3.3695108266547322` |
| peak_inflight_requests | `128` |
| avg_inflight_requests | `127.3437071612432` |
| client_queue_before_first_start_sec | `8.323416113853455e-05` |
| token_tps_bucket_sec | `1.0` |
| request_token_tps_min | `9.49332841774844` |
| request_token_tps_max | `9.832881781079383` |
| start_delay_avg_sec | `0.012871562757936772` |
| start_delay_p95_sec | `0.023949986323714256` |

## Output Files

- summary_json: `/tmp/rwkv7_fp8_experimental_perf_20260821/fp8_c128_3/summary.json`
- requests_jsonl: `/tmp/rwkv7_fp8_experimental_perf_20260821/fp8_c128_3/requests.jsonl`
- config_json: `/tmp/rwkv7_fp8_experimental_perf_20260821/fp8_c128_3/config.json`
