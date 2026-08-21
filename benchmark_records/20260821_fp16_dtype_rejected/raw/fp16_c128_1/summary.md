# fp16_c128_1

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
- wall_time_sec: `3.4991481360048056`
- active_window_sec: `3.499065446667373`
- request_throughput_rps: `36.58033184789528`
- active_request_throughput_rps: `36.581196308263245`
- token_throughput_tps: `1170.570619132649`
- active_output_tps: `1170.5982818644238`
- token_throughput_tps_avg: `1169.8140121545628`
- token_throughput_tps_min: `1163.6299592157798`
- token_throughput_tps_max: `1175.645130012627`
- request_token_tps_avg: `9.184727578223645`
- request_token_tps_p50: `9.183122217042275`
- request_token_tps_p95: `9.205694614354606`
- request_token_tps_weighted_avg: `9.184568069002346`

| metric | value |
|---|---:|
| latency_avg_sec | `3.484105050949438` |
| latency_p50_sec | `3.4848728524520993` |
| latency_p95_sec | `3.495793984271586` |
| latency_p99_sec | `3.4981660647317767` |
| peak_inflight_requests | `128` |
| avg_inflight_requests | `127.45273082739291` |
| client_queue_before_first_start_sec | `8.268933743238449e-05` |
| token_tps_bucket_sec | `1.0` |
| request_token_tps_min | `9.147582563499519` |
| request_token_tps_max | `9.464606585849383` |
| start_delay_avg_sec | `0.010297535234712996` |
| start_delay_p95_sec | `0.019721725024282932` |

## Output Files

- summary_json: `/tmp/rwkv7_fp16_experimental_perf_20260821/fp16_c128_1/summary.json`
- requests_jsonl: `/tmp/rwkv7_fp16_experimental_perf_20260821/fp16_c128_1/requests.jsonl`
- config_json: `/tmp/rwkv7_fp16_experimental_perf_20260821/fp16_c128_1/config.json`
