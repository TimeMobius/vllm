# fp8_c1_2

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
- wall_time_sec: `1.0309984823688865`
- active_window_sec: `1.030903747305274`
- request_throughput_rps: `0.9699335324939932`
- active_request_throughput_rps: `0.9700226646900307`
- token_throughput_tps: `31.03787303980778`
- active_output_tps: `31.040725270080983`
- token_throughput_tps_avg: `31.040725270080983`
- token_throughput_tps_min: `31.040725270080983`
- token_throughput_tps_max: `31.040725270080983`
- request_token_tps_avg: `31.040725270080983`
- request_token_tps_p50: `31.040725270080983`
- request_token_tps_p95: `31.040725270080983`
- request_token_tps_weighted_avg: `31.040725270080983`

| metric | value |
|---|---:|
| latency_avg_sec | `1.030903747305274` |
| latency_p50_sec | `1.030903747305274` |
| latency_p95_sec | `1.030903747305274` |
| latency_p99_sec | `1.030903747305274` |
| peak_inflight_requests | `1` |
| avg_inflight_requests | `1.0` |
| client_queue_before_first_start_sec | `9.473506361246109e-05` |
| token_tps_bucket_sec | `1.0` |
| request_token_tps_min | `31.040725270080983` |
| request_token_tps_max | `31.040725270080983` |
| start_delay_avg_sec | `9.473506361246109e-05` |
| start_delay_p95_sec | `9.473506361246109e-05` |

## Output Files

- summary_json: `/tmp/rwkv7_fp8_experimental_perf_20260821/fp8_c1_2/summary.json`
- requests_jsonl: `/tmp/rwkv7_fp8_experimental_perf_20260821/fp8_c1_2/requests.jsonl`
- config_json: `/tmp/rwkv7_fp8_experimental_perf_20260821/fp8_c1_2/config.json`
