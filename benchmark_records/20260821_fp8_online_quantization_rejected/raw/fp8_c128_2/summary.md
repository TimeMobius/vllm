# fp8_c128_2

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
- wall_time_sec: `3.3504562256857753`
- active_window_sec: `3.3503830125555396`
- request_throughput_rps: `38.20375237817077`
- active_request_throughput_rps: `38.20458721296067`
- token_throughput_tps: `1222.5200761014646`
- active_output_tps: `1222.5467908147414`
- token_throughput_tps_avg: `1220.9454793475925`
- token_throughput_tps_min: `1212.6867563753842`
- token_throughput_tps_max: `1227.7859796344574`
- request_token_tps_avg: `9.592077965894202`
- request_token_tps_p50: `9.589710842911705`
- request_token_tps_p95: `9.620354076305633`
- request_token_tps_weighted_avg: `9.591972532119447`

| metric | value |
|---|---:|
| latency_avg_sec | `3.336122981258086` |
| latency_p50_sec | `3.3369856318458915` |
| latency_p95_sec | `3.3468910781666636` |
| latency_p99_sec | `3.3482014341279864` |
| peak_inflight_requests | `128` |
| avg_inflight_requests | `127.45520139063689` |
| client_queue_before_first_start_sec | `7.3213130235672e-05` |
| token_tps_bucket_sec | `1.0` |
| request_token_tps_min | `9.555675921155261` |
| request_token_tps_max | `9.88970335639127` |
| start_delay_avg_sec | `0.010053902180516161` |
| start_delay_p95_sec | `0.01904547680169344` |

## Output Files

- summary_json: `/tmp/rwkv7_fp8_experimental_perf_20260821/fp8_c128_2/summary.json`
- requests_jsonl: `/tmp/rwkv7_fp8_experimental_perf_20260821/fp8_c128_2/requests.jsonl`
- config_json: `/tmp/rwkv7_fp8_experimental_perf_20260821/fp8_c128_2/config.json`
