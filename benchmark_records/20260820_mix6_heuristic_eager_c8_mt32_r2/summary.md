# 20260820_mix6_heuristic_eager_c8_mt32_r2

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
- wall_time_sec: `3.8265564516186714`
- active_window_sec: `3.826465337537229`
- request_throughput_rps: `4.18130509827755`
- active_request_throughput_rps: `4.1814046616446925`
- token_throughput_tps: `129.3591264779617`
- active_output_tps: `129.36220671963267`
- token_throughput_tps_avg: `128.8285736281614`
- token_throughput_tps_min: `117.06188391615221`
- token_throughput_tps_max: `133.72693336904155`
- request_token_tps_avg: `16.593355657642924`
- request_token_tps_p50: `16.732425136189548`
- request_token_tps_p95: `16.742038883550798`
- request_token_tps_weighted_avg: `16.654899635477495`

| metric | value |
|---|---:|
| latency_avg_sec | `1.857561479031574` |
| latency_p50_sec | `1.9124543955549598` |
| latency_p95_sec | `1.9135225294157863` |
| latency_p99_sec | `1.9138874411582947` |
| peak_inflight_requests | `8` |
| avg_inflight_requests | `7.767216227714233` |
| client_queue_before_first_start_sec | `9.111408144235611e-05` |
| token_tps_bucket_sec | `1.0` |
| request_token_tps_min | `14.512021607412976` |
| request_token_tps_max | `16.74343899777607` |
| start_delay_avg_sec | `0.957320473564323` |
| start_delay_p95_sec | `1.914041175507009` |

## Output Files

- summary_json: `benchmark_records/20260820_mix6_heuristic_eager_c8_mt32_r2/summary.json`
- requests_jsonl: `benchmark_records/20260820_mix6_heuristic_eager_c8_mt32_r2/requests.jsonl`
- config_json: `benchmark_records/20260820_mix6_heuristic_eager_c8_mt32_r2/config.json`
