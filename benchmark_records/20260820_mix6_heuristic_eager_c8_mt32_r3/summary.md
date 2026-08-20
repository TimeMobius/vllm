# 20260820_mix6_heuristic_eager_c8_mt32_r3

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
- wall_time_sec: `3.8392490362748504`
- active_window_sec: `3.8391547612845898`
- request_throughput_rps: `4.167481673844344`
- active_request_throughput_rps: `4.167584011290643`
- token_throughput_tps: `128.9314642845594`
- active_output_tps: `128.93463034930426`
- token_throughput_tps_avg: `128.44488538293797`
- token_throughput_tps_min: `116.75534620567265`
- token_throughput_tps_max: `133.2096383817879`
- request_token_tps_avg: `16.539880293715928`
- request_token_tps_p50: `16.67829515836501`
- request_token_tps_p95: `16.687486764405588`
- request_token_tps_weighted_avg: `16.600195239905673`

| metric | value |
|---|---:|
| latency_avg_sec | `1.863682899682317` |
| latency_p50_sec | `1.918661331757903` |
| latency_p95_sec | `1.9203496687114239` |
| latency_p99_sec | `1.920417519286275` |
| peak_inflight_requests | `8` |
| avg_inflight_requests | `7.767055054831807` |
| client_queue_before_first_start_sec | `9.427499026060104e-05` |
| token_tps_bucket_sec | `1.0` |
| request_token_tps_min | `14.50757242453281` |
| request_token_tps_max | `16.694010986943816` |
| start_delay_avg_sec | `0.9610379841178656` |
| start_delay_p95_sec | `1.9216030593961477` |

## Output Files

- summary_json: `benchmark_records/20260820_mix6_heuristic_eager_c8_mt32_r3/summary.json`
- requests_jsonl: `benchmark_records/20260820_mix6_heuristic_eager_c8_mt32_r3/requests.jsonl`
- config_json: `benchmark_records/20260820_mix6_heuristic_eager_c8_mt32_r3/config.json`
