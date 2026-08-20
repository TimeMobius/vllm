# 20260820_baseline_eager_c8_mt32

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
- wall_time_sec: `4.071533577516675`
- active_window_sec: `4.071433926932514`
- request_throughput_rps: `3.9297232100339894`
- active_request_throughput_rps: `3.9298193921704305`
- token_throughput_tps: `117.40048089976543`
- active_output_tps: `117.40335434109161`
- token_throughput_tps_avg: `113.42531096032583`
- token_throughput_tps_min: `95.98299721117591`
- token_throughput_tps_max: `125.06988317399895`
- request_token_tps_avg: `15.443309855933014`
- request_token_tps_p50: `15.34323485782372`
- request_token_tps_p95: `16.137707810276737`
- request_token_tps_weighted_avg: `15.532148901377756`

| metric | value |
|---|---:|
| latency_avg_sec | `1.9234299252275378` |
| latency_p50_sec | `2.038300024345517` |
| latency_p95_sec | `2.088382345624268` |
| latency_p99_sec | `2.109986314550042` |
| peak_inflight_requests | `8` |
| avg_inflight_requests | `7.558732219640099` |
| client_queue_before_first_start_sec | `9.965058416128159e-05` |
| token_tps_bucket_sec | `1.0` |
| request_token_tps_min | `13.737719583569325` |
| request_token_tps_max | `16.139801431619556` |
| start_delay_avg_sec | `0.9776370511390269` |
| start_delay_p95_sec | `2.0885044559836388` |

## Output Files

- summary_json: `benchmark_records/20260820_baseline_eager_c8_mt32/summary.json`
- requests_jsonl: `benchmark_records/20260820_baseline_eager_c8_mt32/requests.jsonl`
- config_json: `benchmark_records/20260820_baseline_eager_c8_mt32/config.json`
