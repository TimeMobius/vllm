# 20260820_mix6_baseline_eager_c8_mt32_r2

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
- wall_time_sec: `4.087055891752243`
- active_window_sec: `4.086978357285261`
- request_throughput_rps: `3.914798432849501`
- active_request_throughput_rps: `3.9148727008742608`
- token_throughput_tps: `124.05017534091856`
- active_output_tps: `124.05252870895313`
- token_throughput_tps_avg: `121.49368932982395`
- token_throughput_tps_min: `110.03950174761599`
- token_throughput_tps_max: `125.34021831806827`
- request_token_tps_avg: `15.626307720476479`
- request_token_tps_p50: `15.621835598349744`
- request_token_tps_p95: `15.958663417781937`
- request_token_tps_weighted_avg: `15.62186002894219`

| metric | value |
|---|---:|
| latency_avg_sec | `2.028407625039108` |
| latency_p50_sec | `2.0587407229468226` |
| latency_p95_sec | `2.080760716460645` |
| latency_p99_sec | `2.1259419564157724` |
| peak_inflight_requests | `8` |
| avg_inflight_requests | `7.940957637510798` |
| client_queue_before_first_start_sec | `7.753446698188782e-05` |
| token_tps_bucket_sec | `1.0` |
| request_token_tps_min | `15.052151308002` |
| request_token_tps_max | `15.961088094576102` |
| start_delay_avg_sec | `1.0092621878138743` |
| start_delay_p95_sec | `2.081746934913099` |

## Output Files

- summary_json: `benchmark_records/20260820_mix6_baseline_eager_c8_mt32_r2/summary.json`
- requests_jsonl: `benchmark_records/20260820_mix6_baseline_eager_c8_mt32_r2/requests.jsonl`
- config_json: `benchmark_records/20260820_mix6_baseline_eager_c8_mt32_r2/config.json`
