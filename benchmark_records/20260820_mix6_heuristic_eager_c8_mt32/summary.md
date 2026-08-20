# 20260820_mix6_heuristic_eager_c8_mt32

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
- wall_time_sec: `4.339881730265915`
- active_window_sec: `4.339797965250909`
- request_throughput_rps: `3.6867364122891986`
- active_request_throughput_rps: `3.6868075721757583`
- token_throughput_tps: `116.13219698710975`
- active_output_tps: `116.13443852353637`
- token_throughput_tps_avg: `115.07002354675849`
- token_throughput_tps_min: `108.07315636479221`
- token_throughput_tps_max: `128.05542802935`
- request_token_tps_avg: `14.811383608876449`
- request_token_tps_p50: `15.198363021925685`
- request_token_tps_p95: `16.342085627473327`
- request_token_tps_weighted_avg: `14.708520699563422`

| metric | value |
|---|---:|
| latency_avg_sec | `2.1416157779167406` |
| latency_p50_sec | `2.105489910580218` |
| latency_p95_sec | `2.380250330083072` |
| latency_p99_sec | `2.3803058806806803` |
| peak_inflight_requests | `8` |
| avg_inflight_requests | `7.895725266714516` |
| client_queue_before_first_start_sec | `8.376501500606537e-05` |
| token_tps_bucket_sec | `1.0` |
| request_token_tps_min | `13.443650355915254` |
| request_token_tps_max | `16.346390919994437` |
| start_delay_avg_sec | `1.1586493297363631` |
| start_delay_p95_sec | `2.3817025125026703` |

## Output Files

- summary_json: `benchmark_records/20260820_mix6_heuristic_eager_c8_mt32/summary.json`
- requests_jsonl: `benchmark_records/20260820_mix6_heuristic_eager_c8_mt32/requests.jsonl`
- config_json: `benchmark_records/20260820_mix6_heuristic_eager_c8_mt32/config.json`
