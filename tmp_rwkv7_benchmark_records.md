# RWKV7 Benchmark Records

这份文件专门记录 RWKV7 在 vLLM 上的 benchmark 结果，方便后续横向比较不同模型大小、运行模式、并发档位和输出长度。

## Column Guide

- `run_id`: 本次 benchmark 的唯一标识，建议包含日期、模式、模型大小、输出长度
- `model_name`: checkpoint 名称
- `model_size`: 参数规模，便于做 `0.1B` / `0.4B` 对照
- `mode`: `eager` / `piecewise` / `compile_no_cg` / 其他
- `max_tokens`: 每个请求的生成长度
- `concurrency`: 并发请求数
- `round0_tps` / `round1_tps`: 各轮 aggregate TPS
- `avg_tps`: 多轮 aggregate TPS 平均值
- `all_match_serial_baseline`: 并发输出是否逐条对齐串行 baseline

## Prompt Sets

| prompt_set_id | prompts | source |
| --- | --- | --- |
| `default_mixed_8` | `i am`; `北京是`; `The capital of France is`; `Once upon a time`; `In a shocking finding, scientists discovered`; `人工智能的未来`; `Write a short haiku about the sea`; `The theory of relativity says` | [tmp_rwkv7_long_benchmark.py](/home/liu/vllm/tmp_rwkv7_long_benchmark.py) |
| `rwkv7_ttft_seed_repeat` | exact token prefixes cut from repeated tokenization of `The capital of France is Paris. 北京是中国的首都。 RWKV7 is a recurrent world model for language generation.` | [tmp_rwkv7_ttft_benchmark.py](/home/liu/vllm/tmp_rwkv7_ttft_benchmark.py) |
| `rwkv7_exact_long_repeat` | exact token prefixes cut from repeated tokenization of `The capital of France is Paris. Beijing is the capital of China. RWKV7 is a recurrent world model for language generation.` | [tmp_rwkv7_exact_long_input_bench.py](/home/liu/vllm/tmp_rwkv7_exact_long_input_bench.py) |
| `rwkv7_mixed_exact_repeat` | exact token prefixes with lengths `64/128/256/512/768/1024/1536/1984`, all cut from the same repeated seed text buffer | [tmp_rwkv7_mixed_exact_prompt_bench.py](/home/liu/vllm/tmp_rwkv7_mixed_exact_prompt_bench.py) |

## Run Index

| run_id | date | model_name | model_size | mode | dtype | max_tokens | rounds | warmup | concurrency_levels | prompt_set_id | raw_json | server_log | notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `2026-04-13_eager_0p4b_mt64` | `2026-04-13` | `RWKV7-Goose-World2.9-0.4B-HF` | `0.4B` | `eager` | `auto` | `64` | `2` | `1` | `1/2/4/8` | `default_mixed_8` | [rwkv7_bench_0p4b_eager_64_20260413.json](/tmp/rwkv7_bench_0p4b_eager_64_20260413.json) | [vllm_rwkv7_eager_bench_20260413.log](/tmp/vllm_rwkv7_eager_bench_20260413.log) | `tmp_rwkv7_long_benchmark.py --enforce-eager` 基线复跑；全部并发轮次都与串行 baseline 一致 |
| `2026-04-13_piecewise_0p4b_mt16_smoke_packedprefill` | `2026-04-13` | `RWKV7-Goose-World2.9-0.4B-HF` | `0.4B` | `piecewise` | `auto` | `16` | `1` | `1` | `4/8` | `default_mixed_8` | [rwkv7_long_piecewise_packedprefill_20260413.json](/tmp/rwkv7_long_piecewise_packedprefill_20260413.json) | [vllm_rwkv7_long_piecewise_packedprefill_20260413.log](/tmp/vllm_rwkv7_long_piecewise_packedprefill_20260413.log) | packed/varlen prefill 落地后的真实服务 smoke；主要用于确认并发输出仍与串行 baseline 一致，不作为 before/after 性能结论 |
| `2026-04-13_eager_0p4b_exact_long_mt64` | `2026-04-13` | `RWKV7-Goose-World2.9-0.4B-HF` | `0.4B` | `eager` | `auto` | `64` | `1` | `1` | `1/4/8` | `rwkv7_exact_long_repeat` | [rwkv7_exact_long_eager_20260413.json](/tmp/rwkv7_exact_long_eager_20260413.json) | [vllm_rwkv7_exact_long_eager_20260413.log](/tmp/vllm_rwkv7_exact_long_eager_20260413.log) | exact token 长输入 probe；单轮大 sweep 可用于趋势判断，但不如 focused rerun 稳定 |
| `2026-04-13_piecewise_0p4b_exact_long_mt64` | `2026-04-13` | `RWKV7-Goose-World2.9-0.4B-HF` | `0.4B` | `piecewise` | `auto` | `64` | `1` | `1` | `1/4/8` | `rwkv7_exact_long_repeat` | [rwkv7_exact_long_piecewise_20260413.json](/tmp/rwkv7_exact_long_piecewise_20260413.json) | [vllm_rwkv7_exact_long_piecewise_20260413.log](/tmp/vllm_rwkv7_exact_long_piecewise_20260413.log) | exact token 长输入 probe；`1024` mixed-scenario 结果波动很大，需要 focused rerun 才能看 steady-state |
| `2026-04-13_eager_0p4b_exact_long_1024_c8_r2` | `2026-04-13` | `RWKV7-Goose-World2.9-0.4B-HF` | `0.4B` | `eager` | `auto` | `64` | `2` | `1` | `8` | `rwkv7_exact_long_repeat` | [rwkv7_exact_long_eager_1024_c8_r2_20260413.json](/tmp/rwkv7_exact_long_eager_1024_c8_r2_20260413.json) | [vllm_rwkv7_exact_long_eager_1024_c8_r2_20260413.log](/tmp/vllm_rwkv7_exact_long_eager_1024_c8_r2_20260413.log) | focused rerun；用于估计 `1024 + 64` 的 steady-state c8 带宽 |
| `2026-04-13_piecewise_0p4b_exact_long_1024_c8_r2` | `2026-04-13` | `RWKV7-Goose-World2.9-0.4B-HF` | `0.4B` | `piecewise` | `auto` | `64` | `2` | `1` | `8` | `rwkv7_exact_long_repeat` | [rwkv7_exact_long_piecewise_1024_c8_r2_20260413.json](/tmp/rwkv7_exact_long_piecewise_1024_c8_r2_20260413.json) | [vllm_rwkv7_exact_long_piecewise_1024_c8_r2_20260413.log](/tmp/vllm_rwkv7_exact_long_piecewise_1024_c8_r2_20260413.log) | focused rerun；显示 `1024` 档 steady-state 已接近 eager，而不是 mixed-scenario probe 里那种异常慢 |
| `2026-04-13_eager_0p4b_exact_long_1984_c8` | `2026-04-13` | `RWKV7-Goose-World2.9-0.4B-HF` | `0.4B` | `eager` | `auto` | `64` | `1` | `1` | `8` | `rwkv7_exact_long_repeat` | [rwkv7_exact_long_eager_1984_c8_20260413.json](/tmp/rwkv7_exact_long_eager_1984_c8_20260413.json) | [vllm_rwkv7_exact_long_eager_1984_c8_20260413.log](/tmp/vllm_rwkv7_exact_long_eager_1984_c8_20260413.log) | focused rerun；用于和 `PIECEWISE` 对照 packed prefill 在超长 prompt 上的收益 |
| `2026-04-13_piecewise_0p4b_exact_long_1984_c8` | `2026-04-13` | `RWKV7-Goose-World2.9-0.4B-HF` | `0.4B` | `piecewise` | `auto` | `64` | `1` | `1` | `8` | `rwkv7_exact_long_repeat` | [rwkv7_exact_long_piecewise_1984_c8_20260413.json](/tmp/rwkv7_exact_long_piecewise_1984_c8_20260413.json) | [vllm_rwkv7_exact_long_piecewise_1984_c8_20260413.log](/tmp/vllm_rwkv7_exact_long_piecewise_1984_c8_20260413.log) | focused rerun；`1984` 档明显受益于 packed prefill |
| `2026-04-13_eager_0p4b_exact_long_mt64_decodefused_seq` | `2026-04-13` | `RWKV7-Goose-World2.9-0.4B-HF` | `0.4B` | `eager` | `auto` | `64` | `2` | `2` | `8` | `rwkv7_exact_long_repeat` | [rwkv7_exact_long_eager_decodefused_seq_20260413.json](/tmp/rwkv7_exact_long_eager_decodefused_seq_20260413.json) | [vllm_rwkv7_exact_long_eager_decodefused_seq_20260413.log](/tmp/vllm_rwkv7_exact_long_eager_decodefused_seq_20260413.log) | decode fused 后的串行 benchmark；用于替代同机并行启动 eager/piecewise 的无效样本 |
| `2026-04-13_piecewise_0p4b_exact_long_mt64_decodefused_seq` | `2026-04-13` | `RWKV7-Goose-World2.9-0.4B-HF` | `0.4B` | `piecewise` | `auto` | `64` | `2` | `2` | `8` | `rwkv7_exact_long_repeat` | [rwkv7_exact_long_piecewise_decodefused_seq_20260413.json](/tmp/rwkv7_exact_long_piecewise_decodefused_seq_20260413.json) | [vllm_rwkv7_exact_long_piecewise_decodefused_seq_20260413.log](/tmp/vllm_rwkv7_exact_long_piecewise_decodefused_seq_20260413.log) | decode fused 后的串行 benchmark；与 eager 使用同一口径对照 |
| `2026-04-13_eager_0p4b_exact_long_mt64_prefixcache` | `2026-04-13` | `RWKV7-Goose-World2.9-0.4B-HF` | `0.4B` | `eager` | `auto` | `64` | `2` | `2` | `8` | `rwkv7_exact_long_repeat` | [rwkv7_exact_long_eager_prefixcache_20260413.json](/tmp/rwkv7_exact_long_eager_prefixcache_20260413.json) | [vllm_rwkv7_exact_long_eager_prefixcache_20260413.log](/tmp/vllm_rwkv7_exact_long_eager_prefixcache_20260413.log) | `--enable-prefix-caching`；serial baseline 先暖 cache，再测 cached throughput |
| `2026-04-13_piecewise_0p4b_exact_long_mt64_prefixcache` | `2026-04-13` | `RWKV7-Goose-World2.9-0.4B-HF` | `0.4B` | `piecewise` | `auto` | `64` | `2` | `2` | `8` | `rwkv7_exact_long_repeat` | [rwkv7_exact_long_piecewise_prefixcache_20260413.json](/tmp/rwkv7_exact_long_piecewise_prefixcache_20260413.json) | [vllm_rwkv7_exact_long_piecewise_prefixcache_20260413.log](/tmp/vllm_rwkv7_exact_long_piecewise_prefixcache_20260413.log) | `--enable-prefix-caching`；日志显示 Mamba cache 进入 experimental `align` mode |
| `2026-04-13_eager_0p4b_exact_long_mt128` | `2026-04-13` | `RWKV7-Goose-World2.9-0.4B-HF` | `0.4B` | `eager` | `auto` | `128` | `2` | `2` | `8` | `rwkv7_exact_long_repeat` | [rwkv7_exact_long_eager_mt128_20260413.json](/tmp/rwkv7_exact_long_eager_mt128_20260413.json) | [vllm_rwkv7_exact_long_eager_mt128_20260413.log](/tmp/vllm_rwkv7_exact_long_eager_mt128_20260413.log) | longer-output exact-long control；使用 `1024/1920`，因为 `1984 + 128` 超出当前 `2048` 上限 |
| `2026-04-13_piecewise_0p4b_exact_long_mt128` | `2026-04-13` | `RWKV7-Goose-World2.9-0.4B-HF` | `0.4B` | `piecewise` | `auto` | `128` | `2` | `2` | `8` | `rwkv7_exact_long_repeat` | [rwkv7_exact_long_piecewise_mt128_20260413.json](/tmp/rwkv7_exact_long_piecewise_mt128_20260413.json) | [vllm_rwkv7_exact_long_piecewise_mt128_20260413.log](/tmp/vllm_rwkv7_exact_long_piecewise_mt128_20260413.log) | longer-output exact-long control；使用 `1024/1920` 口径 |
| `2026-04-13_compile_no_cg_0p4b_exact_long_mt128` | `2026-04-13` | `RWKV7-Goose-World2.9-0.4B-HF` | `0.4B` | `compile_no_cg` | `auto` | `128` | `2` | `2` | `8` | `rwkv7_exact_long_repeat` | [rwkv7_exact_long_compile_no_cg_mt128_20260413.json](/tmp/rwkv7_exact_long_compile_no_cg_mt128_20260413.json) | [vllm_rwkv7_exact_long_compile_no_cg_mt128_20260413.log](/tmp/vllm_rwkv7_exact_long_compile_no_cg_mt128_20260413.log) | longer-output exact-long control；用于和 eager / `PIECEWISE` 对照 `no-cg` 在长输出下的位置 |
| `2026-04-13_eager_0p4b_mixed_exact_mt64` | `2026-04-13` | `RWKV7-Goose-World2.9-0.4B-HF` | `0.4B` | `eager` | `auto` | `64` | `2` | `1` | `8` | `rwkv7_mixed_exact_repeat` | [rwkv7_mixed_exact_eager_20260413.json](/tmp/rwkv7_mixed_exact_eager_20260413.json) | [vllm_rwkv7_mixed_exact_eager_20260413.log](/tmp/vllm_rwkv7_mixed_exact_eager_20260413.log) | exact token mixed prompt lengths；无 prefix caching |
| `2026-04-13_piecewise_0p4b_mixed_exact_mt64` | `2026-04-13` | `RWKV7-Goose-World2.9-0.4B-HF` | `0.4B` | `piecewise` | `auto` | `64` | `2` | `1` | `8` | `rwkv7_mixed_exact_repeat` | [rwkv7_mixed_exact_piecewise_20260413.json](/tmp/rwkv7_mixed_exact_piecewise_20260413.json) | [vllm_rwkv7_mixed_exact_piecewise_20260413.log](/tmp/vllm_rwkv7_mixed_exact_piecewise_20260413.log) | exact token mixed prompt lengths；无 prefix caching；首轮受 compile warmup 影响较大 |
| `2026-04-13_eager_0p4b_mixed_exact_mt64_prefixcache` | `2026-04-13` | `RWKV7-Goose-World2.9-0.4B-HF` | `0.4B` | `eager` | `auto` | `64` | `2` | `1` | `8` | `rwkv7_mixed_exact_repeat` | [rwkv7_mixed_exact_eager_prefixcache_20260413.json](/tmp/rwkv7_mixed_exact_eager_prefixcache_20260413.json) | [vllm_rwkv7_mixed_exact_eager_prefixcache_20260413.log](/tmp/vllm_rwkv7_mixed_exact_eager_prefixcache_20260413.log) | exact token mixed prompt lengths；开启 prefix caching |
| `2026-04-13_piecewise_0p4b_mixed_exact_mt64_prefixcache` | `2026-04-13` | `RWKV7-Goose-World2.9-0.4B-HF` | `0.4B` | `piecewise` | `auto` | `64` | `2` | `1` | `8` | `rwkv7_mixed_exact_repeat` | [rwkv7_mixed_exact_piecewise_prefixcache_20260413.json](/tmp/rwkv7_mixed_exact_piecewise_prefixcache_20260413.json) | [vllm_rwkv7_mixed_exact_piecewise_prefixcache_20260413.log](/tmp/vllm_rwkv7_mixed_exact_piecewise_prefixcache_20260413.log) | exact token mixed prompt lengths；开启 prefix caching；Mamba cache `align` mode |
| `2026-04-13_compile_no_cg_0p4b_mt128` | `2026-04-13` | `RWKV7-Goose-World2.9-0.4B-HF` | `0.4B` | `compile_no_cg` | `auto` | `128` | `2` | `1` | `8` | `default_mixed_8` | [rwkv7_long_compile_no_cg_mt128_20260413.json](/tmp/rwkv7_long_compile_no_cg_mt128_20260413.json) | [vllm_rwkv7_long_compile_no_cg_mt128_20260413.log](/tmp/vllm_rwkv7_long_compile_no_cg_mt128_20260413.log) | 直接回查历史 `no-cg 128/c8` mixed 场景；当前实现下未复现 mismatch |
| `2026-04-13_eager_0p4b_mt64_refresh_seq` | `2026-04-13` | `RWKV7-Goose-World2.9-0.4B-HF` | `0.4B` | `eager` | `auto` | `64` | `2` | `1` | `1/2/4/8` | `default_mixed_8` | [rwkv7_bench_eager_refresh_seq_20260413.json](/tmp/rwkv7_bench_eager_refresh_seq_20260413.json) | [vllm_rwkv7_eager_bench_refresh_seq_20260413.log](/tmp/vllm_rwkv7_eager_bench_refresh_seq_20260413.log) | 当前代码状态下重新串行复跑的 eager 对照；用于和最新 `PIECEWISE` 直接横比 |
| `2026-04-13_piecewise_0p4b_mt64_refresh_seq` | `2026-04-13` | `RWKV7-Goose-World2.9-0.4B-HF` | `0.4B` | `piecewise` | `auto` | `64` | `2` | `1` | `1/2/4/8` | `default_mixed_8` | [rwkv7_bench_piecewise_refresh_seq_20260413.json](/tmp/rwkv7_bench_piecewise_refresh_seq_20260413.json) | [vllm_rwkv7_piecewise_bench_refresh_seq_20260413.log](/tmp/vllm_rwkv7_piecewise_bench_refresh_seq_20260413.log) | 当前代码状态下重新串行复跑的 `compile + cg (PIECEWISE)` 对照 |

## Throughput Table

| run_id | model_name | model_size | mode | max_tokens | concurrency | round0_tps | round1_tps | avg_tps | all_match_serial_baseline |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `2026-04-13_eager_0p4b_mt64` | `RWKV7-Goose-World2.9-0.4B-HF` | `0.4B` | `eager` | `64` | `1` | `35.047` | `34.912` | `34.980` | `true` |
| `2026-04-13_eager_0p4b_mt64` | `RWKV7-Goose-World2.9-0.4B-HF` | `0.4B` | `eager` | `64` | `2` | `68.610` | `71.110` | `69.860` | `true` |
| `2026-04-13_eager_0p4b_mt64` | `RWKV7-Goose-World2.9-0.4B-HF` | `0.4B` | `eager` | `64` | `4` | `132.553` | `132.926` | `132.739` | `true` |
| `2026-04-13_eager_0p4b_mt64` | `RWKV7-Goose-World2.9-0.4B-HF` | `0.4B` | `eager` | `64` | `8` | `215.131` | `214.844` | `214.987` | `true` |
| `2026-04-13_piecewise_0p4b_mt16_smoke_packedprefill` | `RWKV7-Goose-World2.9-0.4B-HF` | `0.4B` | `piecewise` | `16` | `4` | `18.689` | `n/a` | `18.689` | `true` |
| `2026-04-13_piecewise_0p4b_mt16_smoke_packedprefill` | `RWKV7-Goose-World2.9-0.4B-HF` | `0.4B` | `piecewise` | `16` | `8` | `208.104` | `n/a` | `208.104` | `true` |
| `2026-04-13_compile_no_cg_0p4b_mt128` | `RWKV7-Goose-World2.9-0.4B-HF` | `0.4B` | `compile_no_cg` | `128` | `8` | `277.310` | `274.227` | `275.768` | `true` |
| `2026-04-13_eager_0p4b_mt64_refresh_seq` | `RWKV7-Goose-World2.9-0.4B-HF` | `0.4B` | `eager` | `64` | `1` | `36.107` | `35.839` | `35.973` | `true` |
| `2026-04-13_eager_0p4b_mt64_refresh_seq` | `RWKV7-Goose-World2.9-0.4B-HF` | `0.4B` | `eager` | `64` | `2` | `68.850` | `73.769` | `71.310` | `true` |
| `2026-04-13_eager_0p4b_mt64_refresh_seq` | `RWKV7-Goose-World2.9-0.4B-HF` | `0.4B` | `eager` | `64` | `4` | `133.740` | `141.088` | `137.414` | `true` |
| `2026-04-13_eager_0p4b_mt64_refresh_seq` | `RWKV7-Goose-World2.9-0.4B-HF` | `0.4B` | `eager` | `64` | `8` | `286.786` | `283.853` | `285.320` | `true` |
| `2026-04-13_piecewise_0p4b_mt64_refresh_seq` | `RWKV7-Goose-World2.9-0.4B-HF` | `0.4B` | `piecewise` | `64` | `1` | `34.830` | `35.667` | `35.249` | `true` |
| `2026-04-13_piecewise_0p4b_mt64_refresh_seq` | `RWKV7-Goose-World2.9-0.4B-HF` | `0.4B` | `piecewise` | `64` | `2` | `69.997` | `70.206` | `70.102` | `true` |
| `2026-04-13_piecewise_0p4b_mt64_refresh_seq` | `RWKV7-Goose-World2.9-0.4B-HF` | `0.4B` | `piecewise` | `64` | `4` | `129.235` | `129.512` | `129.374` | `true` |
| `2026-04-13_piecewise_0p4b_mt64_refresh_seq` | `RWKV7-Goose-World2.9-0.4B-HF` | `0.4B` | `piecewise` | `64` | `8` | `271.006` | `257.948` | `264.477` | `true` |

## Exact Long-Input Throughput Table

| run_id | prompt_len | concurrency | round0_tps | round1_tps | avg_tps | all_match_serial_baseline |
| --- | --- | --- | --- | --- | --- | --- |
| `2026-04-13_eager_0p4b_exact_long_mt64` | `1024` | `1` | `5.438` | `n/a` | `5.438` | `true` |
| `2026-04-13_eager_0p4b_exact_long_mt64` | `1024` | `4` | `21.009` | `n/a` | `21.009` | `true` |
| `2026-04-13_eager_0p4b_exact_long_mt64` | `1024` | `8` | `31.685` | `n/a` | `31.685` | `true` |
| `2026-04-13_eager_0p4b_exact_long_mt64` | `1984` | `1` | `6.517` | `n/a` | `6.517` | `true` |
| `2026-04-13_eager_0p4b_exact_long_mt64` | `1984` | `4` | `10.038` | `n/a` | `10.038` | `true` |
| `2026-04-13_eager_0p4b_exact_long_mt64` | `1984` | `8` | `11.717` | `n/a` | `11.717` | `true` |
| `2026-04-13_piecewise_0p4b_exact_long_mt64` | `1024` | `1` | `4.715` | `n/a` | `4.715` | `true` |
| `2026-04-13_piecewise_0p4b_exact_long_mt64` | `1024` | `4` | `12.378` | `n/a` | `12.378` | `true` |
| `2026-04-13_piecewise_0p4b_exact_long_mt64` | `1024` | `8` | `13.387` | `n/a` | `13.387` | `true` |
| `2026-04-13_piecewise_0p4b_exact_long_mt64` | `1984` | `1` | `22.369` | `n/a` | `22.369` | `true` |
| `2026-04-13_piecewise_0p4b_exact_long_mt64` | `1984` | `4` | `60.396` | `n/a` | `60.396` | `true` |
| `2026-04-13_piecewise_0p4b_exact_long_mt64` | `1984` | `8` | `78.889` | `n/a` | `78.889` | `true` |
| `2026-04-13_eager_0p4b_exact_long_1024_c8_r2` | `1024` | `8` | `131.458` | `124.108` | `127.783` | `true` |
| `2026-04-13_piecewise_0p4b_exact_long_1024_c8_r2` | `1024` | `8` | `120.680` | `123.058` | `121.869` | `true` |
| `2026-04-13_eager_0p4b_exact_long_1984_c8` | `1984` | `8` | `14.594` | `n/a` | `14.594` | `true` |
| `2026-04-13_piecewise_0p4b_exact_long_1984_c8` | `1984` | `8` | `80.053` | `n/a` | `80.053` | `true` |
| `2026-04-13_eager_0p4b_exact_long_mt64_decodefused_seq` | `1024` | `8` | `128.662` | `126.905` | `127.784` | `true` |
| `2026-04-13_piecewise_0p4b_exact_long_mt64_decodefused_seq` | `1024` | `8` | `123.847` | `124.573` | `124.210` | `true` |
| `2026-04-13_eager_0p4b_exact_long_mt64_decodefused_seq` | `1984` | `8` | `82.291` | `84.238` | `83.264` | `true` |
| `2026-04-13_piecewise_0p4b_exact_long_mt64_decodefused_seq` | `1984` | `8` | `84.708` | `88.764` | `86.736` | `true` |
| `2026-04-13_eager_0p4b_exact_long_mt128` | `1024` | `8` | `187.512` | `185.750` | `186.631` | `true` |
| `2026-04-13_piecewise_0p4b_exact_long_mt128` | `1024` | `8` | `180.043` | `182.027` | `181.035` | `true` |
| `2026-04-13_compile_no_cg_0p4b_exact_long_mt128` | `1024` | `8` | `177.782` | `175.871` | `176.827` | `true` |
| `2026-04-13_eager_0p4b_exact_long_mt128` | `1920` | `8` | `137.117` | `137.148` | `137.133` | `true` |
| `2026-04-13_piecewise_0p4b_exact_long_mt128` | `1920` | `8` | `134.320` | `135.409` | `134.864` | `true` |
| `2026-04-13_compile_no_cg_0p4b_exact_long_mt128` | `1920` | `8` | `133.385` | `133.164` | `133.275` | `true` |
| `2026-04-13_eager_0p4b_exact_long_mt64_prefixcache` | `1024` | `8` | `130.217` | `210.408` | `170.313` | `true` |
| `2026-04-13_piecewise_0p4b_exact_long_mt64_prefixcache` | `1024` | `8` | `149.130` | `208.119` | `178.625` | `true` |
| `2026-04-13_eager_0p4b_exact_long_mt64_prefixcache` | `1984` | `8` | `155.799` | `270.292` | `213.046` | `true` |
| `2026-04-13_piecewise_0p4b_exact_long_mt64_prefixcache` | `1984` | `8` | `165.642` | `256.278` | `210.960` | `true` |

## Mixed Exact Prompt Throughput Table

| run_id | batch_prompt_lengths | round0_tps | round1_tps | avg_tps | all_match_serial_baseline |
| --- | --- | --- | --- | --- | --- |
| `2026-04-13_eager_0p4b_mixed_exact_mt64` | `64/128/256/512/768/1024/1536/1984` | `149.064` | `150.716` | `149.890` | `true` |
| `2026-04-13_piecewise_0p4b_mixed_exact_mt64` | `64/128/256/512/768/1024/1536/1984` | `87.009` | `134.828` | `110.918` | `true` |
| `2026-04-13_eager_0p4b_mixed_exact_mt64_prefixcache` | `64/128/256/512/768/1024/1536/1984` | `225.201` | `231.280` | `228.240` | `true` |
| `2026-04-13_piecewise_0p4b_mixed_exact_mt64_prefixcache` | `64/128/256/512/768/1024/1536/1984` | `228.883` | `229.894` | `229.388` | `true` |

## Investigation Note

`aggregate_tps` in the exact long-input probe is defined as `completion_tokens / wall_time`, see [tmp_rwkv7_exact_long_input_bench.py](/home/liu/vllm/tmp_rwkv7_exact_long_input_bench.py:120). That means:

- it penalizes long prefill time very heavily
- it is sensitive to first-run compile/cudagraph warmup
- mixed-scenario one-shot sweeps can under-report `PIECEWISE` if the first relevant shape pays extra runtime setup

So the focused reruns are the better estimate for steady-state:

- `1024 + 64`, concurrency `8`:
  - eager: `131.458 / 124.108`
  - piecewise: `120.680 / 123.058`
- `1984 + 64`, concurrency `8`:
  - eager: `14.594`
  - piecewise: `80.053`

After decode fusion landed, the new same-host sequential control is:

- `1024 + 64`, concurrency `8`:
  - eager: `128.662 / 126.905`
  - piecewise: `123.847 / 124.573`
- `1984 + 64`, concurrency `8`:
  - eager: `82.291 / 84.238`
  - piecewise: `84.708 / 88.764`

With prefix caching enabled, the same exact-long control shifts to:

- `1024 + 64`, concurrency `8`:
  - eager: `130.217 / 210.408`
  - piecewise: `149.130 / 208.119`
- `1984 + 64`, concurrency `8`:
  - eager: `155.799 / 270.292`
  - piecewise: `165.642 / 256.278`

The big round1 jump is expected here: the serial baseline warms the prefix
cache, so round1 is closer to the "cache already populated" steady-state.

For mixed exact prompt lengths (`64/128/256/512/768/1024/1536/1984`):

- without prefix caching:
  - eager: `149.064 / 150.716`
  - piecewise: `87.009 / 134.828`
- with prefix caching:
  - eager: `225.201 / 231.280`
  - piecewise: `228.883 / 229.894`

So this service-level picture is much clearer than the earlier kernel-only view:

- prefix caching is working and dominates throughput gains in repeated-prefix scenarios
- on mixed prompt lengths without prefix caching, eager is still more stable on
  the first `PIECEWISE` round
- on mixed prompt lengths with prefix caching, eager and `PIECEWISE` land in the
  same band, with `PIECEWISE` very slightly ahead in this run

For longer outputs (`max_tokens=128`), exact-long control now uses `1024/1920`
rather than `1984`, because `1984 + 128` exceeds the current `2048` limit.

Longer-output exact-long results:

- `1024 + 128`, concurrency `8`:
  - eager: `187.512 / 185.750`
  - piecewise: `180.043 / 182.027`
  - compile_no_cg: `177.782 / 175.871`
- `1920 + 128`, concurrency `8`:
  - eager: `137.117 / 137.148`
  - piecewise: `134.320 / 135.409`
  - compile_no_cg: `133.385 / 133.164`

So at longer outputs:

- all three paths are correct against the serial baseline
- eager remains slightly ahead on this exact-long workload
- `PIECEWISE` stays close behind
- `compile_no_cg` is now also in the same general band, though still a bit slower

The historical `compile_no_cg 128/c8` tail item was also rerun directly on the
old mixed prompt benchmark:

- `default_mixed_8`, `max_tokens=128`, concurrency `8`
- compile_no_cg aggregate TPS: `277.310 / 274.227`, avg `275.768`
- all requests still matched the serial baseline

That old mismatch is therefore not reproduced on the current decode-fused branch.

Current refreshed `default_mixed_8`, `max_tokens=64` control under the latest code:

- eager:
  - concurrency `1`: `36.107 / 35.839`, avg `35.973`
  - concurrency `2`: `68.850 / 73.769`, avg `71.310`
  - concurrency `4`: `133.740 / 141.088`, avg `137.414`
  - concurrency `8`: `286.786 / 283.853`, avg `285.320`
- `PIECEWISE`:
  - concurrency `1`: `34.830 / 35.667`, avg `35.249`
  - concurrency `2`: `69.997 / 70.206`, avg `70.102`
  - concurrency `4`: `129.235 / 129.512`, avg `129.374`
  - concurrency `8`: `271.006 / 257.948`, avg `264.477`

On a single GPU, do not launch eager and `PIECEWISE` benchmark servers in
parallel. Those runs contend on the same device and should be treated as
invalid samples, not trend data.

## Latency Run Index

| run_id | date | model_name | model_size | mode | benchmark_type | prompt_set_id | rounds | warmup | raw_json | server_log | notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `2026-04-13_eager_0p4b_ttft` | `2026-04-13` | `RWKV7-Goose-World2.9-0.4B-HF` | `0.4B` | `eager` | `ttft_prefill_proxy_decode` | `rwkv7_ttft_seed_repeat` | `2` | `1` | [rwkv7_ttft_0p4b_eager_20260413.json](/tmp/rwkv7_ttft_0p4b_eager_20260413.json) | [vllm_rwkv7_ttft_eager_20260413.log](/tmp/vllm_rwkv7_ttft_eager_20260413.log) | `tmp_rwkv7_ttft_benchmark.py --enforce-eager`；`server_ready_sec=30.034`；prefill 部分使用 streaming `max_tokens=1` 的 TTFT proxy，因为 vLLM 不支持 `max_tokens=0` |
| `2026-04-13_compile_no_cg_0p4b_ttft` | `2026-04-13` | `RWKV7-Goose-World2.9-0.4B-HF` | `0.4B` | `compile_no_cg` | `ttft_prefill_proxy_decode` | `rwkv7_ttft_seed_repeat` | `2` | `1` | [rwkv7_ttft_0p4b_compile_no_cg_20260413.json](/tmp/rwkv7_ttft_0p4b_compile_no_cg_20260413.json) | [vllm_rwkv7_ttft_compile_no_cg_20260413.log](/tmp/vllm_rwkv7_ttft_compile_no_cg_20260413.log) | `tmp_rwkv7_ttft_benchmark.py --compile-no-cg --disable-compile-cache`；`server_ready_sec=38.044`；长 prompt TTFT 没有明显优于 eager |
| `2026-04-13_piecewise_0p4b_ttft` | `2026-04-13` | `RWKV7-Goose-World2.9-0.4B-HF` | `0.4B` | `piecewise` | `ttft_prefill_proxy_decode` | `rwkv7_ttft_seed_repeat` | `2` | `1` | [rwkv7_ttft_0p4b_piecewise_20260413.json](/tmp/rwkv7_ttft_0p4b_piecewise_20260413.json) | [vllm_rwkv7_ttft_piecewise_20260413.log](/tmp/vllm_rwkv7_ttft_piecewise_20260413.log) | `tmp_rwkv7_ttft_benchmark.py --cudagraph-mode piecewise --disable-compile-cache`；`server_ready_sec=108.093`；长 prompt TTFT 开始略优于 eager，但启动仍明显更慢 |
| `2026-04-13_eager_0p4b_ttft_fusedoff_r3w2` | `2026-04-13` | `RWKV7-Goose-World2.9-0.4B-HF` | `0.4B` | `eager` | `ttft_prefill_proxy_decode` | `rwkv7_ttft_seed_repeat` | `3` | `2` | [rwkv7_ttft_0p4b_eager_fusedoff_r3w2_20260413.json](/tmp/rwkv7_ttft_0p4b_eager_fusedoff_r3w2_20260413.json) | [vllm_rwkv7_ttft_eager_fusedoff_r3w2_20260413.log](/tmp/vllm_rwkv7_ttft_eager_fusedoff_r3w2_20260413.log) | 关闭 `RWKV7` fused prefill 的稳定多轮基线；decode ITL 稳定在 `27ms` 左右，用来对照 Python token loop |
| `2026-04-13_eager_0p4b_ttft_fusedon_r3w2` | `2026-04-13` | `RWKV7-Goose-World2.9-0.4B-HF` | `0.4B` | `eager` | `ttft_prefill_proxy_decode` | `rwkv7_ttft_seed_repeat` | `3` | `2` | [rwkv7_ttft_0p4b_eager_fusedon_r3w2_20260413.json](/tmp/rwkv7_ttft_0p4b_eager_fusedon_r3w2_20260413.json) | [vllm_rwkv7_ttft_eager_fusedon_r3w2_20260413.log](/tmp/vllm_rwkv7_ttft_eager_fusedon_r3w2_20260413.log) | 启用 fused prefill 的稳定多轮 eager 复测；长 prefill TTFT 显著下降，decode ITL 小幅回升到 `33~34ms`，未再出现秒级 outlier |
| `2026-04-13_piecewise_0p4b_ttft_fusedon_r3w2` | `2026-04-13` | `RWKV7-Goose-World2.9-0.4B-HF` | `0.4B` | `piecewise` | `ttft_prefill_proxy_decode` | `rwkv7_ttft_seed_repeat` | `3` | `2` | [rwkv7_ttft_0p4b_piecewise_fusedon_r3w2_20260413.json](/tmp/rwkv7_ttft_0p4b_piecewise_fusedon_r3w2_20260413.json) | [vllm_rwkv7_ttft_piecewise_fusedon_r3w2_20260413.log](/tmp/vllm_rwkv7_ttft_piecewise_fusedon_r3w2_20260413.log) | 启用 fused prefill 的稳定多轮 PIECEWISE 复测；长 prefill TTFT 继续下降，decode ITL 回到 eager fused-off 同量级；`server_ready_sec` 受 warm cache 影响，不宜和最早 cold run 直接对比 |

## Prefill Proxy Table

| run_id | model_name | model_size | mode | prompt_len | proxy_type | avg_ttft_ms | median_ttft_ms | avg_latency_ms | successful_rounds |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `2026-04-13_eager_0p4b_ttft` | `RWKV7-Goose-World2.9-0.4B-HF` | `0.4B` | `eager` | `64` | `streaming_max_tokens_1` | `188.986` | `188.986` | `188.986` | `2` |
| `2026-04-13_eager_0p4b_ttft` | `RWKV7-Goose-World2.9-0.4B-HF` | `0.4B` | `eager` | `1024` | `streaming_max_tokens_1` | `2708.602` | `2708.602` | `2708.602` | `2` |
| `2026-04-13_eager_0p4b_ttft` | `RWKV7-Goose-World2.9-0.4B-HF` | `0.4B` | `eager` | `1984` | `streaming_max_tokens_1` | `5409.289` | `5409.289` | `5409.289` | `2` |
| `2026-04-13_compile_no_cg_0p4b_ttft` | `RWKV7-Goose-World2.9-0.4B-HF` | `0.4B` | `compile_no_cg` | `64` | `streaming_max_tokens_1` | `219.612` | `219.612` | `219.612` | `2` |
| `2026-04-13_compile_no_cg_0p4b_ttft` | `RWKV7-Goose-World2.9-0.4B-HF` | `0.4B` | `compile_no_cg` | `1024` | `streaming_max_tokens_1` | `2915.364` | `2915.364` | `2915.364` | `2` |
| `2026-04-13_compile_no_cg_0p4b_ttft` | `RWKV7-Goose-World2.9-0.4B-HF` | `0.4B` | `compile_no_cg` | `1984` | `streaming_max_tokens_1` | `5432.906` | `5432.906` | `5432.906` | `2` |
| `2026-04-13_piecewise_0p4b_ttft` | `RWKV7-Goose-World2.9-0.4B-HF` | `0.4B` | `piecewise` | `64` | `streaming_max_tokens_1` | `229.968` | `229.968` | `229.968` | `2` |
| `2026-04-13_piecewise_0p4b_ttft` | `RWKV7-Goose-World2.9-0.4B-HF` | `0.4B` | `piecewise` | `1024` | `streaming_max_tokens_1` | `2660.877` | `2660.877` | `2660.877` | `2` |
| `2026-04-13_piecewise_0p4b_ttft` | `RWKV7-Goose-World2.9-0.4B-HF` | `0.4B` | `piecewise` | `1984` | `streaming_max_tokens_1` | `4930.740` | `4930.740` | `4930.740` | `2` |
| `2026-04-13_eager_0p4b_ttft_fusedoff_r3w2` | `RWKV7-Goose-World2.9-0.4B-HF` | `0.4B` | `eager` | `64` | `streaming_max_tokens_1` | `577.311` | `578.445` | `577.311` | `3` |
| `2026-04-13_eager_0p4b_ttft_fusedoff_r3w2` | `RWKV7-Goose-World2.9-0.4B-HF` | `0.4B` | `eager` | `1024` | `streaming_max_tokens_1` | `2991.876` | `2991.847` | `2991.876` | `3` |
| `2026-04-13_eager_0p4b_ttft_fusedoff_r3w2` | `RWKV7-Goose-World2.9-0.4B-HF` | `0.4B` | `eager` | `1984` | `streaming_max_tokens_1` | `5031.042` | `5047.797` | `5031.042` | `3` |
| `2026-04-13_eager_0p4b_ttft_fusedon_r3w2` | `RWKV7-Goose-World2.9-0.4B-HF` | `0.4B` | `eager` | `64` | `streaming_max_tokens_1` | `96.296` | `100.860` | `96.296` | `3` |
| `2026-04-13_eager_0p4b_ttft_fusedon_r3w2` | `RWKV7-Goose-World2.9-0.4B-HF` | `0.4B` | `eager` | `1024` | `streaming_max_tokens_1` | `522.067` | `523.143` | `522.067` | `3` |
| `2026-04-13_eager_0p4b_ttft_fusedon_r3w2` | `RWKV7-Goose-World2.9-0.4B-HF` | `0.4B` | `eager` | `1984` | `streaming_max_tokens_1` | `1069.782` | `1059.762` | `1069.782` | `3` |
| `2026-04-13_piecewise_0p4b_ttft_fusedon_r3w2` | `RWKV7-Goose-World2.9-0.4B-HF` | `0.4B` | `piecewise` | `64` | `streaming_max_tokens_1` | `60.013` | `66.210` | `60.013` | `3` |
| `2026-04-13_piecewise_0p4b_ttft_fusedon_r3w2` | `RWKV7-Goose-World2.9-0.4B-HF` | `0.4B` | `piecewise` | `1024` | `streaming_max_tokens_1` | `276.974` | `277.462` | `276.974` | `3` |
| `2026-04-13_piecewise_0p4b_ttft_fusedon_r3w2` | `RWKV7-Goose-World2.9-0.4B-HF` | `0.4B` | `piecewise` | `1984` | `streaming_max_tokens_1` | `530.588` | `529.985` | `530.588` | `3` |

## Decode Latency Table

| run_id | model_name | model_size | mode | prompt_len | max_tokens | avg_ttft_ms | avg_latency_ms | avg_tpot_ms | avg_itl_ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `2026-04-13_eager_0p4b_ttft` | `RWKV7-Goose-World2.9-0.4B-HF` | `0.4B` | `eager` | `64` | `32` | `255.053` | `1118.779` | `27.862` | `27.862` |
| `2026-04-13_eager_0p4b_ttft` | `RWKV7-Goose-World2.9-0.4B-HF` | `0.4B` | `eager` | `64` | `64` | `255.353` | `2061.966` | `28.676` | `28.676` |
| `2026-04-13_compile_no_cg_0p4b_ttft` | `RWKV7-Goose-World2.9-0.4B-HF` | `0.4B` | `compile_no_cg` | `64` | `32` | `412.099` | `1328.407` | `29.558` | `29.558` |
| `2026-04-13_compile_no_cg_0p4b_ttft` | `RWKV7-Goose-World2.9-0.4B-HF` | `0.4B` | `compile_no_cg` | `64` | `64` | `250.975` | `2167.977` | `30.429` | `30.429` |
| `2026-04-13_piecewise_0p4b_ttft` | `RWKV7-Goose-World2.9-0.4B-HF` | `0.4B` | `piecewise` | `64` | `32` | `251.827` | `1111.017` | `27.716` | `27.716` |
| `2026-04-13_piecewise_0p4b_ttft` | `RWKV7-Goose-World2.9-0.4B-HF` | `0.4B` | `piecewise` | `64` | `64` | `275.944` | `2041.429` | `28.024` | `28.024` |
| `2026-04-13_eager_0p4b_ttft_fusedoff_r3w2` | `RWKV7-Goose-World2.9-0.4B-HF` | `0.4B` | `eager` | `64` | `32` | `247.801` | `1102.542` | `27.572` | `27.572` |
| `2026-04-13_eager_0p4b_ttft_fusedoff_r3w2` | `RWKV7-Goose-World2.9-0.4B-HF` | `0.4B` | `eager` | `64` | `64` | `231.665` | `1951.312` | `27.296` | `27.296` |
| `2026-04-13_eager_0p4b_ttft_fusedon_r3w2` | `RWKV7-Goose-World2.9-0.4B-HF` | `0.4B` | `eager` | `64` | `32` | `204.524` | `1240.580` | `33.421` | `33.421` |
| `2026-04-13_eager_0p4b_ttft_fusedon_r3w2` | `RWKV7-Goose-World2.9-0.4B-HF` | `0.4B` | `eager` | `64` | `64` | `85.932` | `2230.139` | `34.035` | `34.035` |
| `2026-04-13_piecewise_0p4b_ttft_fusedon_r3w2` | `RWKV7-Goose-World2.9-0.4B-HF` | `0.4B` | `piecewise` | `64` | `32` | `88.598` | `942.975` | `27.561` | `27.561` |
| `2026-04-13_piecewise_0p4b_ttft_fusedon_r3w2` | `RWKV7-Goose-World2.9-0.4B-HF` | `0.4B` | `piecewise` | `64` | `64` | `70.734` | `1823.606` | `27.823` | `27.823` |

## Repro Commands

### `2026-04-13_eager_0p4b_mt64`

```bash
source ~/miniforge3/etc/profile.d/conda.sh
conda activate vllm-dev
cd /home/liu/vllm
python tmp_rwkv7_long_benchmark.py \
  --model /mnt/d/codes/RWKV7-Goose-World2.9-0.4B-HF \
  --enforce-eager \
  --port 8042 \
  --max-tokens 64 \
  --rounds 2 \
  --warmup 1 \
  --log /tmp/vllm_rwkv7_eager_bench_20260413.log \
  > /tmp/rwkv7_bench_0p4b_eager_64_20260413.json
```

### `2026-04-13_eager_0p4b_ttft`

```bash
source ~/miniforge3/etc/profile.d/conda.sh
conda activate vllm-dev
cd /home/liu/vllm
python tmp_rwkv7_ttft_benchmark.py \
  --model /mnt/d/codes/RWKV7-Goose-World2.9-0.4B-HF \
  --enforce-eager \
  --port 8044 \
  --rounds 2 \
  --warmup 1 \
  --prompt-lengths 64 1024 1984 \
  --decode-prompt-len 64 \
  --decode-output-lengths 32 64 \
  --log /tmp/vllm_rwkv7_ttft_eager_20260413.log \
  > /tmp/rwkv7_ttft_0p4b_eager_20260413.json
```

### `2026-04-13_compile_no_cg_0p4b_ttft`

```bash
source ~/miniforge3/etc/profile.d/conda.sh
conda activate vllm-dev
cd /home/liu/vllm
python tmp_rwkv7_ttft_benchmark.py \
  --model /mnt/d/codes/RWKV7-Goose-World2.9-0.4B-HF \
  --compile-no-cg \
  --disable-compile-cache \
  --port 8045 \
  --rounds 2 \
  --warmup 1 \
  --prompt-lengths 64 1024 1984 \
  --decode-prompt-len 64 \
  --decode-output-lengths 32 64 \
  --log /tmp/vllm_rwkv7_ttft_compile_no_cg_20260413.log \
  > /tmp/rwkv7_ttft_0p4b_compile_no_cg_20260413.json
```

### `2026-04-13_piecewise_0p4b_ttft`

```bash
source ~/miniforge3/etc/profile.d/conda.sh
conda activate vllm-dev
cd /home/liu/vllm
python tmp_rwkv7_ttft_benchmark.py \
  --model /mnt/d/codes/RWKV7-Goose-World2.9-0.4B-HF \
  --cudagraph-mode piecewise \
  --disable-compile-cache \
  --port 8046 \
  --rounds 2 \
  --warmup 1 \
  --prompt-lengths 64 1024 1984 \
  --decode-prompt-len 64 \
  --decode-output-lengths 32 64 \
  --log /tmp/vllm_rwkv7_ttft_piecewise_20260413.log \
  > /tmp/rwkv7_ttft_0p4b_piecewise_20260413.json
```

### `2026-04-13_eager_0p4b_ttft_fusedoff_r3w2`

```bash
source ~/miniforge3/etc/profile.d/conda.sh
conda activate vllm-dev
cd /home/liu/vllm
export RWKV7_DISABLE_FUSED_PREFILL=1
python tmp_rwkv7_ttft_benchmark.py \
  --model /mnt/d/codes/RWKV7-Goose-World2.9-0.4B-HF \
  --enforce-eager \
  --port 8049 \
  --rounds 3 \
  --warmup 2 \
  --prompt-lengths 64 1024 1984 \
  --decode-prompt-len 64 \
  --decode-output-lengths 32 64 \
  --log /tmp/vllm_rwkv7_ttft_eager_fusedoff_r3w2_20260413.log \
  > /tmp/rwkv7_ttft_0p4b_eager_fusedoff_r3w2_20260413.json
```

### `2026-04-13_eager_0p4b_ttft_fusedon_r3w2`

```bash
source ~/miniforge3/etc/profile.d/conda.sh
conda activate vllm-dev
cd /home/liu/vllm
unset RWKV7_DISABLE_FUSED_PREFILL
python tmp_rwkv7_ttft_benchmark.py \
  --model /mnt/d/codes/RWKV7-Goose-World2.9-0.4B-HF \
  --enforce-eager \
  --port 8050 \
  --rounds 3 \
  --warmup 2 \
  --prompt-lengths 64 1024 1984 \
  --decode-prompt-len 64 \
  --decode-output-lengths 32 64 \
  --log /tmp/vllm_rwkv7_ttft_eager_fusedon_r3w2_20260413.log \
  > /tmp/rwkv7_ttft_0p4b_eager_fusedon_r3w2_20260413.json
```

### `2026-04-13_piecewise_0p4b_ttft_fusedon_r3w2`

```bash
source ~/miniforge3/etc/profile.d/conda.sh
conda activate vllm-dev
cd /home/liu/vllm
unset RWKV7_DISABLE_FUSED_PREFILL
python tmp_rwkv7_ttft_benchmark.py \
  --model /mnt/d/codes/RWKV7-Goose-World2.9-0.4B-HF \
  --cudagraph-mode piecewise \
  --disable-compile-cache \
  --port 8051 \
  --rounds 3 \
  --warmup 2 \
  --prompt-lengths 64 1024 1984 \
  --decode-prompt-len 64 \
  --decode-output-lengths 32 64 \
  --log /tmp/vllm_rwkv7_ttft_piecewise_fusedon_r3w2_20260413.log \
  > /tmp/rwkv7_ttft_0p4b_piecewise_fusedon_r3w2_20260413.json
```

### `2026-04-13_piecewise_0p4b_mt16_smoke_packedprefill`

```bash
source ~/miniforge3/etc/profile.d/conda.sh
conda activate vllm-dev
cd /home/liu/vllm
python tmp_rwkv7_long_benchmark.py \
  --model /mnt/d/codes/RWKV7-Goose-World2.9-0.4B-HF \
  --cudagraph-mode piecewise \
  --disable-compile-cache \
  --port 8052 \
  --max-tokens 16 \
  --rounds 1 \
  --warmup 1 \
  --concurrency-levels 4 8 \
  --log /tmp/vllm_rwkv7_long_piecewise_packedprefill_20260413.log \
  > /tmp/rwkv7_long_piecewise_packedprefill_20260413.json
```

### `2026-04-13_eager_0p4b_exact_long_mt64`

```bash
source ~/miniforge3/etc/profile.d/conda.sh
conda activate vllm-dev
cd /home/liu/vllm
python tmp_rwkv7_exact_long_input_bench.py \
  --model /mnt/d/codes/RWKV7-Goose-World2.9-0.4B-HF \
  --enforce-eager \
  --port 8053 \
  --max-tokens 64 \
  --rounds 1 \
  --warmup 1 \
  --prompt-lengths 1024 1984 \
  --concurrency-levels 1 4 8 \
  --log /tmp/vllm_rwkv7_exact_long_eager_20260413.log \
  > /tmp/rwkv7_exact_long_eager_20260413.json
```

### `2026-04-13_piecewise_0p4b_exact_long_mt64`

```bash
source ~/miniforge3/etc/profile.d/conda.sh
conda activate vllm-dev
cd /home/liu/vllm
python tmp_rwkv7_exact_long_input_bench.py \
  --model /mnt/d/codes/RWKV7-Goose-World2.9-0.4B-HF \
  --cudagraph-mode piecewise \
  --disable-compile-cache \
  --port 8054 \
  --max-tokens 64 \
  --rounds 1 \
  --warmup 1 \
  --prompt-lengths 1024 1984 \
  --concurrency-levels 1 4 8 \
  --log /tmp/vllm_rwkv7_exact_long_piecewise_20260413.log \
  > /tmp/rwkv7_exact_long_piecewise_20260413.json
```

### `2026-04-13_eager_0p4b_exact_long_mt64_decodefused_seq`

```bash
source ~/miniforge3/etc/profile.d/conda.sh
conda activate vllm-dev
cd /home/liu/vllm
python tmp_rwkv7_exact_long_input_bench.py \
  --model /mnt/d/codes/RWKV7-Goose-World2.9-0.4B-HF \
  --enforce-eager \
  --port 8055 \
  --max-tokens 64 \
  --rounds 2 \
  --warmup 2 \
  --prompt-lengths 1024 1984 \
  --concurrency-levels 8 \
  --log /tmp/vllm_rwkv7_exact_long_eager_decodefused_seq_20260413.log \
  > /tmp/rwkv7_exact_long_eager_decodefused_seq_20260413.json
```

### `2026-04-13_piecewise_0p4b_exact_long_mt64_decodefused_seq`

```bash
source ~/miniforge3/etc/profile.d/conda.sh
conda activate vllm-dev
cd /home/liu/vllm
python tmp_rwkv7_exact_long_input_bench.py \
  --model /mnt/d/codes/RWKV7-Goose-World2.9-0.4B-HF \
  --cudagraph-mode piecewise \
  --disable-compile-cache \
  --port 8056 \
  --max-tokens 64 \
  --rounds 2 \
  --warmup 2 \
  --prompt-lengths 1024 1984 \
  --concurrency-levels 8 \
  --log /tmp/vllm_rwkv7_exact_long_piecewise_decodefused_seq_20260413.log \
  > /tmp/rwkv7_exact_long_piecewise_decodefused_seq_20260413.json
```

### `2026-04-13_eager_0p4b_exact_long_mt64_prefixcache`

```bash
source ~/miniforge3/etc/profile.d/conda.sh
conda activate vllm-dev
cd /home/liu/vllm
python tmp_rwkv7_exact_long_input_bench.py \
  --model /mnt/d/codes/RWKV7-Goose-World2.9-0.4B-HF \
  --enforce-eager \
  --enable-prefix-caching \
  --port 8057 \
  --max-tokens 64 \
  --rounds 2 \
  --warmup 2 \
  --prompt-lengths 1024 1984 \
  --concurrency-levels 8 \
  --log /tmp/vllm_rwkv7_exact_long_eager_prefixcache_20260413.log \
  > /tmp/rwkv7_exact_long_eager_prefixcache_20260413.json
```

### `2026-04-13_piecewise_0p4b_exact_long_mt64_prefixcache`

```bash
source ~/miniforge3/etc/profile.d/conda.sh
conda activate vllm-dev
cd /home/liu/vllm
python tmp_rwkv7_exact_long_input_bench.py \
  --model /mnt/d/codes/RWKV7-Goose-World2.9-0.4B-HF \
  --enable-prefix-caching \
  --disable-compile-cache \
  --port 8058 \
  --max-tokens 64 \
  --rounds 2 \
  --warmup 2 \
  --prompt-lengths 1024 1984 \
  --concurrency-levels 8 \
  --log /tmp/vllm_rwkv7_exact_long_piecewise_prefixcache_20260413.log \
  > /tmp/rwkv7_exact_long_piecewise_prefixcache_20260413.json
```

### `2026-04-13_eager_0p4b_mixed_exact_mt64`

```bash
source ~/miniforge3/etc/profile.d/conda.sh
conda activate vllm-dev
cd /home/liu/vllm
python tmp_rwkv7_mixed_exact_prompt_bench.py \
  --model /mnt/d/codes/RWKV7-Goose-World2.9-0.4B-HF \
  --enforce-eager \
  --port 8059 \
  --max-tokens 64 \
  --rounds 2 \
  --warmup 1 \
  --log /tmp/vllm_rwkv7_mixed_exact_eager_20260413.log \
  > /tmp/rwkv7_mixed_exact_eager_20260413.json
```

### `2026-04-13_piecewise_0p4b_mixed_exact_mt64`

```bash
source ~/miniforge3/etc/profile.d/conda.sh
conda activate vllm-dev
cd /home/liu/vllm
python tmp_rwkv7_mixed_exact_prompt_bench.py \
  --model /mnt/d/codes/RWKV7-Goose-World2.9-0.4B-HF \
  --disable-compile-cache \
  --port 8060 \
  --max-tokens 64 \
  --rounds 2 \
  --warmup 1 \
  --log /tmp/vllm_rwkv7_mixed_exact_piecewise_20260413.log \
  > /tmp/rwkv7_mixed_exact_piecewise_20260413.json
```

### `2026-04-13_eager_0p4b_mixed_exact_mt64_prefixcache`

```bash
source ~/miniforge3/etc/profile.d/conda.sh
conda activate vllm-dev
cd /home/liu/vllm
python tmp_rwkv7_mixed_exact_prompt_bench.py \
  --model /mnt/d/codes/RWKV7-Goose-World2.9-0.4B-HF \
  --enforce-eager \
  --enable-prefix-caching \
  --port 8061 \
  --max-tokens 64 \
  --rounds 2 \
  --warmup 1 \
  --log /tmp/vllm_rwkv7_mixed_exact_eager_prefixcache_20260413.log \
  > /tmp/rwkv7_mixed_exact_eager_prefixcache_20260413.json
```

### `2026-04-13_piecewise_0p4b_mixed_exact_mt64_prefixcache`

```bash
source ~/miniforge3/etc/profile.d/conda.sh
conda activate vllm-dev
cd /home/liu/vllm
python tmp_rwkv7_mixed_exact_prompt_bench.py \
  --model /mnt/d/codes/RWKV7-Goose-World2.9-0.4B-HF \
  --enable-prefix-caching \
  --disable-compile-cache \
  --port 8062 \
  --max-tokens 64 \
  --rounds 2 \
  --warmup 1 \
  --log /tmp/vllm_rwkv7_mixed_exact_piecewise_prefixcache_20260413.log \
  > /tmp/rwkv7_mixed_exact_piecewise_prefixcache_20260413.json
```

### `2026-04-13_eager_0p4b_exact_long_mt128`

```bash
source ~/miniforge3/etc/profile.d/conda.sh
conda activate vllm-dev
cd /home/liu/vllm
python tmp_rwkv7_exact_long_input_bench.py \
  --model /mnt/d/codes/RWKV7-Goose-World2.9-0.4B-HF \
  --enforce-eager \
  --port 8063 \
  --max-tokens 128 \
  --rounds 2 \
  --warmup 2 \
  --prompt-lengths 1024 1920 \
  --concurrency-levels 8 \
  --log /tmp/vllm_rwkv7_exact_long_eager_mt128_20260413.log \
  > /tmp/rwkv7_exact_long_eager_mt128_20260413.json
```

### `2026-04-13_piecewise_0p4b_exact_long_mt128`

```bash
source ~/miniforge3/etc/profile.d/conda.sh
conda activate vllm-dev
cd /home/liu/vllm
python tmp_rwkv7_exact_long_input_bench.py \
  --model /mnt/d/codes/RWKV7-Goose-World2.9-0.4B-HF \
  --cudagraph-mode piecewise \
  --disable-compile-cache \
  --port 8064 \
  --max-tokens 128 \
  --rounds 2 \
  --warmup 2 \
  --prompt-lengths 1024 1920 \
  --concurrency-levels 8 \
  --log /tmp/vllm_rwkv7_exact_long_piecewise_mt128_20260413.log \
  > /tmp/rwkv7_exact_long_piecewise_mt128_20260413.json
```

### `2026-04-13_compile_no_cg_0p4b_exact_long_mt128`

```bash
source ~/miniforge3/etc/profile.d/conda.sh
conda activate vllm-dev
cd /home/liu/vllm
python tmp_rwkv7_exact_long_input_bench.py \
  --model /mnt/d/codes/RWKV7-Goose-World2.9-0.4B-HF \
  --compile-no-cg \
  --disable-compile-cache \
  --port 8065 \
  --max-tokens 128 \
  --rounds 2 \
  --warmup 2 \
  --prompt-lengths 1024 1920 \
  --concurrency-levels 8 \
  --log /tmp/vllm_rwkv7_exact_long_compile_no_cg_mt128_20260413.log \
  > /tmp/rwkv7_exact_long_compile_no_cg_mt128_20260413.json
```

### `2026-04-13_compile_no_cg_0p4b_mt128`

```bash
source ~/miniforge3/etc/profile.d/conda.sh
conda activate vllm-dev
cd /home/liu/vllm
python tmp_rwkv7_long_benchmark.py \
  --model /mnt/d/codes/RWKV7-Goose-World2.9-0.4B-HF \
  --compile-no-cg \
  --disable-compile-cache \
  --port 8066 \
  --max-tokens 128 \
  --rounds 2 \
  --warmup 1 \
  --concurrency-levels 8 \
  --log /tmp/vllm_rwkv7_long_compile_no_cg_mt128_20260413.log \
  > /tmp/rwkv7_long_compile_no_cg_mt128_20260413.json
```
