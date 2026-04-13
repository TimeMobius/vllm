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

## Run Index

| run_id | date | model_name | model_size | mode | dtype | max_tokens | rounds | warmup | concurrency_levels | prompt_set_id | raw_json | server_log | notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `2026-04-13_eager_0p4b_mt64` | `2026-04-13` | `RWKV7-Goose-World2.9-0.4B-HF` | `0.4B` | `eager` | `auto` | `64` | `2` | `1` | `1/2/4/8` | `default_mixed_8` | [rwkv7_bench_0p4b_eager_64_20260413.json](/tmp/rwkv7_bench_0p4b_eager_64_20260413.json) | [vllm_rwkv7_eager_bench_20260413.log](/tmp/vllm_rwkv7_eager_bench_20260413.log) | `tmp_rwkv7_long_benchmark.py --enforce-eager` 基线复跑；全部并发轮次都与串行 baseline 一致 |

## Throughput Table

| run_id | model_name | model_size | mode | max_tokens | concurrency | round0_tps | round1_tps | avg_tps | all_match_serial_baseline |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `2026-04-13_eager_0p4b_mt64` | `RWKV7-Goose-World2.9-0.4B-HF` | `0.4B` | `eager` | `64` | `1` | `35.047` | `34.912` | `34.980` | `true` |
| `2026-04-13_eager_0p4b_mt64` | `RWKV7-Goose-World2.9-0.4B-HF` | `0.4B` | `eager` | `64` | `2` | `68.610` | `71.110` | `69.860` | `true` |
| `2026-04-13_eager_0p4b_mt64` | `RWKV7-Goose-World2.9-0.4B-HF` | `0.4B` | `eager` | `64` | `4` | `132.553` | `132.926` | `132.739` | `true` |
| `2026-04-13_eager_0p4b_mt64` | `RWKV7-Goose-World2.9-0.4B-HF` | `0.4B` | `eager` | `64` | `8` | `215.131` | `214.844` | `214.987` | `true` |

## Latency Run Index

| run_id | date | model_name | model_size | mode | benchmark_type | prompt_set_id | rounds | warmup | raw_json | server_log | notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `2026-04-13_eager_0p4b_ttft` | `2026-04-13` | `RWKV7-Goose-World2.9-0.4B-HF` | `0.4B` | `eager` | `ttft_prefill_proxy_decode` | `rwkv7_ttft_seed_repeat` | `2` | `1` | [rwkv7_ttft_0p4b_eager_20260413.json](/tmp/rwkv7_ttft_0p4b_eager_20260413.json) | [vllm_rwkv7_ttft_eager_20260413.log](/tmp/vllm_rwkv7_ttft_eager_20260413.log) | `tmp_rwkv7_ttft_benchmark.py --enforce-eager`；`server_ready_sec=30.034`；prefill 部分使用 streaming `max_tokens=1` 的 TTFT proxy，因为 vLLM 不支持 `max_tokens=0` |

## Prefill Proxy Table

| run_id | model_name | model_size | mode | prompt_len | proxy_type | avg_ttft_ms | median_ttft_ms | avg_latency_ms | successful_rounds |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `2026-04-13_eager_0p4b_ttft` | `RWKV7-Goose-World2.9-0.4B-HF` | `0.4B` | `eager` | `64` | `streaming_max_tokens_1` | `188.986` | `188.986` | `188.986` | `2` |
| `2026-04-13_eager_0p4b_ttft` | `RWKV7-Goose-World2.9-0.4B-HF` | `0.4B` | `eager` | `1024` | `streaming_max_tokens_1` | `2708.602` | `2708.602` | `2708.602` | `2` |
| `2026-04-13_eager_0p4b_ttft` | `RWKV7-Goose-World2.9-0.4B-HF` | `0.4B` | `eager` | `1984` | `streaming_max_tokens_1` | `5409.289` | `5409.289` | `5409.289` | `2` |

## Decode Latency Table

| run_id | model_name | model_size | mode | prompt_len | max_tokens | avg_ttft_ms | avg_latency_ms | avg_tpot_ms | avg_itl_ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `2026-04-13_eager_0p4b_ttft` | `RWKV7-Goose-World2.9-0.4B-HF` | `0.4B` | `eager` | `64` | `32` | `255.053` | `1118.779` | `27.862` | `27.862` |
| `2026-04-13_eager_0p4b_ttft` | `RWKV7-Goose-World2.9-0.4B-HF` | `0.4B` | `eager` | `64` | `64` | `255.353` | `2061.966` | `28.676` | `28.676` |

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
