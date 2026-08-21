# RWKV7 Full Decode CUDA Graph Triton masked-state store

- Date: 2026-08-21
- GPU: NVIDIA GeForce RTX 4090 D (sm89)
- Runtime: vllm-sp / Python 3.11.15 / PyTorch 2.11.0+cu130
- Model: `/hikscale/models/XiaoKe/rwkv-0-hf` (`xiaoke-5`, BF16)

## Change

Source checkouts on this host do not expose the optional native
`_C::rwkv7_masked_store` extension and cannot rebuild it because `nvcc` is not
available. The old Full CUDA Graph fallback used `index_select + where +
index_copy_` for each of the three state caches. Besides rereading the FP32
recurrent matrix, its `PAD_SLOT_ID=-1 -> 0` remap relied on duplicate-index
write ordering when a padded lane followed a real slot-zero update.

`rwkv7_masked_store_triton` is a capture-safe, stride-aware Triton scatter. It
writes only non-negative slot IDs, so a padding lane neither reads an old row
nor writes slot zero. `RWKV7_USE_TRITON_MASKED_STORE=0` remains an explicit A/B
escape hatch for the legacy fallback; the Triton path is the default.

## Accuracy gate

The candidate was compared against the pre-change Full Decode trace using 8
fixed prompts, greedy sampling (`temperature=0`, `top_p=1`, `seed=0`), 16
generated tokens, and `logprobs=20`.

Result: 128/128 decode steps had identical text and Top-1 tokens; common Top-K
and selected-token logprob errors were exactly zero. The CUDA unit suite also
covers a real slot-zero update followed by padding and a 128-row block decode
with one padded lane.

See `accuracy.json`.

## Performance gate

The A/B service configuration was identical except for
`RWKV7_USE_TRITON_MASKED_STORE`. It used `/v1/completions`, 128 requests,
concurrency 128, `max_tokens=32`; every run produced 3,824 completion tokens
and all 128 requests succeeded.

| Mode | Aggregate output TPS | Mean |
|---|---:|---:|
| Legacy fallback (`=0`) | 373.69, 383.86 | 378.78 |
| Triton store (`=1`) | 457.52, 460.05 | 458.78 |

**Mean throughput gain: +21.12%.**

The prior direct recurrent-cache T=1 experiment was intentionally discarded:
its strict C=128 A/B result was neutral/slightly negative once this state-store
improvement was present. Only the independent masked-store speedup is retained.

See `performance.json`.

## Reproduction

```bash
CUDA_VISIBLE_DEVICES=0 \
VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 \
VLLM_DISABLE_COMPILE_CACHE=1 \
RWKV7_USE_TRITON_MASKED_STORE=1 \
/mnt/data/anaconda3/envs/vllm-sp/bin/vllm serve \
  /hikscale/models/XiaoKe/rwkv-0-hf \
  --served-model-name xiaoke-5 \
  --reasoning-parser rwkv --enable-auto-tool-choice --tool-call-parser rwkv \
  --dtype bfloat16 --host 0.0.0.0 --port 8030 \
  --max-model-len 1M --max-num-seqs 128 --gpu-memory-utilization 0.92 \
  --enable-prefix-caching --mamba-cache-mode align \
  --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[1,2,4,8,16,32,64,128]}'
```
