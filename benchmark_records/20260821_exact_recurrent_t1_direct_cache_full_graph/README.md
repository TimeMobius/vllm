# RWKV7 exact direct persistent recurrent-cache update (Full Decode CUDA Graph)

- Date: 2026-08-21
- GPU: NVIDIA GeForce RTX 4090 D (sm89)
- Runtime: vllm-sp / Python 3.11.15 / PyTorch 2.11.0+cu130
- Model: `/hikscale/models/XiaoKe/rwkv-0-hf` (`xiaoke-5`; BF16 projections and FP32 recurrent state)

## Change

The previous strict decode path copied each layer's FP32 recurrent state out of
vLLM's state cache, updated a temporary `[B,H,64,64]` tensor, then copied that
tensor back. At C=128, each layer touches a 128 MiB recurrent state row on both
the gather and store paths.

This native CUDA path operates directly on the persistent cache rows for
`mamba_cache_mode=align` only. It preserves the existing numerical contract:

1. `sa = sum(state * -kk, dim=-2)` follows the PyTorch `reduce_kernel<128,4>`
   FP32 multiply/reduction order;
2. pointwise update uses the pre-existing explicit `__fmul_rn` / `__fadd_rn`
   order;
3. final `sum(new_state * r, dim=-2)` follows the same exact reduction order.

It supports the actual vLLM state-cache layout: a strided
`[slots,H,64,64]` view whose outer slot stride can exceed one state row. Padded
`slot_id=-1` lanes read slot zero only for graph execution and never write it.
The cache-all, prefill, eager, CPU, unsupported-dtype, and unsupported-layout
paths retain their original implementation.

The guarded switch is:

```bash
RWKV7_USE_EXACT_RECURRENT_T1_DIRECT_CACHE=1
```

## Accuracy gate

- isolated strided-cache operator, B=1/8/128 and three seeds: valid-lane output
  and all cache rows are bitwise equal to eager gather/update/store;
- explicit padded `slot_id=-1` tests verify slot-zero/cache preservation;
- service greedy regression: 8 prompts × 16 decode tokens, `logprobs=20`:
  text, token IDs, selected logprob, and common Top-K logprob all have **zero
  difference**.

## Performance gate

The same rebuilt `_C.abi3.so` was used in both modes. Workload: C=128, 128
requests, 32 output tokens/request, Full Decode CUDA Graph. The first run is
warm-up; reported result is the mean of runs two and three.

| Concurrency | Baseline TPS | Direct-cache TPS | Change |
| ---: | ---: | ---: | ---: |
| 128 | 791.76 | 1205.23 | **+52.22%** |
| 1 | 25.91 | 26.21 | +1.13% |

Raw C=128 values are 792.51 / 792.04 / 791.49 TPS (baseline) and
1173.79 / 1204.36 / 1206.10 TPS (candidate). C=1 remains dominated by small
BF16 projections, so this is a high-concurrency Full Decode optimization,
not a claim of reaching the single-request target.
