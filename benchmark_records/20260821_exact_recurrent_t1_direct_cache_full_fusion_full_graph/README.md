# RWKV7 exact direct-cache full recurrence fusion

- Date: 2026-08-21
- GPU: NVIDIA GeForce RTX 4090 D (sm89)
- Runtime: vllm-sp; BF16 projections and FP32 recurrent state
- Model: `/hikscale/models/XiaoKe/rwkv-0-hf` (`xiaoke-5`)

## Change

The preceding retained path used two CUDA Graph nodes per RWKV7 layer:

```text
exact sa = sum(state * -kk) from persistent cache
exact cache update + exact output reduction
```

This candidate merges all three recurrence phases in one `32 x 4` CTA layout:

```text
read each FP32 state element once
-> exact sa reduction
-> exact state update
-> exact r/output reduction
-> write each valid cache element once
```

The original state is staged in shared memory (no register spill), while both
reductions deliberately retain the existing four partial accumulators and the
same `4 -> 2 -> 1` shared-memory tree. State pointwise arithmetic still uses
explicit round-to-nearest FP32 operations. Full CUDA Graph padded lanes read
slot zero only for ignored output and never mutate cache.

`RWKV7_USE_EXACT_RECURRENT_T1_FULL_FUSION` now defaults to enabled whenever the
existing direct-cache fast path is enabled. Set it to `0` for an immediate A/B
fallback to the prior two-kernel direct-cache implementation. All unsupported
layout/dtype/device/cache-mode cases retain their existing fallbacks.

## Accuracy gate

- Synthetic direct-cache operator: valid output and full cache are
  `torch.equal` at B=1/8/128 across three single-step seeds, with padded slots.
- New recurrence regression: full-fusion cache and output remain `torch.equal`
  for ten consecutive steps at B=1/8/128.
- Service gate: 8 prompts, greedy, 16 decode steps, `logprobs=20`, Full Decode
  CUDA Graph. Text mismatch, token mismatch, selected logprob error, and common
  Top-K logprob error are all **0**.
- Focused RWKV7 suite with an unloaded service: **81 passed, 2 skipped**.

## Performance gate

Each C=128 measurement runs 128 closed-loop completion requests, 32 generated
tokens/request, and 4096 aggregate output tokens; warmups are excluded. The
baseline and candidate were clean-restarted services with identical startup
arguments except for the full-fusion gate.

| Concurrency | Baseline TPS | Candidate TPS | Change |
| ---: | ---: | ---: | ---: |
| 128 aggregate | 1365.38 | 1541.71 | **+12.91%** |
| 1 | 26.22 | 26.35 | +0.47% |

This is a high-concurrency optimization: it removes one full FP32 state-cache
read plus one graph node per RWKV7 layer. It does not solve the C=1 projection
bandwidth bottleneck; the remaining C=1 target requires an independently strict
BF16 projection/CMix breakthrough.

`raw/` contains all service traces and benchmark summaries used for the result.
