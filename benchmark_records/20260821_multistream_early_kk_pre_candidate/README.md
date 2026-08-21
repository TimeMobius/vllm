# Multistream K/A → kk-pre early-release candidate — rejected

## Hypothesis

The retained C=1 three-child-stream decode path originally issued `K → W → A`
on one child stream and only ran the unchanged FP32 casts plus `kk_pre(K, A)`
after joining every child stream. This candidate changed only stream events:

1. issue `K → A` on the existing child stream;
2. record a K/A-ready event;
3. continue independent W work on that same child stream;
4. let the main stream run the original BF16-to-FP32 casts and the existing
   `kk_pre` while W and the R+G/V+V-gate streams are still active.

No F.linear operands, dtype, reduction, activation, recurrence kernel, cache
write, or output reduction was changed. The candidate was behind
`RWKV7_USE_MULTISTREAM_EARLY_KK_PRE=1` and was fully restored after testing.

## Correctness

- Strict tensor test: `test_rwkv7_multistream_rkv_projection_preserves_bits`
  passed with the early event path enabled for layer 0 and layer 1.
- Full Decode CUDA Graph startup/capture completed for `1,2,4,8,16,32,64,128`.
- Service `8 prompts × 16 greedy, logprobs=20` trace was byte-identical to
  `/tmp/rwkv7_service_trace_before_aux.json`.

## Restart-paired HTTP benchmark

Stable service, 32 output tokens, BF16 model, FP32 recurrent cache,
`mamba-cache-mode=align`, Full Decode CUDA Graph.

| Workload | Stable baseline | Candidate | Change |
|---|---:|---:|---:|
| C=1 output TPS | 28.4527 | 28.6101 | +0.55% |
| C=128 aggregate output TPS | 1564.9534 | 1569.4818 | +0.29% |

## Decision

Rejected and source restored. The scheduling dependency is exact, but its
small improvement is below the 1% materiality gate and does not change the
C=1 bottleneck. The stable path remains the previously retained balanced
three-stream projection schedule without the extra K/A event or reordering.
