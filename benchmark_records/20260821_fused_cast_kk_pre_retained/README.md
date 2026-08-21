# RWKV7 fused BF16-to-FP32 recurrent-input preparation

## Candidate

`RWKV7_USE_FUSED_CAST_KK_PRE=1` fuses five existing BF16-to-FP32 recurrent
input materializations with the existing exact Triton `k` adjustment / `kk`
normalization preparation. It is restricted to the checkpoint's equal 64-wide
R/W/K/A/V head layout; all other layouts retain the original fallback.

The kernel preserves the existing `rwkv7_kk_pre` Triton expression exactly:
BF16 values are converted to FP32 before the same `kk_raw`, squared-sum,
`rsqrt`, and k-adjustment operations. It removes the separately materialized
FP32 k/a tensors and the standalone kk-pre launch.

## Precision gates

- Isolated B=1, 8, 128: all six outputs are `torch.equal` to five individual
  BF16-to-FP32 casts followed by the existing `rwkv7_kk_pre`.
- New unit coverage: 3 parameterized cases passed.
- Full focused RWKV7 regression: **92 passed, 4 skipped**.
- Full Decode graph capture sizes 1, 2, 4, 8, 16, 32, 64, 128: successful.
- Service 8 prompts x 16 greedy / `logprobs=20`: byte-identical to
  `/tmp/rwkv7_service_trace_before_aux.json`.

## Throughput

With retained bulk projection streams enabled, the candidate measured:

| workload | result |
|---|---:|
| C=1, four post-warmup samples | **28.9035 TPS** |
| C=128 first two samples | 1603.253, 1608.442 TPS |
| C=128 second two samples | 1605.468, 1599.020 TPS |
| C=128 four-sample mean | **1604.05 aggregate TPS** |

The immediately preceding retained bulk-stream configuration measured
1584.997 aggregate TPS in its clean-restart paired run. The fused preparation
therefore adds about **+1.20%** C=128 throughput on top of that path, while
C=1 improves from roughly 28.46 to 28.90 TPS (about **+1.56%**).

## Decision

Retained. This is a strict, graph-safe reduction in recurrent-input launch and
materialization overhead, with output equality established before service
benchmarking.
