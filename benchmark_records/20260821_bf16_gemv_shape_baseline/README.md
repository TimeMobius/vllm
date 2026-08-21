# BF16 M=1 projection baseline and output-row split probe

## Goal

Measure the actual isolated `torch.nn.functional.linear` contract used by the
RWKV7 CUDA decode fast path on the target RTX 4090 D (SM89), then test the
lowest-risk fusion-like alternative: partitioning only output rows. Output-row
partitioning does not split a dot-product reduction and can therefore be
bit-exact for some shapes, unlike K-dimension splitting/atomic accumulation.

All probes use CUDA BF16 input/weights, M=1, `allow_tf32=False`, and compare
candidate output with `torch.equal(F.linear(x, weight))`.

## Shape baseline

| Projection family | Shape `(out, in)` | Median isolated latency |
|---|---:|---:|
| Attention R/K/V/O | 4096 × 4096 | 17.32 µs |
| FFN up | 16384 × 4096 | 155.97 µs |
| FFN down | 4096 × 16384 | 157.64 µs |
| low-rank 128/192/384 input or output | target shapes | 7.72–7.89 µs |

The exact values vary with GPU clock state; the row-split comparisons in
`row_split_probe.json` are measured in the same process as their baseline.

For the 61-layer target checkpoint, the static direct-linear path contains 852
block-internal linear calls. Summing these isolated families gives an
approximately 29 ms/token projection lower-bound before norms, mix, FP32
recurrent work, cache state movement, sampling, graph/runtime overhead, and
imperfect overlap. This is consistent with the observed C=1 service rate near
28.5 TPS.

Consequently, 50 TPS (20 ms/token) cannot be reached by reducing launch/event
metadata alone: even treating non-projection work as fixed at about 6 ms,
projection time must fall by roughly 52%.

## Output-row split result

- Two-way output-row splitting is bit-exact for FFN up/down and attention in
  this probe, but slower than the original single `F.linear`:
  - FFN up: 133.17 µs baseline vs 136.01 µs two-way parallel;
  - FFN down: 157.20 µs baseline vs 160.23 µs two-way parallel;
  - attention: 16.08 µs baseline vs 57.50 µs two-way parallel.
- Four-way FFN-up splitting is not bit-exact because the altered output shape
  selects a different BF16 GEMV reduction implementation.
- More partitions increase launch/event overhead and regress further.

## Decision

Do not integrate row splitting. It is a useful fail-closed result: simple
Albatross-style stream slicing cannot provide a strict-compatible speedup.
The remaining viable high-return path is a single-launch, shape-specialized
BF16 GEMV backend that reproduces the selected PyTorch output bit pattern for
each supported shape; its first target should be the two FFN families, which
dominate the isolated projection budget.
