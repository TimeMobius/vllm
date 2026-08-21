# BF16 dense GEMV reduction-tree reconstruction — rejected (2026-08-21)

## Why this was investigated

The strict FFN sparse path is blocked by the exact dense `F.linear` reduction
contract, not merely by using BF16 operands. A prior temporary native BF16 GEMV
object was known to match the target FFN shapes exactly, but it is not source
available and must never become a runtime/build dependency. This investigation
attempted to reconstruct its arithmetic contract from first principles before
trying any further Albatross-style zero skipping.

## Source-backed matrix

A temporary CUDA extension tested sixteen deterministic variants for M=1 BF16
GEMV:

- 64/128/256 threads per output row;
- scalar versus 2/4/8 independent FP32 accumulation chains;
- explicit `__fmaf_rn` and `__fadd_rn` arithmetic;
- warp reduction plus serial or warp-tree cross-warp reduction;
- optional volatile accumulator materialization.

Each candidate was compared with `torch.nn.functional.linear` using four
independent seeds for FFN-down `16384→4096` normal input, FFN-down SqReLU
sparse input, and FFN-up `4096→16384` input.

| Case | Lowest total differing BF16 elements (4 seeds) | Best variants |
| --- | ---: | --- |
| FFN down, normal | 2 | [4, 5, 10, 11, 14] |
| FFN down, SqReLU sparse | 2 | [9, 10] |
| FFN up, normal | 3 | [14] |

No reconstructed variant reached `torch.equal`; maximum individual BF16 output
delta was `0.25` for FFN down and `0.03125` for FFN up. The test source and raw
matrix are included in this record. No repository runtime source was changed.

## Diagnostic control

The temporary historical object was separately checked against the same twelve
synthetic cases (three shapes × four seeds) and returned `torch.equal` in all
cases. That proves the desired contract is achievable, but **does not license
use of the object**: its source is absent, and this repository must remain
source-buildable and self-contained.

## Decision

Reject this reconstruction family before model/service integration. Albatross'
FP16 atomic tiled sparse kernel necessarily changes both precision and reduction
order; the naïve BF16 scalar/tree variants also miss the PyTorch contract by a
few outputs. A future strict sparse implementation must start from a
source-backed GEMV schedule that is already exact for this PyTorch build, then
skip zeros without altering that schedule. Do not use prebuilt object fragments
in production.
