# BF16 ordered sparse CMix down projection — rejected after service precision gate

- Date: 2026-08-21
- GPU: NVIDIA GeForce RTX 4090 D (sm89)
- Model: `/hikscale/models/XiaoKe/rwkv-0-hf`
- Candidate scope: Full Decode CUDA Graph, BF16, `M=1` only, tensor parallel size 1.

## What was tried

A native CUDA candidate based on Albatross CMix sparsity was implemented as an
**opt-in** `RWKV7_USE_SPARSE_CMIX_DOWN=1` path. It preserved increasing FFN
intermediate indices rather than using atomic/non-deterministic compaction:

1. count positive SqReLU elements in 256-wide tiles;
2. prefix-scan the tile counts;
3. compact active indices in increasing order;
4. run the BF16 value/down projection using FP32 accumulators and cast once to
   BF16.

The fixed-size index workspace makes it compatible with CUDA graph capture.
The candidate was intentionally restricted to one decode row: at batch sizes
above one, dense tensor-core GEMM is much faster.

## Isolated result

At synthetic 94% sparsity (matching the RWKV7 measured SqReLU envelope), the
operator can reduce an isolated `[1, 16384] @ [4096, 16384]^T` down projection
from about `0.160 ms` to `0.111 ms` (~30%). It was very close but not always
bit-identical to `F.linear`: over 20 seeds the maximum BF16 output difference
was `0.0625`, with at most two of 4096 output elements differing in a seed.

This retained FP32 accumulation; the remaining difference is the reduction
schedule versus PyTorch/CUTLASS BF16 GEMV, not FP16 accumulation or unordered
atomics.

## Service result and decision

C=1 API benchmark (32 generated tokens, greedy, ignore EOS):

| mode | TPS |
|---|---:|
| Stable baseline mean (prior three rounds) | 25.74 |
| Sparse candidate round 1 | 27.26 |
| Sparse candidate round 2 | 27.33 |
| Sparse candidate round 3 | 27.56 |
| Sparse candidate mean | **27.38** |

This is a promising `+6.40%` C=1 endpoint throughput signal, but it fails the
required 8 prompts × 16 greedy-token `logprobs=20` service gate:

- text mismatch: **2 / 8**;
- token-id mismatch: **2 / 8**;
- max selected-token logprob error: **2.630697**;
- max common Top-K logprob error: **11.437075**.

The candidate is therefore **not retained**, source changes are reverted, and
the stable strict-parity service is restored. This is not a rollback over a
minor numerical delta: RWKV7 applies 61 recurrent layers per generated token,
so the sparse projection's small per-layer BF16 reduction differences compound
into greedy token divergence.

## Follow-up

Do not re-enable this custom sparse GEMV as a default path. Any retry must
match the dense BF16 GEMV reduction contract (for example, a CUTLASS/Tensor
Core-compatible sparse formulation) before service integration. The currently
higher-confidence C=1 direction remains reducing the number of small
projections/fusing low-rank projection chains without changing their reduction
semantics.

Raw candidate response and benchmark summaries are in `raw/`.
