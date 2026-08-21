# FFN sparse BF16 reduction feasibility — rejected (2026-08-21)

## Why this was tested

The previously rejected sparse CMix down-projection showed a real C=1 gain but
changed the BF16 GEMV reduction result. This probe pursued the requested
"reduce the error before reverting" direction: recover the historical strict
native BF16 GEMV implementation, then test whether skip-zero FFN-down work can
retain its reduction contract.

## Recovered strict dense operator

A temporary fragment linked the historical `rwkv7_bf16_gemv_out` object in an
isolated namespace and tested it against PyTorch BF16 `F.linear`.

| shape | exact output | PyTorch µs | native µs | result |
|---|---|---:|---:|---|
| FFN up, `4096 -> 16384` | yes | 165.30 | 167.71 | -1.44% |
| FFN down, `16384 -> 4096` | yes | 166.78 | 173.12 | -3.66% |
| LoRA input, `4096 -> 192` | yes | 15.73 | 10.01 | +57.2% isolated only |

This confirms the strict output contract is feasible for FFN shapes, but that
one-row/one-output-block GEMV is not faster than PyTorch/cuBLAS for dense FFN.

## Sparse skip-zero down-projection prototype

A separate temporary CUDA kernel preserved BF16 inputs/weights and FP32 FMA,
checking `SqReLU` zeros before loading a weight. It was compared directly with
PyTorch `F.linear(4096, 16384)` for controlled input sparsity.

| zero fraction | exact? | differing elements | PyTorch µs | prototype µs |
|---:|---|---:|---:|---:|
| 0% | no | 3 | 159.77 | 207.47 |
| ~50% | yes | 0 | 160.37 | 213.07 |
| ~90% | yes | 0 | 160.11 | 235.51 |
| ~94% | no | 2 | 159.74 | 218.65 |
| ~98% | yes | 0 | 160.53 | 160.09 |

## Decision

Reject. The hand-written sparse kernel has two independent blockers:

1. its reduction tree does not reliably reproduce the PyTorch BF16 GEMV result
   at all sparsity patterns; and
2. its one-output-row block mapping is slower through the realistic ~94%
   sparsity regime despite skipping weights.

The experiment supports the existing conclusion: Albatross-style CMix sparsity
is a valuable performance reference, but a strict implementation must reproduce
cuBLAS/PyTorch's reduction tree **and** use a much better output tiling. No
source or runtime dependency was retained.
