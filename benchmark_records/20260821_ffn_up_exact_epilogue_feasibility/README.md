# Strict BF16 FFN-up + ReLU² epilogue feasibility — rejected (2026-08-21)

## Question

Can the historical strict native BF16 GEMV for RWKV7 FFN-up
`[1, 4096] × [16384, 4096]^T` become worthwhile by fusing its following
ReLU² epilogue, while preserving the existing `F.linear` and `relu2` outputs
bit-for-bit?

## Controlled isolated gate

The exact native out-op was loaded only from the historical temporary fragment;
it was **not** connected to the model or the production service. For four
independent BF16 seeds it produced exactly the same GEMV output as
`torch.nn.functional.linear`, and applying the existing native `_C.relu2`
op to either output was also `torch.equal`.

The timing comparison used steady CUDA events on the RTX 4090 D:

| Path | Mean latency |
| --- | ---: |
| `F.linear → relu2` | 157.878 µs |
| exact native GEMV → `relu2` | 166.487 µs |
| exact native GEMV alone | 164.268 µs |

The final row gives a hard optimistic bound for a perfect zero-cost fused
ReLU² epilogue: it would still be **4.05% slower** than the current
`F.linear → relu2` path. Thus the saved ~3.62 µs activation kernel cannot
recover the exact GEMV backend's deficit.

## Decision

**Reject before model integration.** This retains strict numerical quality but
avoids a service restart, CUDA Graph recapture, and regression surface for a
candidate already losing in the narrowest favorable benchmark. The next FFN
proposal must improve the dense BF16 GEMV itself (for example an exact
Tensor-Core/layout formulation), not merely attach an epilogue to the current
row-oriented native GEMV.
