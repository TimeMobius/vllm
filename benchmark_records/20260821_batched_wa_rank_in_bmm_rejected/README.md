# Strict BF16 W/A rank-in batched BMM — rejected (2026-08-21)

## Hypothesis

The target XiaoKe checkpoint uses equal W/A LoRA rank-in shapes
`[192, 4096]`. Two independent M=1 BF16 GEMVs are launch-bound. Packing those
existing parameter tensors into adjacent `[2, 192, 4096]` storage after model
loading (without retaining a second permanent copy) permits one
`torch.bmm([2, 1, 4096], [2, 4096, 192])` for the rank-in stage.

## Accuracy gates passed

- Actual checkpoint layers 0, 1, and 10, four BF16 input seeds each: the
  batched rank-in result was `torch.equal` to the two original
  `F.linear` calls.
- Focused CUDA regression tests passed:
  `test_rwkv7_batched_wa_rank_in_decode_preserves_bits` and
  `test_rwkv7_multistream_rkv_projection_preserves_bits`.
- The candidate vLLM service captured all Full Decode CUDA Graph sizes
  `1,2,4,8,16,32,64,128` successfully.
- Eight prompts × 16 greedy decode tokens with `logprobs=20` were byte exact:
  no text/token mismatch, no top-1 difference, and max common logprob error
  `0.0` across 2,560 compared values.

## Why it was not retained

The isolated pair itself improved from `18.207 µs` to `8.074 µs` on actual
layer-0 weights. However, production already overlaps the W/A chain with the
larger K projection on a child stream. Its local saving is mostly hidden by
that critical path:

| Workload | Strict baseline | Candidate | Change |
| --- | ---: | ---: | ---: |
| C=1 HTTP output TPS | 28.9083 | 28.9973 | +0.31% |
| C=2 aggregate TPS | 52.1629 | 52.8156 | +1.25% |
| C=8 aggregate TPS | 203.4767 | 203.6209 | +0.07% |
| C=32 aggregate TPS | 632.4850 | 632.3580 | -0.02% |

The deployment gate is at least 1% repeatable gain on the primary C=1 target
and no high-concurrency regression. The C=1 result is below that threshold;
C=8/C=32 have no material improvement. The candidate source and startup flag
were therefore reverted. The experiment is retained only as evidence that
small LoRA rank-in launches are not the practical C=1 bottleneck while the
three-stream projection schedule is enabled.
