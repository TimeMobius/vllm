# Shared align state-index CUDA Graph candidate — rejected

## Hypothesis

In `mamba-cache-mode=align` with `FULL_DECODE_ONLY` CUDA Graphs, each of the
61 RWKV blocks recomputes the same replay-time state slot from persistent
`seq_lens` and `block_table_tensor` using `clamp + gather`. The candidate
replaced those repeated graph nodes with one exact in-place CUDA metadata
kernel on layer zero, then reused the captured state-index scratch for all
later layers.

The candidate deliberately left all BF16 projections, FP32 recurrent state,
recurrent update/reduction order, cache layout, and padding-store guards
unchanged. Its CUDA unit test covered int32/int64 indices, zero-length padded
rows, block boundaries, and strided outputs against the original
`clamp/gather` reference.

## Correctness

- CUDA custom-op unit: `4 passed`.
- Full Decode graph startup/capture completed for sizes `1,2,4,8,16,32,64,128`.
- Service trace: `8 prompts x 16 greedy, logprobs=20` is byte-identical to
  `/tmp/rwkv7_service_trace_before_aux.json`.

## Paired HTTP benchmark

Same host, restart-paired stable service, 32 output tokens, BF16 model,
FP32 recurrent cache, `mamba-cache-mode=align`, Full Decode CUDA Graph.

| Workload | Stable baseline | Candidate | Change |
|---|---:|---:|---:|
| C=1 output TPS | 28.4512 | 28.6072 | +0.55% |
| C=128 aggregate output TPS | 1565.3460 | 1573.9400 | +0.55% |

Raw JSON is retained alongside this record. The external dashboard was then
run on the restored stable service and completed successfully at
`2026-08-21 17:36:48` with 8 saved records.

## Decision

Rejected and fully restored. The exact candidate does remove graph metadata
work, but the measured gain is below the 1% materiality gate and cannot move
the C=1 target meaningfully. The dominant C=1 cost remains BF16 GEMV/
projection work, not state-slot metadata.
