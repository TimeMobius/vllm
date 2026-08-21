# Persistent-output native BF16 LoRA-input GEMV — rejected

## Why this follow-up existed

The previous low-rank custom GEMV was isolated-kernel fast and service exact,
but regressed end-to-end. This candidate removed its only obvious framework
overhead: each `RWKV7LoRA` module owned a CUDA-Graph-stable `[1, rank]` BF16
scratch buffer, and an out-variant custom op wrote directly into that buffer.
No output allocation occurred during replay. The next activation and original
second `F.linear` consumed that scratch immediately.

## Correctness

- Target-shape CUDA out-op unit: `3 passed` (`rank=128,192,384`, eight BF16
  inputs per shape, `torch.equal`).
- Full Decode CUDA Graph capture/replay completed for
  `1,2,4,8,16,32,64,128`.
- Service `8 prompts × 16 greedy, logprobs=20`: byte-identical to
  `/tmp/rwkv7_service_trace_before_aux.json`.

## Restart-paired benchmark

| Workload | Stable baseline | Persistent-output candidate | Change |
|---|---:|---:|---:|
| C=1 output TPS | 28.4484 | 28.0959 | -1.24% |
| C=128 aggregate output TPS | 1567.2270 | 1562.4224 | -0.31% |

## Decision

Rejected and restored. Preallocating the output does not remove the C=1
regression, so the problem is not allocator overhead: for this graph and
stream schedule, the independent custom low-rank GEMV launch itself is slower
in context than vLLM/PyTorch's selected path despite its isolated timing.

This closes the standalone LoRA-input replacement avenue. The next strict
GEMV experiment must reduce the number of launches by fusing a complete
producer-consumer unit (for example input GEMV + activation + output GEMV)
while explicitly reproducing the original BF16 reduction and activation
rounding contracts.
