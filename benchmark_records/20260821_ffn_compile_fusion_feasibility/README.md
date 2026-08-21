# FFN producer-consumer `torch.compile` feasibility — rejected (2026-08-21)

## Goal

The remaining C=1 bottleneck is the repeated FFN path:

```text
BF16 F.linear([1,4096] -> [1,16384])
-> fused relu²
-> BF16 F.linear([1,16384] -> [1,4096])
```

This experiment checked the smallest strict compile candidate before modifying the
model: retain both GEMV operands and use a fullgraph Inductor function only to
capture/fuse the producer-consumer expression.

## Numerical gate

On CUDA BF16, with the same random input and weights:

- native expression `F.linear -> torch.relu().square() -> F.linear` was
  `torch.equal` to the current vLLM custom CUDA `relu2` path;
- the compiled result was also `torch.equal` to that current path;
- both maximum absolute differences were exactly `0.0`.

So this narrow expression preserves the activation output contract.

## Timing

Target: RTX 4090 D / SM89, 50 steady CUDA-event iterations, one row,
`4096 -> 16384 -> 4096`, BF16.

| path | mean ms | median ms | result |
|---|---:|---:|---|
| current eager `F.linear + relu2 + F.linear` | 0.32584 | 0.32563 | baseline |
| eager native expression | 0.32664 | 0.32608 | +0.25% slower |
| `torch.compile(fullgraph=True)` native expression | 0.35719 | 0.35635 | **+9.62% slower** |

## Decision

Reject; do not add a model flag. This verifies that fullgraph Inductor does not
turn the two cuBLAS BF16 GEMVs plus activation into an advantageous fused
execution on this workload. A source-level `torch.compile` wrapper would add
warmup/cache complexity while slowing the bottleneck.

The next FFN/LoRA fusion candidate must reduce a real launch or intermediate
memory dependency below the current custom `relu2` path while retaining the
individual BF16 dot-product/reduction contract; a generic compile wrapper is
not sufficient.
