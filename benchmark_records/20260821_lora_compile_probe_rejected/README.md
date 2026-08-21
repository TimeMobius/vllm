# RWKV7 LoRA torch.compile probe

This isolates the attention LoRA direct chain:

```text
BF16 [B,4096] @ [192,4096]^T -> tanh -> [B,192] @ [4096,192]^T
```

`torch.compile(fullgraph=True)` preserved `torch.equal` output for B=1 and
B=128 but did not fuse the two GEMMs into a beneficial kernel.

| batch | eager | compiled | result |
|---:|---:|---:|---|
| 1 | 30.37 us | 67.80 us | -55.2% |
| 128 | 34.38 us | 83.30 us | -58.7% |

Decision: reject. This is distinct from, and reinforces, the prior generic FFN
compile result: Inductor preserves numerical output here but adds graph/runtime
cost around the cuBLAS/CUTLASS work. Do not add compile wrappers to individual
RWKV7 LoRA chains.
