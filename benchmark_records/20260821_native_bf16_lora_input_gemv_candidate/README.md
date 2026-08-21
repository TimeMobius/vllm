# Native BF16 LoRA-input GEMV candidate — exact but rejected for end-to-end speed

## Candidate

A CUDA warp-per-output-row BF16 M=1 GEMV prototype was built for the RWKV7
bias-free LoRA input projections only:

```text
[1, 4096] × [128|192|384, 4096]^T → [1, 128|192|384]
```

It preserves BF16 input/weight and FP32 FMA accumulation, then rounds output
to BF16. The model dispatch was deliberately restricted to CUDA, TP=1,
contiguous BF16 weights, one decode token, no bias, and the three validated
checkpoint shape families. All other linear layers kept `F.linear`.

## Isolated result

The single custom kernel was faster in isolation:

| Family | F.linear | candidate | speedup |
|---|---:|---:|---:|
| 4096 × 4096 | 17.39 µs | 11.92 µs | 1.46× |
| 192/384/128 LoRA input | ~8.02 µs | ~5.33 µs | ~1.50× |

It was slower for FFN up/down, so those were never considered for model
integration. Full isolated data is in `isolated_shape_probe.json`.

## Accuracy gates

- CUDA unit, 8 generated BF16 inputs for each 128/192/384 target shape:
  `3 passed` with `torch.equal`.
- Actual checkpoint input projections: all 243 RWKV LoRA-input weights
  (122×192, 61×384, 60×128), each against 8 independent BF16 inputs:
  **0 differing output elements**. See
  `actual_weight_input_projection_check.json`.
- Full Decode graph captured/replayed normally at
  `1,2,4,8,16,32,64,128`.
- Service `8 prompts × 16 greedy, logprobs=20`: byte-identical to
  `/tmp/rwkv7_service_trace_before_aux.json`.

## Restart-paired HTTP benchmark

| Workload | Stable baseline | Candidate | Change |
|---|---:|---:|---:|
| C=1 output TPS | 28.4445 | 28.1004 | -1.21% |
| C=128 aggregate output TPS | 1565.1864 | 1566.7799 | +0.10% |

The C=128 path is intentionally unchanged because the kernel supports M=1
only; its tiny difference is noise. Despite its faster isolated latency and
exact service output, the extra custom-op/output-buffer integration cost in
the full graph more than erased the low-rank launch saving.

## Decision

Rejected and fully restored. This is a useful result rather than an accuracy
rollback: a strict-compatible CUDA BF16 GEMV is feasible and can be bit-exact
for the validated LoRA-input families, but it must reduce end-to-end graph
node/buffer overhead to be viable. Future GEMV work should use persistent
output buffers or fuse the following activation/second low-rank projection;
a stand-alone allocating replacement is not sufficient.
