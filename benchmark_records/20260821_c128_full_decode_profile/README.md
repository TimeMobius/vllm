# RWKV7 C=128 Full Decode CUDA Graph profile

- Date: 2026-08-21
- GPU: NVIDIA GeForce RTX 4090 D (sm89)
- Model: `/hikscale/models/XiaoKe/rwkv-0-hf`
- Server mode: BF16 projection, FP32 RWKV state, `FULL_DECODE_ONLY`, capture
  sizes `1,2,4,8,16,32,64,128`.

## Reproducible idle C=128 baseline

The benchmark uses 128 closed-loop completion requests, 32 generated tokens
per request, greedy sampling, and `ignore_eos=true`. It must be run when the
external dashboard is idle: running another 1→128 concurrency sweep against
the same server materially contaminates the client-side result.

| round | aggregate API TPS |
|---:|---:|
| 1 | 704.096 |
| 2 | 708.556 |
| 3 | 706.666 |
| mean | **706.440** |
| rounds 2–3 mean | **707.611** |

The raw summaries are in `raw/`.

## Profile setup

A separate server was launched with PyTorch profiler configured for five wait
steps, two warmup steps, and ten active steps. A C=128 request batch was first
run unprofiled, then the profile was started and another C=128 batch was run.
The 72 MB Chrome trace remains temporary at
`/tmp/rwkv7_c128_torch_profile_20260821/`; `raw/profiler_out_0.txt` is the
committed profiler aggregate table.

The ten active `generation_128` CUDA Graph replays totalled **1.7395 s**:
**173.95 ms per graph** or roughly **735.8 raw aggregate tok/s**. The lower
end-to-end API TPS is expected because it includes scheduling, prompt work,
HTTP/client dispatch, and batch ramp-up.

## CUDA time per 128-token decode graph

| category | ms/graph | share of graph CUDA time |
|---|---:|---:|
| BF16 CUTLASS projection kernels | 45.551 | 23.94% |
| ATen elementwise kernels | 41.526 | 21.83% |
| ATen reduction kernels | 27.658 | 14.54% |
| FP32 recurrent-state native gather | 20.375 | 10.70% |
| FP32 recurrent-state native masked store | 18.223 | 9.58% |
| exact recurrent state update CUDA kernel | 18.719 | 9.84% |

The recurrent cache shape is `[B, 64, 64, 64]` FP32. The trace contains 61
large gathers and 61 large stores per decode graph; each transfers 128 MiB for
`B=128`. The two small 4096-element shift-state copies are not a material
fraction. Therefore state-cache tuning is still meaningful for C=128, but the
large state read/write is bandwidth-bound and cannot be removed safely by a
copy-only micro-optimization alone.

A 16-byte `uint4` bit-copy prototype was tested on the exact 128 MiB recurrent
state workload. It was bit-exact but did not improve the scalar kernel
reliably (`gather 326.10 us vs 321.18 us`; `store 327.97 us vs 323.55 us` in a
matched isolated measurement), so it was discarded before model integration.

## Consequence for next candidates

The largest remaining C=128 costs are dense projection/reduction and the
recurrent state data path. The next candidate must either:

1. reduce projection launch/dispatch without changing BF16 GEMV reduction
   semantics; or
2. remove a recurrent-cache materialization only while preserving the exact
   ATen reduction order used by RWKV7's recursive decode.

The prior full alternate recurrent kernel and ordered sparse CMix candidates do
not meet this contract and remain rejected.
