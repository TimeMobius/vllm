# Full Decode model compile — C=1 rejected for no stable gain

- Date: 2026-08-21
- GPU: NVIDIA GeForce RTX 4090 D (sm89)
- Model: `/hikscale/models/XiaoKe/rwkv-0-hf`
- Candidate: `RWKV7_COMPILE_WITH_FULL_CUDAGRAPH=1`
- Common server settings: BF16 projection, FP32 recurrent state, `FULL_DECODE_ONLY`
  graph capture sizes `[1,2,4,8,16,32,64,128]`.

This is the existing model-level `torch.compile` opt-in. It retains the native
final LayerNorm custom-op boundary to preserve RWKV recursive decode numerics.
It had already been neutral at C=128; this record measures the separate C=1
case, where reducing eager launch overhead was still plausible.

## Precision gate

The 8 prompts × 16 generated tokens greedy `logprobs=20` comparison passed
exactly:

```text
text mismatch: 0
token mismatch: 0
max selected-token logprob error: 0
max common Top-K logprob error: 0
```

## C=1 API benchmark

Each round generates 32 tokens with `temperature=0` and `ignore_eos=true`.
The baseline was restarted after the candidate so both modes used an otherwise
identical clean service process.

| mode | round 1 | round 2 | round 3 | rounds 2–3 mean |
|---|---:|---:|---:|---:|
| Full-model compile opt-in | 25.895 | 25.809 | 25.895 | **25.852** |
| Strict stable path | 25.281 | 25.959 | 25.963 | **25.961** |

The compile candidate is `-0.42%` after warmup. This is inside normal service
noise and in the unfavorable direction, so it is not enabled by default.

The raw accuracy and benchmark summaries are in `raw/`. The normal strict
Full Decode CUDA Graph service was restored after this measurement.
