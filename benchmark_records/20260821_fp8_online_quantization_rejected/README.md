# Rejected: RWKV7 online FP8 quantization (Full Decode CUDA Graph)

- Date: 2026-08-21
- GPU: NVIDIA GeForce RTX 4090 D (sm89)
- Model: `/hikscale/models/XiaoKe/rwkv-0-hf`
- Candidate launch: `--quantization fp8 --dtype bfloat16`, online FP8 weight quantization and dynamic FP8 activations; Full Decode CUDA Graph sizes `1,2,4,8,16,32,64,128`; `mamba_cache_mode=align`.

## Why evaluated

The remaining C=1 bottleneck is BF16 small projection/GEMV. FP8 can reduce projection weight traffic without changing the FP32 RWKV recurrent-state contract, so it is a useful high-upside deployment experiment.

## Accuracy gate — failed

Against the strict direct-cache service trace (8 greedy prompts × 16 tokens, `logprobs=20`):

- text mismatch: **5 / 8**;
- token-ID mismatch: **5 / 8**;
- maximum selected-logprob absolute error: **3.2099708358582575**;
- maximum common Top-K logprob absolute error: **13.641592100262642**.

This is a model-quality change, not a harmless last-bit difference. The candidate is therefore **not** used by the stable service and is not a default optimization.

## Performance gate

| Workload | Strict direct-cache | FP8 candidate | Change |
| --- | ---: | ---: | ---: |
| C=1, 32 output tokens (mean of rounds 2–3) | 26.21 TPS (short reference) | 30.89 TPS | +17.9% |
| C=1, 512 output tokens (strict long-run confirmation) | 27.18 TPS | — | — |
| C=128, 128×32 outputs (mean of rounds 2–3) | 1205.23 TPS | 1218.58 TPS | +1.11% |

The C=1 speed potential is real but remains well below the requested 50 TPS and does not satisfy the precision gate. The high-concurrency gain is within a small range and not enough to justify the quality regression.

## Decision

- Stable server restored to BF16 projections + exact FP32 recurrent cache update on port 8030.
- Do not enable online FP8 in the production/stable command.
- FP8 may only be reconsidered with a calibration/serialized checkpoint that passes the same service-quality gate; it is not an answer to strict numerical optimization.

Raw request summaries and candidate service trace are stored under `raw/`.
