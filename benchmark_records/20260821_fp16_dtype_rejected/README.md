# Rejected: RWKV7 FP16 runtime dtype (Full Decode CUDA Graph)

- Date: 2026-08-21
- GPU: NVIDIA GeForce RTX 4090 D (sm89)
- Model: `/hikscale/models/XiaoKe/rwkv-0-hf`
- Candidate launch: `--dtype float16`, unquantized model weights, FP16 projections and existing exact FP32 recurrent-state kernels; Full Decode CUDA Graph and `mamba_cache_mode=align`.

## Purpose

Albatross publishes FP16 standalone results, so this test isolates whether simply changing vLLM's projection dtype from the stable BF16 deployment to FP16 can close the C=1 gap without invasive kernel work.

## Accuracy gate — failed

Against the stable BF16 direct-cache trace (8 greedy prompts × 16 tokens, `logprobs=20`):

- text mismatch: **2 / 8**;
- token-ID mismatch: **2 / 8**;
- maximum selected-logprob absolute error: **2.7019357681274414**;
- maximum common Top-K logprob absolute error: **10.360044971108437**.

Even though this diverges less often than online FP8, it is still a multi-token quality change and fails the deployment gate.

## Performance gate

| Workload | Stable BF16 direct-cache | FP16 candidate | Change |
| --- | ---: | ---: | ---: |
| C=1, 32 output tokens (mean rounds 2–3) | 26.21 TPS | 26.17 TPS | -0.16% |
| C=128, 128×32 outputs (mean rounds 2–3) | 1205.23 TPS | 1198.66 TPS | -0.55% |

FP16 is not faster on this RTX 4090 D/vLLM stack, so it cannot explain Albatross's standalone results and should not be used as an optimization direction.

## Decision

The strict BF16 stable service was restored on port 8030. Do not change the stable launch command to FP16. Raw request summaries and the candidate trace are retained under `raw/`.
