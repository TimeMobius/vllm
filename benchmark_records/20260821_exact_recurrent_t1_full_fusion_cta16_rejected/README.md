# RWKV7 exact full-fusion CTA x=16 tuning — rejected

- Date: 2026-08-21
- GPU: NVIDIA GeForce RTX 4090 D (sm89)
- Runtime: vllm-sp; BF16 projections and FP32 recurrent state
- Model: `/hikscale/models/XiaoKe/rwkv-0-hf` (`xiaoke-5`)

## Candidate

The retained direct-cache full-fusion recurrence uses a `32 x 4` CTA. This
experiment changed **only** the full-fusion CTA to `16 x 4` and adjusted the
launch/grid mapping consistently. One CTA then owns exactly one `[batch, head,
64]` output vector (four values per x lane). The four-y-lane FP32 reduction
contract, pointwise round-to-nearest operations, direct persistent cache
mutation, and CUDA-graph padding policy are unchanged.

This was intentionally not the earlier invalid x=16 prototype: both the
kernel-internal `kThreadsX` and the launcher use 16, so every output position
is covered. `cuobjdump` confirms the expected static-resource change:

| Variant | Registers/thread | Shared memory/CTA |
| --- | ---: | ---: |
| Retained x=32 | 64 | 34,816 B |
| Candidate x=16 | 64 | 17,408 B |

## Accuracy gate — passed

- Exact direct-cache operator, B=1/8/128, three single-step seeds, padded
  slot: valid output and complete persistent cache are `torch.equal`.
- Ten recurrent steps at B=1/8/128: output and complete cache remain
  `torch.equal`.
- Full focused RWKV7 suite: **82 passed, 4 skipped**.
- Service gate, 8 prompts × 16 greedy decode tokens with `logprobs=20`, Full
  Decode CUDA Graph: text/token IDs/selected logprobs/common Top-K logprobs are
  byte-for-byte identical to the retained x=32 full-fusion trace.

## Same-session paired performance gate — rejected

The pair used identical clean-started service arguments, request payloads, and
warmup policy. C=128 is 128 closed-loop completion requests × 32 generated
tokens; C=1 uses four post-warmup 32-token requests.

| Metric | Retained x=32 | Candidate x=16 | Change |
| --- | ---: | ---: | ---: |
| C=128 aggregate output TPS | 1537.02 | 1544.47 | +0.48% |
| C=1 output TPS | 27.37 | 27.38 | +0.06% |

The candidate is strictly correct but has no material serving benefit; reduced
shared memory does not move the dominant BF16 projection/GEMV bottleneck. It is
therefore **not retained**. The stable source and service are restored to the
x=32 full-fusion kernel.

Raw service traces and JSON summaries are in `raw/`.
