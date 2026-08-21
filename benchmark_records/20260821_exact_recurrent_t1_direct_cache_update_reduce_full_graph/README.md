# RWKV7 exact direct-cache update/output-reduction fusion

- Date: 2026-08-21
- GPU: NVIDIA GeForce RTX 4090 D (sm89)
- Runtime: vllm-sp, BF16 projections and FP32 RWKV recurrent state
- Model: `/hikscale/models/XiaoKe/rwkv-0-hf` (`xiaoke-5`)

## Change

The retained direct persistent recurrent-cache path previously launched two exact
CUDA kernels after the exact `sa` reduction: one pointwise cache-state update and
one output reduction over the updated state. This change fuses those independent
operations into one kernel.

The candidate intentionally preserves the prior numerical contract:

- the sensitive `sa = state * -kk` reduction remains a separate exact kernel;
- the output reduction keeps the existing `32 x 4` CTA, four partial FP32
  accumulators, and `4 -> 2 -> 1` shared-memory addition tree;
- state pointwise operations retain explicit round-to-nearest FP32 multiply/add
  ordering;
- padded Full Decode CUDA Graph lanes read cache slot zero for discarded output
  and never mutate cache state.

The optimization is limited to the CUDA/FP32/contiguous/direct-cache fast path.
Unsupported layouts, dtype/device mismatches, `cache_all`, eager execution, and
prefill retain existing fallbacks. This is distinct from the older alternate
recurrent kernel experiment, which was faster but failed multi-step service
accuracy and remains disabled.

## Accuracy gate

The candidate was compared with the strict direct-cache reference under Full
Decode CUDA Graph using 8 prompts, greedy generation, 16 decode steps, and
`logprobs=20`:

| Check | Result |
| --- | ---: |
| Text mismatch | 0 |
| Token-ID mismatch | 0 |
| Selected logprob max absolute error | 0.0 |
| Common Top-K logprob max absolute error | 0.0 |

The focused RWKV7 test suite also passed: **78 passed, 2 skipped**. The direct
cache exact operator test passed **3/3** configurations (B=1/8/128, three seeds).

## Performance gate

Both sides used clean service restarts with `mamba_cache_mode=align` and
`FULL_DECODE_ONLY` graphs captured at `1,2,4,8,16,32,64,128`. C=128 uses 128
closed-loop requests, 32 output tokens/request, 4096 aggregate output tokens per
measurement; warmups are excluded.

| Concurrency | Baseline TPS | Candidate TPS | Change |
| ---: | ---: | ---: | ---: |
| 128 aggregate | 1252.19 | 1365.23 | **+9.03%** |
| 1 | 26.08 | 26.22 | +0.54% |

The primary gain is high-concurrency throughput: one graph node and one
recurrent-state read/write synchronization point are removed per layer. C=1 is
still dominated by BF16 projection GEMV kernels, so its small increase is treated
as secondary rather than as evidence that the single-stream bottleneck is solved.

`raw/` stores the accuracy traces and immutable service benchmark summaries.
