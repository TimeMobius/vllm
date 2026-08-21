# Rejected cache-all prefill-only shift + mix6 fusion

This experiment retried the exact native-BF16 shift/mix6 kernel only in
`RWKV7Attention.forward_prefill_cache_all`, deliberately excluding decode and
varlen batch prefill paths. The operator passed exact BF16 tests for T=1, 13,
and 257 with and without a `[hidden]` cache state; a fixed 4K service prompt
also produced identical 16-token output IDs.

Although isolated execution was 24% to 190% faster across the tested shapes,
that gain did not yield a stable service improvement under cold unique raw
prompts (B=1, 16 decode tokens):

| Prompt tokens | Retained prompt tok/s | Candidate prompt tok/s | Change |
| ---: | ---: | ---: | ---: |
| 4,096 | 2239.92 | 2249.93 | +0.45% |
| 16,384 | 3223.94 | 3215.62 | -0.26% |
| 65,536 | 3575.54 | 3568.54 | -0.20% |

The candidate was reverted because its service-level effect is within noise
and non-positive at longer contexts. The measurements preserve the result for
future profiling and a possible broader prefill fusion boundary.
