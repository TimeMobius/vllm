# Rejected native-BF16 token-shift + mix6 fusion

This candidate fused `token_shift_with_cache` and the six RWKV7 mix6
`addcmul` operations into one Triton kernel. Unlike the earlier FP32 fusion,
it used native BF16 arithmetic and passed exact operator-level checks for
`T=1/8/257/1024`, with no cache, `[hidden]` cache, and decode-batch
`[tokens, hidden]` cache layouts (`max_abs_error=0.0`).

The candidate was rejected by the service throughput gate on the fixed
8-request/32-output-token benchmark:

- Retained cache-on path: **137.972 TPS**
- Candidate: **132.485 TPS** (5 rounds; **-3.61%**)
- Additional 8-round candidate run: **132.217 TPS**
- Candidate serial token IDs matched the retained cache-on reference in three
  repeated runs (8 prompts x 32 tokens), so the rejection is performance-only.

The isolated operator benchmark was faster, but that benefit did not survive
model/service execution, likely because the additional fused kernel launch and
register/resource behavior interact poorly with the surrounding RWKV7 decode
pipeline. The candidate source was reverted; raw measurements remain here for
future shape-specific or CUDA implementation work.
