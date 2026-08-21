# RWKV tokenizer invalid UTF-8 preflight

- Date: 2026-08-21
- Runtime: vllm-sp, Full Decode CUDA Graph
- Model: `/hikscale/models/XiaoKe/rwkv-0-hf` (`xiaoke-5`)

## Problem

The optional `pyrwkv_tokenizer` decoder performs a UTF-8 conversion inside a
PyO3 extension. RWKV's base vocabulary includes raw byte fragments: the real
checkpoint vocabulary has 486 token IDs that are individually not valid UTF-8.
A generated prefix containing an incomplete/invalid byte sequence made the
extension panic, print Rust panic output, emit a vLLM warning, then fall back to
the Python byte decoder. The previous C=1→128 dashboard run emitted **778**
`RWKV fast tokenizer decode panicked` warnings.

This is outside the CUDA graph and cannot change logits, but it adds serial
output-handler work and massive log contention under concurrent requests.

## Change

At tokenizer construction, record only the base-vocabulary IDs containing UTF-8
byte fragments. Before invoking the optional fast decoder, reconstruct bytes
only when one of those exceptional IDs occurs. If the complete base-token byte
stream is invalid UTF-8, skip the known-panicking PyO3 call and take the
existing Python fallback directly. Added/special/unknown IDs keep their previous
routing. A byte-fragment sequence that becomes valid after concatenation still
uses the fast decoder.

## Validation

- Tokenizer unit suite: **18 passed**. Tests cover invalid-byte bypass, valid
  byte-fragment concatenation, normal fast decode, and added-token semantics.
- Isolated real-tokenizer probe (100 invalid-byte decodes): **101 Rust panic
  lines → 0**, same text `�`.
- Service accuracy: 8 prompts × 16 greedy tokens with `logprobs=20` is
  byte-identical to the retained strict full-fusion trace: text/token IDs/
  selected logprobs/common Top-K logprobs are all zero-difference.
- Candidate dashboard POST completed successfully at 2026-08-21 15:02:23 with
  all 1→128 points at 100% success and **zero** RWKV fast-tokenizer panic
  warnings. Its sampled response lengths differ from the preceding dashboard
  run, so it is recorded as an end-to-end no-panic/stability confirmation, not
  as a paired GPU-TPS result.

## Performance interpretation

The change deliberately does not touch model math, CUDA kernels, or the
Full Decode graph. Fixed local C=1/C=128 completion controls remain within
normal restart noise. Its retained value is removing deterministic exception
and logging overhead from invalid byte streams; this improves reliability and
prevents output-handler work from contaminating high-concurrency serving.

Raw traces and extracted dashboard metrics are in `raw/`.
