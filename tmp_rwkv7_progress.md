## RWKV7 vLLM adaptation progress

Date: 2026-03-30
Branch: `codex/rwkv7-adapter-align`

### Confirmed fixes

- `vllm/model_executor/models/rwkv7.py`
  - `get_mamba_state_copy_func()` now returns `(conv, temporal, conv)`
  - `compute_logits()` aligns hidden states to `lm_head.weight.dtype`
  - `load_weights()` maps `model.embeddings.weight -> model.embed_tokens.weight`
  - `RWKV7Block` is registered into `compilation_config.static_forward_context`

### Confirmed passing coverage

- `tests/model_executor/test_rwkv7.py`: `6 passed`
- `RWKV7-Goose-World2.8-0.1B-HF`
  - `/v1/completions`
  - one-shot `max_tokens=8/16`
  - step-by-step `max_tokens=1` loop
  - prompts:
    - `i am`
    - `北京是`
    - `The capital of France is`
    - `Once upon a time`
  - result: one-shot matches step-by-step for all tested prompts
- concurrency smoke test
  - `0.1B`, `max_tokens=16`, 3 rounds: matched serial baseline
  - `0.4B`, `max_tokens=8`, 3 rounds: matched serial baseline
  - `0.4B`, `max_tokens=16`, 3 rounds: matched serial baseline
  - note: for `0.4B`, the concurrent baseline itself is still the same one-shot path

### Confirmed remaining issue

- `RWKV7-Goose-World2.9-0.4B-HF`
- prompt: `北京是`
- `max_tokens=8`: pass
- `max_tokens>=9`: fail
- first mismatch is stable at generated token index `8` (the 9th generated token)
- disabling async scheduling does not change the failure

### Strong evidence already collected

- core RWKV7 math is aligned with the reference implementation
- direct model-module prefill + decode is aligned
- direct `RWKV7ForCausalLM` incremental generation is aligned on the `0.4B` failing prompt
- the remaining bug is in the v1 engine/service path for single-request multi-token decode

### Current debugging hypothesis

- the failure is likely at an engine-step handoff boundary instead of in the model math
- the `max_tokens=8` vs `max_tokens=9+` shape suggests the first decode segment is correct and the next segment resumes from an incorrect running state

### Local uncommitted files at this checkpoint

- `tmp_rwkv7_compare.py`
- `tmp_rwkv7_concurrency_check.py`
- `tmp_rwkv7_progress.md`

### Local unrelated debug change intentionally left untouched

- `vllm/v1/core/kv_cache_coordinator.py`
