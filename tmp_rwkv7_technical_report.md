# RWKV7 on vLLM: Technical Report

Date: 2026-03-31

Branch: `codex/rwkv7-adapter-align`

## 1. Objective

The goal of this work was to adapt RWKV7 to the vLLM v1 engine so that:

- RWKV7 checkpoints can be loaded and served by vLLM.
- The recurrent internal state of RWKV7 is integrated with vLLM's state/cache system.
- `/v1/completions` produces the same deterministic output as the reference implementation.
- Concurrent decoding benefits from vLLM batching instead of degenerating into per-request serial execution.

This report summarizes the full adaptation path, the code changes, the validation methodology, the final results, and the remaining optimization opportunities.

## 2. Adaptation Architecture

Adapting RWKV7 to vLLM requires solving four layers of integration.

### 2.1 Model Definition Layer

RWKV7 is not a Transformer KV-cache model. Each block maintains recurrent internal state:

- attention shift state
- recurrent state
- FFN shift state
- value residual path (`v_first`)

Therefore the vLLM model implementation must explicitly expose:

- state shapes
- state dtypes
- state copy semantics
- per-slot state load/store behavior
- block-local recurrent update behavior

### 2.2 Config and Registration Layer

vLLM must be able to:

- parse RWKV7 configs from checkpoints
- map the architecture name to a model implementation
- apply model-specific execution policy

This required:

- `RWKV7Config`
- registration in the transformers config registry
- registration in the vLLM model registry
- a model-specific vLLM config adapter

### 2.3 Engine State Integration Layer

RWKV7 does not use standard attention KV cache. Instead, it uses internal recurrent state. In vLLM this is integrated through the Mamba/linear-attention path, which requires:

- `get_mamba_state_shape_from_config()`
- `get_mamba_state_dtype_from_config()`
- `get_mamba_state_copy_func()`

In addition, the engine must recognize RWKV7 blocks as stateful runtime layers and treat them correctly in the v1 forward context.

### 2.4 Serving and Batching Layer

Even after mathematical parity is achieved, serving correctness still depends on:

- correct prefill/decode state handoff
- correct state copy behavior during cache operations
- correct batching across concurrent decode requests

This layer contained the most subtle bugs and the most important performance issue.

## 3. Implemented Code

### 3.1 New RWKV7 Config and Registration

Files:

- [vllm/transformers_utils/configs/rwkv7.py](/home/liu/vllm/vllm/transformers_utils/configs/rwkv7.py)
- [vllm/transformers_utils/config.py](/home/liu/vllm/vllm/transformers_utils/config.py)
- [vllm/transformers_utils/configs/__init__.py](/home/liu/vllm/vllm/transformers_utils/configs/__init__.py)
- [vllm/model_executor/models/registry.py](/home/liu/vllm/vllm/model_executor/models/registry.py)
- [vllm/model_executor/models/config.py](/home/liu/vllm/vllm/model_executor/models/config.py)

Implemented:

- `RWKV7Config` as a vLLM-visible config type
- registry mapping from `RWKV7Config`
- model registry entry for `RWKV7ForCausalLM`
- model-specific config handling via `RWKV7ForCausalLMConfig`

Notes:

- RWKV7 no longer forces `enforce_eager=True`.
- The config now allows the normal compile-enabled runtime path, including:
  - default PIECEWISE CUDA graphs
  - `cudagraph_mode=none`

### 3.2 New RWKV7 vLLM Runtime Model

File:

- [vllm/model_executor/models/rwkv7.py](/home/liu/vllm/vllm/model_executor/models/rwkv7.py)

Implemented modules:

- `RWKV7LoRA`
- `RWKV7GroupNorm`
- `RWKV7FeedForward`
- `RWKV7Attention`
- `RWKV7Block`
- `RWKV7Model`
- `RWKV7ForCausalLM`

Implemented runtime integration points:

- tensor-parallel-aware linear layers
- pipeline-parallel-compatible model structure
- `IntermediateTensors`
- logits computation
- state shape declaration
- state dtype declaration
- state copy function declaration
- explicit per-slot state load/store for serving

## 4. Key Technical Fixes

### 4.1 State Copy Semantics

RWKV7 has three state tensors:

- attention shift state
- recurrent state
- FFN shift state

The original copy semantics were incorrect for shift states. The final correct mapping is:

- shift state: `conv`
- recurrent state: `temporal`
- shift state: `conv`

Implementation:

- [rwkv7.py#L981](/home/liu/vllm/vllm/model_executor/models/rwkv7.py#L981)

### 4.2 Logits Dtype Alignment

During serving, `hidden_states` and `lm_head.weight` could have mismatched dtypes. This caused runtime failures and inconsistent behavior. `compute_logits()` was changed to cast hidden states to the logits weight dtype before projection.

Implementation:

- [rwkv7.py#L948](/home/liu/vllm/vllm/model_executor/models/rwkv7.py#L948)

### 4.3 Checkpoint Weight Name Mapping

RWKV7 checkpoints used:

- `model.embeddings.weight`

while the vLLM model implementation expects:

- `model.embed_tokens.weight`

This mapping was added during weight loading.

Implementation:

- [rwkv7.py#L986](/home/liu/vllm/vllm/model_executor/models/rwkv7.py#L986)

### 4.4 Static Forward Context Registration

`RWKV7Block` must be registered into the compilation config's static forward context so that the v1 engine recognizes it as a stateful layer in the serving path.

Without this, the engine could mis-handle RWKV7 as if it were not a proper state-carrying layer.

Implementation:

- [rwkv7.py#L577](/home/liu/vllm/vllm/model_executor/models/rwkv7.py#L577)

### 4.5 Hybrid Cache Coordinator Fix

RWKV-style models can expose multiple raw cache groups that coalesce into a single effective attention group. `HybridKVCacheCoordinator` previously asserted that at least two coalesced groups must remain, which is too strict for this case.

The fix allows the single-coalesced-group case and initializes:

- `self.attention_groups`
- `self.lcm_block_size`

Implementation:

- [kv_cache_coordinator.py#L434](/home/liu/vllm/vllm/v1/core/kv_cache_coordinator.py#L434)

### 4.6 FP32 Runtime for Correctness

The 0.4B RWKV7 checkpoint was mathematically correct in `float32` but exhibited drift under the default `dtype=auto`/`bfloat16` serving path. To restore exact service-path alignment, the current correctness-first solution forces RWKV7 runtime weights and cached states to `float32`.

This affects:

- runtime model weights
- runtime LM head weights
- cached recurrent state dtype
- shift state dtype

Implementation:

- [rwkv7.py#L47](/home/liu/vllm/vllm/model_executor/models/rwkv7.py#L47)
- [rwkv7.py#L592](/home/liu/vllm/vllm/model_executor/models/rwkv7.py#L592)
- [rwkv7.py#L920](/home/liu/vllm/vllm/model_executor/models/rwkv7.py#L920)
- [rwkv7.py#L952](/home/liu/vllm/vllm/model_executor/models/rwkv7.py#L952)

### 4.7 Decode Batching Fix

This was the major performance fix.

Before the fix, `RWKV7Block.forward()` handled decode requests one by one:

- load one slot state
- run one-token recurrent update
- store one slot state
- repeat for each concurrent request

This caused concurrent decoding to degenerate into per-request serial execution.

The final fix introduced:

- batched state gather via `index_select`
- batched decode recurrent update
- batched FFN decode path
- batched state writeback via `index_copy_`

New helper paths:

- `RWKV7FeedForward.forward_decode_batch()`
- `RWKV7Attention.forward_decode_batch()`
- `RWKV7Block._get_kv_states()`
- `RWKV7Block._store_kv_states()`
- `RWKV7Block._run_decode_batch()`

Implementation:

- [rwkv7.py#L225](/home/liu/vllm/vllm/model_executor/models/rwkv7.py#L225)
- [rwkv7.py#L440](/home/liu/vllm/vllm/model_executor/models/rwkv7.py#L440)
- [rwkv7.py#L647](/home/liu/vllm/vllm/model_executor/models/rwkv7.py#L647)
- [rwkv7.py#L673](/home/liu/vllm/vllm/model_executor/models/rwkv7.py#L673)
- [rwkv7.py#L690](/home/liu/vllm/vllm/model_executor/models/rwkv7.py#L690)
- [rwkv7.py#L747](/home/liu/vllm/vllm/model_executor/models/rwkv7.py#L747)

### 4.8 Resolved Compile Correctness Bug And Final PIECEWISE Fix

Fresh compile debugging with `VLLM_DISABLE_COMPILE_CACHE=1` identified the first
real compile correctness bug precisely.

The first key finding was:

- under the compiled RWKV7 path, `RWKV7Block.forward()` received
  `attn_metadata=None`
- therefore the block took the fallback sequence path:
  - `_run_sequence(hidden_states, v_first, None, None, None)`
- and never reached:
  - `_store_kv_state()`
  - `_store_kv_states()`

This was confirmed with local debug summaries captured by:

- [tmp_rwkv7_engine_first_step_compare.py](/home/liu/vllm/tmp_rwkv7_engine_first_step_compare.py)
- artifact:
  - [rwkv7_engine_step_1_final_repro.json](/tmp/rwkv7_engine_step_1_final_repro.json)

That explained the previously confusing behavior:

- the first generated token could still match
- but the recurrent state was never committed into vLLM's cache
- so later decode steps eventually diverged

The fix for that first correctness bug was:

- move the block-level runtime stateful dispatch behind:
  - `torch.ops.vllm.rwkv7_block_forward(...)`

That restored live metadata-aware cache load/store behavior on the compile path.

The remaining PIECEWISE CUDA-graph work then exposed two more concrete issues:

- debug instrumentation in `_store_kv_state()` / `_store_kv_states()` called
  `.item()` on CUDA tensors during graph capture
- `vllm::rwkv7_block_forward` was not yet included in the default
  `splitting_ops`, so the stateful runtime boundary could still be captured too
  coarsely

The final fixes were:

- gate detailed store-debug stats so they only run when:
  - `RWKV7_DEBUG_STORE_STATS=1`
  - and the current stream is not being captured
- add `vllm::rwkv7_block_forward` to the default `splitting_ops` list

An additional debugging pitfall was also confirmed:

- vLLM's torch.compile cache can reuse older compiled artifacts even after local
  RWKV7 model-code edits
- compile-path debugging should therefore be run with:
  - `VLLM_DISABLE_COMPILE_CACHE=1`

## 5. Validation Strategy

The validation process used four layers of testing.

### 5.1 Unit Tests

File:

- [tests/model_executor/test_rwkv7.py](/home/liu/vllm/tests/model_executor/test_rwkv7.py)

Coverage:

- block forward without metadata
- static forward context registration
- cached state update path
- state copy function type validation
- runtime state dtype validation
- batched decode equivalence to sequential decode

### 5.2 Reference Parity Tests

Reference implementation:

- FLA RWKV7

Coverage:

- full forward hidden/logits parity
- prefill + decode parity

Acceptance threshold:

- hidden/logits max absolute difference < `5e-5`

### 5.3 Serving Path Correctness Tests

Scripts:

- [tmp_rwkv7_compare.py](/home/liu/vllm/tmp_rwkv7_compare.py)
- [tmp_rwkv7_engine_step_debug.py](/home/liu/vllm/tmp_rwkv7_engine_step_debug.py)

Method:

- `one-shot`: one request with `max_tokens=N`
- `step-by-step`: repeated requests with `max_tokens=1`

Deterministic generation requires the two to match exactly token-by-token.

### 5.4 Concurrency and Throughput Tests

Scripts:

- [tmp_rwkv7_concurrency_check.py](/home/liu/vllm/tmp_rwkv7_concurrency_check.py)
- [tmp_rwkv7_long_benchmark.py](/home/liu/vllm/tmp_rwkv7_long_benchmark.py)

Metrics:

- aggregate TPS
- batch wall time
- per-request latency
- per-request TPS
- equality to serial baseline

## 6. Results

### 6.1 Final Unit and Parity Tests

Command result:

- `pytest -q tests/model_executor/test_rwkv7.py`
- final result: `9 passed, 2 skipped`

### 6.2 Correctness Conclusions

Confirmed:

- RWKV7 full forward math is aligned with the reference implementation.
- RWKV7 direct incremental prefill/decode is aligned.
- Service-path `one-shot` multi-token decode matches `step-by-step`.
- Both `0.1B` and `0.4B` checkpoints pass correctness checks under:
  - default PIECEWISE CUDA graphs
  - `cudagraph_mode=none`

Validated prompts:

- `i am`
- `北京是`
- `The capital of France is`

### 6.3 Final Throughput Results

After decode batching fix:

| max_tokens | concurrency | total output tokens/round | aggregate TPS | avg batch wall time | avg request latency | avg request TPS | matches serial baseline |
|---|---:|---:|---:|---:|---:|---:|---|
| 64 | 1 | 64 | 35.757 | 1.790s | 1.790s | 35.764 | yes |
| 64 | 2 | 128 | 66.715 | 1.919s | 1.917s | 33.387 | yes |
| 64 | 4 | 256 | 124.544 | 2.055s | 2.054s | 31.158 | yes |
| 64 | 8 | 512 | 228.924 | 2.237s | 2.234s | 28.649 | yes |
| 128 | 1 | 128 | 35.180 | 3.638s | 3.638s | 35.184 | yes |
| 128 | 4 | 512 | 139.380 | 3.673s | 3.672s | 34.860 | yes |
| 128 | 8 | 1024 | 261.705 | 3.913s | 3.910s | 32.740 | yes |

Interpretation:

- aggregate throughput now scales with concurrency
- per-request latency grows only modestly
- batching is now effective during decode
- correctness is preserved under concurrent serving

### 6.4 Result Files

Saved benchmark outputs:

- [/tmp/rwkv7_bench_64_after_batch.json](/tmp/rwkv7_bench_64_after_batch.json)
- [/tmp/rwkv7_bench_128_after_batch.json](/tmp/rwkv7_bench_128_after_batch.json)
- [/tmp/vllm_rwkv7_long_bench_after_batch.log](/tmp/vllm_rwkv7_long_bench_after_batch.log)
- [/tmp/vllm_rwkv7_long_bench_after_batch_128.log](/tmp/vllm_rwkv7_long_bench_after_batch_128.log)
- [/tmp/vllm_rwkv7_compare_after_batch.log](/tmp/vllm_rwkv7_compare_after_batch.log)

## 7. Current Technical Status

At the current checkpoint, RWKV7 support in vLLM has achieved:

- end-to-end model loading
- config and registry integration
- recurrent state integration with the v1 engine
- service-path correctness for multi-token decode
- concurrent decode batching
- stable throughput scaling under concurrency

This means the current implementation is not merely runnable; it is functionally correct and operationally viable under serving workloads.

## 8. Remaining Optimization Opportunities

### 8.1 RWKV7 Compile Runtime Policy

RWKV7 no longer needs an unconditional eager-only policy.

After adding the whole-block custom-op boundary in
[rwkv7.py](/home/liu/vllm/vllm/model_executor/models/rwkv7.py), the model now
supports real compile serving paths for both:

- `enforce_eager=False`
- default PIECEWISE CUDA graphs
- `cudagraph_mode=none`

The config policy in
[config.py](/home/liu/vllm/vllm/model_executor/models/config.py) is now:

- do not force eager fallback for RWKV7
- allow the normal compile-enabled runtime path to proceed

This removes the need for local monkeypatching to exercise compile.

What is now confirmed:

- in-proc engine compile/no-cg works
- real `vllm serve` compile/no-cg works
- real `vllm serve` default PIECEWISE works
- `one-shot` vs `step-by-step` matches on the tested `0.1B` and `0.4B` prompt sets

What is still open:

- compile throughput benchmarking after correctness recovery
- broader stress coverage on concurrency, prefix caching, and longer outputs

### 8.2 Relax FP32 Runtime

The current runtime uses `fp32` for correctness. This is safe but expensive.

Potential next step:

- keep only recurrent/state-critical tensors in `fp32`
- move less sensitive paths back toward `bf16`
- identify the exact subgraph that causes numerical drift

### 8.3 Replace PyTorch Recurrent Update with Fused Kernel

The current vLLM implementation uses a custom PyTorch runtime path, not the FLA Triton kernel path. Long-term peak performance likely requires:

- fused recurrent update
- fused token shift / addcmul / K-update
- chunked prefill kernel

This is likely the largest remaining single performance lever.

### 8.4 Batch Prefill More Aggressively

Decode batching is now fixed. Prefill still processes requests in a less optimized way, especially for heterogeneous mixed batches. There is still room to improve prefill throughput.

### 8.5 Extend Feature Coverage

Not yet fully implemented or stress-tested:

- hybrid RWKV7 with transformer attention
- per-layer varying `value_dim`
- broader prefix caching stress coverage
- broader PIECEWISE stress and performance coverage

### 8.6 Compile Correctness Localization Addendum

The compile-path investigation moved from "can it boot?" to "where does
correctness diverge?".

The temporary probe
[tmp_rwkv7_engine_first_step_compare.py](/home/liu/vllm/tmp_rwkv7_engine_first_step_compare.py)
was extended to support:

- `--capture-generated-tokens` to snapshot an arbitrary decode step
- `--prompt-token-ids` for token-id-controlled prompts
- `--append-generated-prefix-from-run-json` for replaying a captured prefix
- `--compare-second-step` for offline base/replay comparison

Using the local checkpoint:

- `/mnt/d/codes/RWKV7-Goose-World2.9-0.4B-HF`

and the compile/no-cudagraph path:

- `enforce_eager=False`
- `cudagraph_mode=none`
- `async_scheduling=False`

the following was observed on prompt `北京是`:

- first generated token still matches:
  - token id `10250`
  - text `一`
- second-step controlled replay also matches:
  - base run captured two tokens: `[10250, 10283]`
  - replay prompt token ids were `[10902, 10362, 13091, 10250]`
  - replay generated token was `10283`
  - replay text was `个`

That localization work was enough to identify the actual bug:

- runtime metadata existed
- but the compiled `RWKV7Block.forward()` path was not executing the
  metadata-aware cache load/store branch during live requests
- so recurrent state never got committed back into cache

The fix was to move the block-level stateful dispatch behind a whole-block
custom op, so the runtime path can always read live `forward_context`
metadata/state regardless of compile specialization.

After that fix:

- layer-local `kv_cache` stopped staying all-zero in compile/no-cg
- runner-level backing cache also became non-zero
- second-step token-controlled replay still matched
- real `vllm serve` one-shot vs step-by-step matched again on:
  - `i am`
  - `北京是`
  - `The capital of France is`

The final PIECEWISE-specific fixes were:

- keep store-debug `.item()` stats out of CUDA graph capture
- add `vllm::rwkv7_block_forward` to default `splitting_ops`

After those fixes:

- real `vllm serve` default PIECEWISE one-shot vs step-by-step also matched on:
  - `i am`
  - `北京是`
  - `The capital of France is`
- the same prompt set also matched on the local `0.1B` checkpoint for:
  - default PIECEWISE
  - `cudagraph_mode=none`

Reference artifacts from the corrected real-entrypoint validation:

- [rwkv7_engine_step_2_base_real_compile.json](/tmp/rwkv7_engine_step_2_base_real_compile.json)
- [rwkv7_engine_step_2_replay_real_compile.json](/tmp/rwkv7_engine_step_2_replay_real_compile.json)
- [rwkv7_engine_step_2_compare_real_compile.json](/tmp/rwkv7_engine_step_2_compare_real_compile.json)
- [vllm_rwkv7_compare_real_compile_no_cg.log](/tmp/vllm_rwkv7_compare_real_compile_no_cg.log)
- [vllm_rwkv7_piecewise_final.log](/tmp/vllm_rwkv7_piecewise_final.log)
- [vllm_rwkv7_compile_no_cg_final.log](/tmp/vllm_rwkv7_compile_no_cg_final.log)
- [vllm_rwkv7_piecewise_0p1b_final.log](/tmp/vllm_rwkv7_piecewise_0p1b_final.log)
- [vllm_rwkv7_compile_no_cg_0p1b_final.log](/tmp/vllm_rwkv7_compile_no_cg_0p1b_final.log)

## 9. Version Checkpoints

Important commits on this branch:

- `5f088df79` Force RWKV7 runtime to fp32 for correctness
- `89fc82ea3` Handle single coalesced RWKV cache group
- `61db3ff93` Batch RWKV7 decode updates across requests
- `c29c30f49` Add engine-step probe and first-step compile debugging notes
- `a5bbd9b797` Localize compile-path metadata/state bug
- `f94fd358b` Restore real compile serving for RWKV7

Current branch:

- `codex/rwkv7-adapter-align`
