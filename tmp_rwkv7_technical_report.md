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

- The current model config forces `enforce_eager=True` because RWKV7 support currently uses eager recurrent updates.

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

### 4.8 Root Cause Of The Current Non-Eager Correctness Bug

Fresh compile debugging with `VLLM_DISABLE_COMPILE_CACHE=1` identified the current
non-eager correctness bug more precisely.

The key finding is:

- under the compiled RWKV7 path, `RWKV7Block.forward()` receives
  `attn_metadata=None`
- therefore the block takes the fallback sequence path:
  - `_run_sequence(hidden_states, v_first, None, None, None)`
- and never reaches:
  - `_store_kv_state()`
  - `_store_kv_states()`

This was confirmed with local debug summaries captured by:

- [tmp_rwkv7_engine_first_step_compare.py](/home/liu/vllm/tmp_rwkv7_engine_first_step_compare.py)
- artifact:
  - [rwkv7_engine_step_1_final_repro.json](/tmp/rwkv7_engine_step_1_final_repro.json)

Observed properties in that artifact:

- the request is still unfinished after the first captured generation step
- `RWKV7Block.debug_last_forward_summary` reports:
  - `attn_metadata_is_none=1`
  - `num_decode_tokens=-1`
  - `num_prefill_tokens=-1`
- `debug_last_store_stats` is still unset
- layer-local cache summaries remain zero
- runner-level cache summaries remain zero

This explains the previously confusing behavior:

- the first generated token can still match between:
  - `max_tokens=1`
  - `max_tokens=8`
- because prompt-time recurrent math still runs inside the same forward pass
- but the recurrent state is never committed into vLLM's cache
- so later decode steps eventually diverge

The current compile bug is therefore not best described as:

- an unknown later-step cache corruption

It is better described as:

- a metadata/stateful-path integration failure in the RWKV7 compile boundary

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
- final result: `8 passed`

### 6.2 Correctness Conclusions

Confirmed:

- RWKV7 full forward math is aligned with the reference implementation.
- RWKV7 direct incremental prefill/decode is aligned.
- Service-path `one-shot` multi-token decode matches `step-by-step`.
- Both `0.1B` and `0.4B` checkpoints pass correctness checks under the current runtime.

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

### 8.1 Remove Forced Eager Mode

Current RWKV7 support is forced into eager execution:

- [config.py#L664](/home/liu/vllm/vllm/model_executor/models/config.py#L664)

This currently disables:

- `torch.compile`
- `CUDAGraph`

Short-term optimization target:

- make decode-only RWKV7 graph-friendly
- attempt non-eager decode runtime first
- evaluate CUDAGraph on stable decode shapes

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
- non-eager compiled runtime

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

This narrows the known divergence:

- it is not on the first decode step
- it is not on the second decode step for this prompt

However, the current layer-local `model.model.layers[*].kv_cache` snapshot
still stays all-zero in this probe, so these results do not yet prove that the
real model-runner-owned backing cache is correct. The next high-value
localization step is therefore:

- inspect `GPUModelRunner.kv_caches` (or the equivalent backing store) on the
  matched second-step pair
- or extend token-id-controlled replay to later decode steps / other prompts

## 9. Version Checkpoints

Important commits on this branch:

- `5f088df79` Force RWKV7 runtime to fp32 for correctness
- `89fc82ea3` Handle single coalesced RWKV cache group
- `61db3ff93` Batch RWKV7 decode updates across requests
- `c29c30f49` Add engine-step probe and first-step compile debugging notes

Current branch:

- `codex/rwkv7-adapter-align`
