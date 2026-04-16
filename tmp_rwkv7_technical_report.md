# RWKV7 on vLLM: Technical Report

Date: 2026-03-31

Branch: `codex/rwkv7-adapter-align`

## 0. Latest Tooling Update (2026-04-14)

The remote serving benchmark utility was upgraded to better measure remote
vLLM saturation instead of the benchmark client's own fixed worker limit.

File:

- [tmp_rwkv7_remote_concurrency_bench.py](/home/liu/vllm/tmp_rwkv7_remote_concurrency_bench.py)

Added:

- `--dispatch-mode burst`
  - launches all benchmark requests immediately
  - useful when the goal is to push queueing into the remote vLLM service
    rather than keeping it in the client-side worker pool
- explicit throughput fields:
  - `token_throughput_tps`
    - flat aliases:
      - `token_throughput_tps_avg`
      - `token_throughput_tps_min`
      - `token_throughput_tps_max`
  - `active_output_tps`
  - `token_throughput_tps_stats`
    - 1-second bucketed `min / avg / max` token throughput during the active
      request window
  - `request_token_throughput_tps`
    - per-request token throughput
    - stored per row in `requests.jsonl`
    - summarized as `avg / p50 / p95 / min / max`
    - plus `weighted_avg` for a more stable per-request view when heavy
      long-tail latency skews the plain arithmetic mean
- explicit queue/pressure diagnostics:
  - `worker_count`
  - `peak_inflight_requests`
  - `avg_inflight_requests`
  - `active_window_sec`
  - `client_queue_delay_before_first_start_sec`
  - `configured_concurrency`
    - only meaningful in `closed_loop`
    - intentionally `null` in `burst`

Interpretation:

- `token_throughput_tps` remains the end-to-end wall-clock token throughput.
- `active_output_tps` is measured from the first actual request start to the
  last request finish, which helps distinguish client queueing from remote
  service processing.
- `closed_loop` remains useful for fixed-concurrency benchmarking.
- `burst` is now the preferred mode when the goal is to locate remote service
  saturation points.

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

### 8.7 Performance And Default-Mode Addendum

The throughput picture is now more nuanced than the earlier quick benchmark
snapshot suggested.

First, inheriting plain `MambaModelConfig` defaults left RWKV7 on:

- `cudagraph_mode=FULL_AND_PIECEWISE`

That mode is still unsafe for RWKV7 under benchmark load. On the local 0.4B
checkpoint it reproduced:

- `indexSelectSmallIndex ... Assertion srcIndex < srcSelectDimSize failed`
- followed by `CUDA error: device-side assert triggered`

The fix was to add an RWKV7-specific post-optimization override in
[config.py](/home/liu/vllm/vllm/model_executor/models/config.py):

- if the inherited default is `FULL_AND_PIECEWISE`
- override it to `PIECEWISE`

This keeps the passing whole-block piecewise path while avoiding the still-unsafe
full decode-graph path.

Second, the earlier quick performance snapshot was replaced by a clean rerun on
the local 0.4B checkpoint. The historical eager-only table in section `6.3`
remains useful as chronology, but current eager-versus-piecewise conclusions
should use the consolidated reruns below.

### 8.7.1 Short-Output Mixed-Prompt Rerun

On `/mnt/d/codes/RWKV7-Goose-World2.9-0.4B-HF`, with
`async_scheduling=False`, the clean short-output artifacts are:

- [rwkv7_bench_0p4b_eager_64.json](/tmp/rwkv7_bench_0p4b_eager_64.json)
- [rwkv7_bench_0p4b_piecewise_64.json](/tmp/rwkv7_bench_0p4b_piecewise_64.json)
- [rwkv7_bench_0p4b_eager_128.json](/tmp/rwkv7_bench_0p4b_eager_128.json)
- [rwkv7_bench_0p4b_piecewise_128.json](/tmp/rwkv7_bench_0p4b_piecewise_128.json)

Aggregate TPS snapshots:

| mode | `max_tokens=64`, c=`1/2/4/8` | `max_tokens=128`, c=`1/2/4/8` |
|---|---|---|
| eager | `28.122 / 54.669 / 104.712 / 191.801` | `27.642 / 48.780 / 110.584 / 203.157` |
| piecewise | `27.566 / 54.985 / 103.009 / 186.164` | `27.856 / 45.786 / 103.951 / 193.628` |

All rows matched the serial baseline.

### 8.7.2 Long-Input Exact-Token Rerun

To separate short-output decode behavior from long-prefill behavior, an exact
token-count benchmark was added:

- script: [/tmp/rwkv7_exact_long_input_bench.py](/tmp/rwkv7_exact_long_input_bench.py)

The model has `max_position_embeddings=2048`, so the second long-input case was
run as:

- prompt length `1984`
- `max_tokens=64`

to stay just below the context cap while still exercising decode.

Artifacts:

- [rwkv7_long_input_eager_1024.json](/tmp/rwkv7_long_input_eager_1024.json)
- [rwkv7_long_input_piecewise_1024.json](/tmp/rwkv7_long_input_piecewise_1024.json)
- [rwkv7_long_input_eager_1984.json](/tmp/rwkv7_long_input_eager_1984.json)
- [rwkv7_long_input_piecewise_1984.json](/tmp/rwkv7_long_input_piecewise_1984.json)

Aggregate TPS snapshots:

| prompt len + decode | mode | c=`1/4/8` |
|---|---|---|
| `1024 + 64` | eager | `11.830 / 14.974 / 15.850` |
| `1024 + 64` | piecewise | `10.146 / 13.967 / 16.376` |
| `1984 + 64` | eager | `7.237 / 9.431 / 10.046` |
| `1984 + 64` | piecewise | `7.863 / 9.449 / 9.529` |

All rows again matched the serial baseline.

### 8.7.3 Current Interpretation

The combined picture on the local `0.4B` checkpoint is:

- `cudagraph_mode=none` remains useful for correctness localization and
  debugging, but it is not the main performance path
- `PIECEWISE` and eager now sit in roughly the same runtime band on this
  machine, rather than `PIECEWISE` showing a stable clear win
- short-output mixed-prompt runs keep `PIECEWISE` close to eager, but not
  consistently faster
- long-input prefill-heavy runs also stay close:
  - `1024`-token prompt: `PIECEWISE` trails at concurrency `1/4`, edges ahead at `8`
  - `1984`-token prompt: `PIECEWISE` leads slightly at `1`, ties at `4`, trails at `8`
- cold-start cost is still much higher for piecewise capture
  - eager init engine: about `9.2s`
  - no-cg init engine: about `13.8s`
  - piecewise init engine: about `94-97s`

So the current recommendation is:

- keep RWKV7 default compile mode on `PIECEWISE`
- avoid FULL decode graphs for now
- treat `compile + PIECEWISE` as the correctness-capable compile path, not yet
  as a stable throughput win over eager on this machine
- measure TTFT / prefill-only and deeper kernelization next if performance is
  the main remaining goal

### 8.7.4 Fused Prefill Recurrent Addendum

The next bottleneck after compile correctness recovery was the Python token loop
inside `RWKV7Attention._forward()`. That loop still sat on the critical path for
sequence prefill in both eager and PIECEWISE serve, which limited long-prefill
TTFT and left compile with little headroom to help.

To address that, a dedicated fused RWKV7 recurrent op was added under:

- [vllm/model_executor/layers/fla/ops/rwkv7.py](/home/liu/vllm/vllm/model_executor/layers/fla/ops/rwkv7.py)

It is adapted from the FLA RWKV7 fused recurrent implementation, but narrowed to
the inference needs here:

- Triton kernel for the recurrent state update
- local Python reference path for parity and fallback
- env kill switch:
  - `RWKV7_DISABLE_FUSED_PREFILL=1`

The model integration was intentionally scoped:

- only the sequence-prefill branch in
  [RWKV7Attention._forward()](/home/liu/vllm/vllm/model_executor/models/rwkv7.py:550)
  was switched from the Python token loop to `fused_mul_recurrent_rwkv7(...)`
- decode batching in `forward_decode_batch()` is still the older tensor path
- metadata-driven packed/varlen prefill is still not implemented

Validation on the local `0.4B` checkpoint:

- unit tests:
  - `python -m pytest -q tests/model_executor/test_rwkv7.py`
  - result: `11 passed, 2 skipped`
- service correctness:
  - `tmp_rwkv7_compare.py --disable-compile-cache`
  - one-shot vs step-by-step still matched on the standard `3` prompts

Stable reruns with higher warmup (`rounds=3`, `warmup=2`) showed:

| mode | prompt `64` TTFT | prompt `1024` TTFT | prompt `1984` TTFT | decode ITL (`64 -> 64`) |
|---|---|---|---|---|
| eager, fused off | `577.311ms` | `2991.876ms` | `5031.042ms` | `27.296ms` |
| eager, fused on | `96.296ms` | `522.067ms` | `1069.782ms` | `34.035ms` |
| piecewise, fused on | `60.013ms` | `276.974ms` | `530.588ms` | `27.823ms` |

Interpretation:

- fused prefill is the first model-specific optimization that produces a clear
  and repeatable long-prefill latency win for RWKV7
- earlier eager post-fused seconds-long ITL spikes disappeared once warmup was
  increased, suggesting those were one-time warmup effects rather than a stable
  runtime regression
- eager fused-on still shows a modest decode-side ITL regression relative to
  the Python-loop baseline
- piecewise fused-on currently gives the best long-prefill latency while keeping
  decode ITL near the historical eager band

This shifts the next bottleneck:

- the main remaining prefill problem is no longer the per-token recurrence
  inside attention
- it is the Python per-prefill-request loop inside `RWKV7Block._forward_runtime()`
- so the next meaningful optimization step is packed/varlen prefill driven by
  `query_start_loc`, followed by a fused decode recurrent backend

### 8.7.5 Packed Prefill Runtime Addendum

The next follow-up after the fused recurrent op was to stop wasting that kernel
behind a Python loop over prefill requests. The fused op already accepted
`cu_seqlens`, but RWKV7 block runtime still handled each prefill request
separately, including individual KV-state loads and stores.

This iteration moved packed/varlen prefill into the model runtime:

- added `token_shift_with_cache_varlen(...)`
- added:
  - `RWKV7Attention.forward_prefill_batch(...)`
  - `RWKV7FeedForward.forward_prefill_batch(...)`
  - `RWKV7Block._run_prefill_batch(...)`
- changed
  [RWKV7Block._forward_runtime()](/home/liu/vllm/vllm/model_executor/models/rwkv7.py)
  to:
  - slice all prefill tokens as one packed token range
  - derive `cu_seqlens` from `query_start_loc`
  - mask out nonexistent initial states with `seq_lens > query_lens`
  - batch-load KV state via `_get_kv_states(...)`
  - batch-store final state via `_store_kv_states(...)`

The old per-request fallback was kept behind:

- `RWKV7_DISABLE_FUSED_PREFILL=1`

Validation on the local `0.4B` checkpoint:

- unit tests:
  - `python -m pytest -q tests/model_executor/test_rwkv7.py`
  - result: `12 passed, 2 skipped`
- new model-level regression guard:
  - `test_rwkv7_block_batches_prefill_tokens_without_changing_results`
- service correctness:
  - `tmp_rwkv7_compare.py --disable-compile-cache`
  - one-shot vs step-by-step still matched on the standard `3` prompts
- real-entrypoint batching smoke:
  - `tmp_rwkv7_long_benchmark.py --cudagraph-mode piecewise --disable-compile-cache --max-tokens 16 --concurrency-levels 4 8`
  - aggregate TPS:
    - concurrency `4`: `18.689`
    - concurrency `8`: `208.104`
  - both rows matched the serial baseline

Interpretation:

- the Python per-prefill-request loop is now removed from the main packed
  prefill path
- this change is necessary for concurrent prefill scaling, but the short-prompt
  smoke benchmark above is still mostly a correctness check rather than a clean
  performance attribution experiment
- the next meaningful benchmark should reuse the long-input exact-token setup
  (`1024` / `1984` prompt lengths at concurrency `1/4/8`) so the packed-prefill
  gain can be quantified directly
- after that, the next remaining hot path is decode recurrence, which still uses
  the older tensor implementation in `forward_decode_batch()`

### 8.7.6 Why The Model Can Still Look "Slow" After Packed Prefill

After packed-prefill landed, the first impression could still be "it is slower"
if one compared the wrong benchmark rows:

- the packed-prefill smoke used short mixed prompts and `max_tokens=16`
- the earlier eager baseline used a different workload and `max_tokens=64`
- exact benchmark `aggregate_tps` is defined as `completion_tokens / wall_time`,
  so it penalizes prefill time hard and is very sensitive to one-time runtime
  setup costs

To resolve that ambiguity, an exact-token long-input benchmark was added:

- [tmp_rwkv7_exact_long_input_bench.py](/home/liu/vllm/tmp_rwkv7_exact_long_input_bench.py)

It uses:

- fixed token-count prompt prefixes
- token-id prompts instead of prompt strings
- exact prompt lengths `1024` and `1984`
- exact output length `64`

The first broad one-shot rerun already showed that `1984`-token prompts were
helping a lot under `PIECEWISE`, but the `1024, c=8` row came out suspiciously
slow (`13.387` TPS). A focused rerun on that exact row showed the broad sweep
was not a good steady-state estimate.

Focused results:

| workload | eager | piecewise |
|---|---|---|
| `1024 + 64`, concurrency `8` | `131.458 / 124.108` TPS | `120.680 / 123.058` TPS |
| `1984 + 64`, concurrency `8` | `14.594` TPS | `80.053` TPS |

Interpretation:

- packed prefill is now doing what it should:
  - it moves the very long prompt case (`1984`) decisively in favor of `PIECEWISE`
- the medium-long case (`1024`) is now roughly at eager parity in steady-state,
  not dramatically slower
- the earlier "very slow" `1024` row was measurement pollution, not the true
  steady-state behavior

So why is there still no uniform win?

- because prefill is no longer the only hot path
- the remaining major RWKV7 bottleneck is decode recurrence, which still goes
  through the older tensor implementation:
  - [RWKV7FeedForward.forward_decode_batch()](/home/liu/vllm/vllm/model_executor/models/rwkv7.py:404)
  - [RWKV7Attention.forward_decode_batch()](/home/liu/vllm/vllm/model_executor/models/rwkv7.py:748)
- once prompt length is not extreme enough to dominate the request, the decode
  side limits how much benefit packed prefill can surface

This gives the next optimization order very clearly:

1. keep the exact-long eager rows as the control
2. fuse the decode recurrent backend
3. rerun the exact-long steady-state rows
4. only then revisit whether `PIECEWISE` should be marketed as a throughput win

### 8.7.7 Fused Decode Recurrent Backend

The next iteration after packed prefill was to remove the remaining explicit
decode recurrence in:

- [RWKV7Attention.forward_decode_batch()](/home/liu/vllm/vllm/model_executor/models/rwkv7.py:748)

This iteration did two things:

1. moved recurrent-input projection and output finalization into shared helpers
2. switched decode batch recurrence to `fused_mul_recurrent_rwkv7(...)` on CUDA

Concretely:

- added shared helpers in `RWKV7Attention`:
  - `_project_recurrent_inputs(...)`
  - `_finalize_attention_output(...)`
- changed decode batch to call the fused recurrent op with:
  - batch dimension = decode batch size
  - sequence length = `1`
- added a generic disable knob:
  - `RWKV7_DISABLE_FUSED_RECURRENT=1`
  - legacy `RWKV7_DISABLE_FUSED_PREFILL=1` still disables the fused recurrent path too
- added CUDA regression coverage:
  - `test_rwkv7_block_batches_decode_tokens_without_changing_results_cuda`

Validation:

- unit tests:
  - `python -m pytest -q tests/model_executor/test_rwkv7.py`
  - result: `13 passed, 2 skipped`
- service correctness:
  - `tmp_rwkv7_compare.py --disable-compile-cache`
  - one-shot vs step-by-step still matched on the standard `3` prompts
- exact long-input benchmark, sequential on one GPU:

| workload | eager | piecewise |
|---|---|---|
| `1024 + 64`, concurrency `8` | `128.662 / 126.905` TPS | `123.847 / 124.573` TPS |
| `1984 + 64`, concurrency `8` | `82.291 / 84.238` TPS | `84.708 / 88.764` TPS |

Interpretation:

- decode recurrence was indeed one of the last material RWKV7 runtime bottlenecks
- after it was fused, the exact-long steady-state rows narrowed substantially:
  - eager remains slightly ahead at `1024`
  - `PIECEWISE` remains slightly ahead at `1984`
- this means the main model-specific adaptation gap is now much smaller than it
  was before decode fusion
- it also means compile no longer shows a dramatic RWKV7-specific throughput win
  on this exact workload; most of the recovered performance came from fixing the
  model path itself

One practical lesson from this round:

- do not launch eager and `PIECEWISE` benchmark servers in parallel on a single
  GPU
- those runs directly contend on the same device and produce invalid TPS samples

### 8.7.8 Prefix Caching And Mixed Prompt-Length Validation

After the recurrent hot paths were fused, the next question was no longer
"can compile run correctly?" but rather "does it behave sensibly in real
service-style workloads?"

Two follow-up probes were added:

1. `tmp_rwkv7_exact_long_input_bench.py --enable-prefix-caching`
2. [tmp_rwkv7_mixed_exact_prompt_bench.py](/home/liu/vllm/tmp_rwkv7_mixed_exact_prompt_bench.py)

The mixed prompt benchmark uses exact token lengths:

- `64`
- `128`
- `256`
- `512`
- `768`
- `1024`
- `1536`
- `1984`

Correctness check first:

- `tmp_rwkv7_compare.py --enable-prefix-caching --disable-compile-cache`
- one-shot vs step-by-step still matched on the standard `3` prompts

Prefix-caching exact-long results:

| workload | eager | piecewise |
|---|---|---|
| `1024 + 64`, concurrency `8` | `130.217 / 210.408` TPS | `149.130 / 208.119` TPS |
| `1984 + 64`, concurrency `8` | `155.799 / 270.292` TPS | `165.642 / 256.278` TPS |

Interpretation:

- prefix caching is working
- the serial baseline warms the prefix cache before the measured concurrent
  rounds, so the large round1 jump is expected
- once the cache is hot, eager and `PIECEWISE` sit in the same band

Mixed exact prompt-length results:

| workload | eager | piecewise |
|---|---|---|
| no cache | `149.064 / 150.716` TPS | `87.009 / 134.828` TPS |
| prefix cache | `225.201 / 231.280` TPS | `228.883 / 229.894` TPS |

Interpretation:

- without prefix caching, mixed prompt lengths still expose some first-round
  `PIECEWISE` warmup cost
- with prefix caching enabled, eager and `PIECEWISE` are again effectively tied
- in these service-level scenarios, prefix caching contributes a much larger
  throughput shift than compile alone

This sharpens the practical conclusion:

- compile support for RWKV7 is now real and correct on the validated path
- but it should not be sold as a guaranteed standalone throughput win over the
  already-fixed eager path
- the biggest remaining work is service/feature coverage, not another deep
  recurrence rewrite

### 8.7.9 Longer Outputs And compile_no_cg Recheck

The next validation round targeted two remaining concerns:

1. do longer outputs reintroduce hidden regressions?
2. does the old `compile_no_cg 128/c8` mismatch still exist?

For longer outputs, the exact-long benchmark was reused with:

- prompt lengths `1024` and `1920`
- output length `128`
- concurrency `8`

Why `1920` instead of `1984`?

- because `1984 + 128` exceeds the current `2048` limit
- the resulting `400 Bad Request` is a workload validity issue, not a model bug

Exact-long `max_tokens=128` results:

| workload | eager | piecewise | compile_no_cg |
|---|---|---|---|
| `1024 + 128`, concurrency `8` | `187.512 / 185.750` TPS | `180.043 / 182.027` TPS | `177.782 / 175.871` TPS |
| `1920 + 128`, concurrency `8` | `137.117 / 137.148` TPS | `134.320 / 135.409` TPS | `133.385 / 133.164` TPS |

Interpretation:

- all three paths still matched the serial baseline
- eager remains slightly ahead on this exact-long workload
- `PIECEWISE` stays close behind
- `compile_no_cg` is now also in the same general band, though still a bit slower

The historical `compile_no_cg 128/c8` tail item was also rerun directly on the
older mixed prompt benchmark:

- `default_mixed_8`
- `max_tokens=128`
- concurrency `8`
- compile_no_cg aggregate TPS:
  - `277.310 / 274.227`
  - avg `275.768`
- all requests matched the serial baseline

Interpretation:

- the old `compile_no_cg 128/c8` mismatch is not reproduced on the current
  decode-fused branch
- this does not promote `compile_no_cg` to the preferred performance path
- but it does substantially reduce the earlier correctness concern around long
  outputs at high concurrency

This changes the project posture again:

- the remaining unanswered questions are increasingly about product-style
  serving mixes and feature parity
- not about an obviously broken RWKV7 recurrent execution path

### 8.7.10 Partial Prefix-Hit Prefix-Caching Workload

The next follow-up was to probe a more realistic cache-reuse pattern than the
previous "cache off vs cache on" experiments. The new helper
[tmp_rwkv7_prefix_hit_bench.py](/home/liu/vllm/tmp_rwkv7_prefix_hit_bench.py)
does the following:

1. warms a small set of long shared prefixes
2. mixes those warmed prefixes with fresh cold prefixes
3. measures concurrent throughput at hit ratios `0.0`, `0.5`, and `1.0`
4. checks every measured round against a serial baseline

One correctness detail mattered here:

- different hit-ratio scenarios must not reuse the same cold prefixes
- otherwise a previous scenario silently warms the next one and contaminates the
  measurement
- the helper was fixed to allocate disjoint cold/tail token ranges per scenario
  before any numbers were collected

Configuration:

- model: `RWKV7-Goose-World2.9-0.4B-HF`
- shared prefix length: `1024`
- tail length: `128`
- output length: `64`
- concurrency: `8`
- rounds: `2`
- prefix caching: enabled

Results:

| hit ratio | eager | piecewise |
|---|---|---|
| `0.0` | `109.572 / 122.895` TPS | `121.039 / 124.047` TPS |
| `0.5` | `166.636 / 172.240` TPS | `163.858 / 174.774` TPS |
| `1.0` | `251.274 / 255.214` TPS | `252.544 / 249.909` TPS |

Interpretation:

- throughput rises strongly with prefix-hit ratio on both paths
- eager:
  - avg `116.233 -> 169.438 -> 253.244`
- piecewise:
  - avg `122.543 -> 169.316 -> 251.227`
- `PIECEWISE` is slightly better at `0%` hit, essentially tied at `50%`, and
  slightly behind at `100%`
- practically, once cache reuse exists, eager and `PIECEWISE` land in the same
  serving band

This result sharpens the current conclusion again:

- compile/cudagraph support for RWKV7 is real and correct
- but the major production-style throughput lever is still prefix caching
  itself, not compile alone

There is still one realism gap left:

- this benchmark is a burst workload with concurrent arrivals inside a round
- it is more realistic than the earlier exact-length cache probe
- but it is not yet an arrival-staggered repeated-prefix serving stream

### 8.7.11 High-Concurrency Stress Sweep

Once the service-style cache probes were in place, the natural next question
was whether the current RWKV7 adaptation could actually tolerate very large
concurrency on a single GPU.

The existing mixed-prompt benchmark was reused for a direct stress sweep:

- model: `RWKV7-Goose-World2.9-0.4B-HF`
- workload: `default_mixed_8`
- output length: `64`
- prefix caching: off
- paths:
  - eager
  - `PIECEWISE`
- concurrency:
  - `1, 2, 4, 8, 16, 32, 64`
  - plus a separate `128` stress pass

Results:

| concurrency | eager | piecewise |
|---|---:|---:|
| `1` | `31.156` TPS | `29.302` TPS |
| `2` | `62.260` TPS | `68.573` TPS |
| `4` | `123.345` TPS | `135.256` TPS |
| `8` | `245.424` TPS | `244.390` TPS |
| `16` | `459.711` TPS | `466.835` TPS |
| `32` | `929.251` TPS | `857.579` TPS |
| `64` | `1284.756` TPS | `1275.735` TPS |
| `128` | `379.127` TPS | `1668.700` TPS |

Latency behavior matters as much as TPS here:

- eager:
  - roughly `~2s` average request latency through `8`
  - `64`: avg `3.171s`, p95 `3.185s`
  - `128`: avg `15.553s`, p95 `21.746s`
- piecewise:
  - roughly `~1.9s` to `~2.4s` through `32`
  - `64`: avg `3.192s`, p95 `3.206s`
  - `128`: avg `4.875s`, p95 `4.945s`

Correctness remained intact:

- every measured round matched the serial baseline
- even at eager `128`, all requests still produced the full `64` output tokens
- so the eager collapse is not explained by truncated outputs

Interpretation:

- up to `64` concurrency, both eager and `PIECEWISE` remain viable on this
  workload and land in a similar throughput band
- at `128`, the execution paths diverge sharply:
  - eager falls off a cliff
  - `PIECEWISE` continues to scale
- this is the strongest evidence so far that compile/cudagraph support for
  RWKV7 is not merely "correctness plumbing"; under sufficiently high burst
  concurrency it can materially improve serving behavior

This does not yet mean "RWKV7 is solved for all large-scale traffic":

- the workload is still synchronized burst traffic
- the next realism step is arrival-staggered high-concurrency serving
- but as a single-GPU stress result, the current adaptation already shows a
  meaningful operational distinction between eager and `PIECEWISE`

### 8.7.12 Remote Concurrency Benchmark Tooling

To make the next validation stage easier, a reusable remote benchmark helper
was added:

- [tmp_rwkv7_remote_concurrency_bench.py](/home/liu/vllm/tmp_rwkv7_remote_concurrency_bench.py)

Its purpose is not to replace the local benchmark helpers, but to bridge the
gap between:

- local isolated benchmarking
- and real remote deployment checks

Key capabilities:

- targets a remote OpenAI-compatible vLLM endpoint
- supports both:
  - `/v1/completions`
  - `/v1/chat/completions`
- supports both:
  - fixed-concurrency closed-loop load
  - staggered arrival-rate-driven load
- loads prompts from inline args or from `.txt`, `.json`, `.jsonl`
- writes durable artifacts for each run:
  - `config.json`
  - `summary.json`
  - `summary.md`
  - `requests.jsonl`

Why this matters:

- the current local results already show that synchronized burst traffic can
  differentiate eager vs `PIECEWISE`
- the next important question is whether a real remote deployment shows the
  same behavior under more production-like arrivals
- this utility makes that next step repeatable and recordable without needing
  to keep rewriting ad hoc shell loops

No remote run was recorded in this iteration because the task here was to add
the tool itself, not to benchmark a specific remote endpoint yet.

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

## 10. PP Startup Fix

### Symptom

The first remote `pipeline_parallel_size=2` eager smoke for
`RWKV7-Goose-World3-2.9B-HF` failed during startup profiling on PP rank 1 with:

- `RuntimeError: expected scalar type BFloat16 but found Float`

The failure happened in the first local block on rank 1:

- `RWKV7Block._run_sequence()`
- `attn_input = self.attn_norm(residual)`

### Root Cause

RWKV7 uses `RWKV7_RUNTIME_DTYPE = torch.float32` for its block/runtime path.
However, non-first PP ranks allocate empty intermediate tensors for dummy/profile
runs through `make_empty_intermediate_tensors(batch_size, dtype, device)`, and
vLLM passed `model_config.dtype` (`bfloat16`) there.

That meant:

- PP rank 1 dummy/profile input `hidden_states` / `v_first` were bf16
- RWKV7 block `LayerNorm` parameters were still float32 in this path
- startup failed before the engine could finish memory profiling

### Fix

The PP boundary handling in `vllm/model_executor/models/rwkv7.py` was updated:

- `RWKV7Model.make_empty_intermediate_tensors()` now allocates PP activations
  in `RWKV7_RUNTIME_DTYPE`
- `RWKV7Model.forward()` now normalizes received and returned PP
  `hidden_states` / `v_first` tensors to `RWKV7_RUNTIME_DTYPE`

### Validation

Local validation after the fix:

```bash
source ~/miniforge3/etc/profile.d/conda.sh
conda activate vllm-dev
cd /home/liu/vllm
python -m py_compile vllm/model_executor/models/rwkv7.py
python -m pytest -q tests/model_executor/test_rwkv7.py
```

Result:

- `13 passed, 2 skipped`

Remote follow-up still required:

- rerun `PP=2 + eager`
- then rerun `PP=2 + PIECEWISE` if eager startup succeeds

## 11. RWKV7 Mamba Prefix Cache All-Mode Support

### Motivation

RWKV7 had working prefix caching only through the experimental Mamba `align`
mode. That mode is correctness-oriented but relatively strict about when a
request can reuse cached state, which matches the observation that some more
realistic workloads still reported `Prefix cache hit rate: 0.0%` for long
stretches.

The next structural step was therefore to move RWKV7 from:

- `align` mode only

to:

- full `all`-mode support at the model/runtime level

### What changed

1. Model capability

- `RWKV7ForCausalLM` now implements `SupportsMambaPrefixCaching`

2. Metadata path

- `vllm/v1/attention/backends/linear_attn.py` was extended so that in
  `mamba_cache_mode == "all"` it no longer collapses the block table down to a
  single slot id
- the builder now propagates:
  - `num_computed_tokens`
  - `block_idx_last_computed_token`
  - `block_idx_first_scheduled_token`
  - `block_idx_last_scheduled_token`

3. Decode path

- RWKV7 decode now reads initial state from the last computed block and writes
  the updated state into the last scheduled block
- this is the key requirement when a one-token decode crosses a mamba cache
  block boundary

4. Prefill path

- RWKV7 prefill now writes additional aligned states for every fully completed
  block crossed during the current scheduled prefill segment
- it also continues to write the final state for the final scheduled block

### Implementation note

The aligned writeback for recurrent states is currently implemented by:

- keeping the normal recurrent output/final-state path on the existing fused
  recurrent op
- explicitly collecting recurrent checkpoint states for aligned block
  boundaries during cache-all prefill handling

This prioritizes correctness and integration with the current RWKV7 runtime
structure. A later optimization pass can revisit whether those checkpoint
states should be produced directly by the fused kernel.

### Validation

Local verification:

```bash
source ~/miniforge3/etc/profile.d/conda.sh
conda activate vllm-dev
cd /home/liu/vllm
python -m py_compile \
  vllm/v1/attention/backends/linear_attn.py \
  vllm/model_executor/models/rwkv7.py \
  tests/model_executor/test_rwkv7.py
python -m pytest -q tests/model_executor/test_rwkv7.py
```

Result:

- `17 passed, 2 skipped`

New unit coverage includes:

- RWKV7 declares mamba prefix caching support
- config chooses `mamba_cache_mode='all'`
- all-mode prefill writes aligned block states correctly
- all-mode decode writes across cache-block boundaries correctly

### Remaining work

- serving smoke to confirm the runtime now selects `all` for RWKV7 with prefix
  caching enabled
- repeated-prefix benchmark to quantify whether real observed hit rate
  improves over the previous `align` baseline
- if needed, optimize checkpoint-state extraction so `all` mode startup/prefill
  overhead is reduced

### Serving validation (`all` vs `align`)

Follow-up service validation was run on the local `0.4B` checkpoint using the
existing repeated-prefix benchmark harness under `PIECEWISE` with prefix
caching enabled. The harness was extended to:

- accept `--mamba-cache-mode`
- capture startup log signals in the emitted JSON

This confirmed that default RWKV7 startup now really selects `all`:

- `Mamba cache mode is set to 'all' for RWKV7ForCausalLM by default when prefix caching is enabled`

and that a forced `--mamba-cache-mode align` run stays in `align`.

Benchmark summary (`shared_prefix_len=1024`, `tail_len=128`, `max_tokens=64`,
`concurrency=8`, `rounds=3`):

| mode | hit ratio | avg aggregate TPS |
|---|---:|---:|
| `all` | `0.0` | `19.404` |
| `all` | `0.5` | `29.758` |
| `all` | `1.0` | `120.398` |
| `align` | `0.0` | `112.421` |
| `align` | `0.5` | `164.175` |
| `align` | `1.0` | `238.235` |

All rounds in both modes still matched the serial baseline.

The service logs also showed that prefix caching is now active in both modes:

- `all` run: prefix-cache hit rate climbed to about `59.2%`
- `align` run: prefix-cache hit rate climbed to about `50.5%`

This is an important result: the new RWKV7 `all`-mode plumbing is not broken
from a correctness or cache-hit perspective, but it is currently much slower
than `align`.

### Revised performance interpretation

The dominant issue is not cache-hit plumbing. The dominant issue is that the
current `all`-mode recurrent checkpoint writeback still requires explicit
checkpoint-state materialization for aligned block boundaries.

I also tried a follow-up packed-prefill refactor after the first service run,
but it did not materially recover throughput and was reverted. That points to
the checkpoint-state extraction itself as the main bottleneck, not just
per-sequence scheduling.

So the updated recommendation is:

- keep the current `all` implementation as correctness-complete and
  experimentally usable
- keep `align` as the throughput recommendation for now
- treat fused/direct checkpoint-state emission as the next real optimization
  milestone for RWKV7 `all` mode

## 2026-04-15 Follow-up: default `align`, keep explicit `all`, and fuse checkpoint emission

The next iteration intentionally changed strategy:

- do not keep RWKV7 defaulted to `all`
- keep explicit `all` support available for correctness and continued
  optimization
- move the default back to `align`
- start reducing `all` overhead with a fused checkpoint-state emission path

### Why the default changed back to `align`

After the first `all`-mode serving validation, the functional result was good
but the throughput regression was still too large to justify keeping `all` as
the default runtime mode.

The config now behaves as follows:

- if the user enables prefix caching and does not explicitly request a mamba
  cache mode, RWKV7 defaults to `align`
- if the user explicitly requests `--mamba-cache-mode all`, RWKV7 keeps `all`

This means the compatibility work for `all` is not discarded, but the serving
default is placed back on the faster path until the remaining overhead is
reduced.

### Checkpoint-state emission work

To reduce the most obvious `all`-mode overhead, I added a checkpoint-capable
RWKV7 fused recurrent path.

Key implementation pieces:

- `vllm/model_executor/layers/fla/ops/rwkv7.py`
  - new checkpoint-capable fused/reference helper entry points
  - fused recurrent kernel path can now emit aligned recurrent checkpoint
    states directly for requested token positions
- `vllm/model_executor/models/rwkv7.py`
  - RWKV7 cache-all prefill can route through the checkpoint-capable fused op
  - packed prefill cache-all handling now builds checkpoint positions/counts
    and consumes the emitted block-boundary states
- `vllm/model_executor/models/config.py`
  - RWKV7 default prefix-cache mode is forced back to `align` unless the user
    explicitly requested another mode

This is not yet the final "direct write to cache slots with zero extra
materialization cost" design, but it is a concrete move away from the older
fully explicit checkpoint extraction flow.

### Validation

Local verification:

```bash
source ~/miniforge3/etc/profile.d/conda.sh
conda activate vllm-dev
cd /home/liu/vllm
python -m py_compile \
  vllm/model_executor/layers/fla/ops/rwkv7.py \
  vllm/model_executor/layers/fla/ops/__init__.py \
  vllm/model_executor/models/rwkv7.py \
  vllm/model_executor/models/config.py \
  tests/model_executor/test_rwkv7.py
python -m pytest -q tests/model_executor/test_rwkv7.py
```

Result:

- `20 passed, 2 skipped`

Additional coverage now includes:

- config defaults back to `align`
- explicit `all` is preserved
- checkpoint-state fused op matches the reference path

### Updated serving smoke

Repeated-prefix smoke was rerun on the local `0.4B` checkpoint under
`PIECEWISE` with prefix caching enabled:

```bash
python tmp_rwkv7_prefix_hit_bench.py \
  --model /mnt/d/codes/RWKV7-Goose-World2.9-0.4B-HF \
  --enable-prefix-caching \
  --mamba-cache-mode all \
  --cudagraph-mode piecewise \
  --concurrency 8 \
  --shared-prefix-len 1024 \
  --tail-len 128 \
  --max-tokens 64 \
  --rounds 1 \
  --warmup 0 \
  --log /tmp/vllm_rwkv7_prefix_hit_all_fused_20260415.log \
  > /tmp/rwkv7_prefix_hit_all_fused_20260415.json

python tmp_rwkv7_prefix_hit_bench.py \
  --model /mnt/d/codes/RWKV7-Goose-World2.9-0.4B-HF \
  --enable-prefix-caching \
  --cudagraph-mode piecewise \
  --concurrency 8 \
  --shared-prefix-len 1024 \
  --tail-len 128 \
  --max-tokens 64 \
  --rounds 1 \
  --warmup 0 \
  --log /tmp/vllm_rwkv7_prefix_hit_default_20260415.log \
  > /tmp/rwkv7_prefix_hit_default_20260415.json
```

Observed startup signals:

- explicit `all`
  - `Prefix caching in Mamba cache 'all' mode is currently enabled`
- default run
  - `Prefix caching in Mamba cache 'align' mode is currently enabled`

Benchmark summary:

| mode | hit ratio | avg aggregate TPS |
|---|---:|---:|
| `all` | `0.0` | `77.784` |
| `all` | `0.5` | `117.627` |
| `all` | `1.0` | `221.835` |
| default (`align`) | `0.0` | `119.735` |
| default (`align`) | `0.5` | `175.788` |
| default (`align`) | `1.0` | `253.456` |

All requests in both runs still matched the serial baseline.

### Updated interpretation

This follow-up changed the RWKV7 `all` story in an important way:

- `all` is no longer just "correct but drastically slower"
- the fused checkpoint-emission path improved `all` substantially
- compared with the previous `2026-04-14` run:
  - `hit=0.0`: `19.404 -> 77.784`
  - `hit=0.5`: `29.758 -> 117.627`
  - `hit=1.0`: `120.398 -> 221.835`

But the final conclusion remains:

- `align` is still the better serving default today
- explicit `all` is now in a much healthier experimental state
- the remaining optimization target is a lighter direct-write checkpoint-state
  path that removes the residual block-boundary extraction overhead

## 2026-04-15 Follow-up: recurrent-only direct write is not runtime-safe

After the fused checkpoint-emission path improved RWKV7 `all` substantially, I
tried the next obvious optimization: direct-writing recurrent checkpoint states
into the live cache slots during cache-all prefill.

### What worked

At the low-level op layer, the direct-write idea itself was not obviously
broken.

I extended the fused RWKV7 recurrent op so that it can:

- emit checkpoint states into a temporary output tensor, or
- write them directly into a caller-supplied cache tensor by slot id

Unit coverage was added for that path, and the direct-write helper worked in
isolated varlen op checks.

### What failed

When I wired recurrent-only direct write into the RWKV7 serving runtime and
reran the repeated-prefix smoke benchmark, service correctness regressed:

- requests no longer consistently matched the serial baseline

This did not look like a pure numerical instability problem in the fused
recurrent op. The more convincing explanation is runtime visibility.

### Root cause hypothesis

A RWKV7 cache slot is not a single tensor. It is the combination of:

- attn shift state
- recurrent state
- ffn shift state

The direct-write experiment only published the recurrent checkpoint state
early. The corresponding attn/ffn shift states were still written later by the
runtime.

That means a partially updated slot could become visible:

- recurrent state already new
- attn/ffn shift states still old

Under real serving concurrency, that is unsafe.

So this iteration established an important constraint:

- a future "direct write" optimization must publish RWKV7 checkpoint state
  atomically across all three slot components, or
- it must use an explicit staging/commit mechanism that prevents partially
  updated slots from being observed

### Safe runtime rollback and retained optimization

I removed the unsafe recurrent-only direct-write wiring from the RWKV7 serving
runtime, but kept two useful outcomes:

1. The low-level direct-write-capable fused op is still available for future
   work.
2. The runtime now avoids the previous large `torch.cat(...)` store assembly.

Instead of concatenating boundary states and final states into one large
temporary tensor bundle, the runtime now performs:

- one `_store_kv_states(...)` for boundary block states
- one `_store_kv_states(...)` for final output-slot states

This keeps cache-slot publication atomic at the runtime level while still
reducing some Python-side tensor assembly overhead.

### Validation

Local verification:

```bash
source ~/miniforge3/etc/profile.d/conda.sh
conda activate vllm-dev
cd /home/liu/vllm
python -m pytest -q tests/model_executor/test_rwkv7.py
```

Result:

- `21 passed, 2 skipped`

Serving smoke:

```bash
python tmp_rwkv7_prefix_hit_bench.py \
  --model /mnt/d/codes/RWKV7-Goose-World2.9-0.4B-HF \
  --enable-prefix-caching \
  --mamba-cache-mode all \
  --cudagraph-mode piecewise \
  --concurrency 8 \
  --shared-prefix-len 1024 \
  --tail-len 128 \
  --max-tokens 64 \
  --rounds 1 \
  --warmup 0 \
  --log /tmp/vllm_rwkv7_prefix_hit_all_nocat_20260415.log \
  > /tmp/rwkv7_prefix_hit_all_nocat_20260415.json
```

Benchmark summary (`all`, safe no-`cat` runtime path):

| hit ratio | avg aggregate TPS | all-match |
|---|---:|---|
| `0.0` | `116.669` | `true` |
| `0.5` | `120.881` | `true` |
| `1.0` | `231.013` | `true` |

Compared with the earlier `2026-04-15` safe baseline:

- `hit=0.0`: `77.784 -> 116.669`
- `hit=0.5`: `117.627 -> 120.881`
- `hit=1.0`: `221.835 -> 231.013`

### Updated conclusion

This iteration narrows the gap again, but it also changes the roadmap:

- the next useful optimization is not "direct write recurrent checkpoints"
- the next useful optimization is "atomically publish full RWKV7 cache slots"

Until that exists:

- `align` should remain the default
- `all` should keep the current safe runtime publication model
- future work should focus on atomic multi-state checkpoint publication rather
  than recurrent-only direct writes

### Align regression recheck

As a follow-up sanity check, I reran the explicit `align` repeated-prefix
benchmark after the failed dedup experiment had been fully rolled back.

Command:

```bash
python tmp_rwkv7_prefix_hit_bench.py \
  --model /mnt/d/codes/RWKV7-Goose-World2.9-0.4B-HF \
  --enable-prefix-caching \
  --mamba-cache-mode align \
  --cudagraph-mode piecewise \
  --concurrency 8 \
  --shared-prefix-len 1024 \
  --tail-len 128 \
  --max-tokens 64 \
  --rounds 1 \
  --warmup 0 \
  --log /tmp/vllm_rwkv7_prefix_hit_align_recheck_20260415.log \
  > /tmp/rwkv7_prefix_hit_align_recheck_20260415.json
```

Observed result:

| hit ratio | avg aggregate TPS | all-match |
|---|---:|---|
| `0.0` | `117.205` | `true` |
| `0.5` | `160.097` | `true` |
| `1.0` | `230.063` | `true` |

Compared with the earlier `2026-04-15` align baseline:

- `0.0`: `119.735 -> 117.205` (`-2.1%`)
- `0.5`: `175.788 -> 160.097` (`-8.9%`)
- `1.0`: `253.456 -> 230.063` (`-9.2%`)

Interpretation:

- no correctness regression is present in the current `align` path
- a mild single-run throughput dip is visible
- because the retained code changes in this phase are concentrated in
  `all`-mode plumbing, and because this was only a one-shot recheck, I do not
  consider this enough evidence to declare a confirmed `align` regression

If a stricter answer is needed later, the right follow-up is to rerun `align`
with multiple rounds and compare medians / spread instead of one-shot values.

### Conservative revert to the `be29a1808`-equivalent code state

After the above experiments, the branch decision was to keep the conservative
`align`-first runtime as the active code path and preserve the later `all`
work only in history. Concretely, I reverted `3218256c8` so the retained code
matches the earlier `be29a1808`-equivalent RWKV7 runtime behavior again, while
leaving the benchmark/report record intact.

Post-revert verification:

```bash
python -m pytest -q tests/model_executor/test_rwkv7.py
```

Result:

- `20 passed, 2 skipped`

Post-revert `align` serving smoke:

```bash
python tmp_rwkv7_prefix_hit_bench.py \
  --model /mnt/d/codes/RWKV7-Goose-World2.9-0.4B-HF \
  --enable-prefix-caching \
  --mamba-cache-mode align \
  --cudagraph-mode piecewise \
  --concurrency 8 \
  --shared-prefix-len 1024 \
  --tail-len 128 \
  --max-tokens 64 \
  --rounds 1 \
  --warmup 0 \
  --log /tmp/vllm_rwkv7_prefix_hit_align_postrevert_20260415.log \
  > /tmp/rwkv7_prefix_hit_align_postrevert_20260415.json
```

Observed result:

| hit ratio | avg aggregate TPS | all-match |
|---|---:|---|
| `0.0` | `115.775` | `true` |
| `0.5` | `165.857` | `true` |
| `1.0` | `221.106` | `true` |

Interpretation:

- the revert preserved correctness
- the post-revert `align` numbers stay in the same overall band as the earlier
  recheck
- current evidence still points to one-shot variance rather than a material
  regression caused by the now-removed `all` experiment code

### Multi-round `align` confirmation after the revert

To replace the earlier one-shot signal with a more stable answer, I reran the
same `align` repeated-prefix benchmark with `rounds=5, warmup=1`.

Command:

```bash
python tmp_rwkv7_prefix_hit_bench.py \
  --model /mnt/d/codes/RWKV7-Goose-World2.9-0.4B-HF \
  --enable-prefix-caching \
  --mamba-cache-mode align \
  --cudagraph-mode piecewise \
  --concurrency 8 \
  --shared-prefix-len 1024 \
  --tail-len 128 \
  --max-tokens 64 \
  --rounds 5 \
  --warmup 1 \
  --log /tmp/vllm_rwkv7_prefix_hit_align_rounds5_20260415.log \
  > /tmp/rwkv7_prefix_hit_align_rounds5_20260415.json
```

Observed result:

| hit ratio | aggregate TPS rounds | median aggregate TPS | all-match |
|---|---|---:|---|
| `0.0` | `121.232, 115.219, 121.002, 133.697, 144.498` | `121.232` | `true` |
| `0.5` | `162.771, 186.301, 171.810, 177.612, 174.989` | `174.989` | `true` |
| `1.0` | `241.200, 255.500, 248.332, 250.331, 242.496` | `248.332` | `true` |

Compared with the earlier high-water `align` baseline:

- `0.0`: `119.735 -> 121.232` (`+1.2%`)
- `0.5`: `175.788 -> 174.989` (`-0.5%`)
- `1.0`: `253.456 -> 248.332` (`-2.0%`)

Interpretation:

- the current reverted `align` code path remains in the same practical
  performance band as before
- this multi-round check is strong enough to stop treating the earlier
  one-shot dip as evidence of a real regression

### Upstream PR hygiene cleanup

Before opening an upstream PR, I removed a small amount of RWKV7-specific
development-only inspection code from `vllm/model_executor/models/rwkv7.py`.

Removed items:

- `debug_last_runtime_metadata_summary`
- `debug_last_forward_summary`
- `debug_last_store_stats`
- `RWKV7_DEBUG_*` environment-variable branches used only for local store-path
  debugging

Rationale:

- these helpers were useful while iterating locally, but they are not part of
  the model behavior or the intended upstream debugging surface
- the local `tmp_rwkv7_*` documents already preserve the experimentation and
  measurement history that motivated them

Verification:

```bash
python -m py_compile vllm/model_executor/models/rwkv7.py
```

Result:

- compile check passed

### PR-worktree concurrency smoke

To de-risk the eventual upstream PR, I validated the cleaned PR worktree
directly instead of relying only on the long-lived local development branch.

Setup:

- worktree: `/home/liu/vllm-rwkv7`
- branch: `codex/rwkv7`
- model: `RWKV7-Goose-World2.8-0.1B-HF`
- engine path: `AsyncLLM.from_engine_args`
- mode: `enforce_eager=True`
- workload: `8` concurrent requests, `32` output tokens each

Observed result:

- `8 / 8` requests finished successfully
- wall time: `1.218 s`
- aggregate output TPS: `210.125`
- average latency: `0.929 s`
- p95 latency: `0.965 s`

Conclusion:

- the upstream-main-based RWKV7 PR worktree still supports normal concurrent
  generation on the model path itself
- this is a stronger PR-readiness signal than a pure import or unit-test check

Two local environment blockers surfaced while trying to do the same check
through the API server:

1. `piecewise` startup currently depends on a newer local `_C` extension than
   the one available in this validation environment
2. the OpenAI API server currently trips over a separate
   `mistral_common.NamedToolChoice` import mismatch in the local package set

These blockers should be treated as local validation-environment issues rather
than as direct evidence of a RWKV7 regression.

### Fresh-install PR validation

I repeated PR validation using a fresh-install path on top of a clean conda
 environment (`vllm-rwkv7`) plus a repo-local `.venv`, which is much closer to
the expected upstream-review workflow than the earlier reused local dev setup.

Installation:

- created `.venv` against `/home/liu/miniforge3/envs/vllm-rwkv7/bin/python`
- installed the PR worktree with:
  `VLLM_USE_PRECOMPILED=1 uv pip install --python .venv/bin/python -e . --torch-backend=auto`
- one large transitive wheel (`flashinfer-cubin`) needed a retry because of a
  transient TLS download failure

Validation outcomes:

1. `tests/model_executor/test_rwkv7.py`
   - `20 passed, 2 skipped`
2. OpenAI API server with `-cc.cudagraph_mode=piecewise`
   - startup succeeded
   - `/health` returned `200`
3. `piecewise` closed-loop concurrency smoke (`32` requests, `c=8`,
   `max_tokens=32`)
   - aggregate TPS: `277.239`
   - weighted request TPS: `34.668`
   - success: `32 / 32`
4. eager comparison smoke with the same workload
   - aggregate TPS: `398.361`
   - weighted request TPS: `49.822`
   - success: `32 / 32`

Technical conclusion:

- the branch is now validated not only at the unit-test level but also at the
  service level for both eager and `piecewise` cudagraph paths
- this closes the earlier local-environment blockers that had prevented
  confident claims about `compile + piecewise` readiness
- the remaining caveat is performance, not functionality:
  compile/`piecewise` is working, but it is still slower than eager on this
  small-model, short-output smoke workload

### PIECEWISE empty-graph warning follow-up

I traced the recurring startup warning
`The CUDA Graph is empty. This usually means that the graph was attempted to be
captured on wrong device or stream.` through the current vLLM cudagraph path.

Findings:

- the warning appears during piecewise wrapper capture, but it does not prevent
  later successful capture and replay setup
- the same startup logs still report:
  - `CompilationMode.VLLM_COMPILE`
  - `CUDAGraphMode.PIECEWISE`
  - successful `Capturing CUDA graphs ...`
  - successful `Graph capturing finished ...`
- this strongly suggests that the warning comes from empty piecewise
  partitions, i.e. partitions whose FX body reduces to view/alias/metadata-only
  work and therefore launches no CUDA kernel

Mitigation added on the dev branch:

- `vllm/compilation/cuda_graph.py` now records warnings raised during
  `torch.cuda.graph(...)` capture
- it suppresses only the known empty-graph warning when the wrapper runtime
  mode is `PIECEWISE`
- any other warning is re-emitted unchanged
- the same empty-graph warning is still re-emitted for `FULL` mode

Validation:

1. New targeted unit tests:
   - `tests/compile/test_cuda_graph.py`
   - result: `3 passed`
2. Targeted lint:
   - `pre-commit run ruff-check --files vllm/compilation/cuda_graph.py tests/compile/test_cuda_graph.py`
   - `pre-commit run ruff-format --files vllm/compilation/cuda_graph.py tests/compile/test_cuda_graph.py`
3. Real RWKV7 `piecewise` startup smoke after the change:
   - model: `RWKV7-Goose-World2.8-0.1B-HF`
   - startup succeeded
   - compile and `piecewise` cudagraph still initialized
   - the empty-graph warning no longer appeared in the captured startup log
