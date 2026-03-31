# RWKV7 vLLM Handoff

## Current Status

- Branch: `codex/rwkv7-adapter-align`
- Latest committed checkpoint before this round: `a5bbd9b7976b97f6a7810c631c2ead979d6c635c`
- Current service status:
  - No `vllm serve` process is running now.
  - No test ports are currently listening.
  - The stable default path remains eager when CUDA graphs are enabled.
  - The real compile path now works when `cudagraph_mode=none`.
  - PIECEWISE CUDA graph capture is still experimental and should not be treated as correct yet.

## Latest Update (2026-03-31)

- Compile/no-cudagraph has now been validated through the real RWKV7 entrypoints:
  - `RWKV7Block.forward()` gained a whole-block custom-op boundary via `torch.ops.vllm.rwkv7_block_forward(...)`
  - the block-level metadata/stateful dispatch moved into `RWKV7Block._forward_runtime(...)`
  - layer-local `kv_cache` and runner-level `kv_caches` now receive real writeback under compile/no-cg
- Root cause is now understood more precisely:
  - the earlier statement "compile path globally receives `attn_metadata=None`" was too strong
  - runtime `forward_context.attn_metadata` did exist
  - the real bug was that compiled `RWKV7Block.forward()` did not execute the metadata-aware cache load/store path during request runtime
  - moving the stateful boundary to a whole-block custom op fixed that integration bug
- RWKV7 config policy is now narrower and safer:
  - when `cudagraph_mode != none`, RWKV7 still falls back to eager by default
  - when `cudagraph_mode=none`, `RWKV7ForCausalLMConfig` now allows the real compile path without monkeypatching config classes
- The engine-step probe is now a real-entrypoint probe:
  - [`tmp_rwkv7_engine_first_step_compare.py`](/home/liu/vllm/tmp_rwkv7_engine_first_step_compare.py) no longer monkeypatches RWKV7 config
- The service compare tool is now reusable for local checkpoints:
  - [`tmp_rwkv7_compare.py`](/home/liu/vllm/tmp_rwkv7_compare.py) now supports `--model` and `--compile-no-cg`
- Fresh compile/no-cg validation on `/mnt/d/codes/RWKV7-Goose-World2.9-0.4B-HF`:
  - engine replay artifacts:
    - [rwkv7_engine_step_2_base_real_compile.json](/tmp/rwkv7_engine_step_2_base_real_compile.json)
    - [rwkv7_engine_step_2_replay_real_compile.json](/tmp/rwkv7_engine_step_2_replay_real_compile.json)
    - [rwkv7_engine_step_2_compare_real_compile.json](/tmp/rwkv7_engine_step_2_compare_real_compile.json)
  - result:
    - prompt `北京是`
    - base generated tokens: `[10250, 10283]`
    - replay prompt token ids: `[10902, 10362, 13091, 10250]`
    - replay generated token: `10283`
    - second-step replay matches
- Real `vllm serve` correctness is now also back on the compile/no-cg path:
  - log: [vllm_rwkv7_compare_real_compile_no_cg.log](/tmp/vllm_rwkv7_compare_real_compile_no_cg.log)
  - command shape:
    - `python tmp_rwkv7_compare.py --model /mnt/d/codes/RWKV7-Goose-World2.9-0.4B-HF --compile-no-cg --no-async-scheduling`
  - result:
    - `i am`: one-shot == step-by-step
    - `北京是`: one-shot == step-by-step
    - `The capital of France is`: one-shot == step-by-step
- Current remaining blocker:
  - compile correctness under `cudagraph_mode=none` is no longer the main problem
  - the next unresolved path is PIECEWISE CUDA graph capture
- New pitfall to remember:
  - nested-shell JSON quoting for `-cc '{"cudagraph_mode":"none"}'` is easy to break
  - prefer either:
    - `-cc.cudagraph_mode=none`
    - or [`tmp_rwkv7_compare.py`](/home/liu/vllm/tmp_rwkv7_compare.py) with `--compile-no-cg`
- Historical debugging trail below this point predates the whole-block custom-op fix and is kept mainly as chronology.

- Landed a first KDA-style compile boundary in [`rwkv7.py`](/home/liu/vllm/vllm/model_executor/models/rwkv7.py):
  - registered `RWKV7Attention` in `static_forward_context`
  - added `torch.ops.vllm.rwkv7_attention(...)`
  - moved the existing recurrent math behind `RWKV7Attention._forward(...)`
- Added unit coverage in [test_rwkv7.py](/home/liu/vllm/tests/model_executor/test_rwkv7.py) for:
  - attention registration in `static_forward_context`
  - custom-op wrapper parity against direct `_forward(...)`
- This change materially improved compile startup:
  - previous bad case: Dynamo tracing the RWKV7 prefill token loop for about `126.55s`
  - new case: `Dynamo bytecode transform time` about `1.74s`
  - compile range `(1, 2048)` build time about `8.89s`
  - total `torch.compile` time about `10.92s`
- Compile without CUDA graphs is now able to boot:
  - with `-cc '{"cudagraph_mode":"none"}'`
  - `/health` reached in about `56s`
  - `/v1/completions` returned successfully
  - reference log: [vllm_rwkv7_8048_compile_no_cg.log](/tmp/vllm_rwkv7_8048_compile_no_cg.log)
- However, correctness is still not acceptable under the non-eager path:
  - on `RWKV7-Goose-World2.9-0.4B-HF`
  - `one-shot max_tokens=8` and `step-by-step max_tokens=1` diverged for all three prompts
    - `i am`
    - `北京是`
    - `The capital of France is`
  - reference log: [vllm_rwkv7_non_eager_compare_8050.log](/tmp/vllm_rwkv7_non_eager_compare_8050.log)
- CUDA graphs remain a separate blocker:
  - PIECEWISE capture progressed past compile and warmup
  - but engine startup still failed during CUDA graph capture around `7/51`
- Because non-eager correctness is still failing, the model config was restored to the stable eager default.
- Added a reusable engine-step probe:
  - [`tmp_rwkv7_engine_first_step_compare.py`](/home/liu/vllm/tmp_rwkv7_engine_first_step_compare.py)
  - supports:
    - single-run capture via `--max-tokens` + `--capture-generated-tokens`
    - token-id-controlled replay via `--prompt-token-ids`
    - replay prompt construction via `--append-generated-prefix-from-run-json`
    - offline comparison via `--run-json-a` and `--run-json-b`
- New first-step result on `RWKV7-Goose-World2.9-0.4B-HF`:
  - prompt: `北京是`
  - mode: compile enabled, `cudagraph_mode=none`, `async_scheduling=False`
  - `max_tokens=1` and `max_tokens=8` produced the same first generated token:
    - token id `10250`
    - text `一`
  - current probe also reports no layer-level state fingerprint difference on the first step
  - reference artifacts:
    - [rwkv7_engine_step_1.json](/tmp/rwkv7_engine_step_1.json)
    - [rwkv7_engine_step_8.json](/tmp/rwkv7_engine_step_8.json)
    - [rwkv7_engine_step_compare_beijing_sync.json](/tmp/rwkv7_engine_step_compare_beijing_sync.json)
- Caveat on that state result:
  - the layer-local `model.model.layers[*].kv_cache` fingerprint remains all-zero in this probe
  - so the useful strong conclusion is:
    - first generated token matches
    - the non-eager mismatch likely occurs after the first decode step, or in a deeper cache path not exposed by those layer-local tensors
- New second-step replay result on the local 0.4B checkpoint:
  - model path: `/mnt/d/codes/RWKV7-Goose-World2.9-0.4B-HF`
  - prompt: `北京是`
  - mode: compile enabled, `cudagraph_mode=none`, `async_scheduling=False`
  - base run:
    - `max_tokens=2`
    - `capture_generated_tokens=2`
    - prompt token ids: `[10902, 10362, 13091]`
    - generated token ids: `[10250, 10283]`
    - generated text: `一个`
  - controlled replay run:
    - prompt token ids: `[10902, 10362, 13091, 10250]`
    - `max_tokens=1`
    - `capture_generated_tokens=1`
    - generated token id: `10283`
    - generated text: `个`
  - conclusion:
    - the second decode token also matches under token-id-controlled replay
    - the current compile mismatch is therefore not on the first or second decode step for this prompt
  - reference artifacts:
    - [rwkv7_engine_step_2_base.json](/tmp/rwkv7_engine_step_2_base.json)
    - [rwkv7_engine_step_2_replay.json](/tmp/rwkv7_engine_step_2_replay.json)
    - [rwkv7_engine_step_2_compare.json](/tmp/rwkv7_engine_step_2_compare.json)
- New root-cause finding from fresh compile debug:
  - when compile cache is disabled via `VLLM_DISABLE_COMPILE_CACHE=1`
  - and the probe captures `RWKV7Block.debug_last_forward_summary`
  - the non-eager compile path still shows:
    - `attn_metadata_is_none=1`
    - `num_decode_tokens=-1`
    - `num_prefill_tokens=-1`
  - this happens even on the first unfinished request step for prompt `北京是`
  - therefore `RWKV7Block.forward()` is taking the `attn_metadata is None` fallback path under compile
  - and `_store_kv_state()` / `_store_kv_states()` never run
  - as a consequence:
    - layer-local `kv_cache` remains all-zero
    - runner-owned `kv_caches` remains all-zero
    - first token can still match, because prefill math runs within the same forward
    - later decode steps diverge because the recurrent state was never committed into cache
  - reference artifact:
    - [rwkv7_engine_step_1_final_repro.json](/tmp/rwkv7_engine_step_1_final_repro.json)
  - the current compile correctness bug is therefore no longer “unknown later-step divergence”
  - it is specifically a metadata/stateful-path integration bug in the RWKV7 compile path

## What Is Already Working

### Correctness and cache integration

- RWKV7 is registered as a vLLM model and loads from local HF checkpoints.
- `RWKV7Block` is registered in `static_forward_context`, so v1 engine treats it as a stateful cached layer.
- RWKV7 three-state cache semantics are wired correctly:
  - `attn_shift`: conv copy semantics
  - `recurrent`: temporal copy semantics
  - `ffn_shift`: conv copy semantics
- The `HybridKVCacheCoordinator` path was fixed for RWKV-style models where multiple raw state groups coalesce into one effective attention group.
- `compute_logits()` handles dtype alignment safely.
- `load_weights()` maps `model.embeddings.weight -> model.embed_tokens.weight`.
- Runtime state and model path currently use correctness-first `fp32`.

### Service behavior

- `one-shot max_tokens=N` and `step-by-step max_tokens=1` were validated to match on RWKV7 `0.1B` and `0.4B` under the stable eager baseline.
- `one-shot max_tokens=8` and `step-by-step max_tokens=1` now also match on RWKV7 `0.4B` under the real compile/no-cg service path for:
  - `i am`
  - `北京是`
  - `The capital of France is`
- Decode-path batching across concurrent requests is implemented and validated.
- Concurrent decode requests no longer serialize one-by-one inside `RWKV7Block`.

### Tests already in place

- Unit and integration coverage in [test_rwkv7.py](/home/liu/vllm/tests/model_executor/test_rwkv7.py):
  - block forward
  - static forward context registration
  - attention custom-op wrapper parity
  - cache/state update behavior
  - batched decode equivalence
  - state copy function types
  - runtime state dtype
  - reference parity for full forward
  - reference parity for prefill + decode
  - config behavior for the default eager fallback

## Stable Baseline

The conservative known-good serving baseline is still the eager path.

There is now a second known-good opt-in path for RWKV7:

- compile enabled
- `cudagraph_mode=none`
- `async_scheduling=False`

What is known-good there:

- cache/state correctness: yes
- one-shot vs step-by-step parity: yes
- concurrent decode batching: yes
- `0.4B` single-request TPS on RTX 3050 6GB: about `32 TPS`
- `0.4B` concurrent 8 total TPS: about `207 TPS`

This baseline is functionally correct, but slower than expected for a `0.4B` model.

## Non-Eager Experiment Status

### Goal

Reduce framework overhead by allowing RWKV7 to use:

- `enforce_eager=False`
- `torch.compile`
- CUDA graphs when possible

### What was changed

- Added `@support_torch_compile` to `RWKV7Model`.
- Added a first KDA-style custom-op boundary around `RWKV7Attention`.
- Added a post-optimization-level hook because vLLM optimization defaults were overwriting the earlier RWKV7-specific cudagraph choice.

### What is confirmed

From the eager control run [vllm_rwkv7_8044_eager.log](/tmp/vllm_rwkv7_8044_eager.log):

- `--enforce-eager` starts successfully in about `25s`.
- `/health` comes up.
- `/v1/completions` returns normally.
- This confirms the model/runtime logic itself is fine.
- The main blocker is the compile path, not generic serving or model initialization.

From the non-eager no-cudagraph run:

- `torch.compile` starts and completes.
- `Dynamo bytecode transform time` is now small enough to be practical.
- service readiness is possible when CUDA graphs are disabled.
- correctness is now restored for the validated `0.4B` prompts under the real service path.

### What is still not confirmed

- compile-path TPS after correctness is restored
- whether CUDA graphs can be safely re-enabled after more refactor/kernel work
- broader prompt/model sweep beyond the current `0.4B` validation set

### Current blocker

Compile startup is no longer the main blocker.

The current blockers are:

- PIECEWISE CUDA graph capture still fails during engine initialization
- compile/no-cg performance has not yet been benchmarked after correctness recovery
- the next implementation step should move from metadata/state correctness to CUDA-graph compatibility

### Additional findings from deeper debug

From the debug probe logs [vllm_rwkv7_8045_debug.log](/tmp/vllm_rwkv7_8045_debug.log):

- WSL memory was not the primary blocker in the debug run.
- Peak memory pressure did not force swap use during the captured non-eager debug attempt.
- The real hot path is Dynamo tracing inside:
  - [`RWKV7Attention.forward`](/home/liu/vllm/vllm/model_executor/models/rwkv7.py#L335)
  - specifically the time-step loop over sequence length around:
    - [`rwkv7.py`](/home/liu/vllm/vllm/model_executor/models/rwkv7.py#L414)
    - [`rwkv7.py`](/home/liu/vllm/vllm/model_executor/models/rwkv7.py#L423)
- The trace log shows Dynamo repeatedly expanding the recurrent update body for each token in the prefill sequence.

An attempted mitigation was tested:

- temporarily marking `RWKV7Attention.forward()` with `@torch.compiler.disable`

Result:

- this does not work with vLLM's current compile wrapper
- because vLLM uses `torch.compile(..., fullgraph=True)`
- non-eager startup then fails fast with:
  - `torch._dynamo.exc.Unsupported: Skip inlining torch.compiler.disable()d function`

This means the simple "graph break this function" approach is not viable for the current RWKV7 compile path.

Later follow-up showed that the custom-op boundary solved the worst Dynamo trace issue, but did not by itself guarantee correctness under compile.

## Main Pitfalls Already Encountered

### 1. Windows workspace path was misleading

Problem:

- The desktop thread started in `D:\codes\vllm`, which is not the real repo checkout being modified.

Handling:

- Always work against `/home/liu/vllm` inside WSL.
- Prefer `bash -lc 'cd /home/liu/vllm && ...'`.

### 2. WSL multiprocessing spawn breaks stdin-style Python launch

Problem:

- Creating `LLM(...)` or similar engine processes from `python - <<'PY'` can fail under WSL spawn mode with:
  - `FileNotFoundError: /home/liu/vllm/<stdin>`

Handling:

- Use real script files or `vllm serve` from a proper shell process.
- Avoid stdin-based engine bootstrap when multiprocessing spawn is involved.

### 3. PowerShell was rewriting WSL shell commands

Problem:

- Background process commands with quoting, redirection, `&&`, or `$!` were being mangled by PowerShell before reaching WSL.

Handling:

- Use `wsl.exe --% bash -lc "..."` when calling into WSL from this desktop thread.
- For longer flows, write a temporary shell script under `/tmp` and run that script.

### 4. Local RWKV7 tokenizer/model requires remote code trust

Problem:

- Startup can fail during tokenizer load without `trust_remote_code=True`.

Handling:

- For local serving probes, pass `--trust-remote-code`.

### 5. RWKV7 config changes were being overwritten later by vLLM defaults

Problem:

- The RWKV7 config hook successfully set `cudagraph_mode=PIECEWISE`, but vLLM optimization defaults later reset it to `FULL_AND_PIECEWISE`.

Handling:

- Added a post-optimization-level hook and applied the RWKV7-specific cudagraph preference again after defaults are applied.

Current state:

- this hook still exists as part of the experiment history
- but the default runtime was moved back to eager after non-eager correctness failed

### 6. RWKV7 was not recognized as torch.compile-capable

Problem:

- vLLM warned that `torch.compile` was enabled but the model did not support it.

Handling:

- Added `@support_torch_compile` to `RWKV7Model`.

### 7. FULL decode graph mode was unsafe for RWKV7

Problem:

- Earlier non-eager experiments with full decode graph behavior hit device-side asserts around dynamic state index selection.

Handling:

- Restrict RWKV7 to `PIECEWISE` CUDA graph mode for now.
- Keep `cudagraph_copy_inputs=True`.

Later finding:

- even PIECEWISE capture is not yet stable
- and non-eager correctness still fails before CUDA graphs are safe to revisit

### 8. WSL-specific performance warnings are expected noise

Observed:

- `pin_memory=False` warning under WSL
- slow tokenizer warning

Handling:

- These are real but not the core correctness issue.
- Keep them in mind when interpreting latency.

### 9. In-proc engine introspection requires forcing true single-process mode

Problem:

- `LLMEngine.from_engine_args(..., enable_multiprocessing=False)` is not enough if `VLLM_ENABLE_V1_MULTIPROCESSING` is still true in the environment.
- In that case `engine.model_executor` is unavailable and internal inspection fails.

Handling:

- set `VLLM_ENABLE_V1_MULTIPROCESSING=0` before importing `LLMEngine`
- use `engine.engine_core.shutdown()` rather than a nonexistent `engine.shutdown()`

### 10. Re-initializing multiple in-proc engines in one Python process is unreliable here

Problem:

- on this WSL + RTX 3050 setup, launching a second in-proc engine in the same Python process could still fail at startup with very low free GPU memory, even after explicit cleanup

Handling:

- run per-step probes as separate Python processes
- save each run to disk and compare them offline

### 11. Use local checkpoint paths for the 0.4B probe

Problem:

- using `RWKV7-Goose-World2.9-0.4B-HF` as a model id can fall back to Hugging Face resolution and fail with `401` / repo-not-found in this environment

Handling:

- use local paths instead:
  - `/mnt/d/codes/RWKV7-Goose-World2.8-0.1B-HF`
  - `/mnt/d/codes/RWKV7-Goose-World2.9-0.4B-HF`

### 12. vLLM compile cache can hide model-code changes during debug

Problem:

- the torch.compile cache can be reused even after local RWKV7 model-code edits
- this can make a new probe look like it exercised fresh instrumentation when it actually loaded an older compiled artifact

Handling:

- for compile-path debugging, run with:
  - `VLLM_DISABLE_COMPILE_CACHE=1`
- only trust instrumentation results after confirming the log says:
  - `vLLM's torch.compile cache is disabled.`

### 13. RWKV7 compile bug is currently “metadata missing”, not “backing cache mismatch”

Problem:

- earlier probe results only showed that layer-local and runner-level cache fingerprints stayed zero
- that left two possibilities:
  - cache writeback happened into some other backing store
  - or the metadata-driven stateful path never ran

Handling:

- a fresh no-cache compile probe with layer-local forward/store summaries showed:
  - `RWKV7Block.forward()` received `attn_metadata=None`
  - `_store_kv_state()` and `_store_kv_states()` were never reached
- this means the current bug is upstream of cache writeback:
  - the compiled RWKV7 block is not seeing live attention metadata
  - so it always falls back to the stateless sequence path

## Current TODO List

### Highest priority

1. Replace the current RWKV7 compile boundary so the block sees live `LinearAttentionMetadata` under non-eager execution.
2. Re-run the first unfinished-step probe with `VLLM_DISABLE_COMPILE_CACHE=1` and confirm:
   - `attn_metadata_is_none=0`
   - `_store_kv_state()` / `_store_kv_states()` run
   - runner-level cache summaries become non-zero
3. Only after that, return to later-step replay / divergent-prompt checks.
4. Keep the default runtime on eager until the compile path is both correct and measurable.

### After readiness is achieved

1. Re-run `/health` and a single `/v1/completions` smoke test.
2. Re-run one-shot vs step-by-step correctness on:
   - `i am`
   - `北京是`
   - `The capital of France is`
3. Re-run throughput benchmarks:
   - single request TPS
   - concurrent 8 total TPS
4. Compare eager baseline vs non-eager path on:
   - cold start time
   - warm start time
   - correctness
   - TPS

### Medium priority

1. Investigate whether compile should be limited to decode-critical subgraphs.
2. Decide whether compile support should move from `RWKV7Model` to a smaller decode-only submodule.
3. Decide whether the non-eager path should become the default or remain experimental.
4. Re-check whether `fp32` can be partially relaxed once the execution path stabilizes.

### Long-term performance work

1. High-performance prefill batching for RWKV7
2. Fused kernel route for RWKV7 recurrent update
3. More aggressive CUDA graph or compile optimization if safe

## Recommended Workflow For Future RWKV7 Iteration

### Step 1. Keep checkpoints small and isolated

- Make one idea per commit.
- Do not mix correctness fixes, performance changes, and documentation in the same commit unless they are inseparable.

### Step 2. Validate the local invariant first

For model code changes, run:

```bash
source ~/miniforge3/etc/profile.d/conda.sh
conda activate vllm-dev
cd /home/liu/vllm
python -m pytest -q tests/model_executor/test_rwkv7.py
```

### Step 3. Validate service correctness before TPS

Always test:

- one-shot `max_tokens=N`
- step-by-step `max_tokens=1`
- same prompts
- deterministic decoding

If these differ, do not trust any benchmark numbers yet.

### Step 4. Only benchmark after correctness passes

Recommended order:

1. single-request latency/TPS
2. concurrent decode TPS
3. longer-output benchmarks
4. GPU utilization sampling

### Step 5. For compile/cudagraph work, inspect logs first

Before calling a result successful, verify in logs:

- `enforce_eager=False`
- no "`model does not support torch.compile`" warning
- expected `cudagraph_mode`
- whether `/health` actually comes up

### Step 6. Preserve probe artifacts

Useful files to keep inspecting:

- [vllm_rwkv7_8041_probe.log](/tmp/vllm_rwkv7_8041_probe.log)
- [vllm_rwkv7_8044_eager.log](/tmp/vllm_rwkv7_8044_eager.log)
- [vllm_rwkv7_8045_debug.log](/tmp/vllm_rwkv7_8045_debug.log)
- [vllm_rwkv7_8046_probe.log](/tmp/vllm_rwkv7_8046_probe.log)
- [vllm_rwkv7_8037.log](/tmp/vllm_rwkv7_8037.log)
- [rwkv7_engine_step_1.json](/tmp/rwkv7_engine_step_1.json)
- [rwkv7_engine_step_8.json](/tmp/rwkv7_engine_step_8.json)
- [rwkv7_engine_step_compare_beijing_sync.json](/tmp/rwkv7_engine_step_compare_beijing_sync.json)
- [rwkv7_engine_step_2_base.json](/tmp/rwkv7_engine_step_2_base.json)
- [rwkv7_engine_step_2_replay.json](/tmp/rwkv7_engine_step_2_replay.json)
- [rwkv7_engine_step_2_compare.json](/tmp/rwkv7_engine_step_2_compare.json)
- [rwkv7_bench_64_after_batch.json](/tmp/rwkv7_bench_64_after_batch.json)
- [rwkv7_bench_128_after_batch.json](/tmp/rwkv7_bench_128_after_batch.json)

## Recommended Next Action

Continue from commit `c15f30216`.

The next concrete experiment should be:

1. keep the eager baseline as the known-good correctness/perf control
2. move RWKV7 metadata/stateful dispatch out of the current fullgraph-sensitive path
3. use `VLLM_DISABLE_COMPILE_CACHE=1` and rerun [rwkv7_engine_step_1_final_repro.json](/tmp/rwkv7_engine_step_1_final_repro.json)-style probes until `attn_metadata_is_none` flips to `0`
4. only then resume later-step replay and end-to-end divergence checks
