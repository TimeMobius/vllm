# RWKV7 vLLM Handoff

## Current Status

- Branch: `codex/rwkv7-adapter-align`
- Latest code checkpoint: `d573675fa` (`Enable non-eager RWKV7 compile path`)
- Current service status:
  - No `vllm serve` process is running now.
  - No test ports are currently listening.
  - The latest non-eager startup probe on port `8041` did not reach `/health`.
  - It was terminated by the outer probe script after a long cold-start compile phase.

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
- Decode-path batching across concurrent requests is implemented and validated.
- Concurrent decode requests no longer serialize one-by-one inside `RWKV7Block`.

### Tests already in place

- Unit and integration coverage in [test_rwkv7.py](/home/liu/vllm/tests/model_executor/test_rwkv7.py):
  - block forward
  - static forward context registration
  - cache/state update behavior
  - batched decode equivalence
  - state copy function types
  - runtime state dtype
  - reference parity for full forward
  - reference parity for prefill + decode
  - config behavior for non-eager path

## Stable Baseline

The current known-good serving baseline is still the eager path before the non-eager experiment.

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
- CUDA graphs in `PIECEWISE` mode

### What was changed

- Added `@support_torch_compile` to `RWKV7Model`.
- Removed RWKV7's unconditional eager requirement.
- Added a RWKV7-specific preference to force:
  - `cudagraph_mode=PIECEWISE`
  - `cudagraph_copy_inputs=True`
- Added a post-optimization-level hook because vLLM optimization defaults were overwriting the earlier RWKV7-specific cudagraph choice.

### What is confirmed

From the latest probe log [vllm_rwkv7_8041_probe.log](/tmp/vllm_rwkv7_8041_probe.log):

- `enforce_eager=False` is really taking effect.
- The old warning "`torch.compile` is turned on, but the model does not support it" is gone.
- Runtime `cudagraph_mode` is really `PIECEWISE`.
- `torch.compile` actually starts compiling RWKV7.

From the eager control run [vllm_rwkv7_8044_eager.log](/tmp/vllm_rwkv7_8044_eager.log):

- `--enforce-eager` starts successfully in about `25s`.
- `/health` comes up.
- `/v1/completions` returns normally.
- This confirms the model/runtime logic itself is fine.
- The main blocker is the compile path, not generic serving or model initialization.

### What is still not confirmed

- Service readiness under the non-eager path
- One-shot vs step-by-step correctness under the non-eager path
- TPS under the non-eager path

### Current blocker

Cold-start compile cost is still too high.

Observed in the latest probe:

- model load: about `11s`
- `Dynamo bytecode transform time`: about `126.55s`

The service did not reach `/health` before the probe timeout window and was later terminated. So this path is not yet ready for normal use.

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

### 8. WSL-specific performance warnings are expected noise

Observed:

- `pin_memory=False` warning under WSL
- slow tokenizer warning

Handling:

- These are real but not the core correctness issue.
- Keep them in mind when interpreting latency.

## Current TODO List

### Highest priority

1. Make the non-eager path actually reach ready state.
2. Avoid letting Dynamo trace RWKV7 prefill token-by-token Python recurrence.
3. Do not rely on `torch.compiler.disable` inside the fullgraph-compiled path.

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
- [rwkv7_bench_64_after_batch.json](/tmp/rwkv7_bench_64_after_batch.json)
- [rwkv7_bench_128_after_batch.json](/tmp/rwkv7_bench_128_after_batch.json)

## Recommended Next Action

Continue from commit `d573675fa`.

The next concrete experiment should be:

1. keep the eager baseline as the known-good correctness/perf control
2. prototype a compile-friendly replacement for the RWKV7 prefill recurrence loop
3. or move compile support to a smaller decode-only subgraph instead of full-model compile
