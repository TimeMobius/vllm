# RWKV7 vLLM Handoff

## Current Status

- Branch: `codex/rwkv7-adapter-align`
- Latest committed checkpoint before this round: `f94fd358bedcc9eaa9cd9d7c72a4a295b2d553ff`
- Current service status:
  - No `vllm serve` process is running now.
  - No test ports are currently listening.
  - RWKV7 compile now works on both:
    - default PIECEWISE CUDA graphs
    - `cudagraph_mode=none`
  - No experimental environment variable is required anymore.
  - The main remaining work is performance/cleanup, not basic compile correctness.
  - Benchmark results are now tracked separately in:
    - [tmp_rwkv7_benchmark_records.md](/home/liu/vllm/tmp_rwkv7_benchmark_records.md)

## Latest Update (2026-04-13)

- Added a dedicated benchmark ledger:
  - [tmp_rwkv7_benchmark_records.md](/home/liu/vllm/tmp_rwkv7_benchmark_records.md)
  - it stores:
    - run metadata
    - prompt-set naming
    - per-concurrency throughput rows
    - raw JSON/log paths
- Re-ran the eager throughput baseline on the local `0.4B` checkpoint:
  - model:
    - `/mnt/d/codes/RWKV7-Goose-World2.9-0.4B-HF`
  - script:
    - [tmp_rwkv7_long_benchmark.py](/home/liu/vllm/tmp_rwkv7_long_benchmark.py)
  - run id:
    - `2026-04-13_eager_0p4b_mt64`
  - config:
    - `--enforce-eager`
    - `max_tokens=64`
    - `rounds=2`
    - `warmup=1`
    - concurrency `1/2/4/8`
  - raw artifacts:
    - [rwkv7_bench_0p4b_eager_64_20260413.json](/tmp/rwkv7_bench_0p4b_eager_64_20260413.json)
    - [vllm_rwkv7_eager_bench_20260413.log](/tmp/vllm_rwkv7_eager_bench_20260413.log)
  - aggregate TPS:
    - `1`: `35.047 / 34.912`, avg `34.980`
    - `2`: `68.610 / 71.110`, avg `69.860`
    - `4`: `132.553 / 132.926`, avg `132.739`
    - `8`: `215.131 / 214.844`, avg `214.987`
  - correctness note:
    - every concurrent round still matched the serial baseline
  - interpretation:
    - this is now the fresh eager reference point for the next round of
      `PIECEWISE` / `compile_no_cg` comparison

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
- RWKV7 config policy is now fully reopened:
  - `RWKV7ForCausalLMConfig` no longer forces eager as a default fallback
  - real compile is now allowed for both default PIECEWISE and `cudagraph_mode=none`
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
  - the old compile/no-cg correctness blocker is resolved
  - the old PIECEWISE CUDA graph blocker is also resolved
  - next work is around performance and broader coverage
- Final CUDA-graph capture fix:
  - temporary RWKV7 store-debug stats used `.item()` on CUDA tensors during cache writeback
  - this caused `CUDA error: operation not permitted when stream is capturing`
  - detailed tensor stats are now only collected when `RWKV7_DEBUG_STORE_STATS=1`
    and the stream is not being captured
- Final PIECEWISE correctness fix:
  - `vllm::rwkv7_block_forward` was added to the default `splitting_ops` set in
    [compilation.py](/home/liu/vllm/vllm/config/compilation.py)
  - this keeps the RWKV7 stateful block boundary out of a single frozen piecewise capture region
  - after that change, real `vllm serve` PIECEWISE one-shot vs step-by-step parity recovered
- Final real-entrypoint validation logs:
  - PIECEWISE:
    - [vllm_rwkv7_piecewise_final.log](/tmp/vllm_rwkv7_piecewise_final.log)
    - [vllm_rwkv7_piecewise_0p1b_final.log](/tmp/vllm_rwkv7_piecewise_0p1b_final.log)
  - compile/no-cg:
    - [vllm_rwkv7_compile_no_cg_final.log](/tmp/vllm_rwkv7_compile_no_cg_final.log)
    - [vllm_rwkv7_compile_no_cg_0p1b_final.log](/tmp/vllm_rwkv7_compile_no_cg_0p1b_final.log)
  - the `0.4B` and `0.1B` local checkpoints both show:
    - `i am`: one-shot == step-by-step
    - `北京是`: one-shot == step-by-step
    - `The capital of France is`: one-shot == step-by-step
- New benchmark finding from the local `0.4B` checkpoint:
  - [`tmp_rwkv7_long_benchmark.py`](/home/liu/vllm/tmp_rwkv7_long_benchmark.py) now supports:
    - `--enforce-eager`
    - `--cudagraph-mode`
    - `--compile-no-cg`
    - `--disable-compile-cache`
  - the earlier quick conclusion that explicit `PIECEWISE` clearly beat eager has
    been superseded by the later clean reruns below
  - `cudagraph_mode=none` is still mainly a correctness/debug path, not the fast path
- New default compile-mode pitfall:
  - inheriting plain `MambaModelConfig` defaults left RWKV7 on `FULL_AND_PIECEWISE`
  - that mode still hit the old unsafe full decode-graph failure under benchmark load:
    - `indexSelectSmallIndex ... Assertion srcIndex < srcSelectDimSize failed`
    - followed by `CUDA error: device-side assert triggered`
  - RWKV7 now overrides that default back to `PIECEWISE` in
    [config.py](/home/liu/vllm/vllm/model_executor/models/config.py)
- Current benchmark snapshot on `/mnt/d/codes/RWKV7-Goose-World2.9-0.4B-HF`
  supersedes the earlier quick throughput snapshot:
  - short-output mixed-prompt clean rerun:
    - eager 64: [rwkv7_bench_0p4b_eager_64.json](/tmp/rwkv7_bench_0p4b_eager_64.json)
    - piecewise 64: [rwkv7_bench_0p4b_piecewise_64.json](/tmp/rwkv7_bench_0p4b_piecewise_64.json)
    - eager 128: [rwkv7_bench_0p4b_eager_128.json](/tmp/rwkv7_bench_0p4b_eager_128.json)
    - piecewise 128: [rwkv7_bench_0p4b_piecewise_128.json](/tmp/rwkv7_bench_0p4b_piecewise_128.json)
    - `max_tokens=64`, aggregate TPS, concurrency `1/2/4/8`:
      - eager: `28.122 / 54.669 / 104.712 / 191.801`
      - piecewise: `27.566 / 54.985 / 103.009 / 186.164`
    - `max_tokens=128`, aggregate TPS, concurrency `1/2/4/8`:
      - eager: `27.642 / 48.780 / 110.584 / 203.157`
      - piecewise: `27.856 / 45.786 / 103.951 / 193.628`
  - long-input exact-token rerun:
    - script: [/tmp/rwkv7_exact_long_input_bench.py](/tmp/rwkv7_exact_long_input_bench.py)
    - eager 1024 + 64 decode: [rwkv7_long_input_eager_1024.json](/tmp/rwkv7_long_input_eager_1024.json)
    - piecewise 1024 + 64 decode: [rwkv7_long_input_piecewise_1024.json](/tmp/rwkv7_long_input_piecewise_1024.json)
    - eager 1984 + 64 decode: [rwkv7_long_input_eager_1984.json](/tmp/rwkv7_long_input_eager_1984.json)
    - piecewise 1984 + 64 decode: [rwkv7_long_input_piecewise_1984.json](/tmp/rwkv7_long_input_piecewise_1984.json)
    - prompt length `1024`, aggregate TPS, concurrency `1/4/8`:
      - eager: `11.830 / 14.974 / 15.850`
      - piecewise: `10.146 / 13.967 / 16.376`
    - prompt length `1984`, aggregate TPS, concurrency `1/4/8`:
      - eager: `7.237 / 9.431 / 10.046`
      - piecewise: `7.863 / 9.449 / 9.529`
  - interpretation:
    - on the local `0.4B` checkpoint, `PIECEWISE` and eager are now in the same
      runtime band rather than showing a stable clear win for `PIECEWISE`
    - short-output mixed-prompt runs keep `PIECEWISE` close to eager, but not
      consistently faster
    - long-input prefill-heavy runs also stay close:
      - `1024`-token prompt: `PIECEWISE` trails at concurrency `1/4`, edges ahead at `8`
      - `1984`-token prompt: `PIECEWISE` leads slightly at `1`, ties at `4`, trails at `8`
    - current value of `compile + PIECEWISE` is therefore:
      - correctness-capable compile serving
      - default-safe compile mode for RWKV7
      - but not yet a stable throughput win over eager on this machine
  - caveat:
    - `no-cg` remains mainly a correctness/debug path and was not part of this
      consolidated rerun
- Cold-start tradeoff is now clear:
  - eager init engine: about `9.2s`
  - no-cg init engine: about `13.8s`
  - piecewise init engine: about `94-97s`
  - so piecewise currently buys compile availability and correctness at the cost
    of much slower startup
- New pitfall to remember:
  - nested-shell JSON quoting for `-cc '{"cudagraph_mode":"none"}'` is easy to break
  - prefer either:
    - `-cc.cudagraph_mode=none`
    - or [`tmp_rwkv7_compare.py`](/home/liu/vllm/tmp_rwkv7_compare.py) / [`tmp_rwkv7_long_benchmark.py`](/home/liu/vllm/tmp_rwkv7_long_benchmark.py)
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
- `one-shot max_tokens=8` and `step-by-step max_tokens=1` also now match on RWKV7 `0.4B` under the default PIECEWISE CUDA-graph service path for:
  - `i am`
  - `北京是`
  - `The capital of France is`
- `one-shot max_tokens=8` and `step-by-step max_tokens=1` also now match on RWKV7 `0.1B` under both:
  - default PIECEWISE CUDA graphs
  - compile with `cudagraph_mode=none`
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
  - config behavior for compile-enable defaults

## Stable Baseline

There are now three known-good serving baselines for RWKV7:

- eager
- default PIECEWISE CUDA graphs
- compile with `cudagraph_mode=none`
- all validated with `async_scheduling=False`
- compile correctness has been revalidated on local:
  - `RWKV7-Goose-World2.8-0.1B-HF`
  - `RWKV7-Goose-World2.9-0.4B-HF`

What is known-good there:

- cache/state correctness: yes
- one-shot vs step-by-step parity: yes
- concurrent decode batching: yes
- `0.4B` single-request TPS on RTX 3050 6GB: about `32 TPS`
- `0.4B` concurrent 8 total TPS: about `207 TPS`

These baselines are functionally correct. The remaining question is mainly
performance headroom, especially for prefill and longer-run compile throughput.

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
- broader prompt/model sweep beyond the current `0.1B` / `0.4B` validation set
- more stress coverage on concurrency / prefix caching

### Current blocker

Compile startup is no longer the main blocker.

The current blockers are:

- no known basic compile correctness blocker is open on the validated short-prompt path
- FULL decode-graph behavior is still unsafe for RWKV7 and should stay disabled
- `cudagraph_mode=none` still shows a longer-output / high-concurrency mismatch (`128` tokens, concurrency `8`)
- the next implementation step should move from "can it run" to:
  - piecewise performance characterization
  - no-cg long-output concurrency debugging

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

- this was part of the earlier experiment history
- current RWKV7 config no longer forces eager fallback
- default PIECEWISE and `cudagraph_mode=none` have both been validated again

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

- the crash was not a fundamental PIECEWISE limitation
- the final fixes were:
  - keep debug `.item()` stats out of capture
  - add `vllm::rwkv7_block_forward` to default `splitting_ops`
- current validated path is:
  - default PIECEWISE CUDA graphs
  - `cudagraph_copy_inputs` not required for the passing regression runs

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
- this isolated the first compile correctness bug as upstream of cache writeback:
  - the compiled RWKV7 block was not seeing live attention metadata
  - so it always fell back to the stateless sequence path
- later fixes:
  - move block-local runtime dispatch behind `torch.ops.vllm.rwkv7_block_forward(...)`
  - add `vllm::rwkv7_block_forward` to `splitting_ops`
- current state:
  - compile/no-cg correctness is restored
  - default PIECEWISE correctness is also restored

## Current TODO List

### Highest priority

1. Benchmark compile throughput now that correctness is restored:
   - default PIECEWISE
   - `cudagraph_mode=none`
   - compare against eager
2. Re-run concurrency correctness and throughput on the restored compile path:
   - concurrent 3
   - concurrent 8
3. Extend service validation beyond the current prompt set if needed:
   - longer outputs
   - prefix caching
   - mixed prompt lengths
4. Decide whether any parts of the current `fp32` correctness-first policy can be relaxed safely.

### After correctness recovery

1. Re-run `/health` and a single `/v1/completions` smoke test.
2. Re-run one-shot vs step-by-step correctness on:
   - `i am`
   - `北京是`
   - `The capital of France is`
3. Re-run throughput benchmarks:
   - single request TPS
   - concurrent 8 total TPS
4. Compare eager baseline vs compile path on:
   - cold start time
   - warm start time
   - correctness
   - TPS

### Medium priority

1. Investigate whether compile should be limited to decode-critical subgraphs.
2. Decide whether compile support should move from `RWKV7Model` to a smaller decode-only submodule.
3. Decide whether the compile-enabled path should remain conservative or become the default recommendation for RWKV7.
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
