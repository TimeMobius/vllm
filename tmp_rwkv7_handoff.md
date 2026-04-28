# RWKV7 vLLM Handoff

## Current Status

- Branch: `codex/rwkv7-adapter-align`
- Latest committed checkpoint before this round: `6d2f95a75`
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

## Workspace Execution Note

- Current Codex shell entrypoint is Windows PowerShell, while this repo and
  `.venv/bin/python` live inside WSL.
- Do **not** run Linux virtualenv executables directly from PowerShell like:
    - `.venv/bin/python -m pytest ...`
    - `.venv/bin/python -m pre_commit ...`
- Doing so may trigger a Windows "how do you want to open this file?" dialog,
  because PowerShell is trying to open a Linux ELF path instead of executing it.
- Use WSL explicitly for all repo-local Python / pre-commit commands:
    - `bash -lc '.venv/bin/python -m pytest ...'`
    - `bash -lc '.venv/bin/python -m pre_commit run ...'`
    - `bash -lc '.venv/bin/python - <<\"PY\" ... PY'`
- Same rule for other Linux-only executables inside the repo:
    - prefer `bash -lc '...'`
    - or `wsl bash -lc '...'`
- Do not inject Linux `export PATH=...:$PATH` snippets directly through a
  PowerShell-wrapped command string.
    - In this Codex setup, `$PATH` can be expanded on the wrong side and pull
      in Windows paths with spaces / parentheses.
    - That can silently contaminate benchmark startup and produce misleading
      failures such as Triton `gcc: cannot execute 'as'`.
- Preferred benchmark launch pattern:
    - write a small bash script
    - set an explicit clean Linux path inside it, for example:
        - `export PATH=/home/liu/vllm/.venv/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin`
    - then run `.venv/bin/python tmp_rwkv7_ttft_benchmark.py ...`
- Another easy mistake in this PowerShell + WSL setup:
    - do not leave raw `|` outside quotes in a repo-root command string
    - one accidental benchmark helper command produced a junk file named:
        - `32|prefill`
    - if a one-off filename like that appears, inspect it first, then remove it
      before committing

## Latest Update (2026-04-28, CMix Probe)

- Added an experimental RWKV7 CMix activation path behind:
    - `RWKV7_USE_FUSED_CMIX=1`
- Scope of the current landing:
    - new generic `_C` op:
        - `relu2`
    - [vllm/model_executor/layers/activation.py](/home/liu/vllm/vllm/model_executor/layers/activation.py)
      now routes `ReLUSquaredActivation` through CUDA instead of native
      `torch.relu(x).square()`
    - [vllm/model_executor/models/rwkv7.py](/home/liu/vllm/vllm/model_executor/models/rwkv7.py)
      now uses that CUDA `sqrelu` path only when:
        - `RWKV7_USE_FUSED_CMIX=1`
        - activation is `sqrelu`
    - note:
        - this is **not** a full official `_CmixLayerV2Fn` port yet
        - `_mix_ffn_inputs()` still uses the existing `addcmul` fallback
- Focused tests that passed:
    - `tests/kernels/core/test_activation.py::test_activation -v`
        - result:
            - `72 passed`
    - `tests/model_executor/test_rwkv7.py -k fused_cmix_activation -v`
        - result:
            - `2 passed`
- Local microbenchmark verdict:
    - generic `relu2` op on `0.4B`-style `intermediate_size=4096` shapes:
        - mostly flat
        - typical range was roughly:
            - `0.86x ~ 1.08x`
    - direct `RWKV7FeedForward._apply_ffn()` microbench:
        - tokens `64`: `1.00x`
        - tokens `256`: `1.12x`
        - tokens `1024`: `0.99x`
        - tokens `4096`: `1.00x`
    - interpretation:
        - the activation-only slice is too small to produce a strong,
          stable hook-level win by itself
- Real `0.4B` isolated serial benchmark verdict:
    - model:
        - `/mnt/d/codes/RWKV7-Goose-World2.9-0.4B-HF`
    - clean rerun artifacts:
        - baseline JSON:
            - `/tmp/rwkv7_cmix_baseline_2.json`
        - fused JSON:
            - `/tmp/rwkv7_cmix_fused_2.json`
    - recorded A/B summary:
        - prefill proxy:
            - `64`: `54.491ms -> 49.507ms` (`+9.15%`)
            - `256`: `51.221ms -> 54.318ms` (`-6.05%`)
            - `1024`: `143.342ms -> 141.674ms` (`+1.16%`)
            - `1984`: `259.712ms -> 259.664ms` (flat)
        - decode `64 -> 32`:
            - TTFT `86.556ms -> 78.211ms` (`+9.64%`)
            - latency `1049.604ms -> 1074.779ms` (`-2.40%`)
            - TPOT `31.301ms -> 32.147ms` (`-2.70%`)
        - decode `64 -> 64`:
            - TTFT `92.038ms -> 75.254ms` (`+18.24%`)
            - latency `2391.228ms -> 2068.131ms` (`+13.51%`)
            - TPOT `36.495ms -> 31.633ms` (`+13.32%`)
    - interpretation:
        - this path can help decode-heavy cases
        - but the benefit is much noisier and less universal than:
            - `mix6`
            - `kk-pre`
            - fused attention epilogue
        - keep it behind the feature flag
        - if CMix is revisited, it should be as a **larger-region FFN fuse**,
          not just CUDA `sqrelu`

## Latest Update (2026-04-28)

- Added an experimental alternate RWKV7 recurrent CUDA path behind:
    - `RWKV7_USE_ALT_RECURRENT_KERNEL=1`
- Scope of the current landing:
    - new `_C` op:
        - `rwkv7_alt_recurrent`
    - only routed from:
        - `_run_recurrent_sequence()`
        - `_run_recurrent_decode_batch()`
    - varlen / checkpoint-state paths still use existing Triton recurrent
- Focused tests that passed:
    - `tests/model_executor/test_rwkv7.py -k "alt_recurrent or fused_recurrent_matches_reference or checkpoint_states_match_reference" -v`
    - result:
        - `4 passed`
- Important build note:
    - full editable reinstall was noisy / expensive for this iteration
    - a reliable local bring-up path was:
        - configure a dedicated manual build dir
        - build only target `_C`
        - install component `_C` back into the repo prefix
    - commands used:
        - `cmake -G Ninja -DCMAKE_MAKE_PROGRAM=/home/liu/vllm/.venv/bin/ninja -DCMAKE_BUILD_TYPE=RelWithDebInfo -DVLLM_TARGET_DEVICE=cuda -DVLLM_PYTHON_EXECUTABLE=/home/liu/vllm/.venv/bin/python -DFETCHCONTENT_BASE_DIR=/home/liu/vllm/.deps -DNVCC_THREADS=1 -DCMAKE_JOB_POOL_COMPILE:STRING=compile -DCMAKE_JOB_POOLS:STRING=compile=1 -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc ..`
        - `cmake --build . -j=1 --target=_C`
        - `cmake --install . --prefix /home/liu/vllm --component _C`
- Local microbenchmark verdict on representative `0.4B`-style shapes:
    - `B=1,T=1,H=16,K=64,V=64`:
        - current Triton `0.0341ms`
        - alt CUDA `0.0179ms`
        - alt `+47.35%`
    - `B=1,T=256,H=16,K=64,V=64`:
        - current Triton `0.3609ms`
        - alt CUDA `0.2501ms`
        - alt `+30.71%`
- Real `0.4B` isolated serial benchmark verdict:
    - model:
        - `/mnt/d/codes/RWKV7-Goose-World2.9-0.4B-HF`
    - focused decode-heavy rerun:
        - prompt `64`
        - output `256`
        - rounds `4`
    - baseline:
        - TTFT `93.020ms`
        - latency `9807.160ms`
        - TPOT `38.095ms`
    - `RWKV7_USE_ALT_RECURRENT_KERNEL=1`:
        - TTFT `82.514ms`
        - latency `9456.780ms`
        - TPOT `36.762ms`
    - interpretation:
        - end-to-end gain exists, but is only modest on local `0.4B`
        - keep the flag off by default
        - validate on larger checkpoints / longer decode before promoting it

## Latest Update (2026-04-27)

- RWKV7 perf flags landed and were benchmarked in isolation:
    - `RWKV7_USE_FUSED_MIX6`
    - `RWKV7_USE_FUSED_KK_PRE`
- Earlier failed TTFT startup was traced to benchmark launch contamination,
  not Triton / vLLM itself:
    - direct `vllm serve` works
    - isolated `spawn` + Triton `CudaUtils()` works
    - the bad path was a PowerShell-mediated command that polluted Linux
      `PATH`
- Clean serial TTFT benchmark recipe on local `0.4B`:
    - model:
        - `/mnt/d/codes/RWKV7-Goose-World2.9-0.4B-HF`
    - script:
        - [tmp_rwkv7_ttft_benchmark.py](/home/liu/vllm/tmp_rwkv7_ttft_benchmark.py)
    - config:
        - `--enforce-eager`
        - `--gpu-memory-utilization 0.8`
        - `--rounds 2`
        - `--warmup 1`
        - prompt lengths: `64 / 1024 / 1984`
        - decode: prompt `64`, output `32 / 64`
- Clean serial benchmark summary:
    - baseline:
        - ready: `36.045s`
        - prefill TTFT proxy:
            - `64`: `63.519ms`
            - `1024`: `180.642ms`
            - `1984`: `269.706ms`
        - decode:
            - `32 tok`: TTFT `105.311ms`, TPOT `38.565ms`
            - `64 tok`: TTFT `90.251ms`, TPOT `36.891ms`
    - `mix6` only:
        - ready: `32.041s`
        - prefill TTFT proxy:
            - `64`: `46.759ms`
            - `1024`: `188.143ms`
            - `1984`: `262.678ms`
        - decode:
            - `32 tok`: TTFT `83.037ms`, TPOT `34.156ms`
            - `64 tok`: TTFT `85.123ms`, TPOT `34.285ms`
    - `kk-pre` only:
        - ready: `44.054s`
        - prefill TTFT proxy:
            - `64`: `61.461ms`
            - `1024`: `155.154ms`
            - `1984`: `255.132ms`
        - decode:
            - `32 tok`: TTFT `76.515ms`, TPOT `30.689ms`
            - `64 tok`: TTFT `79.815ms`, TPOT `34.709ms`
    - `mix6 + kk-pre`:
        - ready: `32.034s`
        - prefill TTFT proxy:
            - `64`: `49.576ms`
            - `1024`: `171.582ms`
            - `1984`: `248.333ms`
        - decode:
            - `32 tok`: TTFT `81.696ms`, TPOT `32.168ms`
            - `64 tok`: TTFT `76.448ms`, TPOT `33.565ms`
- Interpretation:
    - `mix6` helps short prompt TTFT and decode latency
    - `kk-pre` helps decode most clearly and also improves longer prompt TTFT
    - combined flags remain net positive on the clean serial run
    - next perf item can move forward to fused `CMix`, but keep the same
      isolated benchmark discipline

## Latest Update (2026-04-13)

- Upgraded the remote serving benchmark utility:
    - [tmp_rwkv7_remote_concurrency_bench.py](/home/liu/vllm/tmp_rwkv7_remote_concurrency_bench.py)
    - new capabilities:
        - `--dispatch-mode burst`
            - launches all requests immediately so queueing moves from the client
        worker pool to the remote vLLM service
        - explicit `token_throughput_tps`
            - now also exported as flat aliases:
                - `token_throughput_tps_avg`
                - `token_throughput_tps_min`
                - `token_throughput_tps_max`
        - per-request `request_token_throughput_tps`
            - stored on every row in `requests.jsonl`
            - summary stats include `avg / p50 / p95 / min / max`
            - `weighted_avg` is also exported to reduce confusion from plain
        arithmetic averaging under heavy long-tail latency
        - `token_throughput_tps_stats`
            - 1-second bucketed `min / avg / max` token TPS within the active window
        - `active_output_tps`
            - measured from first request start to last request finish
            - helps separate client-side submission queue time from server busy time
        - `worker_count`
        - `peak_inflight_requests`
        - `avg_inflight_requests`
        - `configured_concurrency` is now only meaningful in `closed_loop`
      mode; in `burst` mode it will be `null`
    - intended usage:
        - `closed_loop`:
            - fixed client concurrency benchmark
        - `burst`:
            - better for finding the remote service saturation point

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
- Added a dedicated TTFT / prefill-heavy benchmark tool:
    - [tmp_rwkv7_ttft_benchmark.py](/home/liu/vllm/tmp_rwkv7_ttft_benchmark.py)
    - this script records:
        - `server_ready_sec`
        - long-prompt streaming TTFT proxy
        - fixed-prompt decode latency / ITL / TPOT
    - important limitation:
        - vLLM rejects `max_tokens=0`
        - so the script uses streaming `max_tokens=1` TTFT as a prefill-heavy proxy,
      not true zero-decode prefill-only latency
- Re-ran an eager TTFT baseline on the local `0.4B` checkpoint:
    - run id:
        - `2026-04-13_eager_0p4b_ttft`
    - raw artifacts:
        - [rwkv7_ttft_0p4b_eager_20260413.json](/tmp/rwkv7_ttft_0p4b_eager_20260413.json)
        - [vllm_rwkv7_ttft_eager_20260413.log](/tmp/vllm_rwkv7_ttft_eager_20260413.log)
    - server ready:
        - `30.034s`
    - prefill-heavy TTFT proxy, `max_tokens=1`, avg TTFT:
        - prompt len `64`: `188.986ms`
        - prompt len `1024`: `2708.602ms`
        - prompt len `1984`: `5409.289ms`
    - decode profile, prompt len `64`:
        - `max_tokens=32`:
            - avg TTFT `255.053ms`
            - avg latency `1118.779ms`
            - avg TPOT / ITL `27.862ms`
        - `max_tokens=64`:
            - avg TTFT `255.353ms`
            - avg latency `2061.966ms`
            - avg TPOT / ITL `28.676ms`
    - interpretation:
        - the eager path now shows a very strong prompt-length sensitivity on the
      first token
        - while short-prompt decode stays near `28ms/token`
        - so the current next compile comparison should focus on whether
      `PIECEWISE` actually reduces the long-prompt first-token cost, not just
      aggregate throughput
- Added the first compile TTFT comparison on the local `0.4B` checkpoint:
    - raw artifacts:
        - compile/no-cg:
            - [rwkv7_ttft_0p4b_compile_no_cg_20260413.json](/tmp/rwkv7_ttft_0p4b_compile_no_cg_20260413.json)
            - [vllm_rwkv7_ttft_compile_no_cg_20260413.log](/tmp/vllm_rwkv7_ttft_compile_no_cg_20260413.log)
        - piecewise:
            - [rwkv7_ttft_0p4b_piecewise_20260413.json](/tmp/rwkv7_ttft_0p4b_piecewise_20260413.json)
            - [vllm_rwkv7_ttft_piecewise_20260413.log](/tmp/vllm_rwkv7_ttft_piecewise_20260413.log)
    - server ready:
        - eager: `30.034s`
        - compile/no-cg: `38.044s`
        - piecewise: `108.093s`
    - prefill-heavy TTFT proxy, prompt len `64 / 1024 / 1984`:
        - eager:
            - `188.986 / 2708.602 / 5409.289 ms`
        - compile/no-cg:
            - `219.612 / 2915.364 / 5432.906 ms`
        - piecewise:
            - `229.968 / 2660.877 / 4930.740 ms`
    - decode profile, prompt len `64`, avg ITL:
        - eager:
            - `32 tok`: `27.862ms`
            - `64 tok`: `28.676ms`
        - compile/no-cg:
            - `32 tok`: `29.558ms`
            - `64 tok`: `30.429ms`
        - piecewise:
            - `32 tok`: `27.716ms`
            - `64 tok`: `28.024ms`
    - interpretation:
        - `compile/no-cg` 仍然更像 correctness/debug 路径：
            - 启动更慢
            - TTFT 没明显改善
            - decode ITL 还略差
        - `PIECEWISE` 对短 prompt 没有优势
        - 但对长 prompt 首 token 已经开始出现改善：
            - `1024` token prompt 略优于 eager
            - `1984` token prompt 改善更明显
        - `PIECEWISE` 的 decode ITL 目前基本和 eager 同量级
        - 所以下一步最值当的方向是：
            - 继续盯长 prefill TTFT
            - 而不是继续投资 `compile/no-cg`

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

### 14. Fused RWKV7 prefill kernel is now wired into the sequence path

Problem:

- RWKV7 prefill still spent most of its time in the Python token loop inside
  `RWKV7Attention._forward()`
- that path limited both eager long-prefill latency and the upside of
  `PIECEWISE`, because compile still had to route prefill through a slow
  per-token recurrence in Python

Handling:

- added a dedicated RWKV7 fused recurrent op at
  [vllm/model_executor/layers/fla/ops/rwkv7.py](/home/liu/vllm/vllm/model_executor/layers/fla/ops/rwkv7.py)
- exported it through
  [vllm/model_executor/layers/fla/ops/**init**.py](/home/liu/vllm/vllm/model_executor/layers/fla/ops/__init__.py)
- switched the sequence-prefill branch of
  [RWKV7Attention._forward()](/home/liu/vllm/vllm/model_executor/models/rwkv7.py:550)
  from the Python token loop to `fused_mul_recurrent_rwkv7(...)`
- kept the old Python recurrence as an env-guarded fallback:
    - `RWKV7_DISABLE_FUSED_PREFILL=1`

Validation:

- unit tests:
    - `python -m pytest -q tests/model_executor/test_rwkv7.py`
    - result: `11 passed, 2 skipped`
- service correctness:
    - `python tmp_rwkv7_compare.py --model /mnt/d/codes/RWKV7-Goose-World2.9-0.4B-HF --disable-compile-cache`
    - result: one-shot vs step-by-step still matches on:
        - `i am`
        - `北京是`
        - `The capital of France is`
- op parity:
    - `tests/model_executor/test_rwkv7.py` now compares the fused kernel against a
    reference recurrence on CUDA

Artifacts:

- [rwkv7_ttft_0p4b_eager_fusedoff_r3w2_20260413.json](/tmp/rwkv7_ttft_0p4b_eager_fusedoff_r3w2_20260413.json)
- [rwkv7_ttft_0p4b_eager_fusedon_r3w2_20260413.json](/tmp/rwkv7_ttft_0p4b_eager_fusedon_r3w2_20260413.json)
- [rwkv7_ttft_0p4b_piecewise_fusedon_r3w2_20260413.json](/tmp/rwkv7_ttft_0p4b_piecewise_fusedon_r3w2_20260413.json)
- [vllm_rwkv7_compare_piecewise_fused_20260413.log](/tmp/vllm_rwkv7_compare_piecewise_fused_20260413.log)
- [vllm_rwkv7_ttft_eager_fusedoff_r3w2_20260413.log](/tmp/vllm_rwkv7_ttft_eager_fusedoff_r3w2_20260413.log)
- [vllm_rwkv7_ttft_eager_fusedon_r3w2_20260413.log](/tmp/vllm_rwkv7_ttft_eager_fusedon_r3w2_20260413.log)
- [vllm_rwkv7_ttft_piecewise_fusedon_r3w2_20260413.log](/tmp/vllm_rwkv7_ttft_piecewise_fusedon_r3w2_20260413.log)

Current interpretation:

- eager fused-on vs fused-off:
    - prompt `1024`: TTFT proxy `2991.876ms -> 522.067ms`
    - prompt `1984`: TTFT proxy `5031.042ms -> 1069.782ms`
    - decode ITL: `27.3ms -> 33~34ms`, so prefill gains came with a small decode-side regression
- piecewise fused-on:
    - prompt `1024`: `276.974ms`
    - prompt `1984`: `530.588ms`
    - decode ITL returned to `27.6~27.8ms`, close to the old eager band
- the earlier eager post-fused seconds-long ITL spikes were not reproduced once
  warmup was increased to `2`; those look more like one-time warmup noise than a
  persistent regression

### 15. `.venv`-first validation was attempted, but the local reproducible path is still the prepared conda env

Problem:

- repo instructions prefer `uv` + `.venv/bin/python`
- the local `.venv` was missing core test/runtime dependencies
- hydrating it fully hit repeated PyPI timeouts while downloading transitive test
  or build dependencies

Handling:

- successfully installed a small subset into `.venv`:
    - `pytest`
    - `aiohttp`
    - `tblib`
- but:
    - editable `vllm` install timed out while fetching `cmake`
    - full `requirements/test.txt` timed out while fetching `pyogrio`
- so the practical validation path remains the already-prepared user env:
    - `source ~/miniforge3/etc/profile.d/conda.sh`
    - `conda activate vllm-dev`

Recommendation:

- continue using `vllm-dev` for end-to-end RWKV7 work unless/until `.venv`
  hydration is cached locally
- if a future PR workflow needs strict `.venv` parity, retry after warming the
  package cache or mirroring the missing wheels

### 16. Packed/varlen prefill is now wired through `RWKV7Block._forward_runtime()`

Problem:

- after landing the fused recurrent op, prefill was only faster inside a single
  sequence
- `RWKV7Block._forward_runtime()` still looped over each prefill request in
  Python:
    - sliced `query_start_loc` one request at a time
    - loaded/stored KV state one request at a time
    - never exercised the fused op's existing `cu_seqlens` support

Handling:

- added a varlen shift helper:
    - `token_shift_with_cache_varlen(...)`
- added packed-prefill entrypoints:
    - `RWKV7Attention.forward_prefill_batch(...)`
    - `RWKV7FeedForward.forward_prefill_batch(...)`
- added `RWKV7Block._run_prefill_batch(...)`
- switched the prefill branch of
  [RWKV7Block._forward_runtime()](/home/liu/vllm/vllm/model_executor/models/rwkv7.py)
  to:
    - slice all prefill tokens as one flat range
    - build per-request `cu_seqlens` from `query_start_loc`
    - mask out nonexistent initial states using `seq_lens > query_lens`
    - batch-load KV state with `_get_kv_states(...)`
    - batch-store final state with `_store_kv_states(...)`
- kept the old per-request loop behind the existing env switch:
    - `RWKV7_DISABLE_FUSED_PREFILL=1`

Validation:

- unit tests:
    - `python -m pytest -q tests/model_executor/test_rwkv7.py`
    - result: `12 passed, 2 skipped`
- new coverage:
    - `test_rwkv7_block_batches_prefill_tokens_without_changing_results`
- service correctness:
    - `python tmp_rwkv7_compare.py --model /mnt/d/codes/RWKV7-Goose-World2.9-0.4B-HF --disable-compile-cache`
    - one-shot vs step-by-step still matches on:
        - `i am`
        - `北京是`
        - `The capital of France is`
- service batching smoke:
    - `tmp_rwkv7_long_benchmark.py --cudagraph-mode piecewise --disable-compile-cache --max-tokens 16 --concurrency-levels 4 8`
    - both `4` and `8` concurrency matched the serial baseline

Artifacts:

- [rwkv7_long_piecewise_packedprefill_20260413.json](/tmp/rwkv7_long_piecewise_packedprefill_20260413.json)
- [vllm_rwkv7_long_piecewise_packedprefill_20260413.log](/tmp/vllm_rwkv7_long_piecewise_packedprefill_20260413.log)
- [vllm_rwkv7_compare_piecewise_packedprefill_20260413.log](/tmp/vllm_rwkv7_compare_piecewise_packedprefill_20260413.log)

Current interpretation:

- the Python per-prefill-request loop is no longer on the main RWKV7 runtime
  path when fused prefill is enabled
- single-request long-prefill latency was already fixed by the fused recurrent
  op; this change is about letting concurrent prefills share that path
- the short-prompt concurrency smoke is a correctness check, not proof of a
  throughput win, because it is not prefill-heavy enough to isolate the gain
- the next missing measurement is a true long-prompt concurrent sweep to
  quantify the packed-prefill payoff

### 17. The "still slower" impression mostly came from benchmark mismatch and first-run instability

Problem:

- the first packed-prefill smoke used:
    - `max_tokens=16`
    - short mixed prompts
    - one round only
- that was compared informally against the earlier eager baseline which used:
    - `max_tokens=64`
    - a different scenario mix
- a later one-shot exact long-input sweep also produced an anomalously bad
  `PIECEWISE` row for:
    - prompt `1024`
    - concurrency `8`
    - aggregate TPS `13.387`

Handling:

- added a reusable exact-token benchmark:
    - [tmp_rwkv7_exact_long_input_bench.py](/home/liu/vllm/tmp_rwkv7_exact_long_input_bench.py)
- reran eager and `PIECEWISE` on exact token lengths:
    - prompt `1024`
    - prompt `1984`
    - concurrency `1/4/8`
    - `max_tokens=64`
- then did focused reruns on the suspicious rows:
    - `1024`, concurrency `8`, `rounds=2`
    - `1984`, concurrency `8`

Key findings:

- focused steady-state `1024 + 64`, concurrency `8`:
    - eager: `131.458 / 124.108` TPS
    - piecewise: `120.680 / 123.058` TPS
- focused `1984 + 64`, concurrency `8`:
    - eager: `14.594` TPS
    - piecewise: `80.053` TPS

Interpretation:

- `PIECEWISE` is not uniformly "still slower" after packed-prefill
- on moderately long prompts (`1024`) it is now roughly in the same band as eager
- on very long prompts (`1984`) it is materially faster, which is exactly where
  packed prefill should help most
- the old `13.387` TPS row is not a good steady-state estimate; the focused rerun
  shows the same scenario near `121-123` TPS
- the short `max_tokens=16` smoke also should not be compared directly to the old
  `max_tokens=64` throughput baseline

Why performance is not a clean win everywhere:

- packed prefill only improves the prefill-heavy side of the runtime
- the decode hot path is still the older tensor implementation:
    - [RWKV7FeedForward.forward_decode_batch()](/home/liu/vllm/vllm/model_executor/models/rwkv7.py:404)
    - [RWKV7Attention.forward_decode_batch()](/home/liu/vllm/vllm/model_executor/models/rwkv7.py:748)
- so once prompt length is not extreme enough to dominate the request,
  `PIECEWISE` only gets part of the total request time back
- exact benchmark `aggregate_tps` is also defined as `completion_tokens / wall_time`,
  so it penalizes prefill time hard and is very sensitive to first-run setup costs

### 18. Fused decode recurrent backend is now in place

Implementation:

- `RWKV7Attention.forward_decode_batch()` now runs its recurrent update through
  `fused_mul_recurrent_rwkv7(...)` on CUDA instead of the older explicit tensor
  recurrence
- recurrent input projection and output finalization are now shared between:
    - `_forward(...)`
    - `forward_prefill_batch(...)`
    - `forward_decode_batch(...)`
- the fused op accepts a generic disable knob:
    - `RWKV7_DISABLE_FUSED_RECURRENT=1`
    - legacy `RWKV7_DISABLE_FUSED_PREFILL=1` still disables the fused path too
- added a CUDA regression guard:
    - `test_rwkv7_block_batches_decode_tokens_without_changing_results_cuda`

Validation:

- unit tests:
    - `python -m pytest -q tests/model_executor/test_rwkv7.py`
    - result: `13 passed, 2 skipped`
- service correctness:
    - `tmp_rwkv7_compare.py --disable-compile-cache`
    - standard `3` prompts still match one-shot vs step-by-step
    - log:
        - [vllm_rwkv7_compare_default_decodefused_20260413.log](/tmp/vllm_rwkv7_compare_default_decodefused_20260413.log)
- exact long-input benchmark, sequential on a single GPU:
    - eager:
        - `1024 + 64`, `c=8`: `128.662 / 126.905`, avg `127.784`
        - `1984 + 64`, `c=8`: `82.291 / 84.238`, avg `83.264`
    - piecewise:
        - `1024 + 64`, `c=8`: `123.847 / 124.573`, avg `124.210`
        - `1984 + 64`, `c=8`: `84.708 / 88.764`, avg `86.736`

Interpretation:

- decode recurrence was indeed one of the remaining RWKV7 bottlenecks
- after it was fused, the current exact-long steady-state rows cluster tightly:
    - eager is still slightly ahead at `1024`
    - `PIECEWISE` is slightly ahead at `1984`
- this means the largest RWKV7 model-specific runtime gaps are now much smaller
  than they were before decode fusion
- it also means compile is no longer showing a huge model-side throughput win
  on this exact workload; most of the gain came from fixing the model path itself

Benchmark hygiene note:

- do not launch eager and `PIECEWISE` benchmark servers in parallel on the same
  GPU
- that invalidates TPS by introducing direct device contention
- only the sequential reruns should be used as the current control rows

### 19. Prefix caching and mixed prompt-length validation are now covered

Correctness:

- `tmp_rwkv7_compare.py --enable-prefix-caching --disable-compile-cache`
- standard `3` prompts still match one-shot vs step-by-step
- log:
    - [vllm_rwkv7_compare_prefixcache_20260413.log](/tmp/vllm_rwkv7_compare_prefixcache_20260413.log)

Prefix-caching throughput:

- exact-long benchmark now supports `--enable-prefix-caching`
- eager:
    - `1024 + 64`, `c=8`: `130.217 / 210.408`, avg `170.313`
    - `1984 + 64`, `c=8`: `155.799 / 270.292`, avg `213.046`
- piecewise:
    - `1024 + 64`, `c=8`: `149.130 / 208.119`, avg `178.625`
    - `1984 + 64`, `c=8`: `165.642 / 256.278`, avg `210.960`

Interpretation:

- prefix caching is working
- the serial baseline before the measured rounds warms the cache, so the large
  round1 jump is expected and useful
- once the prefix cache is hot, eager and `PIECEWISE` are in the same band on
  this workload

Mixed prompt-length throughput:

- added:
    - [tmp_rwkv7_mixed_exact_prompt_bench.py](/home/liu/vllm/tmp_rwkv7_mixed_exact_prompt_bench.py)
- prompt lengths:
    - `64/128/256/512/768/1024/1536/1984`
- without prefix caching:
    - eager: `149.064 / 150.716`, avg `149.890`
    - piecewise: `87.009 / 134.828`, avg `110.918`
- with prefix caching:
    - eager: `225.201 / 231.280`, avg `228.240`
    - piecewise: `228.883 / 229.894`, avg `229.388`

Interpretation:

- mixed prompt lengths without prefix caching still expose some first-round
  `PIECEWISE` warmup cost
- once prefix caching is enabled, eager and `PIECEWISE` are effectively tied
  again on this mixed service-style workload
- at this point, compile is better understood as:
    - supported
    - correct on the validated path
    - not a guaranteed standalone speedup over an already-fixed eager path
    - compatible with service features like prefix caching

### 20. Longer outputs and the old compile_no_cg 128/c8 tail item are now covered

Longer-output exact-long benchmark:

- used:
    - prompt lengths `1024` and `1920`
    - `max_tokens=128`
    - concurrency `8`
- note:
    - `1984 + 128` is invalid under the current `2048` cap, so `1920` is the
    longest exact-long row that still fits

Results:

- eager:
    - `1024 + 128`: `187.512 / 185.750`, avg `186.631`
    - `1920 + 128`: `137.117 / 137.148`, avg `137.133`
- piecewise:
    - `1024 + 128`: `180.043 / 182.027`, avg `181.035`
    - `1920 + 128`: `134.320 / 135.409`, avg `134.864`
- compile_no_cg:
    - `1024 + 128`: `177.782 / 175.871`, avg `176.827`
    - `1920 + 128`: `133.385 / 133.164`, avg `133.275`

Interpretation:

- all three paths still match the serial baseline at longer outputs
- eager remains slightly ahead on this exact-long workload
- `PIECEWISE` stays close behind
- `compile_no_cg` is also now in the same general band, though still slightly slower

Historical tail-item recheck:

- reran the old mixed-prompt scenario directly:
    - `tmp_rwkv7_long_benchmark.py`
    - `compile_no_cg`
    - `max_tokens=128`
    - concurrency `8`
- result:
    - aggregate TPS `277.310 / 274.227`, avg `275.768`
    - all requests matched the serial baseline

Conclusion:

- the old `compile_no_cg 128/c8` mismatch is not reproduced on the current
  decode-fused branch
- `compile_no_cg` should still be treated as a secondary/debug path, but its
  earlier long-output correctness concern is materially reduced now

### 21. Current 1/2/4/8 concurrency numbers have been refreshed on the latest branch state

Workload:

- prompt set: `default_mixed_8`
- `max_tokens=64`
- model:
    - `RWKV7-Goose-World2.9-0.4B-HF`
- rounds `2`, warmup `1`
- eager and `PIECEWISE` rerun sequentially on one GPU

Results:

- eager:
    - `1`: `36.107 / 35.839`, avg `35.973`
    - `2`: `68.850 / 73.769`, avg `71.310`
    - `4`: `133.740 / 141.088`, avg `137.414`
    - `8`: `286.786 / 283.853`, avg `285.320`
- piecewise:
    - `1`: `34.830 / 35.667`, avg `35.249`
    - `2`: `69.997 / 70.206`, avg `70.102`
    - `4`: `129.235 / 129.512`, avg `129.374`
    - `8`: `271.006 / 257.948`, avg `264.477`

Interpretation:

- current `compile + cg` is usable and close to eager on this benchmark
- eager is still ahead on the latest no-cache mixed throughput control
- the gap is modest at low concurrency and larger at `8`
- this is one more sign that compile is now primarily a supported execution path,
  not a guaranteed throughput win over the already-fixed eager path

## Current TODO List

### Highest priority

1. Decide whether any parts of the current `fp32` correctness-first policy can be relaxed safely.
2. Start moving from core-kernel work to feature-parity work:
   - async scheduling coverage
   - TP/PP coverage
   - interface support such as LoRA if needed
3. If you want a stronger compile value story, test more realistic cache-hit
   ratios and repeated-prefix serving mixes instead of only kernel-isolated probes

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
2. Fused decode recurrent backend for RWKV7
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

The fused prefill route, packed-prefill runtime path, and fused decode recurrent
backend are now all in place. The next high-value move is broader service
validation and feature coverage rather than another RWKV7-specific recurrent
kernel rewrite.

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

The next concrete experiment should be:

1. keep the decode-fused exact-long sequential rows as the current control
2. validate:
   - partial prefix-hit ratios
   - more production-like repeated-prefix mixes
3. if the service matrix is stable, move to feature-parity work instead of more
   low-level recurrent-kernel churn

## Partial Prefix-Hit Prefix-Caching Validation (2026-04-13)

This round added a new benchmark helper:

- [tmp_rwkv7_prefix_hit_bench.py](/home/liu/vllm/tmp_rwkv7_prefix_hit_bench.py)

Why this probe exists:

- the earlier prefix-caching checks proved correctness and "cache on vs off"
  throughput, but they did not model partial hit ratios
- this helper warms a small set of shared long prefixes, mixes them with fresh
  cold prefixes, and measures `0.0 / 0.5 / 1.0` hit ratios under concurrency `8`
- one subtle bug was fixed before running it: different hit-ratio scenarios now
  use disjoint cold/tail token ranges so a previous scenario cannot silently
  pre-warm a later one

Commands and raw artifacts:

- eager JSON:
    - [rwkv7_prefix_hit_eager_20260413.json](/tmp/rwkv7_prefix_hit_eager_20260413.json)
- eager log:
    - [vllm_rwkv7_prefix_hit_eager_20260413.log](/tmp/vllm_rwkv7_prefix_hit_eager_20260413.log)
- piecewise JSON:
    - [rwkv7_prefix_hit_piecewise_20260413.json](/tmp/rwkv7_prefix_hit_piecewise_20260413.json)
- piecewise log:
    - [vllm_rwkv7_prefix_hit_piecewise_20260413.log](/tmp/vllm_rwkv7_prefix_hit_piecewise_20260413.log)

Steady-state summary:

- eager:
    - hit ratio `0.0`: `109.572 / 122.895`, avg `116.233`
    - hit ratio `0.5`: `166.636 / 172.240`, avg `169.438`
    - hit ratio `1.0`: `251.274 / 255.214`, avg `253.244`
- piecewise:
    - hit ratio `0.0`: `121.039 / 124.047`, avg `122.543`
    - hit ratio `0.5`: `163.858 / 174.774`, avg `169.316`
    - hit ratio `1.0`: `252.544 / 249.909`, avg `251.227`

Correctness and log checks:

- all measured rounds matched the serial baseline
- both logs still report experimental Mamba cache `align` mode

Interpretation:

- prefix caching remains the dominant service-level lever for RWKV7
- throughput rises almost monotonically with prefix-hit ratio:
    - eager: `116.233 -> 169.438 -> 253.244`
    - piecewise: `122.543 -> 169.316 -> 251.227`
- `PIECEWISE` is a bit better at `0%` hit, essentially tied at `50%`, and
  slightly behind at `100%`; practically they sit in the same band once cache
  reuse is present
- this strengthens the current project-level conclusion:
    - compile/cudagraph is supported and correct on RWKV7
    - the biggest serving gain still comes from prefix caching itself, not from
    compile alone

Important caveat:

- this benchmark is still bursty concurrent traffic
- it is more realistic than the old exact-length cache probe, but it is not yet
  an arrival-staggered streaming workload

Updated recommended next action:

1. extend the new helper into repeated-prefix arrival-staggered mixes
2. if that is stable, move to feature-parity work:
   - LoRA if needed
   - quantization matrix
   - disaggregated prefill/decode validation

## High-Concurrency Stress Validation (2026-04-13)

The next request was straightforward:

- can the current RWKV7 adaptation actually tolerate very high concurrency on a
  single GPU?

To answer that, the existing mixed-prompt benchmark was reused with:

- model: `RWKV7-Goose-World2.9-0.4B-HF`
- workload: `default_mixed_8`
- output length: `64`
- prefix caching: disabled
- execution paths:
    - eager
    - `PIECEWISE`
- concurrency:
    - `1, 2, 4, 8, 16, 32, 64`
    - then `128` in a separate stress pass

Raw artifacts:

- piecewise `1..64`:
    - [rwkv7_piecewise_high_conc_20260413.json](/tmp/rwkv7_piecewise_high_conc_20260413.json)
- piecewise `128`:
    - [rwkv7_piecewise_c128_20260413.json](/tmp/rwkv7_piecewise_c128_20260413.json)
- eager `1..64`:
    - [rwkv7_eager_high_conc_20260413.json](/tmp/rwkv7_eager_high_conc_20260413.json)
- eager `128`:
    - [rwkv7_eager_c128_20260413.json](/tmp/rwkv7_eager_c128_20260413.json)

Summary table:

| concurrency | eager TPS | piecewise TPS |
| --- | ---: | ---: |
| `1` | `31.156` | `29.302` |
| `2` | `62.260` | `68.573` |
| `4` | `123.345` | `135.256` |
| `8` | `245.424` | `244.390` |
| `16` | `459.711` | `466.835` |
| `32` | `929.251` | `857.579` |
| `64` | `1284.756` | `1275.735` |
| `128` | `379.127` | `1668.700` |

Latency observations:

- eager:
    - stays near `~2.05s` through `8`
    - `64`: avg `3.171s`, p95 `3.185s`
    - `128`: avg `15.553s`, p95 `21.746s`
- piecewise:
    - stays near `~1.87s` to `~2.38s` through `32`
    - `64`: avg `3.192s`, p95 `3.206s`
    - `128`: avg `4.875s`, p95 `4.945s`

Correctness:

- every measured round still matched the serial baseline
- even eager `128` returned full `64` output tokens per request
- so the eager `128` collapse is not a "generated fewer tokens" artifact

Interpretation:

- yes, the current RWKV7 adaptation can handle very large concurrency on one
  GPU
- on the validated workload, `PIECEWISE` remains healthy through `128`
- eager is also fine through `64`, but shows a sharp throughput/latency cliff at
  `128`
- this is the clearest evidence so far that compile/cudagraph support for RWKV7
  is not just "compatible": at sufficiently high concurrency it can materially
  improve service behavior

Practical deployment readout from this round:

1. if you expect peak concurrency around `<=64`, eager and `PIECEWISE` are both
   usable on this workload
2. if you expect larger bursts, `PIECEWISE` is now the safer path
3. the next realism upgrade should be arrival-staggered high-concurrency traffic
   rather than another synchronized burst-only sweep

## Remote Concurrency Benchmark Utility (2026-04-14)

To make it easier to validate a remote vLLM deployment under more realistic
traffic, a reusable helper was added:

- [tmp_rwkv7_remote_concurrency_bench.py](/home/liu/vllm/tmp_rwkv7_remote_concurrency_bench.py)

What it does:

- targets a remote OpenAI-compatible vLLM endpoint
- supports:
    - `/v1/completions`
    - `/v1/chat/completions`
- supports:
    - fixed-concurrency closed-loop load
    - arrival-rate-based staggered load for more production-like traffic
- accepts prompts from:
    - inline CLI args
    - `.txt`
    - `.json`
    - `.jsonl`
- saves:
    - per-run config
    - summary JSON
    - summary Markdown
    - per-request JSONL records

Default output directory:

- `/home/liu/vllm/tmp_rwkv7_remote_bench_runs/<run_name>/`

Recommended usage patterns:

1. closed-loop saturation test:

```bash
source ~/miniforge3/etc/profile.d/conda.sh
conda activate vllm-dev
cd /home/liu/vllm
python tmp_rwkv7_remote_concurrency_bench.py \
  --base-url http://YOUR_HOST:8000 \
  --model /mnt/d/codes/RWKV7-Goose-World2.9-0.4B-HF \
  --endpoint completions \
  --num-requests 256 \
  --concurrency 32 \
  --max-tokens 64 \
  --return-token-ids
```

1. staggered "more real" serving test:

```bash
source ~/miniforge3/etc/profile.d/conda.sh
conda activate vllm-dev
cd /home/liu/vllm
python tmp_rwkv7_remote_concurrency_bench.py \
  --base-url http://YOUR_HOST:8000 \
  --model /mnt/d/codes/RWKV7-Goose-World2.9-0.4B-HF \
  --endpoint completions \
  --prompt-file /path/to/prompts.jsonl \
  --num-requests 512 \
  --concurrency 64 \
  --arrival-rate 12 \
  --arrival-jitter-sec 0.2 \
  --max-tokens 64 \
  --return-token-ids
```

Notes:

- this helper does not launch a local server; it assumes the remote vLLM
  instance is already up
- if the endpoint sits behind an API gateway, `--api-key` can be provided
- if the remote stack does not expose `/health`, add `--skip-health-check`
- if usage is not returned, `--return-token-ids` helps recover output token
  counts for `/v1/completions`

## 2026-04-14: PP smoke failure diagnosis

Remote `PP=2` eager smoke on `RWKV7-Goose-World3-2.9B-HF` failed during
engine startup profiling with:

- `RuntimeError: expected scalar type BFloat16 but found Float`

The stack pointed to the first local `LayerNorm` on PP rank 1:

- `RWKV7Block._run_sequence()` -> `self.attn_norm(residual)`

Root cause:

- RWKV7 keeps block/runtime activations in `RWKV7_RUNTIME_DTYPE`
  (`torch.float32`)
- PP dummy/profile runs for non-first ranks were allocating
  `IntermediateTensors` with `model_config.dtype` (`bfloat16`)
- rank 1 therefore entered the first block with bf16 intermediate tensors,
  while RWKV7 `LayerNorm` weights are still float32 in this path

Fix applied:

- `RWKV7Model.make_empty_intermediate_tensors()` now allocates
  `hidden_states` and `v_first` in `RWKV7_RUNTIME_DTYPE`
- PP receive/return boundaries in `RWKV7Model.forward()` now normalize
  `hidden_states` and `v_first` to `RWKV7_RUNTIME_DTYPE`

Local verification:

- `python -m py_compile vllm/model_executor/models/rwkv7.py`
- `python -m pytest -q tests/model_executor/test_rwkv7.py`
- result: `13 passed, 2 skipped`

Next remote retry:

```bash
CUDA_VISIBLE_DEVICES=0,1 vllm serve /mnt/data/Models/Huggingface/RWKV7-Goose-World3-2.9B-HF \
  --trust-remote-code \
  --host 0.0.0.0 \
  --port 8033 \
  --gpu-memory-utilization 0.8 \
  --pipeline-parallel-size 2 \
  --distributed-executor-backend mp \
  --enforce-eager
```

## 2026-04-14: RWKV7 Mamba prefix cache `all` mode

RWKV7 now advertises `SupportsMambaPrefixCaching`, so when prefix caching is
enabled the config path no longer falls back from `all` to `align`.

Implementation summary:

- extended `LinearAttentionMetadata` so `mamba_cache_mode=all` can carry:
    - `num_computed_tokens`
    - `block_idx_last_computed_token`
    - `block_idx_first_scheduled_token`
    - `block_idx_last_scheduled_token`
- `LinearAttentionMetadataBuilder` now emits the full block table in `all`
  mode instead of collapsing to a single slot id
- RWKV7 decode runtime now reads from the last computed block and writes to the
  last scheduled block when a decode step crosses a cache block boundary
- RWKV7 prefill runtime now writes:
    - intermediate aligned block-boundary states
    - plus the final per-sequence state
- the block-boundary writeback currently reuses the existing fused recurrent
  output path for normal outputs/final state and computes checkpoint recurrent
  states explicitly for cache writeback correctness

Validation completed locally:

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

New unit coverage:

- RWKV7 now declares mamba prefix caching support
- config chooses `mamba_cache_mode='all'` when prefix caching is enabled
- cache-all prefill writes aligned block states correctly
- cache-all decode writes into the next block slot correctly at boundaries

Still pending:

- optimize `all`-mode checkpoint-state writeback so throughput is competitive
  with `align`
- re-run repeated-prefix / mixed-length serving benchmark after a fused or
  direct-write checkpoint-state path exists

## 2026-04-14: RWKV7 `all` mode serving validation

What was added:

- `tmp_rwkv7_prefix_hit_bench.py` now supports:
    - `--mamba-cache-mode`
    - startup log signal extraction in the JSON output
- `tests/model_executor/test_rwkv7.py` now also covers multi-sequence cache-all
  prefill correctness

Validation run:

```bash
source ~/miniforge3/etc/profile.d/conda.sh
conda activate vllm-dev
cd /home/liu/vllm
python -m py_compile tmp_rwkv7_prefix_hit_bench.py tests/model_executor/test_rwkv7.py
python -m pytest -q tests/model_executor/test_rwkv7.py
```

Result:

- `18 passed, 2 skipped`

Serving benchmark:

```bash
python tmp_rwkv7_prefix_hit_bench.py \
  --model /mnt/d/codes/RWKV7-Goose-World2.9-0.4B-HF \
  --enable-prefix-caching \
  --cudagraph-mode piecewise \
  --concurrency 8 \
  --shared-prefix-len 1024 \
  --tail-len 128 \
  --max-tokens 64 \
  --rounds 3 \
  --warmup 1 \
  --log /tmp/vllm_rwkv7_prefix_hit_all_20260414.log \
  > /tmp/rwkv7_prefix_hit_all_20260414.json

python tmp_rwkv7_prefix_hit_bench.py \
  --model /mnt/d/codes/RWKV7-Goose-World2.9-0.4B-HF \
  --enable-prefix-caching \
  --mamba-cache-mode align \
  --cudagraph-mode piecewise \
  --concurrency 8 \
  --shared-prefix-len 1024 \
  --tail-len 128 \
  --max-tokens 64 \
  --rounds 3 \
  --warmup 1 \
  --log /tmp/vllm_rwkv7_prefix_hit_align_20260414.log \
  > /tmp/rwkv7_prefix_hit_align_20260414.json
```

Observed startup signals:

- default RWKV7 startup now says:
    - `Mamba cache mode is set to 'all' for RWKV7ForCausalLM by default when prefix caching is enabled`
- forced `align` stays in `align`

Benchmark summary:

- `all`, hit ratio `0.0 / 0.5 / 1.0`
    - avg aggregate TPS: `19.404 / 29.758 / 120.398`
- `align`, hit ratio `0.0 / 0.5 / 1.0`
    - avg aggregate TPS: `112.421 / 164.175 / 238.235`
- both modes stayed `all_match_serial_baseline=true` in every round

Prefix-cache hit-rate notes from service logs:

- `all` no longer stays at `0.0%`; this repeated-prefix run climbed to about
  `59.2%`
- `align` also no longer stays at `0.0%`; the same workload climbed to about
  `50.5%`

Interpretation:

- the new RWKV7 `all` mode plumbing is functionally live:
    - default config selects `all`
    - repeated-prefix serving yields non-zero prefix-cache hit rate
    - outputs remain correct
- but current throughput is much worse than `align`
- the most likely cause is the current cache-all recurrent checkpoint writeback,
  which still requires an explicit second recurrent pass to materialize aligned
  block-boundary states

Important implementation note:

- I tried a follow-up packed-prefill refactor locally after this run
- it did not recover throughput because the dominant cost appears to be the
  checkpoint-state extraction itself
- that attempt was reverted and not kept in the working tree

Updated recommendation:

- keep RWKV7 `all` mode as implemented and tested for correctness
- do not switch throughput-sensitive serving guidance from `align` to `all`
  yet
- the next real optimization target is a fused or direct-write path for
  checkpoint-state emission at cache block boundaries

## 2026-04-15 Update: default `align` + fused checkpoint emission follow-up

I kept explicit RWKV7 `all` support, but changed the default back to `align`
when prefix caching is enabled and the user did not explicitly request a mode.
This keeps throughput-sensitive serving on the faster path while preserving the
new `all` plumbing behind `--mamba-cache-mode all`.

Code status:

- RWKV7 config now defaults prefix caching to `align`, not `all`
- explicit `--mamba-cache-mode all` is still preserved
- the fused recurrent op now has a checkpoint-state emission path used by
  RWKV7 cache-all prefill handling
- packed cache-all prefill is wired through the new checkpoint-capable fused op

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

- explicit `all` run:
    - `Prefix caching in Mamba cache 'all' mode is currently enabled`
- default run:
    - `Prefix caching in Mamba cache 'align' mode is currently enabled`

Updated repeated-prefix summary (`c=8`, `shared_prefix_len=1024`,
`tail_len=128`, `max_tokens=64`):

- explicit `all`
    - hit `0.0 / 0.5 / 1.0`: `77.784 / 117.627 / 221.835`
- default `align`
    - hit `0.0 / 0.5 / 1.0`: `119.735 / 175.788 / 253.456`
- all requests still matched the serial baseline

Interpretation:

- the new checkpoint-capable fused path materially improved `all`
- `all` no longer crashes on the smoke repeated-prefix workload
- `align` is still faster and should remain the default serving recommendation
- the remaining work is to shrink the residual `all` overhead, not to revisit
  whether `align` should be the default right now

## 2026-04-15 Update: safe `all` follow-up after direct-write experiment

I continued from the new fused checkpoint-emission path and tried the next
obvious step: direct-writing recurrent checkpoint states into the live RWKV7
cache slots during cache-all prefill.

What happened:

- low-level direct-write worked in isolated varlen op checks
- but service-level repeated-prefix smoke started diverging from the serial
  baseline
- the issue was not the fused recurrent math itself; it was runtime visibility

Most likely root cause:

- a cache slot for RWKV7 is a composite of:
    - attn shift state
    - recurrent state
    - ffn shift state
- direct-writing only the recurrent component made a partially updated slot
  observable before the boundary attn/ffn shift states were written
- under real serving concurrency, that is unsafe even if the direct-written
  recurrent tensor itself is numerically fine

Updated runtime decision:

- keep the low-level direct-write capability in the fused op for future work
- do not use recurrent-only direct-write in RWKV7 serving runtime yet
- keep boundary slot publication atomic at the runtime level
- still keep the safe no-`torch.cat` store optimization:
    - boundary block states are stored with one `_store_kv_states(...)`
    - final output slots are stored with a second `_store_kv_states(...)`

Validation:

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

Updated repeated-prefix summary:

- `all` (`safe no-cat`)
    - hit `0.0 / 0.5 / 1.0`: `116.669 / 120.881 / 231.013`
    - all requests matched the serial baseline

Interpretation:

- this is a safer and better runtime point than the earlier `77.784 / 117.627 /
  221.835`
- the `hit=0.0` gap to `align` is now small
- `hit=0.5` and `hit=1.0` still leave visible room for improvement
- the next optimization should target a truly atomic multi-state checkpoint
  publication design, not recurrent-only direct writes

## 2026-04-15 Update: align regression check

After the failed "keep last slot write" experiment was rolled back, I reran the
explicit `align` repeated-prefix benchmark to make sure the serving-fast path
had not picked up an accidental regression.

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

Result:

- `align`, hit `0.0 / 0.5 / 1.0`
    - `117.205 / 160.097 / 230.063`
- all requests still matched the serial baseline

Comparison to the earlier `2026-04-15` align baseline:

- `0.0`: `119.735 -> 117.205` (`-2.1%`)
- `0.5`: `175.788 -> 160.097` (`-8.9%`)
- `1.0`: `253.456 -> 230.063` (`-9.2%`)

Interpretation:

- there is no correctness regression in `align`
- a small single-run throughput dip is visible
- because the current retained code changes are concentrated in `all` plumbing,
  this is not enough evidence to call it a confirmed align-path regression
- if we need a stronger answer later, we should rerun `align` with
  `rounds>=3, warmup>=1` and compare medians rather than one-shot values

## 2026-04-15 Update: reverted `3218256c8` for a more conservative align-first state

Per the latest decision, I reverted the code changes from `3218256c8`
(`Keep RWKV7 all-mode checkpoint writes atomic`) so the codebase is back to the
more conservative `be29a1808`-equivalent runtime state for RWKV7 prefix cache
behavior.

Important note:

- I preserved the benchmark / report history from the reverted experiment in
  the docs
- only the code path was reverted

Post-revert local verification:

```bash
source ~/miniforge3/etc/profile.d/conda.sh
conda activate vllm-dev
cd /home/liu/vllm
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

- `align`, hit `0.0 / 0.5 / 1.0`
    - `115.775 / 165.857 / 221.106`
- all requests matched the serial baseline

Interpretation:

- the revert does not introduce a correctness issue
- the post-revert numbers remain in the same rough band as the earlier
  `align` recheck
- the current evidence still points to run-to-run variance, not a strong
  align-path regression caused by the now-reverted `all` experiment code

Current practical state:

- if we want the most conservative branch state for ongoing `align` work,
  this revert is the cleaner baseline
- if we revisit `all` later, the experiment history is still preserved in the
  docs and commit history

## 2026-04-15 Update: multi-round `align` recheck confirms no meaningful regression

To move beyond one-shot noise, I reran the explicit `align` repeated-prefix
benchmark with `rounds=5, warmup=1` after the revert to the conservative
`be29a1808`-equivalent code state.

Command:

```bash
source ~/miniforge3/etc/profile.d/conda.sh
conda activate vllm-dev
cd /home/liu/vllm
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

- hit `0.0`
    - rounds: `121.232, 115.219, 121.002, 133.697, 144.498`
    - median: `121.232`
- hit `0.5`
    - rounds: `162.771, 186.301, 171.810, 177.612, 174.989`
    - median: `174.989`
- hit `1.0`
    - rounds: `241.200, 255.500, 248.332, 250.331, 242.496`
    - median: `248.332`
- all requests matched the serial baseline

Interpretation:

- compared with the earlier strong `align` baseline (`119.735 / 175.788 /
  253.456`), the new medians are `+1.2% / -0.5% / -2.0%`
- this is close enough to treat the current reverted `align` path as being in
  the same effective performance band
- the branch can keep the simpler conservative code state without claiming a
  meaningful `align` regression

## 2026-04-15 Update: upstream PR hygiene cleanup

To prepare for an upstream PR, I did a lightweight RWKV7 code hygiene pass with
one concrete goal: keep local debugging/reporting workflow intact in local
docs, but avoid carrying purely development-only inspection hooks into the PR.

Applied cleanup:

- removed RWKV7-only `debug_last_*` snapshot fields from
  `vllm/model_executor/models/rwkv7.py`
- removed `RWKV7_DEBUG_*` environment-variable branches used only for local
  store-path debugging
- confirmed the main RWKV7 code/test files do not contain Chinese comments

Verification:

```bash
source ~/miniforge3/etc/profile.d/conda.sh
conda activate vllm-dev
cd /home/liu/vllm
python -m py_compile vllm/model_executor/models/rwkv7.py
```

Result:

- compile check passed

Interpretation:

- this change is non-functional cleanup for PR readiness
- local benchmark / handoff / todo / report files remain the place to keep
  experimental notes and iterative debugging history

## 2026-04-15 Update: PR worktree concurrency smoke

I created a clean upstream-PR worktree:

- worktree: `/home/liu/vllm-rwkv7`
- branch: `codex/rwkv7`
- intended PR title: `rwkv7`

I then ran a direct engine-level concurrency smoke there instead of relying on
the OpenAI API server, because the server path exposed local environment drift
that is unrelated to the RWKV7 model code itself.

Smoke summary:

- model: `RWKV7-Goose-World2.8-0.1B-HF`
- path: `AsyncLLM.from_engine_args(..., enforce_eager=True)`
- concurrency: `8`
- max tokens: `32`
- result: `8 / 8` requests completed successfully
- aggregate output TPS: `210.125`
- average latency: `0.929 s`
- p95 latency: `0.965 s`

Interpretation:

- the current upstream-main-based PR worktree still supports normal concurrent
  RWKV7 generation in eager mode
- this is a useful smoke result for PR readiness because it validates the
  model path directly on the cleaned PR branch

Local validation blockers discovered:

1. `piecewise` startup on the PR worktree currently hits a local prebuilt
   extension mismatch:
   - missing op: `torch.ops._C.silu_and_mul_per_block_quant`
   - this points to the local `_C` binary being older than current
     `upstream/main`
2. OpenAI API server startup in eager mode hits an environment package
   mismatch:
   - `ImportError: cannot import name 'NamedToolChoice' from mistral_common...`

These blockers are specific to the local PR-validation environment and should
not be confused with a confirmed RWKV7 concurrency failure.

## 2026-04-16 Update: fresh-env PR validation

I validated the PR worktree again, this time using a fresh install path that is
much closer to what another developer would see after cloning the branch.

Setup:

- conda env: `vllm-rwkv7`
- repo: `/home/liu/vllm-rwkv7`
- local env inside repo: `.venv`
- install path:
    - `uv venv --python /home/liu/miniforge3/envs/vllm-rwkv7/bin/python --seed`
    - `VLLM_USE_PRECOMPILED=1 uv pip install --python .venv/bin/python -e . --torch-backend=auto`

Validation results:

1. RWKV7 targeted tests:
   - `20 passed, 2 skipped`
2. `piecewise` server startup:
   - startup succeeded
   - `/health` returned `200`
3. `piecewise` concurrency smoke (`32` requests, `c=8`, `max_tokens=32`):
   - success: `32 / 32`
   - aggregate TPS: `277.239`
   - weighted request TPS: `34.668`
4. eager comparison smoke with the same workload:
   - success: `32 / 32`
   - aggregate TPS: `398.361`
   - weighted request TPS: `49.822`

Interpretation:

- this is the strongest validation so far that the PR worktree is functionally
  complete enough for upstream review
- the branch can now be described as:
    - RWKV7 eager: working
    - RWKV7 compile + `piecewise` cudagraph: working
    - RWKV7 targeted tests: passing
- performance still follows the known pattern that eager can be faster than
  `piecewise` on smaller / shorter workloads

## 2026-04-16 Update: empty PIECEWISE CUDA-graph warning

I double-checked the recurring
`The CUDA Graph is empty. This usually means that the graph was attempted to be
captured on wrong device or stream.` warning that showed up during RWKV7
`piecewise` startup.

Conclusion:

- this was not a compile/cudagraph enablement failure
- the runtime still completed real capture successfully (`51 / 51` capture
  descriptors and `Graph capturing finished ...` in the startup logs)
- the warning is consistent with harmless empty piecewise partitions
  (view/alias-only subgraphs with no CUDA kernel launches), not with a broken
  RWKV7 implementation

Mitigation added on the dev branch:

- `vllm/compilation/cuda_graph.py` now suppresses only this specific empty-graph
  warning when the wrapper runtime mode is `PIECEWISE`
- other warnings are still re-emitted unchanged
- `FULL` mode continues to surface the empty-graph warning

Validation:

1. new targeted unit tests in `tests/compile/test_cuda_graph.py`
   - `3 passed`
2. targeted lint
   - `pre-commit run ruff-check --files ...`: passed
   - `pre-commit run ruff-format --files ...`: passed
3. real RWKV7 `piecewise` startup smoke on the dev branch
   - model: `RWKV7-Goose-World2.8-0.1B-HF`
   - startup succeeded
   - the startup log still showed compile + cudagraph enablement
   - the empty-graph warning no longer appeared

## 2026-04-17 Update: native `.pth` checkpoint + txt tokenizer support

Native RWKV7 `.pth` support is now wired through on the dev branch while
keeping the original HF loading path intact.

What was added:

- direct config inference from local `.pt` / `.pth`
- single-file `.pth` / `.pt` loading in the default loader
- native RWKV7 weight-name remapping in `rwkv7.py`
- native RWKV txt tokenizer support
- native RWKV renderer support for chat/completions
- metadata guards so local `.pth` / `.txt` are not mistreated as HF repos

Validation:

1. targeted tests
   - `tests/renderers/test_rwkv.py`
   - `tests/tokenizers/test_rwkv.py`
   - `tests/transformers_utils/test_config.py`
   - `tests/model_executor/model_loader/test_default_loader.py`
   - `tests/model_executor/test_rwkv7.py`
   - result: `28 passed, 2 skipped`
2. real 1.5B checkpoint remap sanity
   - checkpoint: `rwkv7-g1e-1.5b-20260309-ctx8192.pth`
   - `state_keys=798`
   - `mapped_params=795`
   - `ignored=3` (`blocks.0.att.v0/v1/v2`)
   - no shape mismatches
3. real 0.4B native eager generation
   - checkpoint: `rwkv7-g1d-0.4b-20260210-ctx8192.pth`
   - tokenizer: `rwkv_vocab_v20250609.txt`
   - prompt `"The capital of France is"` produced coherent text beginning
     with `" Paris."`
   - prompt `"Hello, my name is"` also produced coherent text
4. real 0.4B native chat generation
   - offline `LLM.chat(...)` succeeded
   - output began with `" Hello! How can I assist you today?"`
5. real 0.4B native `compile + piecewise`
   - offline `LLM(...)` with
     `compilation_config={'cudagraph_mode': 'piecewise'}`
   - generation succeeded
   - logs confirmed `CompilationMode.VLLM_COMPILE`
   - logs confirmed `CUDAGraphMode.PIECEWISE`
   - logs confirmed successful piecewise graph capture

HF regression sanity:

- `get_config('/mnt/d/codes/RWKV7-Goose-World2.9-0.4B-HF')` still resolves as
  normal HF RWKV7 config with expected dimensions

## 2026-04-17 native .pth support landed locally

- Branch: `codex/rwkv7-adapter-align`
- Commit: `b060c05d6` (`Add native RWKV7 .pth checkpoint support`)
- Scope:
    - native RWKV7 `.pt/.pth` config inference
    - single-file `.pt/.pth` loader support
    - native RWKV7 weight-name remap into vLLM RWKV7 module layout
    - native RWKV `.txt` tokenizer support
    - native RWKV renderer for chat templating without HF tokenizer dependency
    - preserved existing HF RWKV7 loading path
- Validation:
    - `pre-commit run` on staged code/tests: passed
    - `pytest -q tests/renderers/test_rwkv.py tests/tokenizers/test_rwkv.py tests/transformers_utils/test_config.py tests/model_executor/model_loader/test_default_loader.py tests/model_executor/test_rwkv7.py -k 'rwkv or native_pth or txt_tokenizer or single_pth'`
        - `28 passed, 2 skipped, 3 deselected`
    - HF config sanity still resolves for `/mnt/d/codes/RWKV7-Goose-World2.9-0.4B-HF`
    - native 1.5B `.pth` config inference resolves to `hidden_size=2048`, `layers=24`, `heads=32`, `head_dim=64`, `vocab=65536`, `ctx=8192`
    - native 0.4B `.pth + .txt` eager generate produces coherent text
    - native 0.4B `.pth + .txt` compile + piecewise generate also produces coherent text

## 2026-04-17 native RWKV txt tokenizer hot-path cleanup

- Branch: `codex/rwkv7-adapter-align`
- Scope:
    - precompute `id -> latin-1 token string` and `string -> id` maps
    - cache `get_vocab()` / `get_added_vocab()` results
    - reuse a cached special-id set instead of rebuilding it per call
- Validation:
    - `pytest -q tests/tokenizers/test_rwkv.py`
        - `2 passed`
    - `.venv/bin/pre-commit run ruff-check --files vllm/tokenizers/rwkv.py tests/tokenizers/test_rwkv.py`
        - passed
    - `.venv/bin/pre-commit run ruff-format --files vllm/tokenizers/rwkv.py tests/tokenizers/test_rwkv.py`
        - passed
- Local microbenchmark deltas:
    - native `convert_ids_to_tokens([tid])`: `0.0409s -> 0.0196s`
    - native `convert_tokens_to_string(tokens)`: `0.0596s -> 0.0437s`
    - native incremental detokenization loop: `0.0214s -> 0.0193s`
- Local end-to-end eager check on the same HF weights:
    - HF tokenizer path: `880.821 output tok/s`
    - native `.txt` tokenizer path: `879.511 output tok/s`
    - result: native `.txt` path is now effectively at parity on this workload

## 2026-04-22 HF RWKV fast tokenizer adapter

- Branch: `codex/rwkv7-adapter-align`
- Scope:
    - local HF RWKV tokenizer directories are now detected as `tokenizer_mode=rwkv`
      when `tokenizer_config.json` points at `hf_rwkv_tokenizer.RwkvTokenizer`
    - `RWKVTokenizer` now uses optional `pyrwkv_tokenizer.WorldTokenizer` as the
      fast Rust backend when available
    - vLLM still owns the HF compatibility layer:
        - `added_tokens.json`
        - `added_tokens_decoder`
        - `special_tokens_map.json`
        - chat-template bos prefix
    - this preserves the important HF RWKV behavior where `"\n\n"` maps to
      added special id `65530`, while raw Rust base-vocab tokenization would map
      it to base id `261`
- Added tool:
    - [tmp_rwkv7_tokenizer_speed_smoke.py](/home/liu/vllm/tmp_rwkv7_tokenizer_speed_smoke.py)
    - purpose: real 0.4B slow-vs-fast tokenizer offline generation smoke

Validation:

```bash
.venv/bin/python -m pytest -q tests/tokenizers/test_rwkv.py tests/renderers/test_rwkv.py
.venv/bin/python -m pytest -q tests/model_executor/test_rwkv7.py
.venv/bin/pre-commit run ruff-check --files \
  vllm/tokenizers/rwkv.py \
  vllm/tokenizers/registry.py \
  tests/tokenizers/test_rwkv.py \
  tests/renderers/test_rwkv.py \
  tmp_rwkv7_tokenizer_speed_smoke.py
.venv/bin/pre-commit run ruff-format --files \
  vllm/tokenizers/rwkv.py \
  vllm/tokenizers/registry.py \
  tests/tokenizers/test_rwkv.py \
  tests/renderers/test_rwkv.py \
  tmp_rwkv7_tokenizer_speed_smoke.py
```

Results:

- tokenizer/renderer tests: `6 passed`
- RWKV7 model tests: `23 passed, 2 skipped`
- ruff check/format: passed
- real HF 0.1B and 0.4B tokenizer parity:
    - `resolve_tokenizer_args(...) -> ('rwkv', model_path)`
    - `get_tokenizer(...).is_fast == True`
    - token ids match original HF slow tokenizer for:
        - `北京是`
        - `User: hi\n\nAssistant:`
        - `a\n\nb`
        - `<|rwkv_tokenizer_end_of_text|>x`
        - mixed English/Chinese text
    - chat-template rendered text matches the original HF slow tokenizer

Tokenizer-only speed on the 0.4B HF directory:

- workload: `3000` mixed English/Chinese prompt strings
- slow HF encode loop: best `0.097766s`
- fast RWKV encode loop: best `0.004333s`
- encode-loop speedup: about `22.6x`
- slow HF batch `__call__`: best `0.075943s`
- fast RWKV batch `__call__`: best `0.008332s`
- batch speedup: about `9.1x`

Real 0.4B offline `LLM.generate` results:

- long-output workload:
    - `18` prompts, `369` input tokens, `1152` output tokens per round
    - fast auto median output TPS: `447.021`
    - slow HF median output TPS: `454.831`
    - conclusion: no meaningful output-throughput difference; model decode
      dominates this workload
- prompt-heavy workload:
    - `120` prompts, `2460` input tokens, `120` output tokens per round
    - fast auto median output TPS: `296.304`
    - slow HF median output TPS: `303.403`
    - conclusion: still no stable end-to-end output TPS win; tokenizer rendering
      is faster, but prefill/model work dominates measured wall time

Practical interpretation:

- fast tokenizer materially reduces tokenizer CPU overhead
- correctness matches the HF slow tokenizer on the tested RWKV7 paths
- do not expect a visible output-token/s gain for normal offline generation on
  the 0.4B model unless the workload is tokenizer-bound or server-side request
  admission/rendering is the bottleneck

## 2026-04-22 Rust tokenizer local-backend update

User request: make the Rust tokenizer understand the HF RWKV added-token overlay
so vLLM can use the local `/home/liu/rwkv-tokenizer` repository as the tokenizer
backend source, instead of relying on a separately installed external package.

Rust repo changes in `/home/liu/rwkv-tokenizer`:

- `rwkv-tokenizer/src/trie.rs`
    - changed trie terminal ids from `u16` sentinel `0` to `Option<u16>`
    - this makes token id `0` valid instead of impossible to encode
- `rwkv-tokenizer/src/lib.rs`
    - parse vocab rows by explicit id instead of append order
    - added `from_buffer(&[u8])`
    - added assigned-id tracking so sparse id gaps do not appear in `get_vocab`
    - decode now works for explicit id `0` and appended HF overlay ids
- `bindings/python/src/lib.rs`
    - exposed `WorldTokenizer.from_buffer(bytes)`
    - constructor/decode now return Python exceptions instead of unwrap panics

vLLM integration:

- installed the local binding into the vLLM `.venv` with:

```bash
uv pip install --python .venv/bin/python -e /home/liu/rwkv-tokenizer/bindings/python
```

- `vllm/tokenizers/rwkv.py` now builds an augmented in-memory vocab buffer:
    - base `rwkv_vocab*.txt`
    - plus HF `added_tokens`/special-token overlay rows
    - appended rows intentionally override duplicate trie paths such as
      `"\n\n"` so Rust returns HF id `65530` instead of base id `261`
- when `WorldTokenizer.from_buffer` exists, vLLM encode/batch encode/decode can
  go directly through Rust for the HF RWKV tokenizer semantics
- fallback remains:
    - old Rust binding: Python splits specials and uses Rust only for plain text
    - no Rust binding: pure Python trie path

Smoke result:

- Rust direct:
    - `WorldTokenizer.from_buffer(...)`
    - `encode("\n\n") == [65530]`
    - `decode([0, 65530]) == "<|rwkv_tokenizer_end_of_text|>\n\n"`
- vLLM direct:
    - `get_tokenizer("/mnt/d/codes/RWKV7-Goose-World2.9-0.4B-HF").is_fast`
      is `True`
    - `encode("\n\n") == [65530]`
    - `decode([0, 65530]) == "<|rwkv_tokenizer_end_of_text|>\n\n"`

Validation:

```bash
cd /home/liu/rwkv-tokenizer
source ~/.cargo/env
cargo fmt --manifest-path rwkv-tokenizer/Cargo.toml
cargo test --manifest-path rwkv-tokenizer/Cargo.toml
cargo fmt --manifest-path bindings/python/Cargo.toml
cargo check --manifest-path bindings/python/Cargo.toml

cd /home/liu/vllm
.venv/bin/python -m pytest -q tests/tokenizers/test_rwkv.py tests/renderers/test_rwkv.py
.venv/bin/python -m pytest -q tests/model_executor/test_rwkv7.py
.venv/bin/pre-commit run ruff-format --files vllm/tokenizers/rwkv.py tests/tokenizers/test_rwkv.py
.venv/bin/pre-commit run ruff-check --files vllm/tokenizers/rwkv.py tests/tokenizers/test_rwkv.py
.venv/bin/pre-commit run check-forbidden-imports --files vllm/tokenizers/rwkv.py
.venv/bin/pre-commit run mypy-local --files vllm/tokenizers/rwkv.py tests/tokenizers/test_rwkv.py
```

Results:

- Rust tokenizer crate tests: `6 passed`
- Python binding cargo check: passed
- vLLM tokenizer/renderer tests: `6 passed`
- vLLM RWKV7 model tests: `23 passed, 2 skipped`
- targeted pre-commit: passed

## 2026-04-22 long-text tokenizer benchmark and boundary fix

While benchmarking long prompts, found one important correctness edge case:

- direct whole-string encode through the augmented Rust trie is not equivalent to
  HF added-token matching
- example shape: `。\n\nAssistant`
    - HF slow tokenizer prioritizes added special token `"\n\n" -> 65530`
    - raw longest-match trie can consume the base token `。\n` first, then leave
      a single `\n`
- fix:
    - keep Python-side HF special-token boundary splitting for encode/batch
      encode
    - ordinary non-special spans still go through Rust `encode`/`encode_batch`
    - decode still goes directly through Rust when the augmented backend is
      available
- regression test added:
    - `test_rwkv_tokenizer_prioritizes_hf_added_token_boundaries`

Long-text tokenizer-only benchmark on
`/mnt/d/codes/RWKV7-Goose-World2.9-0.4B-HF`:

- single-text encode speedup:
    - `1K chars`: `16.4x`
    - `8K chars`: `17.1x`
    - `32K chars`: `17.4x`
    - `128K chars`: `23.0x`
    - `512K chars`: `17.6x`
- decode speedup:
    - `1K chars / 329 tokens`: `16.2x`
    - `8K chars / 2686 tokens`: `17.0x`
    - `32K chars / 10691 tokens`: `15.7x`
    - `128K chars / 42751 tokens`: `21.5x`
    - `512K chars / 170965 tokens`: `20.9x`
- batch long-prompt encode speedup:
    - `16 x 8K chars`: `13.7x`
    - `8 x 32K chars`: `16.3x`
    - `4 x 128K chars`: `16.4x`

Validation after the fix:

- `tests/tokenizers/test_rwkv.py tests/renderers/test_rwkv.py`:
    - `7 passed`
- `tests/model_executor/test_rwkv7.py`:
    - `23 passed, 2 skipped`
- targeted ruff format/check, mypy-local, forbidden-imports:
    - passed

## 2026-04-22 real 0.4B long-prompt generation benchmark

Ran real `LLM.generate` on
`/mnt/d/codes/RWKV7-Goose-World2.9-0.4B-HF` with long prompts near the model's
configured context limit:

- model config `max_position_embeddings`: `2048`
- vLLM args:
    - `max_model_len=2048`
    - `enforce_eager=True`
    - `gpu_memory_utilization=0.60`
    - fast path: `tokenizer_mode=auto`
    - slow path: `tokenizer_mode=slow`
- benchmark logs:
    - `/tmp/rwkv_real_long_auto_single.log`
    - `/tmp/rwkv_real_long_slow_single.log`
    - `/tmp/rwkv_real_long_auto_batch4.log`
    - `/tmp/rwkv_real_long_slow_batch4.log`

Results:

- single long prompt:
    - prompt tokens: `1800`
    - output tokens: `8`
    - fast median wall: `0.484391s`
    - slow median wall: `0.469076s`
    - interpretation: same practical band; slow was about `3.3%` faster in
      this small sample, likely run-to-run/model scheduling noise
- batch long prompts:
    - prompts: `4`
    - prompt tokens per prompt: `1800`
    - total prompt tokens: `7200`
    - output tokens: `4`
    - fast median wall: `0.878082s`
    - slow median wall: `0.900331s`
    - interpretation: same practical band; fast was about `2.5%` faster by
      median, but average wall time was effectively equal

Conclusion:

- tokenizer-only long text is still clearly faster with Rust
- real 0.4B long-prompt generation does not show a meaningful end-to-end win
  because model prefill dominates the measured wall time
- this matches the earlier short/prompt-heavy real-model observation

## 2026-04-22 server native vocab bytes-literal parser fix

Observed on server with native `.pth` model and tokenizer
`rwkv_vocab_v20250609.txt`:

```bash
VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 vllm serve \
  /mnt/data/Models/RWKV-7-30B/Mobius-r7-base-30B-1230.pth \
  --tokenizer /mnt/data/Codes/RWKV/RWKV_tokenizer/rwkv_vocab_v20250609.txt \
  --load-format pt \
  --trust-remote-code \
  --tokenizer-mode rwkv
```

Failure:

- Rust tokenizer panicked in `rwkv-tokenizer/src/lib.rs`
- stack reached `FastWorldTokenizer(str(self.vocab_path))`
- root cause: native `v20250609` vocab contains Python `bytes` repr tokens that
  are not pure `\xNN` sequences
- old parser only handled pure hex byte reprs, so tokens like `b'\n'`,
  `b'abc'`, `b'\\'`, `b'\''`, or octal escapes could hit
  `Option::unwrap()`

Fix committed in `/home/liu/rwkv-tokenizer`:

- `aa6f61a Parse Python bytes vocab tokens`
- added a Python bytes-literal parser for:
    - plain ASCII bytes
    - `\n`, `\r`, `\t`, `\a`, `\b`, `\f`, `\v`
    - escaped quotes and backslashes
    - `\xNN`
    - octal escapes
- changed parser failures and byte-length mismatches into `io::Error` instead
  of panic
- validation:
    - `cargo test --manifest-path rwkv-tokenizer/Cargo.toml`: `7 passed`
    - `cargo check --manifest-path bindings/python/Cargo.toml`: passed

Server fix after this commit is pushed:

```bash
cd /mnt/data/Codes/RWKV/vllm/vllm_rwkv7/rwkv-tokenizer
git pull
source ~/.cargo/env
cd bindings/python
python -m pip install -e .
```

Quick verification:

```bash
python - <<'PY'
from pyrwkv_tokenizer import WorldTokenizer
tok = WorldTokenizer("/mnt/data/Codes/RWKV/RWKV_tokenizer/rwkv_vocab_v20250609.txt")
print("from_buffer:", hasattr(WorldTokenizer, "from_buffer"))
print("vocab_size:", tok.vocab_size())
print("encode smoke:", tok.encode("hello\n\nworld")[:16])
PY
```

## 2026-04-27 RWKV7 attention epilogue Triton path

Implemented a feature-flagged Triton path for the attention epilogue:

- flag: `RWKV7_USE_FUSED_LNX_RKVRES_XG=1`
- fused region:
    - groupnorm over each local value head
    - `r*k*r_k` residual correction
    - output gate `* g`
- deliberately not fused:
    - `o_proj`, which stays on vLLM `RowParallelLinear`
    - this keeps tensor parallel and quantization boundaries unchanged

Validation commands:

```bash
cd /home/liu/vllm
export PATH=/home/liu/vllm/.venv/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
.venv/bin/python -m pytest \
  tests/model_executor/test_rwkv7.py::test_rwkv7_perf_flags_from_env \
  tests/model_executor/test_rwkv7.py::test_rwkv7_mix6_triton_matches_reference \
  tests/model_executor/test_rwkv7.py::test_rwkv7_kk_pre_triton_matches_reference \
  tests/model_executor/test_rwkv7.py::test_rwkv7_lnx_rkvres_xg_triton_matches_reference \
  tests/model_executor/test_rwkv7.py::test_rwkv7_perf_hooks_match_reference_formulas \
  tests/model_executor/test_rwkv7.py::test_rwkv7_attention_mix6_flag_matches_reference \
  tests/model_executor/test_rwkv7.py::test_rwkv7_attention_kk_pre_flag_matches_reference \
  tests/model_executor/test_rwkv7.py::test_rwkv7_attention_lnx_rkvres_xg_flag_matches_reference \
  -q
```

Result:

- `10 passed`

Real 0.4B isolated benchmark commands:

```bash
cd /home/liu/vllm
export PATH=/home/liu/vllm/.venv/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin

unset RWKV7_USE_FUSED_MIX6 RWKV7_USE_FUSED_KK_PRE \
  RWKV7_USE_FUSED_LNX_RKVRES_XG RWKV7_USE_FUSED_CMIX \
  RWKV7_USE_ALT_RECURRENT_KERNEL
.venv/bin/python tmp_rwkv7_ttft_benchmark.py \
  --model /mnt/d/codes/RWKV7-Goose-World2.9-0.4B-HF \
  --dtype auto \
  --gpu-memory-utilization 0.6 \
  --port 8071 \
  --rounds 2 \
  --warmup 1 \
  --prompt-lengths 64 1024 1984 \
  --decode-output-lengths 32 64 \
  --decode-prompt-len 64 \
  --warmup-prompt-len 64 \
  --enforce-eager \
  --log /tmp/vllm_rwkv7_epilogue_baseline.log

unset RWKV7_USE_FUSED_MIX6 RWKV7_USE_FUSED_KK_PRE \
  RWKV7_USE_FUSED_CMIX RWKV7_USE_ALT_RECURRENT_KERNEL
export RWKV7_USE_FUSED_LNX_RKVRES_XG=1
.venv/bin/python tmp_rwkv7_ttft_benchmark.py \
  --model /mnt/d/codes/RWKV7-Goose-World2.9-0.4B-HF \
  --dtype auto \
  --gpu-memory-utilization 0.6 \
  --port 8072 \
  --rounds 2 \
  --warmup 1 \
  --prompt-lengths 64 1024 1984 \
  --decode-output-lengths 32 64 \
  --decode-prompt-len 64 \
  --warmup-prompt-len 64 \
  --enforce-eager \
  --log /tmp/vllm_rwkv7_epilogue_fused.log
```

Results:

- prefill proxy:
    - `64`: `61.615ms -> 52.390ms`
    - `1024`: `165.422ms -> 165.970ms`
    - `1984`: `263.834ms -> 247.408ms`
- decode:
    - `64 -> 32`: TTFT `108.119ms -> 75.901ms`, TPOT `37.934ms -> 30.600ms`
    - `64 -> 64`: TTFT `92.324ms -> 78.167ms`, TPOT `38.243ms -> 33.744ms`

Operational note:

- When launching benchmark commands from Codex/PowerShell, keep the WSL `PATH`
  explicit as above.
- Do not run `.venv/bin/python` directly from PowerShell on the WSL path; that
  can trigger the Windows "choose how to open Python" dialog.

## 2026-04-27 RWKV7 recurrent core evaluation

Compared official CUDA `rwkv7_clampw` against vLLM's current Triton
`fused_mul_recurrent_rwkv7` on the subset where they are directly comparable:

- `B=1`
- contiguous prefill
- `K=V=64`
- zero initial recurrent state
- no varlen
- no checkpoint-state output
- fp32 inputs

Important mapping:

- official CUDA receives raw `w`, `-kk`, and `kk*a`
- vLLM Triton receives `LOG_DECAY_SCALE * sigmoid(raw_w)`, `kk`, and `a`
- with model-like stable inputs, outputs match closely:
    - max abs diff: about `2.2e-8` to `5.2e-8`
    - mean abs diff: about `1.5e-9`

Prototype benchmark:

- seq `16`: official `0.0198ms`, current `0.0519ms`, official `2.62x`
- seq `64`: official `0.0883ms`, current `0.1932ms`, official `2.19x`
- seq `256`: official `0.3209ms`, current `0.7580ms`, official `2.36x`
- seq `1024`: official `1.1598ms`, current `2.5130ms`, official `2.17x`

Decision:

- Worth implementing, but not as a direct drop-in.
- Official `rwkv7_clampw` starts from zero state and does not expose vLLM-style
  final recurrent state.
- First safe vLLM implementation should add initial/final state support and
  route only behind `RWKV7_USE_ALT_RECURRENT_KERNEL` for exact supported shapes.

Rejected Triton alternative:

- A stateful Triton prototype with one program per `(batch, head, value_channel)`
  matched current outputs/final states, but was much slower than the current
  Triton kernel.
- Example timings:
    - `(B=1,T=1024)`: current `1.5512ms`, probe `8.4710ms`
    - `(B=64,T=1)`: current `0.4928ms`, probe `1.5707ms`
- Conclusion:
    - if we pursue this optimization, use a vLLM-owned CUDA op based on the
      official shared-memory block structure
    - do not spend more time on this Triton shape unless a fundamentally
      different blocking strategy is available
