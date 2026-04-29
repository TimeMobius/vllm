# RWKV7 KDA-Style Adaptation Plan

## Conclusion

结论是：**可行，而且比继续硬推“整模型 fullgraph compile”更合理。**

补充性能迁移清单见：

- [tmp_rwkv7_official_perf_todo.md](/home/liu/vllm/tmp_rwkv7_official_perf_todo.md)

## Progress Update (2026-04-27)

- `RWKV7_USE_FUSED_MIX6` 与 `RWKV7_USE_FUSED_KK_PRE` 已接入。
- 这两项都已经做完：
    - Triton / reference 数值等价测试
    - model hook 接入测试
    - isolated serial end-to-end benchmark
- benchmark 方法也校正了：
    - 不再并行跑多组 flag
    - 不再通过 PowerShell 拼接 `export PATH=...:$PATH`
    - 统一改成：
        - 单独 bash 脚本
        - 显式 clean Linux `PATH`
        - 一次只测一组 flag
- clean serial benchmark 结论：
    - baseline
        - prefill TTFT proxy:
            - `64`: `63.519ms`
            - `1024`: `180.642ms`
            - `1984`: `269.706ms`
        - decode:
            - `64 -> 32`: TTFT `105.311ms`, TPOT `38.565ms`
            - `64 -> 64`: TTFT `90.251ms`, TPOT `36.891ms`
    - `mix6` only
        - short TTFT / decode 有正收益
    - `kk-pre` only
        - decode 收益最明显
        - longer prompt TTFT 也改善
    - `mix6 + kk-pre`
        - clean serial run 下整体仍然 net positive
- 当前下一步：
    - 继续按 [tmp_rwkv7_official_perf_todo.md](/home/liu/vllm/tmp_rwkv7_official_perf_todo.md)
      做 `P1: Fused CMix / FFN`

## Progress Update (2026-04-28)

- `RWKV7_USE_ALT_RECURRENT_KERNEL` 已完成首轮接入：
    - vLLM-owned CUDA op 已加入 `_C`
    - 只在受限路径尝试启用：
        - `_run_recurrent_sequence()`
        - `_run_recurrent_decode_batch()`
    - varlen / checkpoint / shape 不满足时仍走现有 Triton fallback
- 这一步也已经做完：
    - direct op 数值等价测试
    - attention hook 接入测试
    - focused pytest
    - local `0.4B` isolated serial benchmark
- benchmark 结论要分两层看：
    - 内核层：
        - `B=1,T=1,H=16,K=64,V=64`
        - alt CUDA 比当前 Triton 快约 `47%`
    - 服务层：
        - focused `64 -> 256` decode-heavy rerun
        - TTFT / latency / TPOT 只有温和收益
        - 大致在 `+3% ~ +11%` 这个量级
- 当前判断：
    - 这条路径值得保留，但只能算“实验性正收益”
    - 还不足以直接改默认
    - 下一步不应该继续围着 recurrent 小修小补
    - 更值得回到：
        - 更大粒度 `CMix / FFN`
        - 或 `P3 runtime integration cleanup`

## Progress Update (2026-04-28, CMix Probe)

- `RWKV7_USE_FUSED_CMIX` 已完成一轮更保守的 probe：
    - 这次不是直接硬搬官方 `_CmixLayerV2Fn`
    - 而是先把其中最独立的一块：
        - `sqrelu`
      抽成通用 CUDA `_C.relu2`
    - RWKV7 FFN 只在 flag 打开时走这个 CUDA activation
- 这一步已经做完：
    - CUDA op correctness 测试
    - RWKV7 hook-level 开关测试
    - direct microbenchmark
    - local `0.4B` isolated serial benchmark
- 当前 benchmark 结论：
    - activation op 本体：
        - 大多只有近似持平
    - direct `RWKV7FeedForward._apply_ffn()`：
        - 只有个别 token 数有小幅改善
        - 整体并没有像 recurrent / epilogue 那样明确
    - real `0.4B` serial benchmark：
        - `64 -> 64` decode 有正收益
        - `64 -> 32` 只有 TTFT 改善，latency/TPOT 带噪声
        - longer prefill proxy 基本接近平盘
- 当前判断：
    - 这条路径可以保留为实验开关
    - 但不能当作已经验证过的主收益项
    - 如果继续做 `CMix / FFN`，方向应该是：
        - 更大粒度 fuse
        - 而不是只盯着 `sqrelu`
- 当前下一步建议：
    - 优先回到 [tmp_rwkv7_official_perf_todo.md](/home/liu/vllm/tmp_rwkv7_official_perf_todo.md)
      里的：
        - 更完整 `CMix / FFN` 区域融合评估
        - 或 `P3 runtime cleanup`

## Progress Update (2026-04-28, Runtime Cleanup)

- `P3 runtime cleanup` 已完成一个很具体的首轮落点：
    - `cache_all + packed prefill`
    - 把原来 Python 侧逐 request 拼 checkpoint metadata 的逻辑，
      改成了张量化 helper
- 这一步不是改 attention/FFN 数学：
    - 只是在 `_forward_runtime()` 里减少：
        - `.item()`
        - Python `for` 循环
        - list append / `torch.cat(...)`
- 当前 benchmark 结论很明确：
    - metadata helper 单独测：
        - CPU 约 `22.5x`
        - CUDA 约 `26.5x`
    - 但收益范围要说清楚：
        - 只影响 `cache_all` 路径
        - 不等于普通 decode TPS 会直接同步涨这么多
- 当前判断：
    - 这是值得保留的 runtime cleanup
    - 也说明 “减少 Python / metadata 开销” 这条路线是对的
    - 后续如果继续做 `P3`，优先看：
        - 还有没有类似的 per-request metadata 组装逻辑
        - 是否能继续把 prefix-caching 相关路径去 Python 化

## Progress Update (2026-04-13)

- 远程压测脚本已升级：
    - [tmp_rwkv7_remote_concurrency_bench.py](/home/liu/vllm/tmp_rwkv7_remote_concurrency_bench.py)
    - 已补：
        - `burst` 发压模式
        - 显式 `token_throughput_tps`
            - 平铺别名：
                - `token_throughput_tps_avg/min/max`
        - 单请求 `request_token_throughput_tps`
            - 每条请求记录都有
            - summary 会给 `avg / p50 / p95 / min / max`
            - 另补 `weighted_avg`，避免长尾下 `avg(每请求TPS)` 误导
        - `token_throughput_tps_stats`
            - 1 秒桶的 `min / avg / max tok/s`
        - `active_output_tps`
        - `peak_inflight_requests`
    - 现在可以区分：
        - 固定 client worker 的 closed-loop benchmark
        - 更像“把请求直接压到 vLLM 队列里”的 burst benchmark
    - 这更适合继续测：
        - 远程服务真正的饱和并发点
        - 队列积压后的尾延迟

- 新增了独立 benchmark 台账文件：
    - [tmp_rwkv7_benchmark_records.md](/home/liu/vllm/tmp_rwkv7_benchmark_records.md)
- 这份台账现在固定记录：
    - run id
    - 模型名 / 模型大小
    - 运行模式
    - `max_tokens`
    - 并发档位
    - 每轮 TPS 和平均 TPS
    - 串行 baseline 一致性
    - 原始 JSON / log 路径
- 已补一条新的 eager 基线：
    - run id: `2026-04-13_eager_0p4b_mt64`
    - model: `RWKV7-Goose-World2.9-0.4B-HF`
    - mode: `eager`
    - `max_tokens=64`
    - concurrency `1/2/4/8`
    - avg aggregate TPS:
        - `34.980 / 69.860 / 132.739 / 214.987`
- 当前在做的事也更明确了：
    - 以后每次 benchmark 先落原始 JSON / log
    - 再把摘要追加到 benchmark 台账
    - 然后再在 todo / handoff 里只写结论和下一步
- 已新增 TTFT / prefill-heavy benchmark 工具：
    - [tmp_rwkv7_ttft_benchmark.py](/home/liu/vllm/tmp_rwkv7_ttft_benchmark.py)
- 已补一条新的 eager TTFT 基线：
    - run id: `2026-04-13_eager_0p4b_ttft`
    - server ready: `30.034s`
    - prefill-heavy TTFT proxy:
        - prompt len `64`: `188.986ms`
        - prompt len `1024`: `2708.602ms`
        - prompt len `1984`: `5409.289ms`
    - decode profile, prompt len `64`:
        - `max_tokens=32`: avg TTFT `255.053ms`, avg ITL `27.862ms`
        - `max_tokens=64`: avg TTFT `255.353ms`, avg ITL `28.676ms`
- 新 benchmark 口径说明：
    - vLLM 服务口径下 `max_tokens` 不能是 `0`
    - 所以当前所谓 prefill-only benchmark 更准确地说是：
        - streaming `max_tokens=1` 的 prefill-heavy TTFT proxy
- 已补首轮 compile TTFT 对照：
    - `compile/no-cg`
    - `PIECEWISE`
    - 对照结论：
        - `compile/no-cg` 没有表现出 TTFT 优势
        - `PIECEWISE` 对短 prompt 没优势
        - 但在长 prompt 首 token 上已经开始比 eager 更好
        - `PIECEWISE` decode ITL 目前与 eager 基本同量级
- 下一步直接接这组 TTFT 对照继续补：
    - 更稳定多轮统计
    - prefix caching
    - mixed prompt lengths
    - 更长输出

## CUDA Graph Assessment (2026-04-13)

### 当前判断

- RWKV7 其实已经不是“完全没有 CUDA graph”：
    - 默认 `PIECEWISE` 已经能正确跑通
    - `compile/no-cg` 也已经能正确跑通
- 现在真正缺的不是“能不能 capture”，而是：
    - 能不能把 `PIECEWISE` 路径做成值得长期默认使用的主线
    - 能不能在不回退正确性的前提下，把 cold start / TTFT / 吞吐做上去
    - 要不要继续追更激进的 full decode graph
- 现阶段不建议直接追 `FULL_AND_PIECEWISE`：
    - 当前历史结论已经证明它对 RWKV7 仍然不安全
    - 更合理的主线仍然是：
        - 先把 `PIECEWISE` 做强
        - 再决定要不要重新打开 full decode graph

### 代码层面的真实 blocker

1. prefill recurrence 还是 Python token loop：
   - [`RWKV7Attention._forward()`](/home/liu/vllm/vllm/model_executor/models/rwkv7.py#L490)
   - 里面仍有：
     - `for idx in range(hidden_states.shape[0])`
   - 这说明 compile 虽然被 custom op boundary 挡住了
   - 但真正最重的 prefill recurrence 还没有变成 varlen/chunk kernel
   - 所以 `PIECEWISE` 很难稳定打赢 eager

2. runtime metadata dispatch 还是 Python-side loop + `.item()`：
   - [`RWKV7Block._forward_runtime()`](/home/liu/vllm/vllm/model_executor/models/rwkv7.py#L945)
   - 当前 prefill 还是逐段：
     - 读 `query_start_loc`
     - 读 `seq_lens`
     - 读 `state_indices_tensor`
     - 再逐条 `_run_sequence(...)`
   - 这条路径现在对 correctness 已经足够
   - 但对更激进的 graph capture / 更低 CPU overhead 并不理想

3. decode 还是普通张量实现，不是 fused recurrent kernel：
   - [`RWKV7Attention.forward_decode_batch()`](/home/liu/vllm/vllm/model_executor/models/rwkv7.py#L633)
   - 这条路径已经正确
   - 但还没有变成类似 FLA / KDA / Mamba 风格的 fused recurrent backend

4. 当前 compile policy 还是“安全优先”而不是“最大 capture”：
   - [`config.py`](/home/liu/vllm/vllm/model_executor/models/config.py#L675)
   - RWKV7 现在会把：
     - `FULL_AND_PIECEWISE`
     收紧到：
     - `PIECEWISE`
   - 这说明当前策略是：
     - 先保住正确性
     - 还没有准备好重新打开 full decode graph

5. 现有 benchmark 结论说明性能目标还没达成：
   - [tmp_rwkv7_handoff.md](/home/liu/vllm/tmp_rwkv7_handoff.md)
   - 当前结论仍然是：
     - `PIECEWISE` 和 eager 大致同一量级
     - 没有稳定显著胜出
     - piecewise cold start 还明显更慢

### 推荐实现顺序

#### Stage A. 先把目标定义清楚

- 把“实现 CUDA graph”拆成两个目标：
    - Goal 1:
        - 让 `PIECEWISE` 成为稳定、可解释、可回归的 compile 主线
    - Goal 2:
        - 只有在 Goal 1 达成后，才考虑 full decode graph
- 当前推荐优先级：
    - `PIECEWISE` > `compile/no-cg` > `FULL_AND_PIECEWISE`

#### Stage B. 先补性能判断依据

- 补 TTFT / prefill-only benchmark：
    - 区分 cold start
    - 区分 prefill
    - 区分 decode
- 当前已完成 eager 基线工具化：
    - [tmp_rwkv7_ttft_benchmark.py](/home/liu/vllm/tmp_rwkv7_ttft_benchmark.py)
- 把 benchmark 扩成更稳定多轮统计：
    - 不只看一次 aggregate TPS
- 补这几类 coverage：
    - prefix caching
    - mixed prompt lengths
    - 更长输出
    - concurrency `3/8`
- 如果这些 benchmark 还显示 `PIECEWISE` 没有明确收益
    - 就不要急着继续追 full decode graph

#### Stage C. 再做真正的 compile backend 优化

1. prefill kernelization：
   - 优先把 [`RWKV7Attention._forward()`](/home/liu/vllm/vllm/model_executor/models/rwkv7.py#L490)
     里的 Python recurrence 换成：
     - packed / varlen / chunk prefill backend
   - 重点参考：
     - `/home/liu/flash-linear-attention`
     - KDA / Mamba 在 vLLM 内的 metadata 接法

2. decode fused recurrent：
   - 保持现在的接口不变
   - 先把 [`forward_decode_batch()`](/home/liu/vllm/vllm/model_executor/models/rwkv7.py#L633)
     的内部替换成 fused recurrent kernel

3. metadata path 去 Python 化：
   - 尽量减少 `_forward_runtime()` 里的：
     - `.item()`
     - Python per-prefill loop
   - 目标不是马上删光所有 Python
   - 而是把最热路径挪到 backend/kernel 里

4. 只在上面三步完成后，再重新评估：
   - 是否值得重新尝试 `FULL_AND_PIECEWISE`
   - 或者长期把 RWKV7 固定在 `PIECEWISE`

### 验收标准

- Correctness：
    - `0.1B` / `0.4B`
    - one-shot vs step-by-step
    - `PIECEWISE`
    - `compile/no-cg`
    - prefix caching
    - mixed prompt lengths
- Performance：
    - eager vs `PIECEWISE`
    - TTFT
    - prefill-only
    - decode throughput
    - cold start
- Policy：
    - 如果 `FULL_AND_PIECEWISE` 不能稳定通过 correctness + benchmark
    - 就继续保持：
        - RWKV7 默认 `PIECEWISE`
        - full decode graph 关闭

## Progress Update (2026-03-31)

- 同一天的后续推进已经把 compile/no-cudagraph 路径真正打通：
    - 新增了 whole-block custom op：
        - `torch.ops.vllm.rwkv7_block_forward(...)`
    - `RWKV7Block.forward()` 在 compile 路径下不再依赖可能被 trace 固化的 Python metadata 分支
    - runtime metadata/stateful dispatch 迁移到了 `RWKV7Block._forward_runtime(...)`
    - layer-local `kv_cache` 和 runner-level `kv_caches` 现在都会在 compile/no-cg 路径里被真正写回
- 根因结论也修正了：
    - 之前“`attn_metadata` 在 compile 路径里全局缺失”的说法不够准确
    - 更准确的根因是：
        - runtime `forward_context.attn_metadata` 实际存在
        - 但 compiled `RWKV7Block.forward()` 没有在真实请求时走到 metadata-aware cache path
        - whole-block custom op 把这条 runtime stateful path 拉回来了
- 配置策略现在已经正式放开：
    - `RWKV7ForCausalLMConfig` 不再强制 eager fallback
    - RWKV7 现在已经可以通过真实入口直接跑：
        - 默认 PIECEWISE CUDA 图
        - `cudagraph_mode=none`
- probe / 回归工具链同步更新：
    - [`tmp_rwkv7_engine_first_step_compare.py`](/home/liu/vllm/tmp_rwkv7_engine_first_step_compare.py) 已移除 compile monkeypatch
    - [`tmp_rwkv7_compare.py`](/home/liu/vllm/tmp_rwkv7_compare.py) 现在支持：
        - `--model`
        - `--compile-no-cg`
- 最新验证结果：
    - 单测：
        - `python -m pytest -q tests/model_executor/test_rwkv7.py`
        - 结果：`9 passed, 2 skipped`
    - engine real compile replay：
        - base: [rwkv7_engine_step_2_base_real_compile.json](/tmp/rwkv7_engine_step_2_base_real_compile.json)
        - replay: [rwkv7_engine_step_2_replay_real_compile.json](/tmp/rwkv7_engine_step_2_replay_real_compile.json)
        - compare: [rwkv7_engine_step_2_compare_real_compile.json](/tmp/rwkv7_engine_step_2_compare_real_compile.json)
        - 结论：`北京是` 的第二个 decode token 仍然完全一致
    - `vllm serve` real compile/no-cg correctness：
        - 日志：[vllm_rwkv7_compare_real_compile_no_cg.log](/tmp/vllm_rwkv7_compare_real_compile_no_cg.log)
        - `i am`、`北京是`、`The capital of France is`
        - one-shot `max_tokens=8` 与 step-by-step `max_tokens=1` 全部一致
- 默认 PIECEWISE correctness：
    - 日志：[vllm_rwkv7_piecewise_final.log](/tmp/vllm_rwkv7_piecewise_final.log)
    - `i am`、`北京是`、`The capital of France is`
    - one-shot `max_tokens=8` 与 step-by-step `max_tokens=1` 全部一致
- `0.1B` 本地 checkpoint 也已补测：
    - 默认 PIECEWISE：
        - [vllm_rwkv7_piecewise_0p1b_final.log](/tmp/vllm_rwkv7_piecewise_0p1b_final.log)
    - compile/no-cg：
        - [vllm_rwkv7_compile_no_cg_0p1b_final.log](/tmp/vllm_rwkv7_compile_no_cg_0p1b_final.log)
    - `i am`、`北京是`、`The capital of France is`
    - one-shot `max_tokens=8` 与 step-by-step `max_tokens=1` 全部一致
- 当前 compile correctness blocker 已经清零
- PIECEWISE 最终修复点：
    - 把 `vllm::rwkv7_block_forward` 加进了默认 `splitting_ops`
    - 让 RWKV7 的 whole-block stateful boundary 不会被错误冻进单个 piecewise capture 区域
- 另一个 graph-capture 直接坑：
    - `_store_kv_state()` / `_store_kv_states()` 里的 debug `.item()` 会导致：
        - `CUDA error: operation not permitted when stream is capturing`
    - 现在详细 stats 只在：
        - `RWKV7_DEBUG_STORE_STATS=1`
        - 且当前 stream 不在 capture
    时才会启用
- 新 benchmark 结论：
    - `cudagraph_mode=none` 仍然主要是 correctness/debug 路径，不是最快路径
    - 当前通用 Mamba 默认 `FULL_AND_PIECEWISE` 对 RWKV7 仍然不安全
        - benchmark 下会复现 `indexSelectSmallIndex` / device-side assert
    - 因此 RWKV7 现在新增了 post-optimization-level 覆写：
        - 默认把 `FULL_AND_PIECEWISE` 收紧回 `PIECEWISE`
    - 当前 `0.4B` clean rerun 更准确的结论是：
        - `PIECEWISE` 和 eager 在短输出 mixed-prompt benchmark 上处于同一量级
        - `PIECEWISE` 没有稳定、显著地跑赢 eager
        - 长输入 `1024` / `1984` token benchmark 也没有看到稳定 clear win
    - 现阶段 `compile + PIECEWISE` 的主要价值更偏：
        - compile correctness
        - 默认策略安全性
        - 而不是已经明确领先 eager 的吞吐

- Phase 1 的第一步已经落地：
    - `RWKV7Attention` 注册进了 `static_forward_context`
    - 新增了 `torch.ops.vllm.rwkv7_attention(...)`
    - 当前 recurrent 实现已经搬到 `RWKV7Attention._forward(...)`
- 这一步确实解决了最初的 compile 卡点：
    - `Dynamo bytecode transform time` 从约 `126.55s` 降到了约 `1.74s`
    - compile range `(1, 2048)` 的图编译约 `8.89s`
    - 总 `torch.compile` 时间约 `10.92s`
- 这段中间结论后来已经被后续修复覆盖：
    - whole-block custom op 把 runtime metadata/stateful dispatch 拉回到了 live path
    - capture 期间的 debug `.item()` 被收敛成仅在非 capture 下启用
    - `vllm::rwkv7_block_forward` 加入默认 `splitting_ops`
    - 所以现在的最终状态已经不是“能启动但不正确”，而是：
        - 默认 PIECEWISE 可用
        - `cudagraph_mode=none` 可用
        - 两条 compile 路径都能通过真实服务正确性回归
- 新增了一个 engine-step probe：
    - [`tmp_rwkv7_engine_first_step_compare.py`](/home/liu/vllm/tmp_rwkv7_engine_first_step_compare.py)
    - 现在支持：
        - `--capture-generated-tokens` 抓任意 decode step 的快照
        - `--prompt-token-ids` 做受控 token-id prompt
        - `--append-generated-prefix-from-run-json` 从上一次 run JSON 直接拼 replay prompt
        - `--compare-second-step` 做 base/replay 的离线第二步对比
- 当前 probe 的新结论：
    - 在 `RWKV7-Goose-World2.9-0.4B-HF` 上
    - prompt `北京是`
    - `async_scheduling=False`
    - `max_tokens=1` 和 `max_tokens=8` 的第一个生成 token 一致：
        - token id `10250`
        - text `一`
    - 当前 probe 看到的 layer-level state fingerprint 也没有第一步差异
    - 但 layer-local `kv_cache` 在这个 probe 里保持全零，所以这个 state 结果只能当粗粒度信号
    - 目前更可靠的判断是：
        - compile mismatch 不在第一个 decode step
        - 更可能发生在后续 step，或者发生在 model-runner/scheduler 拥有但未暴露给 layer-local `kv_cache` 的 cache 路径
- 第二步 replay 的新结论：
    - 在本地 checkpoint `/mnt/d/codes/RWKV7-Goose-World2.9-0.4B-HF` 上
    - base run：
        - prompt token ids: `[10902, 10362, 13091]`
        - `max_tokens=2`
        - `capture_generated_tokens=2`
        - generated token ids: `[10250, 10283]`
        - text: `一个`
    - controlled replay run：
        - prompt token ids: `[10902, 10362, 13091, 10250]`
        - `max_tokens=1`
        - `capture_generated_tokens=1`
        - generated token id: `10283`
        - text: `个`
    - 结论：
        - 第二个 decode token 也没有分叉
        - 因而当前 non-eager mismatch 至少不在 `北京是` 这个 case 的前两个 decode step
    - 参考结果：
        - [rwkv7_engine_step_2_base.json](/tmp/rwkv7_engine_step_2_base.json)
        - [rwkv7_engine_step_2_replay.json](/tmp/rwkv7_engine_step_2_replay.json)
        - [rwkv7_engine_step_2_compare.json](/tmp/rwkv7_engine_step_2_compare.json)
- 新定位结论：
    - 在 `VLLM_DISABLE_COMPILE_CACHE=1` 的 fresh compile 下
    - `RWKV7Block.forward()` 的调试摘要显示：
        - `attn_metadata_is_none=1`
        - `num_decode_tokens=-1`
        - `num_prefill_tokens=-1`
    - 同时：
        - `_store_kv_state()` / `_store_kv_states()` 没有被调用
        - layer-local `kv_cache` 仍然全零
        - runner-level `kv_caches` 也仍然全零
    - 这说明当前 non-eager bug 已经可以明确表述为：
        - **RWKV7 block 在 compile 路径里没有拿到 live `LinearAttentionMetadata`**
        - **所以它始终退回 `attn_metadata is None` 的纯 sequence path**
        - **首 token 还能对，但 recurrent state 从未写回 cache，后续 decode 必然分叉**
    - 参考结果：
        - [rwkv7_engine_step_1_final_repro.json](/tmp/rwkv7_engine_step_1_final_repro.json)
- 额外坑：
    - vLLM 的 torch.compile cache 会掩盖本地模型代码修改
    - 调 compile correctness 时必须显式加：
        - `VLLM_DISABLE_COMPILE_CACHE=1`
    - 否则很容易误以为新 instrumentation 已经生效

RWKV7 当前 compile 路径的核心问题，不是 cache 语义，也不是服务路径，而是：

- [`RWKV7Attention.forward()`](/home/liu/vllm/vllm/model_executor/models/rwkv7.py#L348) 在 prefill 路径里包含 Python 时间步循环
- vLLM 的 `support_torch_compile` 封装走的是 `torch.compile(..., fullgraph=True)`
- Dynamo 会试图把这段 recurrence 整体追进图里，导致：
    - 冷启动编译非常慢
    - 或者直接在 graph-break / disable 方案上失败
- 并且当前 RWKV7 还有一个更直接的 correctness blocker：
    - [`RWKV7Model.forward()`](/home/liu/vllm/vllm/model_executor/models/rwkv7.py#L1019) 在 compiled 路径里把 layer-local `attn_metadata` 解析成了 `None`
    - 导致 [`RWKV7Block.forward()`](/home/liu/vllm/vllm/model_executor/models/rwkv7.py#L863) 永远走不到 metadata-aware cache path

而 vLLM 里已经成熟的前沿线性/混合注意力实现，像：

- [`KimiDeltaAttention`](/home/liu/vllm/vllm/model_executor/layers/kda.py)
- [`Qwen3NextGatedDeltaNet`](/home/liu/vllm/vllm/model_executor/models/qwen3_next.py#L373)
- [`OlmoHybridGatedDeltaNet`](/home/liu/vllm/vllm/model_executor/models/olmo_hybrid.py#L129)
- [`MambaMixer2`](/home/liu/vllm/vllm/model_executor/layers/mamba/mamba_mixer2.py)

它们的共同点不是“数学一样”，而是**实现结构**一样：

1. 模型外壳里的 projection / norm / residual 保持 compile-friendly
2. 真正带 recurrence 的状态更新，不让 Dynamo 去追 Python 循环
3. recurrence 放进：
   - custom op 边界
   - 或 fused / varlen kernel
4. prefill 和 decode 分成不同路径
5. 利用 `static_forward_context` + `forward_context.no_compile_layers` 访问层实例和 cache

对 RWKV7 来说，这条路线是最值得模仿的。

## Why KDA-Style Is Better For RWKV7

### 当前 RWKV7 的主要矛盾

- cache/state 逻辑已经基本跑通
- decode 并发 batching 已经修好
- eager 路径已经证明模型和服务逻辑本身是对的
- compile 路径失败点集中在 prefill recurrence 的 Python 循环

所以现在最应该优化的，不是继续调 `enforce_eager` 开关，而是**改实现形状**。

### KDA/GDN 路线的优势

KDA 的关键模式在 [`kda.py`](/home/liu/vllm/vllm/model_executor/layers/kda.py)：

- `forward()` 只做 projection 和输出 buffer 准备
- 真正状态更新通过 `torch.ops.vllm.kda_attention(...)` 进入 custom op 边界
- custom op 内部再通过 `forward_context.no_compile_layers[layer_name]` 找到层实例
- `_forward()` 里再根据 metadata 分 prefill / decode
- prefill 用 chunk/varlen kernel
- decode 用 fused recurrent kernel

这套结构对 RWKV7 非常有参考意义，因为它解决的是**编译边界问题**，不是某个模型专用细节。

### 为什么不继续走“给 RWKV7Attention.forward 打 disable”

已经验证过这条路不行：

- vLLM 当前 compile 包装是 `fullgraph=True`
- `torch.compiler.disable` 会直接触发 `torch._dynamo.exc.Unsupported`
- 这不是性能差一点，而是启动直接失败

所以正确方向不是“graph break 这个函数”，而是**把 recurrence 抽到 compile 图之外的 op 边界**。

## Recommended Target Architecture

RWKV7 推荐改成两层结构：

### 第一层：compile-friendly shell

保留在 Python/模型层里的内容：

- embedding / lm_head
- norm / residual
- q/k/v/r/a/g projection
- output projection
- FFN
- `RWKV7Block` 外壳

这些内容大多是普通张量算子，适合继续放在 compile 图里。

但当前要注意一条边界约束：

- **不要再让 compiled `RWKV7Model.forward()` 直接决定某一层的 `LinearAttentionMetadata` 是否存在**
- 这一步已经验证会把 RWKV7 block 固化到 `attn_metadata=None` 路径
- metadata/stateful dispatch 需要移到：
    - whole-block custom op
    - 或 block-local no-compile/runtime helper
    - 或更小的、明确不被 fullgraph 固化的边界

### 第二层：stateful recurrent backend

迁移出去的内容：

- shift state 读写
- recurrent state 更新
- prefill 时间步循环
- decode 单步 recurrent update
- varlen / query_start_loc / state_indices 相关逻辑

这些内容应该放进：

- custom op 包装
- 以及后续的 fused / varlen kernel

## FLA RWKV7 Reference Map

这条路线最值得借鉴的，不只是 KDA 的“custom op 边界”，还包括 FLA 里已经存在的 RWKV7 专用算子。

可直接参考的 FLA 文件：

- [`fla/layers/rwkv7.py`](/home/liu/flash-linear-attention/fla/layers/rwkv7.py)
- [`fla/ops/rwkv7/fused_recurrent.py`](/home/liu/flash-linear-attention/fla/ops/rwkv7/fused_recurrent.py)
- [`fla/ops/rwkv7/chunk.py`](/home/liu/flash-linear-attention/fla/ops/rwkv7/chunk.py)
- [`fla/modules/token_shift.py`](/home/liu/flash-linear-attention/fla/modules/token_shift.py)

建议的对照关系：

### 1. shift state

FLA:

- `token_shift(...)`

RWKV7 in vLLM:

- 当前是 [`token_shift_with_cache`](/home/liu/vllm/vllm/model_executor/models/rwkv7.py#L49) + 三份 state cache 里的 shift state

意义：

- 这是最容易先对齐的部分
- FLA 已经支持 `cu_seqlens`
- 也支持 initial/final cache 语义

### 2. prefill recurrence

FLA:

- `chunk_rwkv7(...)`

RWKV7 in vLLM:

- 当前还是 Python 时间步循环
- 正是 compile 路径最主要的阻塞点

意义：

- 这是后续最重要的性能目标
- 也是最接近 KDA/Mamba 风格实现的部分

### 3. decode recurrence

FLA:

- `fused_mul_recurrent_rwkv7(...)`

RWKV7 in vLLM:

- 当前已有 batched decode 数学实现
- 但还是普通张量路径，不是 fused kernel

意义：

- 当前 decode correctness 已经成立
- 后面可以先保持接口不变，再把内部替换成 fused kernel

### 4. input mixing / projection helper ops

FLA:

- `fused_addcmul_rwkv7`
- `fused_k_rwkv7`
- `gate_output_correction`

RWKV7 in vLLM:

- 当前都在 Python / PyTorch 逻辑里展开

意义：

- 这些不是第一优先级
- 但如果后面做 kernel 化，它们是最自然的第二层优化点

## Recommended Policy For Reusing FLA

推荐策略不是“直接把 FLA 作为运行时依赖接进 vLLM”，而是：

1. **先参考 FLA 的接口与张量布局**
2. **在 vLLM 内部做一层适配**
3. **必要时把最小 kernel 迁进 vLLM 侧维护**

原因：

- vLLM 有自己固定的 metadata 体系
    - `query_start_loc`
    - `seq_lens`
    - `state_indices_tensor`
- vLLM 有自己的 cache 生命周期和 slot 管理
- 直接把整层 FLA runtime 拉进来，耦合太高
- 但 FLA 的 RWKV7 算子已经足够当“实现蓝图”

一句话策略：

- **学 FLA 的 RWKV7 kernel 形状**
- **学 KDA/GDN 的 vLLM 集成方式**
- **在 vLLM 里拼出一条 RWKV7 专用的 compile-friendly backend**

## Recommended Implementation Strategy

推荐分两期做，而不是直接一步到位上 RWKV7 专用 Triton kernel。

### Phase 1: Compile-Unblock Refactor

目标：

- 不先追求极致性能
- 先让非 eager 路径能真正启动
- 先避免 Dynamo 去 trace RWKV7 prefill recurrence

做法：

1. `[done]` 仿照 KDA，新增一个 RWKV7 custom op 包装
2. `[partially done]` `RWKV7Attention.forward()` 已变成 compile-friendly shell，但当前 shell 仍然包着整段 attention，而不是只保留 projection
3. `[done]` recurrent 更新搬进 `_forward(...)`
4. `[done]` `torch.ops.vllm.rwkv7_attention(...)` 调 `_forward(...)`
5. `[done]` `_forward(...)` 通过 `forward_context.no_compile_layers[self.prefix]` 访问层实例和 cache
6. `[done]` prefill 先复用当前正确的 eager 数学实现
7. `[done]` decode 先复用当前正确的 batched decode 实现

### Immediate Next Steps

1. 补 compile 路径的吞吐 benchmark：
   - 以 [tmp_rwkv7_benchmark_records.md](/home/liu/vllm/tmp_rwkv7_benchmark_records.md)
     里的 `2026-04-13_eager_0p4b_mt64` 为当前 eager 基线
   - eager vs 默认 PIECEWISE vs `cudagraph_mode=none`
2. 补 compile 路径的并发正确性与吞吐：
   - concurrent 3
   - concurrent 8
3. 扩大服务回归覆盖：
   - 更长输出
   - prefix caching
   - mixed prompt lengths
4. 评估是否能在保持正确性的前提下放宽部分 `fp32` 路径

当前这套 KDA-style / whole-block compile boundary 的核心收益是：

- compile 图不再直接吞下 RWKV7 prefill 的 Python 时间步循环
- compile 服务路径已经能真正跑通并保持 token 级一致性

### Phase 2: Kernelization / Varlen Optimization

目标：

- 再把性能真正提上去
- 对齐 Mamba/KDA 一类实现的成熟度

做法：

1. 给 RWKV7 prefill 做 varlen/chunk 路径
2. 给 RWKV7 decode 做 fused recurrent 路径
3. 尽量利用：
   - `query_start_loc`
   - `seq_lens`
   - `state_indices_tensor`
4. 最终把 Python 时间步循环替换掉
5. 这一步优先参考 FLA 的：
   - `token_shift`
   - `chunk_rwkv7`
   - `fused_mul_recurrent_rwkv7`

## Step-by-Step TODO

### Phase 0. Freeze Validation Baseline

- [x] 固定 correctness prompts：
    - `i am`
    - `北京是`
    - `The capital of France is`
- [x] 固定回归口径：
    - one-shot vs step-by-step correctness
    - compile/no-cg
    - 默认 PIECEWISE

### Phase 1. Build RWKV7 Compile Boundary

- [x] 新增 RWKV7 custom-op boundary
- [x] 保留 attention-level custom op 骨架：
    - `torch.ops.vllm.rwkv7_attention(...)`
- [x] 新增 whole-block runtime boundary：
    - `torch.ops.vllm.rwkv7_block_forward(...)`
- [x] 把 runtime metadata/stateful dispatch 移到 live block runtime helper

### Phase 2. Restore Compile Correctness

- [x] fresh compile probe 定位 `attn_metadata=None`
- [x] second-step token-id-controlled replay 跑通
- [x] compile/no-cg correctness 恢复
- [x] default PIECEWISE correctness 恢复
- [x] 把 `vllm::rwkv7_block_forward` 加入默认 `splitting_ops`
- [x] 修复 graph capture 期间 debug `.item()` 崩溃

### Phase 3. Service Regression Validation

- [x] 单测：
    - [test_rwkv7.py](/home/liu/vllm/tests/model_executor/test_rwkv7.py)
- [x] `0.4B` compile/no-cg one-shot vs step-by-step
- [x] `0.4B` 默认 PIECEWISE one-shot vs step-by-step
- [x] `0.1B` compile/no-cg one-shot vs step-by-step
- [x] `0.1B` 默认 PIECEWISE one-shot vs step-by-step

### Phase 4. Next Performance Work

- [x] eager vs 默认 PIECEWISE / 显式 PIECEWISE / `cudagraph_mode=none` 首轮吞吐对比
- [x] `0.4B` 长输入 benchmark：
    - prompt len `1024`
    - prompt len `1984`
    - concurrency `1/4/8`
    - eager vs `PIECEWISE`
- [x] 新增独立 benchmark 台账文件：
    - [tmp_rwkv7_benchmark_records.md](/home/liu/vllm/tmp_rwkv7_benchmark_records.md)
- [x] 补 `2026-04-13` eager `0.4B` / `max_tokens=64` 基线记录
- [x] 新增 TTFT / prefill-heavy benchmark 脚本：
    - [tmp_rwkv7_ttft_benchmark.py](/home/liu/vllm/tmp_rwkv7_ttft_benchmark.py)
- [x] 补 `2026-04-13` eager `0.4B` TTFT 基线记录
- [x] 用 TTFT benchmark 补首轮 `PIECEWISE` / `compile_no_cg` 对照
- [x] 把 benchmark 扩成更稳定的多轮统计，减少单次波动：
    - [x] eager fused-off `rounds=3`, `warmup=2`
    - [x] eager fused-on `rounds=3`, `warmup=2`
    - [x] `PIECEWISE` fused-on `rounds=3`, `warmup=2`
- [ ] 继续查 `no-cg` 的长输出高并发分叉：
    - `max_tokens=128`
    - concurrency `8`
- [x] 做 TTFT / prefill-heavy benchmark，把长 prefill 和 decode 开销分开看
- [x] 做 exact long-input throughput probe：
    - [x] `1024 + 64`
    - [x] `1984 + 64`
    - [x] eager vs `PIECEWISE`
    - [x] concurrency `1/4/8`
- [x] 对异常慢的 `1024 + 64, c=8` 做 focused rerun
- [x] 做 prefix caching correctness 覆盖
- [x] 做 prefix caching exact-long throughput 覆盖
- [x] 做 mixed prompt lengths throughput 覆盖
- [x] 做 longer outputs exact-long 覆盖：
    - [x] eager `1024/1920 + 128, c=8`
    - [x] `PIECEWISE 1024/1920 + 128, c=8`
    - [x] `compile_no_cg 1024/1920 + 128, c=8`

### Phase 5. Kernelization / Varlen Optimization

- [x] 新增 RWKV7 fused recurrent op：
    - [rwkv7.py](/home/liu/vllm/vllm/model_executor/layers/fla/ops/rwkv7.py)
- [x] 用 fused recurrent kernel 替换 sequence prefill 的 Python token loop
- [x] 为 fused recurrent op 补 CUDA/reference 单测
- [x] 借鉴 Mamba/KDA 的 metadata 使用方式
- [x] 用 `query_start_loc` 做 packed / varlen prefill
- [x] 对照 FLA 的 `chunk_rwkv7` / `fused_mul_recurrent_rwkv7`
- [x] 评估并落地首版独立 Triton kernel
- [x] 把 fused kernel 接到 packed / varlen prefill，而不是只接单 sequence path
- [x] packed prefill 的真实服务 smoke：
    - [x] `PIECEWISE`
    - [x] concurrency `4/8`
    - [x] 输出对齐串行 baseline
- [x] 为 decode batch 接入 fused recurrent backend
- [x] 给 decode batch 补 CUDA 回归测试
- [x] 用 exact long-input benchmark 复测 decode fused 后的 steady-state：
    - [x] eager `1024 + 64, c=8`
    - [x] eager `1984 + 64, c=8`
    - [x] `PIECEWISE 1024 + 64, c=8`
    - [x] `PIECEWISE 1984 + 64, c=8`

### Phase 6. Precision / Policy Cleanup

- [ ] 评估哪些路径能从 `fp32` 回到更轻量 dtype
- [ ] 评估 compile-enabled 路径是否可以作为默认推荐
- [x] RWKV7 默认 compile cudagraph 策略收紧到 `PIECEWISE`
- [ ] FULL decode graph 是否还有必要单独支持，还是直接长期禁用
- [ ] 做更广 prompt/model sweep 后再决定最终默认策略

## Files Most Likely To Change

### Immediate

- [rwkv7.py](/home/liu/vllm/vllm/model_executor/models/rwkv7.py)
- [test_rwkv7.py](/home/liu/vllm/tests/model_executor/test_rwkv7.py)
- [config.py](/home/liu/vllm/vllm/model_executor/models/config.py)
- [rwkv7.py](/home/liu/vllm/vllm/model_executor/layers/fla/ops/rwkv7.py)

### Likely new or adjacent files

- new RWKV7 custom-op helper under vLLM runtime layer
- possibly a dedicated RWKV7 backend/op file under:
    - [`model_executor/layers/`](/home/liu/vllm/vllm/model_executor/layers)
    - or [`model_executor/layers/fla/ops/`](/home/liu/vllm/vllm/model_executor/layers/fla/ops)

### Probably unchanged initially

- [kv_cache_coordinator.py](/home/liu/vllm/vllm/v1/core/kv_cache_coordinator.py)
- state copy semantics in RWKV7
- service routing layer

## Validation Gates

每一步都建议卡住这些 gate：

### Gate A. 单测必须先过

```bash
source ~/miniforge3/etc/profile.d/conda.sh
conda activate vllm-dev
cd /home/liu/vllm
python -m pytest -q tests/model_executor/test_rwkv7.py
```

### Gate B. 非 eager 服务必须能启动

至少确认：

- `/health` 200
- 一条 `/v1/completions` 正常返回

### Gate C. 正确性必须先于性能

如果：

- one-shot != step-by-step

则：

- 不看 TPS
- 先修 correctness

### Gate D. 并发不能回退

必须确认 decode batching 还存在：

- aggregate TPS 随并发增长
- 不再退回之前那种“总 TPS 几乎不变”的串行表现

## Recommended Next Task

下一步最值得直接开始的是：

- [ ] 从“核心内核补齐”切到“vLLM 特性覆盖矩阵”：
    - async scheduling
    - TP/PP
    - 需要的话再评估 LoRA / 量化接口
- [ ] 如果要证明 compile 的现实价值，补更真实的 repeated-prefix / partial-cache-hit workload
- [x] partial prefix-hit workload：
    - [x] 新增 [tmp_rwkv7_prefix_hit_bench.py](/home/liu/vllm/tmp_rwkv7_prefix_hit_bench.py)
    - [x] 用 warmed shared prefixes + fresh cold prefixes 跑 `0.0 / 0.5 / 1.0`
    hit ratio
    - [x] eager 和 `PIECEWISE` 都完成 `concurrency=8`、`1024+128 -> 64`
    的部分命中率验证
    - [x] 所有测量轮次继续对齐串行 baseline
    - [x] 结论明确：
        - prefix caching 仍然是最大的服务级增益来源
        - `PIECEWISE` 在 partial-hit workload 下已经和 eager 基本同档
- [ ] 更真实的 repeated-prefix / partial-cache-hit workload：
    - [ ] 做 arrival-staggered 请求流，而不是每轮同时送一批
    - [ ] 补 partial hit ratio 随时间变化的场景，而不是固定 `0.0 / 0.5 / 1.0`
    - [ ] 加入更接近线上分布的 prompt length mix
- [x] 高并发压力测试：
    - [x] 单卡 `default_mixed_8, max_tokens=64`
    - [x] eager / `PIECEWISE` 都完成 `1/2/4/8/16/32/64`
    - [x] 额外补 `128` 并发 stress pass
    - [x] 所有轮次继续对齐串行 baseline
    - [x] 结论：
        - `PIECEWISE` 可以稳定扛到 `128`
        - eager 在 `128` 出现明显吞吐 cliff 和队列延迟上升
- [x] 远程并发压测工具：
    - [x] 新增 [tmp_rwkv7_remote_concurrency_bench.py](/home/liu/vllm/tmp_rwkv7_remote_concurrency_bench.py)
    - [x] 支持 remote OpenAI-compatible vLLM endpoint
    - [x] 支持 `/v1/completions` 和 `/v1/chat/completions`
    - [x] 支持固定并发和 arrival-rate staggered workload
    - [x] 自动保存：
        - config.json
        - summary.json
        - summary.md
        - requests.jsonl
- [ ] 用新脚本对远程 RWKV7 实例补真实服务压测：
    - [ ] closed-loop saturation
    - [ ] arrival-staggered workload
    - [ ] repeated-prefix prompt set
    - [ ] 把远程结果补回 benchmark records

compile 路径已经不是“能不能跑通”的问题了。现在最该区分的是：

- 纯 `PIECEWISE` 已经是可用且正确的主线
- `FULL_AND_PIECEWISE` 仍然不安全
- fused prefill 已经把长 prompt TTFT 明显打下来
- packed-prefill runtime 已经接上 `query_start_loc`
- fused decode backend 也已经接上 `forward_decode_batch()`
- 当前 exact-long steady-state：
    - eager `1024 + 64, c=8`: avg `127.784`
    - `PIECEWISE 1024 + 64, c=8`: avg `124.210`
    - eager `1984 + 64, c=8`: avg `83.264`
    - `PIECEWISE 1984 + 64, c=8`: avg `86.736`
- prefix caching 已经验证可用：
    - exact-long cached `1984 + 64, c=8`:
        - eager avg `213.046`
        - `PIECEWISE` avg `210.960`
- mixed prompt lengths 已经完成首轮覆盖：
    - no-cache：
        - eager avg `149.890`
        - `PIECEWISE` avg `110.918`
    - prefix-cache：
        - eager avg `228.240`
        - `PIECEWISE` avg `229.388`
- partial prefix-hit workload 已完成：
    - eager：
        - hit ratio `0.0`: avg `116.233`
        - hit ratio `0.5`: avg `169.438`
        - hit ratio `1.0`: avg `253.244`
    - `PIECEWISE`：
        - hit ratio `0.0`: avg `122.543`
        - hit ratio `0.5`: avg `169.316`
        - hit ratio `1.0`: avg `251.227`
    - 两边都随 hit ratio 提升而阶梯式增速，且全部对齐串行 baseline
- longer outputs exact-long 也已完成：
    - `1024 + 128, c=8`:
        - eager avg `186.631`
        - `PIECEWISE` avg `181.035`
        - `compile_no_cg` avg `176.827`
    - `1920 + 128, c=8`:
        - eager avg `137.133`
        - `PIECEWISE` avg `134.864`
        - `compile_no_cg` avg `133.275`
- 旧的 `compile_no_cg 128/c8` mixed mismatch 在当前分支未复现：
    - avg `275.768`
    - all_match `true`
- 当前最新 `default_mixed_8, max_tokens=64` refreshed control：
    - eager:
        - `1/2/4/8` = `35.973 / 71.310 / 137.414 / 285.320`
    - `PIECEWISE`:
        - `1/2/4/8` = `35.249 / 70.102 / 129.374 / 264.477`
- 高并发 `default_mixed_8, max_tokens=64` stress：
    - eager:
        - `16/32/64/128` = `459.711 / 929.251 / 1284.756 / 379.127`
    - `PIECEWISE`:
        - `16/32/64/128` = `466.835 / 857.579 / 1275.735 / 1668.700`
    - 单卡上 `PIECEWISE` 到 `128` 仍然健康，eager 在 `128` 出现明显 cliff
- 模型特定热点已经不像之前那样突出，接下来更需要补服务矩阵和特性覆盖
- [x] PP 启动 smoke 首轮排障：
    - 远端 `PP=2 + eager` 首次启动失败，报错
    `expected scalar type BFloat16 but found Float`
    - 已定位到 RWKV7 PP dummy/profile run 的 `IntermediateTensors` dtype
    与模型 runtime dtype 不一致
    - 已补 PP 边界 dtype 护栏，本地验证：
        - `python -m py_compile vllm/model_executor/models/rwkv7.py`
        - `python -m pytest -q tests/model_executor/test_rwkv7.py`
        - `13 passed, 2 skipped`
- [ ] 远端复测 `PP=2 + eager`：
    - 目标：确认 engine 能完整启动并通过 `/health`
    - 然后补一条 completion smoke
- [ ] 若 `PP=2 + eager` 通过，再测 `PP=2 + PIECEWISE`
- [x] RWKV7 Mamba prefix cache `align -> all` plumbing：
    - [x] RWKV7 挂 `SupportsMambaPrefixCaching`
    - [x] `LinearAttentionMetadata` 补 `all mode` 所需 block-index 元数据
    - [x] decode 路径支持跨 block 读旧 slot / 写新 slot
    - [x] prefill 路径支持 aligned block-boundary state writeback
    - [x] 本地验证：
        - `python -m py_compile vllm/v1/attention/backends/linear_attn.py vllm/model_executor/models/rwkv7.py tests/model_executor/test_rwkv7.py`
        - `python -m pytest -q tests/model_executor/test_rwkv7.py`
        - `17 passed, 2 skipped`
- [x] 补 `all mode` 的服务级验证：
    - [x] 起服务确认日志里不再把 RWKV7 降回 `align`
    - [x] repeated-prefix workload 对比 `all` vs `align`
    - [x] 观察到 `Prefix cache hit rate` 不再长期卡在 `0.0%`
    - [x] benchmark / handoff 已回填真实数据
    - [x] 结果结论：
        - `all` 确实生效，但当前吞吐显著差于 `align`
        - `all` repeated-prefix avg TPS：
            - hit ratio `0.0 / 0.5 / 1.0` = `19.404 / 29.758 / 120.398`
        - `align` repeated-prefix avg TPS：
            - hit ratio `0.0 / 0.5 / 1.0` = `112.421 / 164.175 / 238.235`
        - `all` 的日志 hit rate 峰值约 `59.2%`，`align` 约 `50.5%`
        - 因此当前瓶颈不是“命中没接通”，而是 `all` 的 checkpoint
      writeback 成本
- [ ] `all mode` 性能化：
    - [x] 默认模式改回 `align`，显式 `all` 仍可保留
    - [x] 首版 fused checkpoint-state emission 已接进 RWKV7 `all` mode prefill
    - [x] 显式 `all` repeated-prefix smoke 已恢复可运行，不再像上一轮那样
    直接在服务级 benchmark 中崩溃
    - [x] 最新 repeated-prefix smoke：
        - `all`: `77.784 / 117.627 / 221.835`
        - `default align`: `119.735 / 175.788 / 253.456`
        - 三档 hit ratio 都保持 `all_match_serial_baseline=true`
    - [x] direct-write runtime feasibility check：
        - [x] 低层 fused op direct-write 在 isolated varlen 对拍中可工作
        - [x] 真实服务 runtime 中，recurrent-only direct-write 会引入
      partial-slot visibility，导致 repeated-prefix smoke 分叉
        - [x] 这条 unsafe runtime wiring 已回退，不保留在当前 serving 路径
    - [x] 保留安全 no-`torch.cat` runtime 写回优化：
        - `all` repeated-prefix 现为 `116.669 / 120.881 / 231.013`
        - 所有请求重新回到 `all_match_serial_baseline=true`
    - [x] `align` recheck：
        - `117.205 / 160.097 / 230.063`
        - correctness 正常
        - 单轮略低于旧基线，但还不足以认定为 confirmed regression
    - [x] 代码回退到 `be29a1808` 等价稳态：
        - 已 revert `3218256c8`
        - 文档保留实验历史，代码回到更保守的 `align`-first 状态
    - [x] post-revert `align` smoke：
        - `115.775 / 165.857 / 221.106`
        - correctness 正常
        - 仍然更像 run-to-run 波动，而不是回退前实验代码留下的确定性影响
    - [x] post-revert `align` 多轮确认（`rounds=5, warmup=1`）：
        - median `121.232 / 174.989 / 248.332`
        - 相比旧高点基线 `119.735 / 175.788 / 253.456`，差异约 `+1.2% / -0.5% / -2.0%`
        - 现阶段可以认为 `align` 没有实质性性能回退
    - [x] upstream PR hygiene cleanup：
        - 去掉了 `rwkv7.py` 里纯开发期的 `RWKV7_DEBUG_*` 分支和 `debug_last_*` 状态快照
        - 核心 RWKV7 代码/测试文件已确认没有中文注释
        - 本地实验文档继续保留，但不作为上游 PR 内容
    - [x] upstream-PR worktree concurrency smoke：
        - clean PR worktree `/home/liu/vllm-rwkv7` created on branch `codex/rwkv7`
        - direct engine-level RWKV7 smoke on the PR worktree passed with
      `8 / 8` finished requests (`32` output tokens each)
        - aggregate output TPS `210.125`, avg latency `0.929 s`, p95 `0.965 s`
        - current local PR-validation blockers are environment drift issues:
      old `_C` extension for `piecewise`, and `mistral_common` mismatch in
      OpenAI API server startup
    - [x] fresh-env PR validation：
        - repo-local `.venv` on top of `conda` env `vllm-rwkv7` now installs
      successfully with `VLLM_USE_PRECOMPILED=1`
        - `tests/model_executor/test_rwkv7.py`: `20 passed, 2 skipped`
        - `compile + piecewise cudagraph` service now starts successfully in the
      fresh env and passes a closed-loop concurrency smoke (`32` req, `c=8`)
        - eager comparison smoke with the same workload also passes
        - current functional status for the PR branch can be treated as complete
      enough for upstream PR preparation
    - [x] PIECEWISE empty-graph warning investigation：
        - confirmed this is not a failed RWKV7 cudagraph enablement path
        - startup still completes real `piecewise` capture successfully
        - warning is consistent with harmless empty piecewise partitions that
      contain no CUDA kernels
        - dev-branch mitigation added to suppress only this known warning for
      `PIECEWISE`, while keeping unrelated warnings and `FULL`-mode warnings
      visible
        - targeted tests: `tests/compile/test_cuda_graph.py` -> `3 passed`
    - [x] native RWKV7 `.pth` + txt tokenizer support：
        - config inference from local `.pth`
        - single-file `.pth` loader path
        - native weight remap
        - native txt tokenizer
        - native RWKV renderer
    - [x] real native checkpoint validation：
        - 1.5B remap sanity: `798 -> 795 mapped + 3 ignored`, no shape mismatches
        - 0.4B eager generation: coherent text
        - 0.4B eager chat: coherent text
        - 0.4B `compile + piecewise`: coherent text, capture confirmed
    - [ ] 设计真正 atomic 的 multi-state checkpoint publication：
        - 要么一次性发布 attn/recurrent/ffn 三段状态
        - 要么引入不会暴露 partial slot 的 staging/commit 机制
    - [ ] direct-write 优化完成后重跑 repeated-prefix / partial-hit /
    mixed-length serving benchmark
    - [ ] 若 direct-write 优化把 `all` 拉近或追平 `align`，再重新评估默认模式

## 2026-04-17 updates

- Done: native RWKV7 `.pth` checkpoint loading support
- Done: native RWKV `.txt` tokenizer support
- Done: native RWKV renderer path for chat requests
- Done: eager and compile+piecewise offline smoke on native `.pth`
- Remaining if needed later:
    - API-server smoke for native `.pth + .txt` on a roomier GPU / cleaner detached-process setup
    - compare native `.pth` outputs against original RWKV reference implementation token-by-token for a longer sample

## 2026-04-22 updates

- [x] HF RWKV slow tokenizer -> vLLM RWKV fast tokenizer adapter:
    - local HF RWKV tokenizer dirs now auto-resolve to `tokenizer_mode=rwkv`
    - `RWKVTokenizer` uses `pyrwkv_tokenizer.WorldTokenizer` when installed
    - Python layer preserves HF `added_tokens` semantics
    - verified `"\n\n"` keeps HF added id `65530`, not raw base-vocab id `261`
    - chat-template bos prefix is preserved for HF RWKV dirs
- [x] Targeted correctness validation:
    - `tests/tokenizers/test_rwkv.py`
    - `tests/renderers/test_rwkv.py`
    - `tests/model_executor/test_rwkv7.py`
    - real 0.1B / 0.4B HF tokenizer parity against original slow tokenizer
- [x] Tokenizer speed check:
    - 0.4B HF tokenizer-only encode loop: about `22.6x` faster
    - 0.4B HF batch `__call__`: about `9.1x` faster
- [x] Real 0.4B generation check:
    - long-output `18 x 64` workload:
        - fast median `447.021 tok/s`
        - slow median `454.831 tok/s`
    - prompt-heavy `120 x 1` workload:
        - fast median `296.304 tok/s`
        - slow median `303.403 tok/s`
    - current conclusion:
        - tokenizer CPU time is clearly improved
        - real offline output TPS does not meaningfully move on this 0.4B
      workload because model prefill/decode dominates
- [ ] Optional follow-up:
    - if API-server request admission becomes CPU-bound under high QPS, rerun
      fast vs slow through `/v1/completions` with concurrent clients rather than
      offline `LLM.generate`

## 2026-04-22 Rust tokenizer overlay follow-up

- [x] Rust tokenizer can now represent explicit token id `0`:
    - trie terminal id changed from sentinel `0` to `Option<u16>`
    - required for HF RWKV bos/pad/unk token
      `<|rwkv_tokenizer_end_of_text|> -> 0`
- [x] Rust tokenizer can now load an augmented vocab from memory:
    - `WorldTokenizer::from_buffer(&[u8])`
    - Python binding exposes `WorldTokenizer.from_buffer(bytes)`
    - vLLM uses this to append HF added/special tokens onto the base RWKV vocab
- [x] Rust decode now covers HF overlay ids:
    - `decode([0, 65530])` returns
      `"<|rwkv_tokenizer_end_of_text|>\n\n"`
    - `encode("\n\n")` returns HF id `65530`, not base id `261`, when vLLM
      builds the augmented buffer
- [x] vLLM now uses the local `/home/liu/rwkv-tokenizer` binding source:
    - installed with
      `uv pip install --python .venv/bin/python -e /home/liu/rwkv-tokenizer/bindings/python`
    - this is still a compiled Python extension, but the source of truth is the
      local third-party repo instead of an opaque installed wheel
- [x] Validation:
    - Rust crate `cargo test`: `6 passed`
    - Python binding `cargo check`: passed
    - vLLM tokenizer/renderer pytest: `6 passed`
    - vLLM RWKV7 pytest: `23 passed, 2 skipped`
    - targeted ruff/forbidden-imports/mypy: passed
- [ ] Optional follow-up:
    - add packaging metadata in vLLM if we want fresh environments to install
      `/home/liu/rwkv-tokenizer/bindings/python` automatically instead of
      running the local `uv pip install -e ...` command manually

## 2026-04-22 long-text tokenizer benchmark

- [x] Fixed long-text HF added-token boundary correctness:
    - whole-string augmented Rust encode can mismatch HF slow when a base token
      overlaps an added special token boundary, e.g. `。\n\nAssistant`
    - encode/batch encode now keeps Python HF special splitting first, then Rust
      handles ordinary spans
    - decode still goes through Rust for augmented backend ids
- [x] Added regression coverage:
    - `test_rwkv_tokenizer_prioritizes_hf_added_token_boundaries`
- [x] Long-text tokenizer speed measured on 0.4B HF tokenizer:
    - single encode: `16.4x` to `23.0x` faster across `1K` to `512K` chars
    - decode: `15.7x` to `21.5x` faster across the same text sizes
    - batch long-prompt encode: `13.7x` to `16.4x` faster
- [x] Validation:
    - tokenizer/renderer pytest: `7 passed`
    - RWKV7 model pytest: `23 passed, 2 skipped`
    - targeted ruff, mypy-local, forbidden-imports: passed

## 2026-04-22 real 0.4B long-prompt generation

- [x] Ran actual 0.4B `LLM.generate` long-prompt comparison:
    - model: `/mnt/d/codes/RWKV7-Goose-World2.9-0.4B-HF`
    - context limit: `2048`
    - prompt target: `1800` tokens
    - fast mode: `tokenizer_mode=auto`
    - slow mode: `tokenizer_mode=slow`
- [x] Single long prompt result:
    - `1800` input tokens, `8` output tokens
    - fast median wall: `0.484391s`
    - slow median wall: `0.469076s`
    - practical result: no meaningful win; slow slightly ahead in this run
- [x] Batch long prompt result:
    - `4 x 1800` input tokens, `1` output token each
    - fast median wall: `0.878082s`
    - slow median wall: `0.900331s`
    - practical result: no meaningful win; fast slightly ahead in this run
- [x] Current conclusion:
    - Rust tokenizer helps tokenizer-only long text by about `14x-23x`
    - actual 0.4B long generation remains model-prefill dominated

## 2026-04-22 server native vocab tokenizer panic

- [x] Diagnosed server crash with native `.pth + rwkv_vocab_v20250609.txt`:
    - Rust parser panic came from Python `bytes` repr tokens in the vocab
    - old parser only handled pure `\xNN` bytes
- [x] Fixed in `/home/liu/rwkv-tokenizer`:
    - commit `aa6f61a Parse Python bytes vocab tokens`
    - parser now supports plain ASCII bytes, common escapes, hex escapes, and
      octal escapes
    - parser errors now return `io::Error` instead of panic
- [x] Validation:
    - Rust tokenizer tests: `7 passed`
    - Python binding cargo check: passed
- [ ] Server action:
    - push/pull Rust tokenizer commit `aa6f61a`
    - reinstall binding in the target conda env from `bindings/python`
    - rerun the 30B native `.pth` serve command

## 2026-04-29 larger CMix / FFN probe

- [x] Evaluated a larger `CMix / FFN` half-fusion:
    - custom `_C.rwkv7_cmix_key_relu2`
    - fused region:
        - `mix + key GEMM + relu^2`
    - kept `value` on existing `RowParallelLinear`
- [x] Correctness:
    - direct kernel parity passed
    - hook/fallback CUDA tests passed during the probe
- [x] Direct FFN microbench:
    - actual `0.4B` layer, larger token blocks:
        - `64`: `1.05x`
        - `1024`: `1.10x`
        - `1984`: `1.10x`
    - small-token decode-like blocks:
        - `1`: `0.99x`
        - `4`: `1.07x`
        - `8`: `0.91x`
        - `64`: `1.14x`
- [x] Real `0.4B` isolated eager benchmark:
    - fixed-on flags:
        - `RWKV7_USE_FUSED_MIX6=1`
        - `RWKV7_USE_FUSED_KK_PRE=1`
        - `RWKV7_USE_FUSED_LNX_RKVRES_XG=1`
        - `RWKV7_USE_ALT_RECURRENT_KERNEL=1`
    - only toggled:
        - `RWKV7_USE_FUSED_CMIX`
    - prefill proxy:
        - `64`: `62.888ms -> 54.887ms`
        - `1024`: `132.673ms -> 130.020ms`
        - `1984`: `224.944ms -> 226.809ms`
    - decode `64 -> 32`:
        - TTFT `81.653ms -> 113.158ms`
        - latency `1126.332ms -> 1359.976ms`
        - TPOT `33.699ms -> 40.220ms`
    - decode `64 -> 64`:
        - TTFT `89.750ms -> 90.528ms`
        - latency `2292.257ms -> 2406.679ms`
        - TPOT `34.960ms -> 36.764ms`
- [x] Decision:
    - reject this half-fused FFN path
    - do not keep the code landed
    - likely issue:
        - medium/large token FFN path can improve
        - but small-token decode shape is not reliably better
- [ ] Next:
    - if `CMix / FFN` is revisited, try a more complete region fuse
    - otherwise move to a different runtime/kernel hotspot

## 2026-04-29 full local CMix / FFN probe

- [x] Evaluated a fuller local FFN fused path:
    - custom local `_C.rwkv7_cmix_layer`
    - fused region:
        - `mix + key GEMM + relu^2 + value GEMM`
    - TP design:
        - op returned local row-parallel output
        - TP all-reduce stayed in Python wrapper
- [x] Correctness:
    - direct kernel parity passed
    - hook/fallback CUDA tests passed during the probe
- [x] Direct FFN microbench:
    - decode-like small token blocks:
        - `1`: `1.01x`
        - `4`: `1.01x`
        - `8`: `1.03x`
        - `64`: `1.11x`
    - larger token blocks:
        - `1024`: `1.10x`
        - `1984`: `1.09x`
- [x] Real `0.4B` isolated eager benchmark:
    - fixed-on flags:
        - `RWKV7_USE_FUSED_MIX6=1`
        - `RWKV7_USE_FUSED_KK_PRE=1`
        - `RWKV7_USE_FUSED_LNX_RKVRES_XG=1`
        - `RWKV7_USE_ALT_RECURRENT_KERNEL=1`
    - only toggled:
        - `RWKV7_USE_FUSED_CMIX`
    - prefill proxy:
        - `64`: `45.393ms -> 53.212ms`
        - `1024`: `126.125ms -> 122.227ms`
        - `1984`: `215.026ms -> 210.284ms`
    - decode `64 -> 32`:
        - TTFT `75.940ms -> 80.214ms`
        - latency `1030.894ms -> 1071.333ms`
        - TPOT `30.805ms -> 31.972ms`
    - decode `64 -> 64`:
        - TTFT `69.444ms -> 75.649ms`
        - latency `1941.821ms -> 2068.324ms`
        - TPOT `29.720ms -> 31.630ms`
- [x] Decision:
    - reject this fuller local FFN fused path too
    - direct layer win did not translate to end-to-end serving win
    - this strongly suggests the remaining decode bottleneck is no longer
      “just fuse a bit more FFN math”
- [ ] Next:
    - move to `2. small token decode optimization`
    - treat `CMix / FFN` as low priority unless a substantially different idea appears

## 2026-04-29 small-token decode follow-up

- [x] Probe:
    - test whether storing RWKV7 shift states in cache/model dtype instead of
      fixed `fp32` reduces decode-path copy overhead
- [x] Correctness:
    - focused dtype-selection test passed
    - full `tests/model_executor/test_rwkv7.py -v` passed
- [x] Benchmark:
    - compare old behavior via `--mamba-cache-dtype float32`
    - against default `--mamba-cache-dtype auto`
    - keep all existing RWKV7 perf flags enabled
- [x] Result:
    - mixed prefill result
    - no stable decode win
    - `64 -> 64` decode latency / TPOT regressed
    - `64 -> 32` had slightly better steady-state TPOT on `auto`, but also a
      large first-round TTFT outlier and worse overall latency
- [x] Decision:
    - reject the shift-state cache-dtype change
    - keep all three RWKV7 states on the existing stable dtype path
- [x] Next decode-track candidate:
    - reduce the number of small GEMM launches / wrapper hops in the stacked
      projection and LoRA path
    - landed behind `RWKV7_USE_DIRECT_LINEAR`
    - direct `F.linear(...)` fast path is only enabled for:
        - CUDA
        - tensor parallel size `== 1`
        - unquantized RWKV7 internal linears
    - covered areas:
        - `RWKV7LoRA`
        - attention recurrent-input projections
        - attention `o_proj`
        - FFN `key/value`
- [x] Correctness:
    - full `tests/model_executor/test_rwkv7.py -v` passed
    - added equality tests for attention and FFN direct-linear flag paths
- [x] Benchmark:
    - isolated `0.4B` decode benchmark showed a strong decode win
    - real prompt validation showed:
        - clear gain on short, decode-heavy requests
        - near-flat or slightly negative result on longer prefill-heavy prompts
- [x] Decision:
    - keep `RWKV7_USE_DIRECT_LINEAR`
    - do not default-enable it yet
    - treat it as a decode-oriented experimental flag until more workload
      coverage is profiled
- [x] Follow-up:
    - narrow the direct-linear path to:
        - decode always on
        - prefill only when `num_tokens <= 128`
    - new tests confirm:
        - large prefill no longer calls `_rwkv7_direct_linear`
        - decode still matches reference even when the prefill threshold is
          exceeded
- [x] Updated benchmark:
    - fixed-shape `0.4B` benchmark still shows net decode wins
    - latest mixed real-prompt rerun shows:
        - short / medium / long prompts all improved in total latency
        - TTFT can still regress slightly on some prompt shapes
- [ ] Next follow-up:
    - decide whether this should become decode-only instead of
      decode-plus-small-prefill
    - if not, the next decode-track optimization should look beyond wrapper
      overhead and profile block-level launch structure

## 2026-04-27 RWKV7 official fused-kernel perf track

- [x] `P0` perf hook scaffolding:
    - feature flags are in place for `mix6`, `kk-pre`, epilogue, CMix, and
      future recurrent-kernel experiments
- [x] `P1` `mix6`:
    - Triton path landed behind `RWKV7_USE_FUSED_MIX6`
    - isolated 0.4B benchmark showed positive decode and short-prefill result
- [x] `P1` `kk-pre`:
    - Triton path landed behind `RWKV7_USE_FUSED_KK_PRE`
    - isolated 0.4B benchmark showed strongest decode improvement among the
      first two kernels
- [x] `P1` CMix probe:
    - mix-only fusion was tested and rejected
    - reason: hook-level speed was not better enough to land
    - next CMix attempt should fuse a larger FFN region, not just `addcmul`
- [x] `P2` attention epilogue:
    - Triton `lnx+rkvres+xg` path landed behind
      `RWKV7_USE_FUSED_LNX_RKVRES_XG`
    - correctness:
        - op-level float32/bfloat16 equality passed
        - `_finalize_attention_output` hook equality passed
        - targeted RWKV7 pytest: `10 passed`
    - microbench:
        - op-level: about `3.25x-10.19x`
        - hook-level including `o_proj`: about `1.82x-2.20x`
    - real 0.4B isolated benchmark:
        - decode `64 -> 32` TPOT: `37.934ms -> 30.600ms`
        - decode `64 -> 64` TPOT: `38.243ms -> 33.744ms`
        - long prefill proxy `1984`: `263.834ms -> 247.408ms`
- [ ] Next:
    - evaluate `RWKV7_CLAMPW_CUDA` / recurrent-core alternatives
    - do not replace the recurrent core until decode batch, prefill, varlen,
      prefix-cache, and checkpoint-state behavior have explicit tests

## 2026-04-27 recurrent-core next step

- [x] Evaluated official `rwkv7_clampw` vs current vLLM Triton recurrent kernel:
    - safe subset only:
        - contiguous prefill
        - `K=V=64`
        - zero initial state
        - no varlen/checkpoints
    - correctness:
        - max abs diff about `2.2e-8` to `5.2e-8`
    - speed:
        - official CUDA was about `2.17x-2.62x` faster than current Triton
- [ ] Implement stateful vLLM-owned CUDA op:
    - add initial recurrent state load
    - add final recurrent state store
    - support decode batch `T=1`
    - support non-varlen sequence `T>1`
    - constrain to `head_dim == head_v_dim == 64`
    - keep current Triton path for varlen and checkpoint-state paths
- [ ] Add correctness tests before benchmark:
    - zero initial state
    - nonzero initial state
    - decode batch exactness
    - sequence final-state exactness
    - fallback guard exactness
- [x] Rejected simple Triton scalar-value prototype:
    - correctness passed
    - performance was worse than current Triton:
        - `(B=1,T=1024)`: `1.5512ms -> 8.4710ms`
        - `(B=64,T=1)`: `0.4928ms -> 1.5707ms`
    - conclusion: recurrent-core speedup needs CUDA shared-memory style, not
      this Triton blocking
