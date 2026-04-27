# RWKV7 Official Perf Migration TODO

## Goal

目标不是替换整套官方 `model.py`，而是评估并增量迁移 `/home/liu/RWKV-LM-V7` 与 `/home/liu/RWKV-LM/RWKV-v7/train_temp` 中**可提升 vLLM RWKV7 性能**的 fused 实现。

当前已确认：

- [RWKV-LM-V7/src/model.py](/home/liu/RWKV-LM-V7/src/model.py)
- [RWKV-LM/RWKV-v7/train_temp/src/model.py](/home/liu/RWKV-LM/RWKV-v7/train_temp/src/model.py)
- [RWKV-LM/RWKV-v7/rwkv_v7_demo_rnn.py](/home/liu/RWKV-LM/RWKV-v7/rwkv_v7_demo_rnn.py)
- [vllm/model_executor/models/rwkv7.py](/home/liu/vllm/vllm/model_executor/models/rwkv7.py)

在 RWKV7 数学本体上是同构的，差别主要在：

- 官方 `train_temp` 融合得更激进
- vLLM 需要适配 serving runtime
    - varlen prefill
    - decode batch
    - prefix cache
    - slot/state 管理
    - compile / cudagraph policy

所以迁移原则是：

- 不整套替换官方 `model.py`
- 只迁移局部 kernel / fused epilogue
- 每一步都必须保持 vLLM 现有调度接口与 correctness

## Progress Update (2026-04-27)

- `P0` hook / flag scaffolding 已完成。
- `P1 mix6` 已完成首轮迁移：
    - Triton fused path 已接入
    - correctness tests 已补
    - isolated serial benchmark 显示：
        - short prefill TTFT proxy 改善明显
        - decode TTFT / TPOT 也有正收益
- `P1 kk-pre` 已完成首轮迁移：
    - Triton fused path 已接入
    - correctness tests 已补
    - isolated serial benchmark 显示：
        - decode TTFT / TPOT 改善最明显
        - longer prompt TTFT proxy 也有正收益
- `mix6 + kk-pre` 组合在 clean serial benchmark 下仍是 net positive。
- 当前下一项优先级前移为：
    - `P1: Port Fused CMix / FFN`
    - 在继续做 attention epilogue 之前，先看 FFN 热路径能否带来更稳定的 decode 收益

## Non-Goals

- 不在这一轮里处理 QRWKV checkpoint 质量问题
- 不把官方训练路径直接嵌进 vLLM 主线
- 不为了追求单一 benchmark 提升而破坏：
    - one-shot vs step-by-step 一致性
    - prefix caching
    - mixed prefill/decode
    - native `.pt/.pth` 路径

## Current Bottlenecks In vLLM

基于当前 [rwkv7.py](/home/liu/vllm/vllm/model_executor/models/rwkv7.py) 的实现，最明显的热点是：

1. token mixing 还是 Python / 普通张量表达式拼出来的
2. `kk` 归一化与 `k` 调制没有 fused precompute
3. attention 后处理：
   - groupnorm
   - residual correction
   - output gate
   还是拆开的
4. FFN / CMix 还是普通张量实现
5. recurrent 主核虽然已经正确，但和官方最激进实现还有融合空间

## Priority Plan

### P0: Baseline And Hook Points

#### 优先级

- `P0`

#### 要解决的问题

- 在迁移任何 kernel 之前，先把“替换前后到底快了哪里、没快哪里”说清楚
- 避免 kernel 接上后 correctness 过了，但 TTFT / decode / long prefill 没收益

#### 建议接入点

- [vllm/model_executor/models/rwkv7.py](/home/liu/vllm/vllm/model_executor/models/rwkv7.py)
- [tmp_rwkv7_ttft_benchmark.py](/home/liu/vllm/tmp_rwkv7_ttft_benchmark.py)
- [tmp_rwkv7_benchmark_records.md](/home/liu/vllm/tmp_rwkv7_benchmark_records.md)

#### 实现步骤

1. 明确分 4 类 benchmark：
   - short prompt decode-heavy
   - long prompt prefill-heavy
   - mixed prompt lengths
   - step-by-step vs one-shot correctness replay
2. 给 RWKV7 新 kernel 预留 feature flag：
   - `RWKV7_USE_FUSED_MIX6`
   - `RWKV7_USE_FUSED_KK_PRE`
   - `RWKV7_USE_FUSED_LNX_RKVRES_XG`
   - `RWKV7_USE_FUSED_CMIX`
   - `RWKV7_USE_ALT_RECURRENT_KERNEL`
3. 每个 kernel 都走 A/B benchmark 记录。

#### 验收

- 替换前有基线
- 替换后有同口径对照
- 能按 feature flag 单独开关

### P1: Port `tmix_mix6_bf16_v5`

#### 优先级

- `P1`

#### 来源

- [RWKV-LM/RWKV-v7/train_temp/src/model.py](/home/liu/RWKV-LM/RWKV-v7/train_temp/src/model.py:558)

#### 要解决的问题

- 当前 `xr/xw/xk/xv/xa/xg` 六路 token mixing 是分散张量表达式
- 会带来额外 elementwise kernel launch 和内存带宽开销

#### 建议迁移形态

- 新增 vLLM 自定义 op，输入：
    - `hidden_states`
    - 六个 `x_*`
    - optional cached shift state / delta
- 输出：
    - `xr/xw/xk/xv/xa/xg`
    - 或更进一步直接输出 `delta`

#### 建议落点

- `csrc/rwkv7/` 新增 RWKV7 专属 CUDA op
- Python 侧在 [rwkv7.py](/home/liu/vllm/vllm/model_executor/models/rwkv7.py) 的 `_project_recurrent_inputs()` 接入

#### 实现步骤

1. 先实现纯 contiguous decode-batch 版
2. 再支持 prefill 路径使用的 `delta`
3. 确保和当前 `addcmul` 公式逐元素一致
4. 保留原始 Python fallback

#### 风险

- 只优化 mixing，不一定显著提升 long prefill 总时长
- 如果接口设计得太死，后面接 `varlen` 会返工

#### 验收

- logits 一致
- first-step compare 一致
- decode ITL 有明确改善

#### 当前状态

- `Done (first pass)`
- 本地接入形态：
    - Triton fused `mix6`
    - feature flag: `RWKV7_USE_FUSED_MIX6`
- 当前 isolated serial benchmark 结论：
    - prefill TTFT proxy:
        - prompt `64`: `63.519ms -> 46.759ms` (`+26.39%`)
        - prompt `1024`: `180.642ms -> 188.143ms` (`-4.15%`)
        - prompt `1984`: `269.706ms -> 262.678ms` (`+2.61%`)
    - decode:
        - `64 -> 32`: TTFT `+21.15%`, TPOT `+11.43%`
        - `64 -> 64`: TTFT `+5.68%`, TPOT `+7.06%`
- 结论：
    - 对 short prompt 和 decode 路径是有效优化
    - 对长 prompt prefill 不是主收益点

### P1: Port `tmix_kk_pre_bf16_v5`

#### 优先级

- `P1`

#### 来源

- [RWKV-LM/RWKV-v7/train_temp/src/model.py](/home/liu/RWKV-LM/RWKV-v7/train_temp/src/model.py:603)

#### 要解决的问题

- 当前：
    - `kk = normalize(k * k_k)`
    - `k = k * (1 + (a - 1) * k_a)`
  仍是分开做
- 这部分既有 elementwise 又有 normalize，是 attention 前非常热的路径

#### 建议迁移形态

- fused precompute op：
    - input: `k, k_k, a, k_a`
    - output: `k_adj, neg_kk, kk_a`

#### 为什么先做它

- 它数学边界清晰
- 输入输出张量形状稳定
- 对 decode 和 prefill 都能用

#### 实现步骤

1. 先做单路径 contiguous 版本
2. 用纯 PyTorch reference 做 bitwise/close-allclose 对照
3. 接入 `_project_recurrent_inputs()`
4. 与现有 fallback 并存

#### 风险

- normalize 的 eps、dtype、head reshape 必须和现有实现完全对齐
- 一旦 eps 不同，就容易出现长序列漂移

#### 验收

- long prompt step-by-step 不漂
- pure torch / vLLM / official demo rnn 数值趋势一致

#### 当前状态

- `Done (first pass)`
- 本地接入形态：
    - Triton fused `kk-pre`
    - feature flag: `RWKV7_USE_FUSED_KK_PRE`
- 当前 isolated serial benchmark 结论：
    - prefill TTFT proxy:
        - prompt `64`: `63.519ms -> 61.461ms` (`+3.24%`)
        - prompt `1024`: `180.642ms -> 155.154ms` (`+14.11%`)
        - prompt `1984`: `269.706ms -> 255.132ms` (`+5.40%`)
    - decode:
        - `64 -> 32`: TTFT `+27.34%`, TPOT `+20.42%`
        - `64 -> 64`: TTFT `+11.56%`, TPOT `+5.91%`
- 结论：
    - 目前是两项里对 decode 收益更稳定的一项
    - 对 longer prompt TTFT 也比 `mix6` 更直接

### P1: Port Fused CMix / FFN

#### 优先级

- `P1`

#### 来源

- [RWKV-LM/RWKV-v7/train_temp/src/model.py](/home/liu/RWKV-LM/RWKV-v7/train_temp/src/model.py:665)

#### 要解决的问题

- FFN 路径目前仍是：
    - token-shift
    - linear
    - `relu() ** 2`
    - linear
- 对 decode 来说这是稳定热路径

#### 建议迁移形态

- 新增 fused CMix op：
    - input: `x, x_k, key.weight, value.weight`
    - output: FFN result + final shift state

#### 实现步骤

1. 先做 decode 单 token / decode batch 版
2. 再决定是否扩成 prefill batch
3. 对齐当前：
   - `delta = prev - x`
   - `mixed = x + delta * x_k`
   - `sqrelu`

#### 风险

- 官方 `train_temp` kernel 假设的是训练/GPT contiguous 路径
- 直接照搬可能不适合 vLLM 的 varlen runtime

#### 验收

- decode ITL 改善最明显
- FFN 输出与现有实现严格对齐

### P2: Port `tmix_lnx_rkvres_xg_bf16_v1`

#### 优先级

- `P2`

#### 来源

- [RWKV-LM/RWKV-v7/train_temp/src/model.py](/home/liu/RWKV-LM/RWKV-v7/train_temp/src/model.py:619)

#### 要解决的问题

- 当前 attention 后处理是拆开的：
    - groupnorm
    - `r*k*r_k` correction
    - `* g`
    - output projection 前的 dtype 往返
- 这是高频小 kernel 链

#### 为什么是 P2

- 收益可能不错
- 但更容易踩数值一致性和 dtype 边界

#### 实现步骤

1. 先保留 `o_proj` 在 Python 层，先只融合：
   - groupnorm
   - residual correction
   - gate
2. 等 correctness 稳定后，再考虑把 `o_proj` 前后一起优化
3. 单独测短输出 decode ITL

#### 风险

- eps、weight/bias broadcast、head reshape 任何一点不一致都会导致输出漂移
- 这条路径更难 debug

#### 验收

- 一致性测试稳定
- decode small-batch latency 有可见收益

### P2: Evaluate `RWKV7_CLAMPW_CUDA`

#### 优先级

- `P2`

#### 来源

- [RWKV-LM/RWKV-v7/train_temp/src/model.py](/home/liu/RWKV-LM/RWKV-v7/train_temp/src/model.py:609)

#### 要解决的问题

- recurrent 主核是 RWKV7 最重的数学核心
- 如果替换成功，理论收益最大

#### 为什么不放 P1

- 迁移成本最高
- 和 vLLM runtime 的耦合最深
- 需要同时考虑：
    - decode batch
    - prefill
    - chunked prefill
    - future varlen kernelization

#### 实现步骤

1. 先只评估 decode batch 替换是否值得
2. 再评估 prefill contiguous 场景
3. 最后才考虑 varlen / packed prefill
4. 若只适合训练式 `[B,T,C]` contiguous，不直接替主线

#### 关键判断

- 如果收益主要来自“大段 contiguous prefill”
  而 vLLM 实际热点在 mixed runtime
  那它不一定是最优先移植对象

#### 验收

- 长 prompt prefill TTFT 有清晰提升
- decode 不退化
- 不破坏 prefix cache / mixed scheduling correctness

### P3: Runtime Integration Cleanup

#### 优先级

- `P3`

#### 要解决的问题

- 即使 kernel 更快，如果 `_forward_runtime()` 还是有很多 Python `.item()` 和 per-request loop
  端到端收益也可能被吃掉

#### 建议接入点

- [rwkv7.py](/home/liu/vllm/vllm/model_executor/models/rwkv7.py)

#### 实现步骤

1. 把新增 fused kernel 尽量设计成：
   - decode batch 直接消费 batched states
   - prefill batch 直接消费 packed metadata
2. 避免只优化 layer math，不优化 runtime path
3. 与 compile/no-cg、piecewise benchmark 一起看

#### 验收

- kernel 收益能真正体现在 wall-clock
- 不是只在 isolated microbenchmark 里好看

## Recommended Execution Order

1. `P0` 基线与 feature flag
2. `P1` `tmix_mix6_bf16_v5`
3. `P1` `tmix_kk_pre_bf16_v5`
4. `P1` fused CMix
5. `P2` `tmix_lnx_rkvres_xg_bf16_v1`
6. `P2` `RWKV7_CLAMPW_CUDA`
7. `P3` runtime integration cleanup

## Validation Matrix

每一步都必须过这组验证：

- Correctness
    - one-shot vs step-by-step
    - decode batch vs single decode
    - prefix caching
    - mixed prompt lengths
    - native `.pt/.pth`
    - HF RWKV tokenizer 路径
- Performance
    - TTFT
    - decode ITL
    - short prompt throughput
    - long prompt throughput
    - cold start overhead
- Regression Guard
    - eager
    - `compile/no-cg`
    - `PIECEWISE`

## Exit Criteria

只有满足下面条件，才认为“官方 fused 实现迁移到 vLLM”这条路线成立：

- 至少 1 到 2 个 `P1` kernel 在真实 vLLM serving benchmark 中有稳定收益
- correctness 全部通过
- 代码结构没有把 vLLM RWKV7 runtime 绑死成只能跑 contiguous 训练式路径
- 新 kernel 可以 feature-flag 关闭，方便 bisect 和回滚
