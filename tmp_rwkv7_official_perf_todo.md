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
- `P1 CMix` 已完成 probe：
    - mix-only Triton path correctness 通过
    - microbench/hook-level 收益不足，未落地
    - 后续如继续，需要做更大粒度的 FFN 融合
- `P2 attention epilogue` 已完成首轮迁移：
    - Triton fused path 已接入
    - correctness tests 已补
    - op/hook microbench 显示明确收益
    - real 0.4B isolated serial benchmark 显示 decode TPOT 有实际收益
- `P2 recurrent alt kernel` 已完成首轮迁移与评估：
    - vLLM-owned CUDA op 已接入
    - correctness tests 已补
    - isolated op microbench 显示 recurrent 主核本体明显更快
    - real 0.4B isolated serial benchmark 显示收益较温和，且样本对口径比较敏感
- 当前下一项优先级：
    - 回到更大粒度的 `CMix / FFN` 或 runtime cleanup
    - `ALT recurrent` 先保留在 feature flag 后面，后续再做更大模型 / 更长 decode 验证

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

#### Current Probe (2026-04-27)

- Tried a minimal first pass that only fuses:
    - `mixed = hidden_states + delta * x_k`
- Correctness result:
    - Triton/reference equality passed
    - FFN hook equality passed
- Microbenchmark result:
    - op-level:
        - rows `32`: `0.82x` of PyTorch reference
        - rows `256`: `1.16x`
        - rows `2048`: `0.99x`
    - hook-level:
        - rows `32`: `0.53x`
        - rows `256`: `0.86x`
        - rows `2048`: `0.98x`
- Decision:
    - Do not land the mix-only CMix Triton path.
    - This is too small a slice; launch overhead and PyTorch's existing
      elementwise path are already good enough.
    - A useful CMix migration likely needs to fuse a larger region, at least:
        - token mixing
        - key projection
        - `sqrelu`
      and ideally evaluate whether the second projection can stay on the
      existing `RowParallelLinear` path without losing most of the gain.
- Next action:
    - Inspect whether official `_CmixLayerV2Fn` can be split into a vLLM-safe
      forward-only kernel that respects tensor parallel and quantized linear
      constraints.

#### Current Probe (2026-04-28)

- Landed a more conservative first pass:
    - generic CUDA `_C.relu2`
    - `ReLUSquaredActivation` now has a real CUDA backend
    - RWKV7 FFN routes `sqrelu` through it only when:
        - `RWKV7_USE_FUSED_CMIX=1`
- Correctness:
    - `tests/kernels/core/test_activation.py::test_activation -v`
        - `72 passed`
    - `tests/model_executor/test_rwkv7.py -k fused_cmix_activation -v`
        - `2 passed`
- Microbenchmark:
    - generic `relu2` op on `0.4B`-style `intermediate_size=4096` shapes:
        - roughly `0.86x ~ 1.08x`
    - direct `RWKV7FeedForward._apply_ffn()`:
        - tokens `64`: `1.00x`
        - tokens `256`: `1.12x`
        - tokens `1024`: `0.99x`
        - tokens `4096`: `1.00x`
- Real `0.4B` isolated serial benchmark:
    - model: `/mnt/d/codes/RWKV7-Goose-World2.9-0.4B-HF`
    - rerun artifacts:
        - baseline `/tmp/rwkv7_cmix_baseline_2.json`
        - fused `/tmp/rwkv7_cmix_fused_2.json`
    - summary:
        - prefill proxy:
            - `64`: `+9.15%`
            - `256`: `-6.05%`
            - `1024`: `+1.16%`
            - `1984`: flat
        - decode `64 -> 32`:
            - TTFT `+9.64%`
            - latency `-2.40%`
            - TPOT `-2.70%`
        - decode `64 -> 64`:
            - TTFT `+18.24%`
            - latency `+13.51%`
            - TPOT `+13.32%`
- Decision:
    - Keep this path behind the feature flag.
    - Do **not** treat it as the final CMix migration.
    - The activation-only slice is too small and too noisy.
    - If CMix is revisited, the next serious version should target a larger
      FFN region, not just CUDA `sqrelu`.

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

#### Current Status (2026-04-27)

- `Done (first pass)`
- 本地接入形态：
    - Triton fused `lnx+rkvres+xg`
    - feature flag: `RWKV7_USE_FUSED_LNX_RKVRES_XG`
    - `o_proj` 仍保留在 vLLM `RowParallelLinear` 路径，避免破坏 TP/量化线性层边界
- Correctness:
    - op-level reference equality:
        - `float32`
        - `bfloat16`
    - attention hook equality:
        - `_finalize_attention_output()` fused path matches unfused path
    - targeted pytest:
        - `10 passed`
- Microbenchmark:
    - op-level speedup:
        - tokens `1`: `4.31x`
        - tokens `16`: `10.19x`
        - tokens `64`: `6.31x`
        - tokens `256`: `3.76x`
        - tokens `1024`: `3.25x`
        - tokens `2048`: `3.26x`
    - hook-level speedup including `o_proj`:
        - tokens `1`: `2.20x`
        - tokens `16`: `2.13x`
        - tokens `64`: `1.90x`
        - tokens `256`: `1.82x`
        - tokens `1024`: `1.97x`
        - tokens `2048`: `1.83x`
- Real 0.4B isolated serial benchmark:
    - model: `/mnt/d/codes/RWKV7-Goose-World2.9-0.4B-HF`
    - args:
        - `--enforce-eager`
        - `--gpu-memory-utilization 0.6`
        - rounds `2`, warmup `1`
    - baseline vs `RWKV7_USE_FUSED_LNX_RKVRES_XG=1`:
        - prefill proxy `64`: `61.615ms -> 52.390ms` (`+14.97%`)
        - prefill proxy `1024`: `165.422ms -> 165.970ms` (`-0.33%`)
        - prefill proxy `1984`: `263.834ms -> 247.408ms` (`+6.23%`)
        - decode `64 -> 32`: TTFT `+29.80%`, TPOT `+19.34%`
        - decode `64 -> 64`: TTFT `+15.34%`, TPOT `+11.76%`
- Decision:
    - Land behind feature flag.
    - This is useful on decode and neutral-to-positive on long prefill in the current sample.

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

#### Current Evaluation (2026-04-27)

- Local prototype compared official CUDA `rwkv7_clampw` against current vLLM
  Triton `fused_mul_recurrent_rwkv7` on their shared safe subset:
    - `B=1`
    - contiguous prefill
    - `K=V=64`
    - zero initial state
    - no varlen
    - no checkpoint-state output
    - fp32 inputs
- Stable model-like inputs were used:
    - `kk` normalized over head dim
    - `a = sigmoid(randn)`
    - small `r/k/v`
    - vLLM receives `w = LOG_DECAY_SCALE * sigmoid(raw_w)`
    - official CUDA receives `raw_w`, `-kk`, and `kk*a`
- Correctness on this subset:
    - max abs diff stayed around `2.2e-8` to `5.2e-8`
    - mean abs diff stayed around `1.5e-9`
- Microbenchmark:
    - seq `16`: official `0.0198ms`, current `0.0519ms`, official `2.62x`
    - seq `64`: official `0.0883ms`, current `0.1932ms`, official `2.19x`
    - seq `256`: official `0.3209ms`, current `0.7580ms`, official `2.36x`
    - seq `1024`: official `1.1598ms`, current `2.5130ms`, official `2.17x`
- Interpretation:
    - official CUDA kernel is materially faster for the contiguous zero-state
      recurrent core
    - it is not a direct drop-in for vLLM serving because current vLLM needs:
        - initial recurrent state
        - final recurrent state
        - decode batch
        - varlen prefill
        - checkpoint states for prefix-cache/cache-all paths
    - a safe first landing should target only:
        - decode batch `T=1`
        - non-varlen `_run_recurrent_sequence`
        - `K=V=64`
      while keeping existing Triton for varlen/checkpoint paths
- Next implementation steps:
    - create a vLLM-owned CUDA op, not runtime `torch.utils.cpp_extension.load`
    - add initial-state load and final-state store to the official forward
      pattern
    - register behind `RWKV7_USE_ALT_RECURRENT_KERNEL`
    - route only when shape/dtype constraints are exactly satisfied
    - add tests for:
        - zero initial state
        - nonzero initial state
        - decode batch `T=1`
        - sequence `T>1`
        - fallback when `head_v_dim != head_dim` or varlen/checkpoints are used

#### Current Status (2026-04-28)

- `Done (experimental first pass)`
- 本地接入形态：
    - new vLLM-owned CUDA op:
        - `torch.ops._C.rwkv7_alt_recurrent`
    - feature flag:
        - `RWKV7_USE_ALT_RECURRENT_KERNEL`
    - 当前只在这些路径尝试启用：
        - `_run_recurrent_sequence()`
        - `_run_recurrent_decode_batch()`
    - 这些路径继续保留原 Triton fallback：
        - varlen / packed prefill
        - checkpoint-state collection
        - `head_dim != 64`
        - `head_v_dim != 64`
        - 非 fp32 / 非 contiguous 输入
- Correctness:
    - direct op equality:
        - nonzero initial state
        - `T=1`
        - `T>1`
    - attention hook equality:
        - sequence path matches reference
        - decode-batch path matches reference
        - unsupported head shapes fall back correctly
    - focused pytest:
        - `tests/model_executor/test_rwkv7.py -k "alt_recurrent or fused_recurrent_matches_reference or checkpoint_states_match_reference" -v`
        - `4 passed`
- Local microbenchmark on representative `0.4B`-style shapes (`H=16,K=64,V=64`):
    - decode-like `B=1,T=1`:
        - current Triton: `0.0341ms`
        - alt CUDA: `0.0179ms`
        - alt CUDA `+47.35%`
    - contiguous sequence `B=1,T=256`:
        - current Triton: `0.3609ms`
        - alt CUDA: `0.2501ms`
        - alt CUDA `+30.71%`
- Real `0.4B` isolated serial benchmark:
    - model:
        - `/mnt/d/codes/RWKV7-Goose-World2.9-0.4B-HF`
    - args:
        - `--enforce-eager`
        - `--gpu-memory-utilization 0.8`
    - first mixed run (`prompt 64/1984`, decode `64 -> 256`, rounds `2`):
        - result was noisy / conflicting and did not give a stable conclusion
    - focused decode-heavy rerun (`prompt 64`, decode `64 -> 256`, rounds `4`):
        - baseline:
            - TTFT `93.020ms`
            - latency `9807.160ms`
            - TPOT `38.095ms`
        - `RWKV7_USE_ALT_RECURRENT_KERNEL=1`:
            - TTFT `82.514ms`
            - latency `9456.780ms`
            - TPOT `36.762ms`
        - improvement:
            - TTFT `+11.29%`
            - latency `+3.57%`
            - TPOT `+3.50%`
- Decision:
    - Land behind feature flag only.
    - The recurrent kernel itself is clearly faster, but end-to-end serving gain
      is currently modest on local `0.4B`.
    - Do not make it default yet.
    - Next validation should prefer:
        - larger RWKV7 checkpoints
        - longer decode-heavy runs
        - before spending more time on prefill-side integration

#### Rejected Triton Probe (2026-04-27)

- Tried an official-shaped Triton prototype with one program per
  `(batch, head, value_channel)` and a `[K]` recurrent state vector.
- Correctness:
    - matched current vLLM Triton for output and final state on nonzero initial
      states
- Performance:
    - `(B=1,T=16)`: current `0.0376ms`, probe `0.1828ms`
    - `(B=1,T=64)`: current `0.1323ms`, probe `0.6545ms`
    - `(B=1,T=256)`: current `0.3918ms`, probe `2.0985ms`
    - `(B=1,T=1024)`: current `1.5512ms`, probe `8.4710ms`
    - `(B=16,T=1)`: current `0.1225ms`, probe `0.3766ms`
    - `(B=64,T=1)`: current `0.4928ms`, probe `1.5707ms`
- Decision:
    - Do not land this Triton variant.
    - The official speedup appears to depend on CUDA shared-memory/block
      organization, not just scalarizing the value dimension in Triton.

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

#### Current Status (2026-04-28)

- `Done (first targeted cleanup)`
- 本地首轮落点：
    - `cache_all + packed prefill`
    - 把 `_forward_runtime()` 里的逐 request checkpoint metadata 组装：
        - Python `for` loop
        - repeated `.item()`
        - list append + `torch.cat(...)`
      改成了张量 helper：
        - `_rwkv7_cache_all_packed_checkpoint_metadata(...)`
- Correctness:
    - helper/reference equality:
        - `test_rwkv7_cache_all_packed_checkpoint_metadata_matches_reference`
    - integrated cache-all coverage:
        - aligned-state prefill
        - multi-prefill
        - multi-prefill with nonzero prefix state
        - cache-all decode slot update
    - focused pytest:
        - `5 passed`
- Microbenchmark:
    - workload:
        - `num_prefills=64`
        - `max_blocks=16`
    - CPU:
        - old Python loop `1.917ms`
        - new helper `0.085ms`
        - `22.49x`
    - CUDA:
        - old Python loop `34.149ms`
        - new helper `1.288ms`
        - `26.52x`
- Decision:
    - Keep this cleanup.
    - It is a real runtime-side win for the `cache_all` prefix-caching path.
    - It does not by itself answer the more general “plain decode TPS” question,
      so the next `P3` work should keep targeting other Python-side metadata
      hotspots if they exist.

## 2026-04-29 Evaluation Note: Larger CMix / FFN Half-Fusion

- Evaluated but rejected a larger FFN probe:
    - custom `_C.rwkv7_cmix_key_relu2`
    - fused region:
        - `mix + key projection + relu^2`
    - left `value` on existing `RowParallelLinear`
- Why it is rejected:
    - hook-level correctness passed
    - direct FFN layer microbench looked mildly positive for larger token blocks
    - but real `0.4B` eager serving decode regressed
- Direct FFN layer microbench:
    - larger token counts:
        - `64`: `1.05x`
        - `1024`: `1.10x`
        - `1984`: `1.10x`
    - decode-like small token counts:
        - `1`: `0.99x`
        - `4`: `1.07x`
        - `8`: `0.91x`
        - `64`: `1.14x`
- Real `0.4B` isolated eager benchmark, with:
    - `RWKV7_USE_FUSED_MIX6=1`
    - `RWKV7_USE_FUSED_KK_PRE=1`
    - `RWKV7_USE_FUSED_LNX_RKVRES_XG=1`
    - `RWKV7_USE_ALT_RECURRENT_KERNEL=1`
    - and only toggling `RWKV7_USE_FUSED_CMIX`
- A/B summary:
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
- Conclusion:
    - Do not land the current half-fused FFN path.
    - If `CMix / FFN` is revisited, it should be a more complete region fuse,
      or we should move to a different hotspot instead.

## 2026-04-29 Evaluation Note: Full Local CMix / FFN Fusion

- Evaluated but rejected a fuller FFN probe:
    - custom local `_C.rwkv7_cmix_layer`
    - fused region:
        - `mix + key projection + relu^2 + value projection`
    - retained TP all-reduce outside the custom op
- Why it is rejected:
    - hook-level correctness passed
    - direct FFN layer microbench was consistently positive
    - but real `0.4B` eager serving still regressed on decode
- Direct FFN layer microbench:
    - decode-like small token counts:
        - `1`: `1.01x`
        - `4`: `1.01x`
        - `8`: `1.03x`
        - `64`: `1.11x`
    - larger token counts:
        - `1024`: `1.10x`
        - `1984`: `1.09x`
- Real `0.4B` isolated eager benchmark, with:
    - `RWKV7_USE_FUSED_MIX6=1`
    - `RWKV7_USE_FUSED_KK_PRE=1`
    - `RWKV7_USE_FUSED_LNX_RKVRES_XG=1`
    - `RWKV7_USE_ALT_RECURRENT_KERNEL=1`
    - and only toggling `RWKV7_USE_FUSED_CMIX`
- A/B summary:
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
- Conclusion:
    - Do not land this fuller local FFN fused path either.
    - The remaining decode bottleneck is unlikely to be solved by only fusing
      a bit more FFN math.
    - Move to `small token decode optimization`.

## 2026-04-29 Evaluation Note: shift-state cache dtype for small-token decode

- Evaluated but rejected a runtime-side decode probe:
    - store only the two shift states in `mamba_cache_dtype` / model dtype
    - keep the recurrent state in `fp32`
    - motivation was to reduce decode-path cached-state cast / copy traffic
- Why it is rejected:
    - correctness passed
    - prefill result was mixed
    - decode did not show a robust end-to-end win
- Real `0.4B` serving benchmark:
    - compared:
        - baseline via `--mamba-cache-dtype float32`
        - probe via default `--mamba-cache-dtype auto`
    - all existing RWKV7 perf flags enabled on both sides
- A/B summary:
    - prefill proxy:
        - `64`: `47.862ms -> 49.424ms`
        - `1024`: `133.934ms -> 122.305ms`
        - `1984`: `214.983ms -> 215.300ms`
    - decode `64 -> 32`:
        - TTFT `75.177ms -> 151.431ms`
        - latency `1046.909ms -> 1105.981ms`
        - TPOT `31.346ms -> 30.792ms`
    - decode `64 -> 64`:
        - TTFT `73.992ms -> 75.694ms`
        - latency `1986.962ms -> 2034.393ms`
        - TPOT `30.365ms -> 31.090ms`
- Interpretation:
    - `64 -> 32` included one obvious first-round `auto` outlier, so TTFT is
      noisier than the steady-state TPOT there
    - even so, the probe does not produce a clean decode win
    - it should not be landed as the next small-token decode optimization
- Next decode-track focus:
    - target reduction of small GEMM launch count / wrapper overhead
    - do not spend more time on cache-dtype tweaks unless a different
      state-layout design appears

## 2026-04-29 Evaluation Note: direct-linear path for small-token decode

- Landed a new experimental flag:
    - `RWKV7_USE_DIRECT_LINEAR=1`
- Scope:
    - RWKV7 internal linears only
    - CUDA only
    - tensor parallel size `== 1`
    - unquantized linear method only
- Main idea:
    - bypass the generic linear wrapper path for small RWKV7 internal
      projections and call `F.linear(...)` directly
    - target decode-side wrapper overhead instead of recurrent math itself
- Covered paths:
    - `RWKV7LoRA`
    - attention recurrent-input projections
    - attention `o_proj`
    - FFN `key/value`
- Correctness:
    - full `tests/model_executor/test_rwkv7.py -v`
        - `43 passed, 2 skipped`
    - added direct-path equality coverage for attention and FFN
- Isolated `0.4B` serving benchmark:
    - prefill proxy:
        - `64`: `66.572ms -> 54.778ms`
        - `1024`: `114.386ms -> 118.653ms`
        - `1984`: `200.983ms -> 207.067ms`
    - decode `64 -> 32`:
        - TTFT `86.361ms -> 69.866ms`
        - latency `1265.979ms -> 969.283ms`
        - TPOT `38.052ms -> 29.273ms`
    - decode `64 -> 64`:
        - TTFT `89.619ms -> 73.887ms`
        - latency `2450.318ms -> 1900.853ms`
        - TPOT `37.471ms -> 28.999ms`
- Real prompt validation:
    - mixed workload:
        - `zh_short`
        - `en_code`
        - `long_context`
    - long prompt was shortened to stay below the `2048` context limit while
      still asking for `64` output tokens
    - results:
        - short decode-heavy request improved clearly:
            - `zh_short` TTFT `117.920ms -> 105.838ms`
            - latency `2386.135ms -> 2170.705ms`
            - TPOT `36.584ms -> 32.776ms`
        - medium and long prompts were roughly flat or slightly slower:
            - `en_code` latency `2298.492ms -> 2307.628ms`
            - `long_context` latency `2462.533ms -> 2467.528ms`
- Conclusion:
    - this is the first confirmed win on the `small token decode` follow-up
      track
    - the benefit is workload-dependent and strongest when decode dominates
    - keep it behind a feature flag for now; do not default-enable until a
      broader workload sweep confirms the tradeoff

## Recommended Execution Order

1. `P0` 基线与 feature flag
2. `P1` `tmix_mix6_bf16_v5`
3. `P1` `tmix_kk_pre_bf16_v5`
4. `P1` fused CMix probe
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
