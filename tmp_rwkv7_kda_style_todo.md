# RWKV7 KDA-Style Adaptation Plan

## Conclusion

结论是：**可行，而且比继续硬推“整模型 fullgraph compile”更合理。**

## Progress Update (2026-04-13)

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
- 下一步直接接这条基线继续补：
  - `PIECEWISE`
  - `compile_no_cg`
  的同口径对照

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
- [ ] 把 benchmark 扩成更稳定的多轮统计，减少单次波动
- [ ] 继续查 `no-cg` 的长输出高并发分叉：
  - `max_tokens=128`
  - concurrency `8`
- [ ] 做 TTFT / prefill-only benchmark，把长 prefill 和 decode 开销分开看
- [ ] prefix caching / mixed prompt lengths 覆盖

### Phase 5. Kernelization / Varlen Optimization

- [ ] 借鉴 Mamba/KDA 的 metadata 使用方式
- [ ] 用 `query_start_loc` 做 packed / varlen prefill
- [ ] 对照 FLA 的 `chunk_rwkv7` / `fused_mul_recurrent_rwkv7`
- [ ] 评估是否需要独立 Triton/CUDA kernel

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

- [ ] 做 TTFT / prefill-only benchmark，确认 `PIECEWISE` 在长 prefill 下到底有没有独立收益
- [ ] 把 default/PIECEWISE benchmark 做成更稳定的多轮统计，并持续追加到
  [tmp_rwkv7_benchmark_records.md](/home/liu/vllm/tmp_rwkv7_benchmark_records.md)
- [ ] 评估 prefix caching、长输出、mixed prompt lengths 是否还有隐藏分叉
- [ ] 继续盯 `no-cg` 的 `128 tokens + concurrency 8` mismatch

compile 路径已经不是“能不能跑通”的问题了。现在最该区分的是：
- 纯 `PIECEWISE` 已经是可用且正确的主线
- `FULL_AND_PIECEWISE` 仍然不安全
- `PIECEWISE` 在当前 `0.4B` benchmark 上还不是稳定的吞吐优势路径
- `no-cg` 仍有长输出高并发尾巴要清
