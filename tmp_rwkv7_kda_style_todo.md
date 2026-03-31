# RWKV7 KDA-Style Adaptation Plan

## Conclusion

结论是：**可行，而且比继续硬推“整模型 fullgraph compile”更合理。**

## Progress Update (2026-03-31)

- Phase 1 的第一步已经落地：
  - `RWKV7Attention` 注册进了 `static_forward_context`
  - 新增了 `torch.ops.vllm.rwkv7_attention(...)`
  - 当前 recurrent 实现已经搬到 `RWKV7Attention._forward(...)`
- 这一步确实解决了最初的 compile 卡点：
  - `Dynamo bytecode transform time` 从约 `126.55s` 降到了约 `1.74s`
  - compile range `(1, 2048)` 的图编译约 `8.89s`
  - 总 `torch.compile` 时间约 `10.92s`
- 但这一步还没有把 non-eager 变成“可上线”状态：
  - 关闭 CUDA graphs 后，服务可以启动到 `/health`
  - 但 `one-shot` 与 `step-by-step` 仍然不一致
  - PIECEWISE CUDA graph capture 仍然会在 graph capture 阶段失败
- 因此当前策略是：
  - **保留这轮 custom-op 骨架**
  - **默认配置回到 eager**
  - **下一步继续调 compile correctness，而不是强推默认 non-eager**
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

RWKV7 当前 compile 路径的核心问题，不是 cache 语义，也不是服务路径，而是：

- [`RWKV7Attention.forward()`](/home/liu/vllm/vllm/model_executor/models/rwkv7.py#L348) 在 prefill 路径里包含 Python 时间步循环
- vLLM 的 `support_torch_compile` 封装走的是 `torch.compile(..., fullgraph=True)`
- Dynamo 会试图把这段 recurrence 整体追进图里，导致：
  - 冷启动编译非常慢
  - 或者直接在 graph-break / disable 方案上失败

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

1. 检查真正的 model-runner-owned cache backing store，而不是只看 `model.model.layers[*].kv_cache`
2. 把 token-id-controlled replay 扩到第三步甚至更后面的 decode step
3. 在 `i am` / `The capital of France is` 上重复受控 replay，而不只看 `北京是`
4. 复查当前 custom op fake impl / `mutates_args` / output buffer 语义，确认 compiled graph 没有错误重排 state 输出
5. 复查 `RWKV7Block.forward()` 里 prefill request 循环和 `_get_kv_state(...)` 在 compile 下是否仍有隐藏的 data-dependent 假设
6. 只在 correctness 重新成立之后，再回头试 PIECEWISE CUDA graph capture
7. 在 correctness 成立之前，不要再次把 non-eager 设为默认

这一步的核心收益是：

- compile 图不再直接吞下 RWKV7 prefill 的 Python 时间步循环
- 先把“服务能不能在非 eager 模式起起来”这个问题解决

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

### Phase 0. Freeze Baseline

- [ ] 记录当前 eager 基线为控制组
- [ ] 保留一组固定 prompts:
  - `i am`
  - `北京是`
  - `The capital of France is`
- [ ] 保留一组固定 benchmark 口径:
  - single request TPS
  - concurrent decode TPS
  - one-shot vs step-by-step correctness

### Phase 1. Build RWKV7 Custom-Op Boundary

- [ ] 在 [`rwkv7.py`](/home/liu/vllm/vllm/model_executor/models/rwkv7.py) 中新增 RWKV7 custom op wrapper
- [ ] 参考 [`kda.py`](/home/liu/vllm/vllm/model_executor/layers/kda.py) 的 `direct_register_custom_op(...)` 模式
- [ ] 设计 `rwkv7_attention(...)` 的入参
  - q/r/w/k/v/a/g 或投影后的张量
  - 输出 buffer
  - `layer_name`
- [ ] 添加 fake impl，满足编译期 shape 推断
- [ ] 让 `RWKV7Attention.forward()` 改成：
  - 只做 projection / reshape / buffer allocation
  - 调 `torch.ops.vllm.rwkv7_attention(...)`
  - 不直接写 Python recurrence

### Phase 2. Move Stateful Logic Behind `_forward`

- [ ] 新增 `RWKV7Attention._forward(...)`
- [ ] 在 `_forward(...)` 中通过 `get_forward_context()` 获取 metadata
- [ ] 通过 `forward_context.no_compile_layers[layer_name]` 取回层实例
- [ ] 在 `_forward(...)` 中处理：
  - cache 读取
  - state 写回
  - prefill / decode 分流
- [ ] 保持当前数学逻辑完全一致，先不做 kernel 优化

### Phase 3. Preserve Current Decode Batching

- [ ] 保留现有 `forward_decode_batch()` 路径
- [ ] 在 custom-op 后端里继续复用 batched decode 更新
- [ ] 确认并发 decode 行为不回退成串行

### Phase 4. Re-Validate Non-Eager Startup

- [ ] 再次测试：
  - `enforce_eager=False`
  - `cudagraph_mode=PIECEWISE`
  - `cudagraph_copy_inputs=True`
- [ ] 确认日志里：
  - 不再卡在 Dynamo 展开 RWKV7Attention prefill Python 循环
  - `/health` 能起来
  - `/v1/completions` 能返回

### Phase 5. Correctness Regression

- [ ] 跑单测：
  - [test_rwkv7.py](/home/liu/vllm/tests/model_executor/test_rwkv7.py)
- [ ] 跑服务正确性：
  - one-shot vs step-by-step
  - `0.1B`
  - `0.4B`
- [ ] 跑并发正确性：
  - concurrent 3 / 8
  - 输出是否稳定匹配 baseline

### Phase 6. Varlen Prefill Optimization

- [ ] 借鉴 Mamba/KDA 的 metadata 使用方式
- [ ] 用 `query_start_loc` 做 packed / varlen prefill
- [ ] 先做 RWKV7 prefill backend 的 varlen 版本
- [ ] 对照 FLA 的 `chunk_rwkv7` 接口和状态形状
- [ ] 再决定是否需要独立 Triton/CUDA kernel

### Phase 7. Decode Kernel Optimization

- [ ] 评估当前 batched decode 是否已足够
- [ ] 对照 FLA 的 `fused_mul_recurrent_rwkv7`
- [ ] 若仍慢，给 recurrent update 做 fused kernel
- [ ] 目标是减少 Python/launch overhead，而不是再改 cache 语义

### Phase 7.5. Helper Op Cleanup

- [ ] 评估是否需要把以下子步骤逐步 kernel 化
  - `addcmul` mixing
  - `k` update
  - output correction
- [ ] 对照 FLA 的：
  - `fused_addcmul_rwkv7`
  - `fused_k_rwkv7`
  - `gate_output_correction`

### Phase 8. Revisit Default Runtime Policy

- [ ] 只有当非 eager 路径稳定后，才考虑把它作为默认
- [ ] 在那之前，eager 基线仍然是唯一的稳定控制组
- [ ] 非 eager 应视为实验路径，不应替代当前稳定实现

## Files Most Likely To Change

### Immediate

- [rwkv7.py](/home/liu/vllm/vllm/model_executor/models/rwkv7.py)
- [test_rwkv7.py](/home/liu/vllm/tests/model_executor/test_rwkv7.py)

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

- [ ] 实现 RWKV7 custom op wrapper
- [ ] 把当前 `RWKV7Attention.forward()` 的 recurrence 从 compile 图里剥出来
- [ ] 先追求“non-eager path can boot”

这一步是整个 KDA-style 路线的第一块里程碑。
