# RWKV7 精确 FP32 recurrent state-update 融合（Full Decode CUDA Graph）

- 日期：2026-08-21
- GPU：NVIDIA GeForce RTX 4090 D（sm89）
- 运行时：vllm-sp / Python 3.11.15 / PyTorch 2.11.0+cu130
- 模型：`/hikscale/models/XiaoKe/rwkv-0-hf`（`xiaoke-5`，BF16 投影、FP32 recurrent state）

## 改动

旧的 Albatross 风格整块 recurrent CUDA kernel 将 `sa` 和最终输出的 FP32
reduction 一起重写；其 reduction tree、`exp` 实现和 FMA/加法结合次序与
ATen reference 不同，因此多 token decode 会发生状态漂移。

本次只融合精度安全的 state-pointwise 子图：

```text
new_state = ((exp_w * state) + (kk_a * sa)) + (k * v)
```

其中 `torch.exp(w)`、`kk * a`、`sa` reduction 及 final-output reduction
仍由 eager ATen 执行。native CUDA kernel 用显式 round-to-nearest
`__fmul_rn` / `__fadd_rn` 和 `volatile` 保持 eager 的 materialized
FP32 乘法与左结合加法语义；避免了对巨大 `[B,H,64,64]` state 的五个中间
张量和对应的 global-memory 流量。

默认启用开关：`RWKV7_USE_EXACT_RECURRENT_T1_UPDATE=1`。不满足 CUDA、
FP32 contiguous 或 `[B,H,64,64]` guard 时自动回退到 reference。

## 精度门禁

同模型、同 Full Decode CUDA Graph、greedy sampling（`temperature=0`,
`top_p=1`, `seed=0`）、8 个固定 prompts、每 prompt 16 decode steps、
`logprobs=20`，仅切换该开关：

- text mismatch：**0 / 8**
- Top-1 disagreement：**0 / 128**
- common Top-K logprob 最大绝对误差：**0**
- selected-token logprob 最大绝对误差：**0**

此外，native operator 对 B=1/8/128 逐元素 `torch.equal`，并且 Full
CUDA Graph C=128 padding engine integration（含 compiled full graph）通过。
细节见 `accuracy.json`。

## 性能门禁

服务级 A/B 使用**相同重编译 `_C.abi3.so`**，仅切换
`RWKV7_USE_EXACT_RECURRENT_T1_UPDATE`，负载为 `/v1/completions`、128
requests、C=128、`max_tokens=32`、`ignore_eos=true`，每轮 4,096 输出 token。

| 并发 | Baseline TPS | Candidate TPS | 变化 |
| ---: | ---: | ---: | ---: |
| 128（两轮均值） | 487.91 | 696.58 | **+42.77%** |
| 64（单轮） | 436.11 | 589.25 | **+35.12%** |
| 8（单轮） | 149.68 | 154.28 | **+3.07%** |

C=128 的两轮基线是 487.11 / 488.71 TPS，candidate 是 695.00 /
698.15 TPS。原始运行摘要在 `raw/`，聚合数据在 `performance.json`。

## 验证命令

```bash
export VLLM_RWKV7_ENGINE_TEST_MODEL_PATH=/hikscale/models/XiaoKe/rwkv-0-hf
export RWKV7_USE_EXACT_RECURRENT_T1_UPDATE=1
/mnt/data/anaconda3/envs/vllm-sp/bin/python -m pytest \
  tests/model_executor/test_rwkv7.py -v
```

结果：`66 passed, 2 skipped`。两个 skip 是需要外部 reference runtime 的
既有 parity tests，不是本次优化引入的跳过。

服务命令沿用 Full Decode CUDA Graph 配置，并启用：

```bash
RWKV7_USE_TRITON_MASKED_STORE=1 \
RWKV7_USE_EXACT_RECURRENT_T1_UPDATE=1 \
/mnt/data/anaconda3/envs/vllm-sp/bin/vllm serve ...
```
