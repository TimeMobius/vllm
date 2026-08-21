# RWKV7 精确 FP32 recurrent output-reduction 融合（Full Decode CUDA Graph）

- 日期：2026-08-21
- GPU：NVIDIA GeForce RTX 4090 D（sm89）
- 运行时：vllm-sp / Python 3.11.15 / PyTorch 2.11.0+cu130
- 模型：`/hikscale/models/XiaoKe/rwkv-0-hf`（`xiaoke-5`，BF16 projection、FP32 recurrent state）

## 改动

此前 T=1 exact state-update 已保留 `final_state * r` 的 eager `aten::mul`
和 FP32 D=64 `aten::sum`。此路径在每层都物化一个 `[B,H,64,64]` 临时乘积，
并发射两个 kernel；在 C=128 时这是明显的 recurrent memory/launch 开销。

本次 native CUDA kernel 精确复刻本环境 PyTorch 的：

```text
at::native::reduce_kernel<128, 4, ReduceOp<float, sum, ..., 4, 4>>
grid = [B * H * 64 / 128, 1, 1]
block = [32, 4, 1]
```

其中 4 个 y-lane 按 ATen 相同的四个独立 accumulator 处理 D=64，再按相同
`4 -> 2 -> 1` shared-memory 加法树合并。逐元素 FP32 `__fmul_rn` 保持 eager
`aten::mul` 的 materialized product rounding，因此它既去掉临时乘积和一次
launch，又保留精确 reduction contract。

默认启用：`RWKV7_USE_EXACT_RECURRENT_T1_OUTPUT_REDUCTION=1`。CUDA、FP32、
contiguous、`[B,H,64,64]` / `[B,H,64]` guard 不满足时自动回退 eager ATen；
可以设置该环境变量为 `0` 进行 A/B 或紧急回退。

## 精度门禁

- isolated operator：B=1/8/128，各 5 个随机 seed，`torch.equal=True`；
- 服务：同 Full Decode CUDA Graph、greedy、8 prompts × 16 steps、`logprobs=20`；
  text mismatch、Top-1 disagreement、selected/common Top-K logprob 误差均为 **0**。

## 性能门禁

同一 rebuilt `_C.abi3.so`，仅切换 output-reduction 开关。每轮 128 个请求、
C=128、32 output tokens/request（4096 token/round）；第一轮仅预热，后两轮均值：

| 并发 | Baseline TPS | Candidate TPS | 变化 |
| ---: | ---: | ---: | ---: |
| 128 | 706.63 | 793.18 | **+12.25%** |
| 1 | 25.94 | 26.00 | +0.23% |

C=128 的 baseline 为 709.26 / 706.69 / 706.57 TPS；candidate 为
779.31 / 793.97 / 792.38 TPS。C=1 收益接近测量噪声，因此下一阶段仍应优先
解决 426 个小 BF16 GEMV/projection 的 launch/矩阵乘瓶颈。

原始 summary 位于 `raw/`，聚合结论位于 `performance.json`。
