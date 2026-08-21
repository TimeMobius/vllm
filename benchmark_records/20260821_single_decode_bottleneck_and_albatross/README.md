# RWKV7 单请求 decode 瓶颈与 Albatross 对照

- 日期：2026-08-21
- GPU：NVIDIA GeForce RTX 4090 D（sm89）
- 模型：`/hikscale/models/XiaoKe/rwkv-0-hf` / 同源 `/hikscale/models/XiaoKe/rwkv-0.pth`
- vLLM：当前稳定 Full Decode CUDA Graph 路径；BF16 projection、FP32 recurrent state。

## 结论

当前 C=128 aggregate throughput 已约 705 TPS；用户关注的“50+ TPS”应按 **单请求 decode（C=1）** 单独衡量。稳定 vLLM 的 C=1、32 output tokens、temperature=0、`ignore_eos=true` 三轮平均是 **25.74 TPS**。

2026-08-21 对完整模型的 vLLM Torch trace 显示，31 次 generation step 的主要 CUDA 时间不是 RWKV state cache，而是小矩阵 BF16 **GEMV**：三类 `internal::gemvx` kernel 共 984.67 ms，占 trace 约 82% 的 1.20 s CUDA time；大约 426 次小 linear 调用/step。`rwkv7_recurrent_t1_exact_update`、native gather 和 native masked store 合计远小于这一瓶颈。

因此，为提高 C=1，最高价值的方向是减少/融合 decoder 小 projection（尤其 low-rank projection 与 FFN down projection），而不是继续优化 cache gather/store。

## Albatross 的可比测试

Albatross `faster4_2605_cpp` 已使用**同一个** `rwkv-0.pth`，以 RTX 4090 D / sm89 编译并执行。它是 standalone CUDA Graph forward：不含 vLLM scheduler、OpenAI HTTP、token sampling、BF16 strict regression；其默认 runtime 是 FP16 weights/activation，并非可直接视为严格数值等价的 vLLM 替换。

| 路径 | B=1,T=1 raw graph ms | raw tok/s | 说明 |
|---|---:|---:|---|
| vLLM API stable | 约 38.85 ms/token | 25.74 | C=1 × 32 outputs 的端到端 API 平均；包含 vLLM sampling/scheduler |
| Albatross `--wkv32 --cmix-sparse off` | 31.6378 | 31.61 | standalone，dense CMix |
| Albatross `--wkv32` 默认 `cmix-sparse no-fc` | 24.2992 | 41.15 | standalone，sparse CMix down projection |

Albatross sparse CMix 在同一 checkpoint 的 B=1 graph 测试中把 standalone latency 从 31.64 ms 降至 24.30 ms（约 **+30.2%**）。这证明 FFN sparse down projection 是有实效的性能杠杆；但它尚未通过本仓库的 BF16/logprob/state 对齐，不能直接迁入默认路径。

README 顶部“145+ TPS”是其 **RTX 5090、FP16、standalone** 结果，不能与本机 RTX 4090 D 上包含 vLLM 服务栈的 25.74 TPS 直接比较。当前在本机、同 checkpoint、同 Albatross C++ 程序的复现实测是约 41.15 raw tok/s，而不是 100+。

## FFN 稀疏度证据

用 checkpoint 自带 HuggingFace RWKV7 BF16 model，在单 token stateful decode 的四个 token 上 hook 了 61 层 `ffn.key` 的 preactivation。`preactivation <= 0`（SqReLU 后严格为零）的 layer mean 比例分别为：

- 96.63%（token 65532）
- 92.90%（token 1）
- 91.61%（token 100）
- 93.29%（token 200）

因此 FFN hidden `[16384]` 在 C=1 高度稀疏，值得做一个**受 guard 保护的 BF16 sparse down-projection**候选。

## 下一步门禁

1. 从 Albatross 的 `cmix_sparse_spmv_relu_one_kernel` 做 isolated BF16 prototype，但保留 vLLM BF16 activation 语义；先比较 tensor/output `torch.equal`。
2. 若非 bit-exact，量化各层 output、服务 `logprobs=20`、61 层递归后的 token 分歧；调整 accumulation/rounding 后继续，而不是直接成为默认路径。
3. 只有在 8 prompts × 16 tokens 服务精度门禁、C=1 与 C=128 三轮 benchmark、focused suite 均通过时，才接入并提交。
4. C=128 下 Albatross sparse benefit 本身接近零（B=128 WKV32 dense 1263.82 vs sparse 1264.03 tok/s），因此这个候选优先改善 C=1/小 batch，不应承诺提升 128 并发 aggregate TPS。

`raw/` 保存本次 vLLM C=1 summary、Albatross sparse/dense 输出和 BF16 sparsity 原始日志。完整 Torch profile trace 位于临时路径 `/tmp/rwkv7_c1_torch_profile/`，未提交其 152 MB trace 文件。
