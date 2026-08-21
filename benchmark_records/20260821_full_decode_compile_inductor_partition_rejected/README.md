# Rejected RWKV7 full-graph `torch.compile` partition candidate

- 日期：2026-08-21
- Candidate：`RWKV7_COMPILE_WITH_FULL_CUDAGRAPH=1` 与 compilation config `use_inductor_graph_partition=true`。
- Baseline：当前 stable Full Decode CUDA Graph native state-cache 路径。
- 负载：128 closed-loop requests、每 request 32 outputs、C=128、4,096 aggregate output tokens。

## 精度

服务级 8 prompts × 16 decode tokens、greedy `logprobs=20`：

- text mismatch：0
- token mismatch：0
- selected logprob maximum absolute error：0
- common Top-K logprob maximum absolute error：0

## 性能

| Mode | C=128 TPS runs | mean |
|---|---|---:|
| baseline | 702.17, 707.88, 708.58 | 706.21 |
| compile + Inductor graph partition | 670.94, 706.47, 706.75 | 694.72 |

即使剔除 candidate 首轮启动扰动，后两轮也只有 706.61 TPS，仍没有超过 baseline。该 candidate 数值正确但没有稳定吞吐收益，不设置为默认。

`raw/` 保存六个原始服务 summary。
