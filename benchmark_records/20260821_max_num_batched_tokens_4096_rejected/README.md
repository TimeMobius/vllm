# Rejected: `--max-num-batched-tokens 4096`

After the 8192 scheduler-budget trial did not help, the intermediate 4096
setting was tested to distinguish an extreme-buffer effect from a general
scheduler-budget effect. No model arithmetic changed: the 8-prompt × 16-token
`logprobs=20` trace was byte-identical and Full Decode CUDA Graph capture
succeeded for all configured sizes.

| Workload | Default 2048 | 4096 | Change |
|---|---:|---:|---:|
| C=1, 32-token output | 28.4540 TPS | 28.4699 TPS | +0.06% |
| C=128 aggregate, 32-token output | 1568.15 TPS | 1560.45 TPS | **-0.49%** |

The result confirms the default 2048 budget is the best tested scheduler point
for this workload. The startup script was restored without an override.
