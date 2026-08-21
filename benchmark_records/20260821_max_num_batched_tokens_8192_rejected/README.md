# Rejected: `--max-num-batched-tokens 8192`

The OpenAI-server default on this RTX 4090 D is 2048 scheduler tokens. The
128-request benchmark has 49 prompt tokens/request (6272 prompt tokens total),
so 8192 was evaluated as a scheduler-only high-concurrency candidate. It does
not alter model arithmetic: service trace remained byte-identical and all Full
Decode graph sizes captured successfully.

| Workload | Default 2048 | 8192 | Change |
|---|---:|---:|---:|
| C=1, 32-token output | 28.4540 TPS | 28.4827 TPS | +0.10% |
| C=128 aggregate, 32-token output | 1568.15 TPS | 1565.60 TPS | **-0.16%** |

The small C=1 move is noise and C=128 regressed. The service startup script was
restored without the explicit 8192 override.
