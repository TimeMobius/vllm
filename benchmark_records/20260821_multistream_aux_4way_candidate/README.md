# Four-way auxiliary decode projection streams — rejected (2026-08-21)

## Hypothesis

The retained C=1 auxiliary DAG has three persistent CUDA child streams:

```text
R + G | K + W + A | V + V-gate
```

`A` is an independent rank-192 LoRA chain. This candidate created a fourth
persistent child stream and used:

```text
R + G | K + W | V + V-gate | A
```

No operation changed input, weight, activation, dtype, GEMV reduction order,
cache read/write, or recurrent math. The implementation also made the stream
set topology-aware, so a three-stream call cannot wait on an event belonging
to a previous four-stream graph topology.

## Correctness

- New CUDA unit coverage exercised both layer 0 and layer 1 with three-way and
  four-way paths. It passed with `torch.equal` outputs:

  ```bash
  /mnt/data/anaconda3/envs/vllm-sp/bin/python -m pytest \
    tests/model_executor/test_rwkv7.py::test_rwkv7_multistream_rkv_projection_preserves_bits -q -v
  # 1 passed
  ```

- The candidate service captured Full Decode CUDA Graph at its configured
  sizes and the service gate (eight greedy prompts × sixteen tokens,
  `logprobs=20`) was byte-identical to the strict BF16 reference:
  zero text/token/Top-1 differences and zero selected/common-Top-K logprob
  error.

## Restart-paired throughput

| workload | four-way candidate | three-way baseline | change |
|---|---:|---:|---:|
| C=1, 32 completion tokens | 28.6821 TPS | 28.4534 TPS | +0.804% |
| C=1, 512 completion tokens | 28.7910 TPS | 28.5628 TPS | +0.799% |
| C=128, 32-token aggregate | 1564.99 TPS | 1562.28 TPS | +0.173% |

## Decision

Reject and restore the retained three-way topology. The C=1 gain is real and
repeatable, but is below the repository's 1% retention gate while introducing
a fourth stream/event topology. It has no meaningful high-concurrency benefit,
and cannot materially advance the 50+ TPS target. The experiment is retained
only as evidence that this narrow stream-splitting space is nearly exhausted.
