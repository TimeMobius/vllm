# Selective native LoRA-input BF16 GEMV, W+A only — rejected (2026-08-21)

## Goal

The previous all-LoRA-input native GEMV candidate was strict but slowed C=1.
Instead of discarding the idea without isolation, this experiment recovered its
strict out-variant as a private, temporary `rwkv7exp` extension and restricted
it to the critical child-stream group:

```text
K + W + A child stream
```

Only W- and A-LoRA's first `4096 -> 192` BF16 projection was replaced. The
rank-192 output, activation, second BF16 `F.linear`, all other projections,
FP32 recurrent update, state cache, CUDA-graph settings and scheduler remained
unchanged.

The extension was deliberately loaded from `/tmp` through explicit experimental
environment variables and is not a production dependency.

## Accuracy / Full Graph

- Service startup completed with `FULL_DECODE_ONLY` CUDA Graph capture sizes
  `1,2,4,8,16,32,64,128`.
- Eight greedy prompts × sixteen tokens, `logprobs=20`, were byte-identical to
  the strict BF16 reference: zero text/token/Top-1 differences and zero
  selected/common-Top-K logprob error.

## Throughput

| workload | candidate | strict stable baseline | change |
|---|---:|---:|---:|
| C=1, 32-token output | 28.0572 TPS | 28.4534 TPS | **-1.39%** |
| C=128, 32-token aggregate | 1569.90 TPS | 1562.28 TPS | inactive / noise |

The native operator is intentionally M=1-only, so the C=128 number does not
measure this candidate and must not be attributed to it.

## Decision

Reject. The selected critical W+A group is still slower despite exact output.
This rules out the remaining plausible scheduling explanation for the earlier
all-LoRA input result: this out-variant's graph/operator integration cost and
stream interaction dominate its isolated GEMV saving. The model source was
restored and the regular strict BF16 service restarted.
