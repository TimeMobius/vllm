# Fused RWKV7 decay-scale / cast preparation — rejected (2026-08-21)

## Candidate

The attention direct CUDA path normally materializes this BF16 pointwise
operation before widening the result for FP32 recurrence:

```text
w = LOG_DECAY_SCALE * sigmoid(w_lora(...))
```

The retained `rwkv7_cast_kk_pre` kernel already consumes the BF16 `w` tensor
while materializing FP32 recurrent inputs. This experiment added an opt-in
`RWKV7_USE_FUSED_W_SCALE_CAST=1` path that reproduces the separate BF16
multiply's rounding inside that kernel, removing the standalone scale launch.
It does not alter the BF16 projection, sigmoid, recurrent state, state-cache,
or CUDA Graph settings.

## Exactness

The scale must be narrowed back to BF16 before its FP32 recurrent read; directly
multiplying in FP32 differs for every tested lane. With that explicit writeback
contract, the following gates passed:

- `rwkv7_cast_kk_pre` versus separate BF16 multiply plus cast: `torch.equal`
  for B=1, 8, and 128;
- actual attention recurrent-input projection bundle: `torch.equal` at B=1,
  8, and 128;
- focused CUDA test command: **6 passed**;
- eight greedy OpenAI completions × sixteen decode tokens with `logprobs=20`:
  byte-identical JSON trace.

## Throughput

The baseline and candidate used the retained Full Decode CUDA Graph launch
configuration. C=128's first candidate sample was externally contaminated and
is saved for transparency; the decision uses an idle-GPU repeat.

| Workload | Baseline | Candidate | Change |
| --- | ---: | ---: | ---: |
| C=1, 32 output tokens | 28.9459 TPS | 28.9515 TPS | +0.019% |
| C=128, 32 output tokens/request aggregate | 1609.28 TPS | 1606.44 TPS | -0.177% |

## Decision

**Reject and restore the strict stable service.** The candidate has zero
observable accuracy error, but its C=1 change is noise and the repeatable
C=128 result is a slight regression. Removing one tiny elementwise launch is
not material in the current graph. This closes the simple decay-scale fusion
variant; future work should target a larger strict-compatible dense projection
or fused producer-consumer boundary.
