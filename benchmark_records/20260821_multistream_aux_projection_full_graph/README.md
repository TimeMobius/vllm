# RWKV7 C=1 balanced auxiliary projection streams

## Candidate

`RWKV7_USE_MULTISTREAM_AUX_PROJECTIONS=1` extends the retained C=1 R/K/V
child-stream path without changing the numerical contract of any projection.
It is enabled only when the direct CUDA linear path is selected, the
projection batch has exactly one row, and the caller is `forward_decode_batch`;
prefill (including a one-token prefill) stays on the single-stream route. Batches above one, including the
C=128 Full Decode CUDA Graph replay, remain on the established single-stream
path.

After `mix6`, the attention branches have independent BF16 inputs. The target
checkpoint has 4096-wide R/K/V projections; its W/A LoRAs have rank 192, G has
rank 384, and V-gate has rank 128. The candidate reuses the existing three
persistent per-layer streams with balanced groups:

```text
stream 0: R projection + G LoRA
stream 1: K projection + W LoRA + A LoRA
stream 2: V projection + V-gate LoRA (layers > 0)
main:     wait all groups -> existing BF16-to-FP32 conversion, kk_pre,
          exact FP32 direct-cache recurrent path, tail, and o_proj
```

Every `F.linear`, activation, sigmoid, scaling operation, `lerp`, dtype cast,
and recurrent update retains its original inputs, order, dtype, and reduction
contract. Only the launch stream changes for data-independent branches.

## Correctness gates

- CUDA BF16 unit checks for layer 0 and layer 1 recurrent-input bundles:
  all returned tensors are `torch.equal` to the single-stream route over
  repeated calls.
- Service gate: 8 prompts × 16 greedy tokens with `logprobs=20`; the complete
  response trace (text, token IDs, selected logprobs, and Top-K logprobs) is
  byte-identical to R/K/V-only baseline.
- The configured Full Decode graph sizes `1,2,4,8,16,32,64,128` captured
  successfully on service startup.
- Focused suite:
  `93 passed, 5 skipped`.

## Fixed-payload performance

| Workload | R/K/V-only baseline | Balanced aux streams | Change |
|---|---:|---:|---:|
| C=1, 32-token output | 27.9366 TPS | 28.4571 TPS | **+1.86%** |
| C=1, 512-token steady decode | 27.9596 TPS | 28.4891 TPS | **+1.89%** |
| C=128 aggregate, 32-token output | feature inactive | 1567.52 TPS | not attributed |

The C=1 gain is repeatable and strict, so the candidate is retained. It does
not change high-concurrency math or claim a C=128 benefit. The remaining
barrier to a 50+ C=1 TPS target is the many exact BF16 GEMV/projection launches,
not RWKV recurrence or shift-cache movement.

## External dashboard regression

A fresh POST to the dashboard completed successfully on **2026-08-21 16:45:52**
and saved eight records. C=1/2/4/8/16/32/64/128 all had 100% request success.
Per-request output TPS was **36.2 / 31.2 / 34.2 / 32.5 / 30.8 / 26.2 / 22.2 /
17.3**; overall TPS was **35.6 / 30.7 / 33.5 / 31.8 / 30.0 / 25.4 / 21.5 /
16.8**. Completion lengths are variable, so this is an API-stability regression
and not a comparison against the fixed-payload C=128 aggregate benchmark.
