# RWKV7 model-dtype shift-cache candidate

## Retained optimization

`RWKV7_USE_MODEL_DTYPE_SHIFT_CACHE=1` stores only the attention and FFN
**token-shift** cache entries in the model activation dtype. For the stable
BF16 model, those two entries are BF16. The FP32 recurrent matrix cache remains
FP32 without exception.

The change is lossless because both shift cache values are the previous
model-dtype activation and the existing decode path consumes them as the next
step's `hidden_states.dtype`. The former layout did exactly `BF16 -> FP32 ->
BF16`; the new layout retains the same BF16 bits directly.

## Strict gates

* A four-step isolated BF16 block-decode test compared every returned tensor
  and final cache entry using `torch.equal`.
* The real HTTP service ran 8 prompts × 16 greedy decode tokens with
  `logprobs=20`; the two complete response trace files were byte-identical.
* Focused RWKV model/TP/logprob regression suite: **92 passed, 5 skipped**.

## Fixed-payload A/B

| workload | baseline | candidate | change |
| --- | ---: | ---: | ---: |
| C=1 output TPS | 27.394 | 27.603 | +0.76% |
| C=128 aggregate output TPS | 1542.50 | 1566.23 | +1.54% |

A final clean restart of the retained service reproduced byte-identical HTTP output and reached C=1 **27.588** TPS and C=128 **1572.816** aggregate TPS.

This cannot reach the requested C=1 50 TPS by itself: profiling still assigns
about 82% of C=1 generation CUDA time to the many dense BF16 GEMV projections.
It is retained because it is strictly exact, makes Full Decode CUDA Graph C=128
faster, and saves 999,424 bytes per RWKV state slot (122 MiB at 128 slots;
3,904 MiB at 4,096 slots).

## Dashboard regression

The external dashboard POST completed successfully at **2026-08-21 15:29:59**:
all eight concurrency points (1, 2, 4, 8, 16, 32, 64, 128) had 100% request
success. Its per-request output TPS was respectively **37.8, 33.0, 34.3, 33.7,
31.5, 26.8, 22.7, 17.8**. This is a real OpenAI API stability/latency gate; its
variable completion lengths mean it must not be compared directly with the
fixed-payload C=128 aggregate TPS above.
