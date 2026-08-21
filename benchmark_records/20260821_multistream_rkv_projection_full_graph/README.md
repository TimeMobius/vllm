# RWKV7 C=1 multi-stream R/K/V projection

## Retained optimization

The three direct R/K/V projections consume different mixed inputs and have no
data dependency until RWKV recurrence. With the explicit
`RWKV7_USE_MULTISTREAM_RKV_PROJECTIONS=1` flag, a **single-token decode**
dispatches their existing BF16 `F.linear` operations to three lazily-created
CUDA child streams. A readiness event preserves all preceding dependencies and
the main stream waits for every branch before recurrence.

This does not change the input, weight, bias, dtype, algorithm selection, or
operand/reduction order inside any GEMV. It is deliberately disabled for graph
capture sizes above one, where stream synchronization did not produce a
material throughput gain.

## Strict gates

* Unit test: every child-stream BF16 R/K/V output is `torch.equal` to the
  original same-stream output.
* HTTP service gate: 8 prompts × 16 greedy decode with `logprobs=20`; complete
  baseline and candidate response traces are byte-identical.
* Full Decode CUDA Graph captured all configured sizes successfully.
* Focused RWKV suite: **93 passed, 5 skipped**.

## Paired fixed-payload results

| workload | baseline | candidate | change |
| --- | ---: | ---: | ---: |
| C=1, 32 generated tokens | 27.611 TPS | 27.934 TPS | +1.17% |
| C=1, 512 generated tokens | 27.628 TPS | 27.960 TPS | +1.20% |
| C=128 aggregate | 1563.04 TPS | 1571.71 TPS | +0.55% |

The two independent C=1 measurements both show approximately +1.2%, including
the long steady decode that removes first-token startup effects. This is a
strictly exact retained improvement, but it does **not** by itself reach the
user's 50+ C=1 TPS target. The dominant remaining work is the broader set of
BF16 projection/GEMV launches, not recurrence or state-copy overhead.

## Dashboard regression

The external dashboard completed successfully at **2026-08-21 16:23:03** and
saved eight records. C=1/2/4/8/16/32/64/128 all had 100% request success, with
per-request output TPS **32.5 / 29.3 / 33.9 / 32.8 / 31.5 / 26.3 / 22.7 / 17.5**.
This is a real OpenAI API stability regression; variable completion lengths and
per-request accounting mean it is not directly comparable with fixed-payload
C=128 aggregate TPS.
