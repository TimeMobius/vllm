# RWKV7 Full Decode profile and strict-sparsity follow-up

## Scope

- Device: NVIDIA GeForce RTX 4090 D (sm89)
- Model: `/hikscale/models/XiaoKe/rwkv-0-hf`
- Stable math contract: BF16 projections, FP32 recurrent state, exact direct-cache recurrent reduction/update, Full Decode CUDA Graph.
- Service: 128 sequence slots, graph sizes `1,2,4,8,16,32,64,128`.
- Date: 2026-08-21 (Asia/Shanghai).

This record deliberately separates **aggregate C=128 throughput** from a
single request's token latency. The latter is the source of the perceived
"slow at high concurrency" behavior.

## Reproducible service measurements

The temporary `torch` profiler launch used the same stable environment flags as
`/tmp/start_rwkv7_fullgraph_t1_direct_cache.sh`; it only appended vLLM's
profiler configuration. The service was restored from the stable launch script
before the final health, accuracy, and C=1 checks.

| workload | measured result |
|---|---:|
| stable C=1, 32 greedy completion tokens, post-restore | **28.4517 TPS** |
| profiled C=1, same payload | 28.4099 TPS |
| profiled C=128, 128x32 closed-loop tokens | **1532.89 aggregate TPS** |
| C=128 Full Decode graph GPU span | 73.90--75.22 ms / graph replay |

At C=128, the fixed 32-token closed-loop measurement implies an idealized
average of about `1532.89 / 128 = 11.98` generated tokens/s/request. This is
not a scheduler failure: one graph must compute the state for every active
sequence before any of those requests can receive its next token.

## Bottleneck attribution

The profile captured the final stable implementation, including the retained
single-read/single-write full-fusion direct recurrent-cache kernel.

### C=1

The CUDA graph's GPU span was about 35.27 ms/token. BF16 `internal::gemvx`
projections accounted for 307.62 ms summed CUDA time across eight profiled
decode replays (the sum exceeds wall time because the retained C=1 projection
streams overlap work). The next individual kernels were much smaller:

- exact recurrent full fusion: 2.406 ms / 488 layer calls (4.93 us/call);
- native state gather/store: 1.147 / 0.979 ms / 976 calls;
- layer norm: 5.512 ms / 992 calls.

The CPU `execute_context` annotations were roughly 4 ms while the graph GPU
span was roughly 35 ms. Thus a host-only K-step unroll cannot produce the
requested 50+ strict C=1 TPS by itself; projection weight traffic is the
critical path.

### C=128

For seven profiled `generation_128` Full Decode graph replays:

- BF16 CUTLASS projection kernels: 339.377 ms total, about **48.48 ms/graph**;
- exact FP32 direct-cache recurrent full fusion: 155.119 ms total, about
  **22.16 ms/graph**;
- native state gather/store: 7.086 ms total, about 1.01 ms/graph;
- graph GPU span: 73.90--75.22 ms/graph.

Therefore the two largest strict-path levers are:

1. preserve the BF16 GEMV/GEMM reduction contract while reducing/accelerating
   decoder projection weight traffic; and
2. reduce FP32 recurrent-state materialization bandwidth without changing its
   FP32 state contract.

The retained direct-cache full fusion already addresses (2)'s highest-safe
part: it removed an extra full-state read and reached +12.91% C=128 in its
paired benchmark. The new trace confirms that further small cache-copy rewrites
are not the highest-return route.

## Albatross-style sparse CMix precision follow-up

A new `/tmp`-only tiled active-index prototype was used only to refine the
existing rejected sparse-CMix conclusion; it was **not** linked into vLLM,
loaded by the service, or added to this repository.

For BF16 `[1,16384] @ [4096,16384]^T` at 94% synthetic activation zeros, it
reduced an isolated dense down projection from 166.82 us to 118.36 us (1.41x),
but still differed in 2 of 4096 BF16 outputs (max absolute difference
0.000244140625). This is the same failure class as the prior ordered sparse
candidate: it improves the useful computation but cannot reproduce PyTorch's
BF16 GEMV reduction tree. The earlier full-service candidate had a real +6.40%
C=1 signal but failed the 8 prompts x 16 logprob gate with 2/8 text/token
divergences. No sparse path is retained.

This is not an abandonment of precision work. It narrows the remaining viable
sparse route: a future candidate must use a Tensor-Core/CUTLASS-compatible
formulation that preserves the dense BF16 reduction order (for example, only
skipping mathematically whole tiles while maintaining that kernel's accumulator
schedule), before it may enter service-level testing.

## Upstream review outcome

`upstream/main` was fetched for review only. The current branch is deeply
forked around RWKV7 state management, so no blind merge/rebase is safe. Recent
vLLM graph/speculation optimizations are model-specific (MTP/draft-model or
attention/KV-residency assumptions) and do not provide an immediately portable
RWKV7 decode improvement. The reusable design lessons remain persistent GPU
metadata, fixed graph buffers, padding safety, and warmup/fallback discipline;
all are already present in the retained RWKV7 path.

## Decision

- Keep the strict stable service unchanged.
- Do not retain the tiled sparse prototype because it fails exact output before
  the service-level recursive logprob gate.
- Do not claim that 50+ strict C=1 TPS is reachable on this RTX 4090 D through
  CUDA-graph or scheduler tuning alone. The profile shows the required gain
  must come from a new strict-compatible projection/state algorithm or a
  higher-bandwidth device.
- Next experimental priority: reverse-engineer/match the PyTorch BF16 GEMV
  reduction schedule sufficiently to make a tile-sparse CMix down projection
  bit-exact, then run isolated -> layer -> 8x16 logprob -> Full Graph -> C=1
  and C=128 gates.
