# Recovered sparse CMix source-recovery / layer precision scan (rejected)

- Date: 2026-08-21
- GPU: NVIDIA GeForce RTX 4090 D (sm89)
- Runtime: vllm-sp / Python 3.11.15 / PyTorch 2.11.0+cu130
- Model: `/hikscale/models/XiaoKe/rwkv-0-hf` (`xiaoke-5`, BF16 projections)

## Why this experiment exists

The prior source-backed Albatross-style sparse CMix-down prototype showed a
real C=1 endpoint signal (+6.40%), but its FP32 ordered accumulation did not
match the dense BF16 GEMV reduction contract and therefore diverged during
multi-token service decode. A stale local object file still contained that
candidate, so it was linked **only temporarily** to answer a narrower question:
can an individual FFN layer preserve the service-level numerical envelope?

This was a diagnostic recovery experiment, not a production implementation:
its CMake input was a local `/tmp/...rwkv7_sparse_cmix.cu.o` file. The temporary
CMake, binding, Python, and model changes were fully reverted after testing; no
runtime or build dependency on that object remains in the repository.

## Method

The stable direct-cache Full Decode CUDA Graph server was stopped, rebuilt with
the recovered object, and started with:

```bash
RWKV7_USE_SPARSE_CMIX_DOWN=1
RWKV7_SPARSE_CMIX_DOWN_LAYERS=<one layer index>
```

The candidate only covered CUDA, BF16, TP=1, B=1, a contiguous
`[1, 16384]` FFN activation, and a bias-free down-projection. Four representative
layers (`0`, `1`, `30`, `60`) were tested separately against the stable strict
trace using eight greedy requests × sixteen decode tokens and `logprobs=20`.

## Accuracy result

| Allowlisted FFN layer | Text/token divergence | Max selected-logprob error | Max common Top-K error |
| ---: | --- | ---: | ---: |
| 0 | 0 / 8 | 0.113672 | 0.374853 |
| 1 | 1 / 8 | 1.017645 | 11.437089 |
| 30 | 1 / 8 | 1.816113 | 14.175959 |
| 60 | 0 / 8 | 0.0000122 | 0.0156164 |

Layer 60 is an important precision observation: locating the altered projection
at the final layer limits recursive amplification enough to keep greedy tokens
unchanged for this test, but it still changes logits/logprobs. It therefore does
**not** meet the repository's strict default requirement of zero text, token,
and logprob difference.

## Performance result

A layer-60-only C=1 benchmark (three 32-token runs; post-warm-up mean) measured
`26.2151 TPS` versus the strict direct-cache reference `26.2068 TPS`:
`+0.03%`, below noise and not useful even if its numerical error had been
acceptable.

## Decision and next step

Reject the recovered candidate for the stable default path. This is not a
premature rollback due to a minor tensor error: the layer scan established that
its reduction mismatch persists even in the most favorable placement, while
meaningful sparse coverage would need early/middle FFN layers that already cause
recursive token divergence.

A viable continuation must be source-backed and preserve the dense BF16
CUTLASS/GEMV numerical contract, e.g. a structured/tensor-core sparse
formulation or a projection fusion that does not alter individual GEMV reduction
semantics. Do not resurrect or commit a prebuilt-object dependency.
