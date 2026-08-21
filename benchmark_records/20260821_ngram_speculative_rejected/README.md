# RWKV7 ngram speculative decode (rejected)

- Date: 2026-08-21
- GPU: NVIDIA GeForce RTX 4090 D (sm89)
- Runtime: vllm-sp / Python 3.11.15 / PyTorch 2.11.0+cu130
- Model: `/hikscale/models/XiaoKe/rwkv-0-hf` (`xiaoke-5`, BF16 projection)

## Candidate

To evaluate an exact-verification route toward the C=1 target without changing
model weights, the stable server was restarted with vLLM's non-model-based
ngram proposer:

```json
{
  "method": "ngram",
  "num_speculative_tokens": 4,
  "prompt_lookup_min": 2,
  "prompt_lookup_max": 5
}
```

This is useful to evaluate separately from approximate FP8/sparse kernels:
the proposer can only suggest existing token patterns, while the target model
still verifies them.

## Runtime compatibility result

The candidate is incompatible with the main optimization objective on this
model/runtime. During startup vLLM emitted:

```text
CUDAGraph is not supported for ngram-based speculative decoding;
setting cudagraph_mode=PIECEWISE
```

It also disabled async scheduling for ngram speculation. Therefore the
candidate does **not** retain `FULL_DECODE_ONLY` CUDA Graph, so its draft-token
acceptance must overcome both graph loss and scheduling overhead.

## Accuracy gate

Against the strict direct-cache service trace, eight greedy requests × sixteen
decode tokens produced no text or token-ID divergence, but did produce nonzero
logprob differences:

```text
max selected-token logprob error: 0.00190655
max common Top-K logprob error: 0.12647915
```

This fails the repository's strict default parity contract. The token agreement
is encouraging evidence that target verification works for this limited case,
but it is not sufficient to enable an alternate default serving path.

## C=1 performance gate

Each run generated 32 tokens with `temperature=0`, `ignore_eos=true`, and one
request. The post-warm-up mean is used for comparison.

| Mode | C=1 TPS |
| --- | ---: |
| Strict direct-cache Full Decode CUDA Graph | 26.2068 |
| Ngram speculative decoding | 17.2220 |
| Change | **-34.29%** |

The server did report a mean acceptance length of 3.50 and a 62.5% average
per-position draft acceptance rate during the probe. That acceptance cannot
compensate for losing Full Decode CUDA Graph and async scheduling, and the
ngram method is slower even before considering its nonzero logprob error.

## Decision

Do not retain ngram speculative decoding for RWKV7 on this runtime. The stable
exact direct-cache Full Decode CUDA Graph service was restored after testing.

A future speculative path would need a compatible RWKV draft/MTP model and
preserve the linear-attention state commit/rollback semantics. It should be
considered only after a draft checkpoint exists; it cannot be substituted by
prompt-lookup ngrams for the general C=1 TPS target.
