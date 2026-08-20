# Cached FP32 RWKV7 attention constants

This retained optimization lazily caches the local FP32 forms of `k_k`, `k_a`,
and `r_k` behind `RWKV7_USE_CACHED_FP32_PARAMS=1`.  The BF16 checkpoint
parameters are immutable for normal inference, but eager decode otherwise
performs these small casts for every token and each of the 61 layers.

## Workload

- GPU: NVIDIA GeForce RTX 4090 D (SM89)
- Model: `/hikscale/models/XiaoKe/rwkv-0-hf`
- Service: `--enforce-eager`, retained fused flags and
  `RWKV7_USE_FUSED_RECURRENT_T1=1`
- Requests: eight fixed prompts concurrently, 32 generated tokens each,
  `temperature=0`, `ignore_eos=true`; one warm-up plus five measured rounds

## Result

| Variant | Aggregate TPS | Relative |
| --- | ---: | ---: |
| Cache disabled | 133.775 | baseline |
| Cached FP32 params | 137.972 | **+3.14%** |

The serial cache-off/cache-on token-ID regression is exact for all 8 prompts
(256 generated token IDs).  The dedicated CUDA unit test also confirms numeric
reference parity, cache reuse, and invalidation after a parameter mutation.

The external remote harness also completed successfully with this cache-on service: `Test completed: 8 records saved`; see `remote_service_test.json`. This is an integration/stability result and is not mixed into the local fixed-load TPS A/B.

Raw A/B throughput and output artifacts are retained in this directory.
