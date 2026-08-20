# RWKV7 long-context prefill/decode baseline (4K–256K)

This is the retained eager configuration with cached FP32 attention constants
and the fused T=1 recurrent kernel. Each request uses a unique raw-token
prompt to avoid prefix-cache reuse; B=1 and 16 generated tokens.

| Prompt length | Prefill prompt tok/s | Aggregate output TPS |
| ---: | ---: | ---: |
| 4,096 | 2,250.40 | 8.7906 |
| 16,384 | 3,215.40 | 3.1400 |
| 65,536 | 3,577.33 | 0.8734 |
| 262,144 | 3,668.57 | 0.2239 |

The 1K and 1,984-token baseline matrix is retained in
`benchmark_records/20260820_prefill_decode_baseline_1k_2k/`.
