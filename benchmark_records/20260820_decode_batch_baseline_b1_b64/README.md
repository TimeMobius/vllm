# RWKV7 retained eager decode-batch baseline

Measured on the retained eager configuration: fused mix6 / kk-pre /
lnx-rkvres-xg / CMix / direct linear / alternate recurrent / T=1 recurrent and
cached FP32 attention parameters enabled. The fixed 209-token prompt was
warmed into the prefix cache before three measurements at each client
concurrency. Every request generated exactly 32 tokens with `temperature=0`
and `ignore_eos=true`.

| Client concurrency | Average aggregate TPS | P50 request latency | P95 request latency |
| ---: | ---: | ---: | ---: |
| 1 | 19.162 | 1.675 s | 1.675 s |
| 8 | 141.396 | 1.810 s | 1.820 s |
| 16 | 280.709 | 1.813 s | 1.836 s |
| 32 | 492.295 | 2.070 s | 2.077 s |
| 64 | 749.811 | 2.712 s | 2.726 s |

For B=1 and B=8, generated token-ID hashes were identical in all three rounds.
At B>=16, all output lengths remained 32 but the server's concurrent request
ordering/batch shape produced different deterministic token hash vectors across
rounds. The raw hashes are retained in `results.json`; this throughput baseline
does not treat those variants as an accuracy acceptance result. Serial output
regressions remain token-exact and are used for optimization gates.
