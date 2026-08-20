# RWKV7 1K / 2K prefill-decode baseline

Configuration: eager, prefix caching, `mamba-cache-mode=align`, and all retained fused flags including `RWKV7_USE_FUSED_RECURRENT_T1=1`.

| Prompt tokens | Concurrency | Output tokens | Aggregate TPS | Token parity |
|---:|---:|---:|---:|---|
| 1024 | 1 | 16 | 15.398 | exact |
| 1024 | 4 | 16 | 35.573 | exact |
| 1024 | 8 | 16 | 50.226 | exact |
| 1984 | 1 | 16 | 15.471 | exact |
| 1984 | 4 | 16 | 34.207 | exact |
| 1984 | 8 | 16 | 57.651 | exact |

Raw request results: `results.json`. Server log: `server.log`.
