# External dashboard regression — strict BF16 restored service (2026-08-21)

A dashboard POST to `http://192.168.10.121:3000/api/run-test` completed at
**2026-08-21 18:27:59**. Its final status was:

```json
{"running":false,"progress":1.0,"message":"Test completed: 8 records saved","last_run":"2026-08-21 18:27:59","last_run_status":"success"}
```

All eight concurrency points completed with 100% request success:

| concurrency | TTFT ms | output TPS/request | overall TPS/request | requests | completion tokens |
|---:|---:|---:|---:|---:|---:|
| 1 | 285 | 39.1 | 38.5 | 3 | 2,092 |
| 2 | 342 | 33.4 | 32.8 | 6 | 4,227 |
| 4 | 347 | 35.2 | 34.6 | 12 | 8,403 |
| 8 | 489 | 33.7 | 32.9 | 16 | 11,025 |
| 16 | 605 | 32.3 | 31.4 | 32 | 22,219 |
| 32 | 994 | 27.0 | 25.9 | 32 | 21,814 |
| 64 | 929 | 23.1 | 22.4 | 64 | 43,861 |
| 128 | 1,477 | 17.9 | 17.3 | 128 | 88,070 |

This is an external API stability/user-experience sweep with variable completion
lengths, streaming/client overhead, and per-request metrics. It must not be
compared directly with the repository's fixed-payload closed-loop C=128
aggregate benchmark (about 1,563 output tokens/s): at C=128, GPU time is shared
across the 128 active requests, so per-request TPS intentionally decreases while
aggregate generation rises.

The vLLM engine log during the C=128 point reported roughly 1.69–1.70k aggregate
generation tokens/s for a 10-second rolling window; that is consistent with the
fixed-payload aggregate measurement allowing for different prompt/output mix.
