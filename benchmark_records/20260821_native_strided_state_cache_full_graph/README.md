# Native strided RWKV7 state-cache path

This retained Full Decode CUDA Graph optimization replaces three cache reads per RWKV7 block from `index_select(0, slot_ids)` with a CUDA row-copy gather and broadens the existing native masked-store path to the padded-row cache views used during graph replay.

* **Scope:** CUDA, Full Decode graph only; the dispatch falls back to existing ATen/Triton paths for unsupported layouts or missing extension symbols.
* **Numerics:** gather/store only copy dtype bits. The service logprob comparison is exactly zero, and padded-slot behavior is covered by unit and real engine integration tests.
* **Measured workload:** 128 concurrent closed-loop requests, 32 output tokens each, 4096 aggregate output tokens, C=128 graph replay, RTX 4090 D.
* **Result:** `700.675 → 708.385` aggregate output TPS, **+1.10%** after excluding the first warm-up measurement in each mode.

`raw/` contains the immutable benchmark summaries used to calculate the comparison. The standalone `kk * a` exact-recurrence fusion experiment was bit-exact but did not produce a stable service-level gain, so it is intentionally absent from the code and this retained record.
