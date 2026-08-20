# Rejected recurrent T=1 direct-cache kernel

The direct cache kernel was an opt-in experiment intended to remove the decode `index_select -> recurrent update -> index_copy` path. It was rejected on August 20, 2026.

- Service throughput: 130.499 TPS → 91.954 TPS (-29.54%).
- Deterministic output comparison: 55 / 256 output tokens differed across 3 / 8 prompts.
- The standalone operator parity test passed, but that does not prove vLLM cache ownership/indexing semantics.

The source implementation and unit test were reverted. See `comparison.json` and raw benchmark artifacts for reproducibility.
