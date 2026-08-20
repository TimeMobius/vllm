# CUDA Graph rejected experiment

This directory records CUDA graph experiments that were not retained.

- Reference: `--enforce-eager` + `RWKV7_USE_FUSED_RECURRENT_T1=1`, 134.727 aggregate TPS.
- PIECEWISE: 132.692 TPS (-1.51%), and prompt 4 diverged at 28/32 tokens.
- FULL/FULL_AND_PIECEWISE variants: 156.4–166.5 TPS, but 248/256 output tokens diverged from eager.

The faster graph result is not valid for serving because deterministic token parity failed. See `comparison.json`, `accuracy.json`, and the copied seed-ID artifacts. Raw server logs remain under `/tmp/` as referenced in `comparison.json`.
