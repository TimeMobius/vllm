# Rejected: Triton fused three-cache masked-store fallback

- Date: 2026-08-21
- Reason tested: the source checkout's loaded `vllm._C` extension does not export
  `_C::rwkv7_masked_store`, so Full CUDA Graph falls back to graph-captured
  `index_select`/`where`/`index_copy_` state updates.
- Candidate: a single Triton kernel writing the attention shift, recurrent, and
  FFN shift state while skipping PAD slots.

The standalone PAD-slot parity check passed, but remote C=128 performance was
4.2 aggregate TPS, versus 7.8 for the original Full Decode graph and 11.1 for
Eager. The candidate is reverted. Rebuilding the existing native
`rwkv7_masked_store` extension is blocked because this host has no `nvcc` at
`/usr/local/cuda/bin/nvcc`.
