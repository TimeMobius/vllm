# Rejected FFN CMix decode fusion

This candidate fused the FP32 cached-shift conversion, subtraction and FFN CMix `addcmul` into a Triton pointwise kernel. The target RTX 4090 D BF16 microbenchmark showed a regression for every tested decode batch size (B=1–64), so it was rejected before a costly service A/B run.

The existing PyTorch pointwise path is already compiler/runtime efficient for this small operation. The prototype source and tests were reverted; the immutable measurement artifact is retained here.
