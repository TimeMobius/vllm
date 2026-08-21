# Rejected: paired RWKV7 shift-cache gather/store

The candidate fused the attention-shift and FFN-shift cache copies into one
native CUDA gather and one masked store. It was deliberately restricted to
same-shape, same-dtype rows and directly copied bits, leaving the FP32 recurrent
matrix cache untouched.

It passed isolated CUDA tests for FP32 and BF16, including padded outer strides,
and the service-level greedy/logprob trace was byte-identical. However, paired
fixed-payload results were not material:

| workload | baseline | candidate | change |
| --- | ---: | ---: | ---: |
| C=1 output TPS | 27.613 | 27.609 | -0.01% |
| C=128 aggregate TPS | 1569.19 | 1571.26 | +0.13% |

The source implementation and runtime flag were removed. This documents that
further copy-launch fusion alone is not a meaningful route to the requested
throughput target on this workload.
