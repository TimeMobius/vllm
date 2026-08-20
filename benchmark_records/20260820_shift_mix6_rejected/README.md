# Rejected token-shift + mix6 fusion

A Triton prototype combined `token_shift_with_cache` and the six RWKV7 mix6
pointwise projections to remove the intermediate `delta` allocation and launch.
It handled both `[hidden]` and decode-batch `[tokens, hidden]` shift-state
layouts, but operator-level precision validation failed: BF16 reference values
were not exact due to changed intermediate rounding, with a maximum absolute
error of `0.125` on tested `T=1/8/257/1024`, `hidden=4096` tensors.

The candidate was reverted before model or service performance measurements.
