# Rejected: main-stream R+G with two child streams

This candidate replaced the R+G child stream with the current CUDA stream,
leaving two child streams for K+W+A and V+V-gate. It retained every individual
BF16 `F.linear`, LoRA activation, ordering, dtype, and reduction contract.

Correctness was strict: the service's 8-prompt × 16-token greedy trace with
`logprobs=20` was byte-identical to the baseline and Full Decode CUDA Graph
capture succeeded for sizes 1, 2, 4, 8, 16, 32, 64, and 128.

However, the fixed C=1 32-token benchmark moved from **28.4558 TPS** to
**28.4488 TPS** (**-0.024%**). The removed fork/join overhead did not compensate
for the changed launch concurrency, so the source experiment and startup flag
were removed. The retained three-child-stream schedule remains active.
