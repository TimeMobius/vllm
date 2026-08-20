# Rejected packed-prefill recurrent 2-warp launch

The Triton recurrent prefill kernel was tested with `RWKV7_FUSED_RECURRENT_NUM_WARPS=2` instead of the default four warps. Operator-only timing improved, but service-level acceptance failed.

| Workload | Default 4 warps | Candidate 2 warps | Delta |
| --- | ---: | ---: | ---: |
| 4K cold unique prompt | 8.7906 TPS | 8.7049 TPS | -0.98% |
| 16K cold unique prompt | 3.1400 TPS | 3.1418 TPS | +0.06% |
| 64K cold unique prompt | 0.8734 TPS | 0.8774 TPS | +0.46% |

The 4K service result regressed, and the 16K/64K differences were too small
to establish a robust end-to-end win. The source change was reverted; the
operator timing logs remain for future architecture-specific tuning.
