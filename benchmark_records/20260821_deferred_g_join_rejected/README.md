# Rejected: delayed G-tail join for C=1 auxiliary streams

`G` is not consumed by `kk_pre` or the FP32 RWKV recurrence, so this candidate
recorded an R-ready event before launching the independent G LoRA and delayed
the G event wait until `_finalize_attention_output`. It preserved strict
correctness: the CUDA unit path was bitwise equal, the 8-prompt service trace
with `logprobs=20` was byte-identical, and all Full Decode CUDA Graph sizes
captured successfully.

The fixed C=1 32-token benchmark, however, changed only from **28.4558 TPS**
to **28.4594 TPS** (**+0.013%**). This is well below the 1% materiality gate
and does not justify the extra event/state complexity. The source experiment
and its startup flag were removed; the stable balanced auxiliary-stream route
remains enabled.
