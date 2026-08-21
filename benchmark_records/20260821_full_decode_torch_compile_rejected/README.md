# Rejected: Full Decode CUDA Graph + model torch.compile

- Date: 2026-08-21
- Candidate switch: `RWKV7_COMPILE_WITH_FULL_CUDAGRAPH=1`
- Service: `FULL_DECODE_ONLY`, `max_num_seqs=128`, prefix cache align.

The service compiled and captured successfully, but the logits gate failed against
Eager over 8 prompts × 16 decode steps: 3 output-text mismatches, 26/128 Top-1
disagreements, max selected-logprob absolute error 3.1506. The candidate is not
retained and model-level compile remains disabled for FULL CUDA graphs.
