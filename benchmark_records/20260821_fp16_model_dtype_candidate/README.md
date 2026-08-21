# FP16 model-dtype candidate — rejected (2026-08-21)

## Hypothesis

Albatross uses FP16-oriented kernels, so test whether changing only vLLM's loaded model
projection dtype from the strict BF16 path to FP16 can improve Full Decode CUDA Graph
throughput while preserving service output.

The candidate retained the exact FP32 recurrent-cache kernels and all stable direct-cache,
stream and graph settings. Only `--dtype bfloat16` changed to `--dtype float16`.

## Strict service accuracy gate

Reference: `/tmp/rwkv7_service_trace_before_aux.json` (strict BF16 service), eight greedy
prompts × sixteen tokens, `logprobs=20`.

| metric | FP16 candidate |
|---|---:|
| text mismatches | 2 / 8 |
| token-ID mismatches | 2 / 8 |
| top-1 disagreements | 25 / 128 decode positions |
| max selected-token logprob error | 2.70193577 |
| mean selected-token logprob error | 0.18231092 |
| max common Top-K logprob error | 10.36004497 |

This fails the byte-identical service gate. The divergent prompts were `The capital of
France is` and `Write a short poem about CUDA.`

## Paired HTTP throughput

Both services used Full Decode CUDA Graph capture sizes `1,2,4,8,16,32,64,128`; each
measurement generated 32 tokens per request.

| metric | FP16 candidate | restored strict BF16 | change |
|---|---:|---:|---:|
| C=1 output TPS | 28.4581 | 28.4539 | +0.01% |
| C=128 aggregate output TPS | 1557.3263 | 1562.6169 | -0.34% |

## Decision

Reject. FP16 provides no material speed gain on the target RTX 4090 D under the tested
vLLM graph path and produces recurrent decode divergence. The strict BF16 service was
restored after testing.

## Artifacts

- `service_accuracy.json`: FP16 candidate trace.
- `candidate_c1.json`, `candidate_c128.json`: FP16 throughput.
- `bf16_after_restore_service_accuracy.json`: restored-service trace; byte-identical to
  the strict reference.
- `bf16_after_restore_c1.json`, `bf16_after_restore_c128.json`: paired BF16 throughput.
