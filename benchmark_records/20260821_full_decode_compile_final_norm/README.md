# RWKV7 Full Decode CUDA Graph + model `torch.compile`

- Date: 2026-08-21
- GPU: NVIDIA GeForce RTX 4090 D (sm89)
- Python: vllm-sp / Python 3.11.15 / PyTorch 2.11.0+cu130
- Model: `/hikscale/models/XiaoKe/rwkv-0-hf` (`xiaoke-5`, BF16)

## Change

`RWKV7_COMPILE_WITH_FULL_CUDAGRAPH=1` opt-in enables model-level
`torch.compile` for full CUDA Graph modes. `rwkv7_final_norm` is a
compile-opaque custom op that dispatches to native `F.layer_norm`, preventing
Inductor from changing the final LayerNorm reduction order and causing an early
greedy-decode divergence in RWKV's recurrent state.

## Accuracy gate

The candidate was compared with a `FULL_DECODE_ONLY` CUDA-Graph baseline that
did not use model-level compile. The test used 8 fixed prompts, greedy sampling
(`temperature=0`, `top_p=1`, `seed=0`), 16 generated tokens, and `logprobs=20`.

Result: 0/128 Top-1 disagreements, 0 text mismatches, and exact zero error for
common Top-K and selected-token logprobs. See `accuracy.json`.

## Performance gate

A local `/v1/completions` closed-loop workload with 16 requests, concurrency 8,
and 32 output tokens measured 129.77 aggregate output TPS for full graph
without model compile and 138.70 TPS for the candidate (+6.89%). All 16
requests succeeded. See `performance.json`.

## Reproduction

Candidate server environment:

```bash
CUDA_VISIBLE_DEVICES=0 \
VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 \
VLLM_DISABLE_COMPILE_CACHE=1 \
RWKV7_COMPILE_WITH_FULL_CUDAGRAPH=1 \
/mnt/data/anaconda3/envs/vllm-sp/bin/vllm serve \
  /hikscale/models/XiaoKe/rwkv-0-hf \
  --served-model-name xiaoke-5 \
  --reasoning-parser rwkv \
  --enable-auto-tool-choice \
  --tool-call-parser rwkv \
  --dtype bfloat16 --host 0.0.0.0 --port 8030 \
  --max-model-len 1M --max-num-seqs 128 --gpu-memory-utilization 0.92 \
  --enable-prefix-caching --mamba-cache-mode align \
  --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[1,2,4,8,16,32,64,128]}'
```

The opt-in remains deliberately disabled by default. A padded C=128 workload
and a full model-state parity probe remain required before making it a default.
