# RWKV7 async scheduling (strict parity, no material gain)

- Date: 2026-08-21
- GPU: NVIDIA GeForce RTX 4090 D (sm89)
- Runtime: vllm-sp / Python 3.11.15 / PyTorch 2.11.0+cu130
- Model: `/hikscale/models/XiaoKe/rwkv-0-hf` (`xiaoke-5`)

## Candidate

The normal exact direct-cache Full Decode CUDA Graph server was compared with
an otherwise identical server started with:

```bash
--async-scheduling
```

This is a scheduler-only candidate: it does not alter RWKV weights, BF16 GEMV,
FP32 recurrence, cache layout, or CUDA Graph capture sizes.

## Accuracy

The existing strict greedy service comparison (eight prompts × sixteen decode
tokens, `logprobs=20`) was exactly equal to the stable trace:

```text
text mismatch: 0
token mismatch: 0
max selected-token logprob error: 0
max common Top-K logprob error: 0
```

## Paired performance

The servers were restarted between modes; both retained `FULL_DECODE_ONLY`
CUDA Graph. C=1 uses three 32-token OpenAI-completions runs and reports the
post-warm-up mean. C=128 uses four independent 128-request × 32-token runs.

| Workload | Stable | `--async-scheduling` | Change |
| --- | ---: | ---: | ---: |
| C=1 post-warm-up TPS | 25.9699 | 25.9774 | +0.03% |
| C=128 aggregate TPS | 1251.4257 | 1253.2730 | +0.15% |

Both directions are within normal service noise and below the project's minimum
material-gain threshold. Async scheduling is therefore not added to the stable
launch command. The strict direct-cache server was restored after testing.

## Implication

This negative result reinforces the existing profile conclusion: C=1 is
primarily BF16 projection/weight-bandwidth bound rather than CPU scheduler
bound. Future work should reduce projection kernel nodes or weight traffic while
preserving each dense GEMV's numerical reduction contract.
