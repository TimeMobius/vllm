# RWKV7 bulk decode projection multistream

## Candidate

The existing projection child-stream path was deliberately limited to a single
decode row. This candidate adds the explicit opt-in environment flag
`RWKV7_USE_MULTISTREAM_BULK_DECODE=1`, allowing the already independent
`R+G`, `K+W+A`, and `V+V-gate` projection groups to run on the three persistent
child streams for padded bulk decode rows as well. It preserves every
individual `F.linear`, LoRA operation, dtype conversion, operand order, and
main-stream recurrence; only inter-stream ordering changes.

The flag is checked in both the R/K/V-only and auxiliary grouped paths. It is
never selected unless the existing RKV/AUX stream flags are also enabled.

## Gates

- Focused CUDA unit: `test_rwkv7_multistream_rkv_projection_preserves_bits`
  now includes B=8 BF16 bulk projection outputs. Result: **1 passed**.
- Full focused RWKV7 regression after stopping the service: **89 passed, 4 skipped**.
- Full Decode CUDA Graph: service boot captured sizes 1, 2, 4, 8, 16, 32, 64,
  and 128 successfully with the flag enabled.
- Service accuracy: 8 prompts x 16 greedy, `logprobs=20`, compared with
  `/tmp/rwkv7_service_trace_before_aux.json`: **byte-identical**.

## Paired HTTP C=128 benchmark

Same source/binary, clean restart, 128 closed-loop requests x 32 greedy output
tokens, BF16 Full Decode graph:

| mode | two post-warmup samples | mean aggregate TPS |
|---|---:|---:|
| stable single-stream baseline | 1569.799, 1569.190 | **1569.495** |
| bulk projection streams | 1582.451, 1587.542 | **1584.997** |

Result: **+0.99% aggregate C=128 TPS**. C=1 was unchanged within noise:
28.459 TPS in the retained candidate versus the recent stable 28.45 TPS.

## Decision

Retain this opt-in as part of the stable launch configuration because it is
strictly exact, graph-safe, targets the user-visible high-concurrency path,
and produces a repeatable positive paired C=128 result without hurting C=1.
It is intentionally an incremental improvement, not a solution to the much
larger projection/recurrent bandwidth ceiling identified by the full profile.
