# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Songlin Yang, Yu Zhang
#
# This file contains code adapted from the flash-linear-attention project.
# The original source code was licensed under the MIT license and included
# the following copyright notice:
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang

import os

import torch

from vllm.triton_utils import HAS_TRITON, tl, triton

from .op import exp


def _rwkv7_fused_recurrent_disabled() -> bool:
    return (
        os.getenv("RWKV7_DISABLE_FUSED_RECURRENT") == "1"
        or os.getenv("RWKV7_DISABLE_FUSED_PREFILL") == "1"
    )


@triton.heuristics(
    {
        "USE_INITIAL_STATE": lambda args: args["h0"] is not None,
        "STORE_FINAL_STATE": lambda args: args["ht"] is not None,
        "STORE_CHECKPOINT_STATE": lambda args: args["hc"] is not None,
        "DIRECT_CHECKPOINT_WRITE": lambda args: args["checkpoint_slot_ids"] is not None,
        "IS_VARLEN": lambda args: args["cu_seqlens"] is not None,
    }
)
@triton.jit(do_not_specialize=["T"])
def fused_recurrent_rwkv7_fwd_kernel(
    r,
    w,
    k,
    v,
    kk,
    a,
    o,
    h0,
    ht,
    checkpoint_positions,
    checkpoint_offsets,
    checkpoint_slot_ids,
    hc,
    cu_seqlens,
    scale,
    T,
    B: tl.constexpr,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    STORE_FINAL_STATE: tl.constexpr,
    STORE_CHECKPOINT_STATE: tl.constexpr,
    DIRECT_CHECKPOINT_WRITE: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_v, i_nh = tl.program_id(0).to(tl.int64), tl.program_id(1).to(tl.int64)
    i_n, i_h = i_nh // H, i_nh % H

    if IS_VARLEN:
        bos = tl.load(cu_seqlens + i_n).to(tl.int64)
        eos = tl.load(cu_seqlens + i_n + 1).to(tl.int64)
        T = eos - bos
    else:
        bos, eos = i_n * T, i_n * T + T

    o_k = tl.arange(0, BK)
    o_v = i_v * BV + tl.arange(0, BV)
    p_r = r + bos * H * K + i_h * K + o_k
    p_w = w + bos * H * K + i_h * K + o_k
    p_k = k + bos * H * K + i_h * K + o_k
    p_v = v + bos * H * V + i_h * V + o_v
    p_a = a + bos * H * K + i_h * K + o_k
    p_kk = kk + bos * H * K + i_h * K + o_k
    p_o = o + bos * H * V + i_h * V + o_v

    mask_k = o_k < K
    mask_v = o_v < V
    mask_h = mask_k[:, None] & mask_v[None, :]
    b_h = tl.zeros([BK, BV], dtype=tl.float32)

    if USE_INITIAL_STATE:
        p_h0 = h0 + i_nh * K * V + o_k[:, None] * V + o_v
        b_h += tl.load(p_h0, mask=mask_h, other=0).to(tl.float32)

    if STORE_CHECKPOINT_STATE:
        checkpoint_idx = tl.load(checkpoint_offsets + i_n).to(tl.int64)
        checkpoint_end = tl.load(checkpoint_offsets + i_n + 1).to(tl.int64)
        has_checkpoint = checkpoint_idx < checkpoint_end
        checkpoint_pos = tl.load(
            checkpoint_positions + checkpoint_idx,
            mask=has_checkpoint,
            other=0,
        ).to(tl.int64)

    for t in range(0, T):
        b_r = tl.load(p_r, mask=mask_k, other=0).to(tl.float32) * scale
        b_w = tl.load(p_w, mask=mask_k, other=0).to(tl.float32)
        b_k = tl.load(p_k, mask=mask_k, other=0).to(tl.float32)
        b_v = tl.load(p_v, mask=mask_v, other=0).to(tl.float32)
        b_a = tl.load(p_a, mask=mask_k, other=0).to(tl.float32)
        b_kk = tl.load(p_kk, mask=mask_k, other=0).to(tl.float32)
        b_act_a = -b_kk
        b_b = b_kk * b_a

        b_h = exp(b_w)[:, None] * b_h + b_b[:, None] * tl.sum(
            b_act_a[:, None] * b_h, 0
        )[None, :]
        b_h += b_k[:, None] * b_v[None, :]
        b_o = tl.sum(b_h * b_r[:, None], 0)

        tl.store(p_o, b_o.to(p_o.dtype.element_ty), mask=mask_v)

        if STORE_CHECKPOINT_STATE:
            should_store = has_checkpoint & (t == checkpoint_pos)
            safe_checkpoint_idx = tl.where(has_checkpoint, checkpoint_idx, 0)
            if DIRECT_CHECKPOINT_WRITE:
                checkpoint_slot_id = tl.load(
                    checkpoint_slot_ids + safe_checkpoint_idx,
                    mask=has_checkpoint,
                    other=0,
                ).to(tl.int64)
                p_hc = hc + (checkpoint_slot_id * H + i_h) * K * V + o_k[:, None] * V + o_v
            else:
                p_hc = hc + (safe_checkpoint_idx * H + i_h) * K * V + o_k[:, None] * V + o_v
            tl.store(
                p_hc,
                b_h.to(p_hc.dtype.element_ty),
                mask=mask_h & should_store,
            )
            checkpoint_idx += should_store.to(tl.int64)
            has_checkpoint = checkpoint_idx < checkpoint_end
            checkpoint_pos = tl.load(
                checkpoint_positions + checkpoint_idx,
                mask=has_checkpoint,
                other=checkpoint_pos,
            ).to(tl.int64)

        p_r += H * K
        p_w += H * K
        p_k += H * K
        p_v += H * V
        p_a += H * K
        p_kk += H * K
        p_o += H * V

    if STORE_FINAL_STATE:
        p_ht = ht + i_nh * K * V + o_k[:, None] * V + o_v
        tl.store(p_ht, b_h.to(p_ht.dtype.element_ty), mask=mask_h)


def rwkv7_recurrent_reference(
    r: torch.Tensor,
    w: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    kk: torch.Tensor,
    a: torch.Tensor,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    cu_seqlens: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    out, final_state, _ = _rwkv7_recurrent_reference_impl(
        r=r,
        w=w,
        k=k,
        v=v,
        kk=kk,
        a=a,
        initial_state=initial_state,
        output_final_state=output_final_state,
        cu_seqlens=cu_seqlens,
    )
    return out, final_state


def rwkv7_recurrent_reference_with_checkpoints(
    r: torch.Tensor,
    w: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    kk: torch.Tensor,
    a: torch.Tensor,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    cu_seqlens: torch.Tensor | None = None,
    checkpoint_positions: torch.Tensor | None = None,
    checkpoint_offsets: torch.Tensor | None = None,
    output_checkpoint_states: bool = False,
    checkpoint_state_cache: torch.Tensor | None = None,
    checkpoint_slot_ids: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    return _rwkv7_recurrent_reference_impl(
        r=r,
        w=w,
        k=k,
        v=v,
        kk=kk,
        a=a,
        initial_state=initial_state,
        output_final_state=output_final_state,
        cu_seqlens=cu_seqlens,
        checkpoint_positions=checkpoint_positions,
        checkpoint_offsets=checkpoint_offsets,
        output_checkpoint_states=output_checkpoint_states,
        checkpoint_state_cache=checkpoint_state_cache,
        checkpoint_slot_ids=checkpoint_slot_ids,
    )


def _rwkv7_recurrent_reference_impl(
    r: torch.Tensor,
    w: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    kk: torch.Tensor,
    a: torch.Tensor,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    cu_seqlens: torch.Tensor | None = None,
    checkpoint_positions: torch.Tensor | None = None,
    checkpoint_offsets: torch.Tensor | None = None,
    output_checkpoint_states: bool = False,
    checkpoint_state_cache: torch.Tensor | None = None,
    checkpoint_slot_ids: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    if r.ndim != 4:
        raise ValueError(f"`r` must be 4D, got {r.ndim}.")
    if cu_seqlens is not None and r.shape[0] != 1:
        raise ValueError(
            "When `cu_seqlens` is provided, the batch size must be 1."
        )
    if (output_checkpoint_states or checkpoint_state_cache is not None) and (
        checkpoint_positions is None or checkpoint_offsets is None
    ):
        raise ValueError(
            "`checkpoint_positions` and `checkpoint_offsets` are required "
            "when storing checkpoint states."
        )
    if checkpoint_state_cache is not None and checkpoint_slot_ids is None:
        raise ValueError(
            "`checkpoint_slot_ids` are required when `checkpoint_state_cache` is provided."
        )
    if output_checkpoint_states and checkpoint_state_cache is not None:
        raise ValueError(
            "`output_checkpoint_states` and `checkpoint_state_cache` are mutually exclusive."
        )

    B, T, H, K = r.shape
    V = v.shape[-1]
    N = B if cu_seqlens is None else int(cu_seqlens.numel() - 1)
    out = torch.empty_like(v)

    if output_final_state:
        if initial_state is None:
            final_state = torch.zeros(
                (N, H, K, V),
                device=r.device,
                dtype=torch.float32,
            )
        else:
            final_state = initial_state.to(torch.float32).clone()
    else:
        final_state = None

    if output_checkpoint_states:
        assert checkpoint_offsets is not None
        num_checkpoints = int(checkpoint_offsets[-1].item())
        checkpoint_states = torch.empty(
            (num_checkpoints, H, K, V),
            device=r.device,
            dtype=torch.float32,
        )
    else:
        checkpoint_states = None

    store_checkpoint_state = output_checkpoint_states or checkpoint_state_cache is not None

    for seq_idx in range(N):
        batch_idx = 0 if cu_seqlens is not None else seq_idx
        if cu_seqlens is None:
            start = seq_idx * T
            end = start + T
        else:
            start = int(cu_seqlens[seq_idx].item())
            end = int(cu_seqlens[seq_idx + 1].item())

        if initial_state is None:
            state = torch.zeros((H, K, V), device=r.device, dtype=torch.float32)
        else:
            state = initial_state[seq_idx].to(torch.float32).clone()

        if store_checkpoint_state:
            assert checkpoint_positions is not None
            assert checkpoint_offsets is not None
            checkpoint_idx = int(checkpoint_offsets[seq_idx].item())
            checkpoint_end = int(checkpoint_offsets[seq_idx + 1].item())
        else:
            checkpoint_idx = 0
            checkpoint_end = 0

        for tok_idx in range(start, end):
            local_token_idx = tok_idx - start
            tensor_token_idx = tok_idx if cu_seqlens is not None else local_token_idx
            sa = (
                state * (-kk[batch_idx, tensor_token_idx]).unsqueeze(-1)
            ).sum(dim=-2)
            state = (
                torch.exp(w[batch_idx, tensor_token_idx]).unsqueeze(-1) * state
                + (
                    kk[batch_idx, tensor_token_idx]
                    * a[batch_idx, tensor_token_idx]
                ).unsqueeze(-1)
                * sa.unsqueeze(-2)
                + k[batch_idx, tensor_token_idx].unsqueeze(-1)
                * v[batch_idx, tensor_token_idx].unsqueeze(-2)
            )
            out[batch_idx, tensor_token_idx] = (
                state * r[batch_idx, tensor_token_idx].unsqueeze(-1)
            ).sum(dim=-2).to(out.dtype)

            if (
                checkpoint_idx < checkpoint_end
                and local_token_idx == int(checkpoint_positions[checkpoint_idx].item())
            ):
                if checkpoint_state_cache is not None:
                    assert checkpoint_slot_ids is not None
                    slot_id = int(checkpoint_slot_ids[checkpoint_idx].item())
                    checkpoint_state_cache[slot_id].copy_(
                        state.to(checkpoint_state_cache.dtype)
                    )
                elif checkpoint_states is not None:
                    checkpoint_states[checkpoint_idx] = state
                checkpoint_idx += 1

        if final_state is not None:
            final_state[seq_idx] = state

    return out, final_state, checkpoint_states


def fused_mul_recurrent_rwkv7(
    r: torch.Tensor,
    w: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    kk: torch.Tensor,
    a: torch.Tensor,
    scale: float = 1.0,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    cu_seqlens: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    out, final_state, _ = _fused_mul_recurrent_rwkv7_impl(
        r=r,
        w=w,
        k=k,
        v=v,
        kk=kk,
        a=a,
        scale=scale,
        initial_state=initial_state,
        output_final_state=output_final_state,
        cu_seqlens=cu_seqlens,
    )
    return out, final_state


def fused_mul_recurrent_rwkv7_with_checkpoints(
    r: torch.Tensor,
    w: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    kk: torch.Tensor,
    a: torch.Tensor,
    checkpoint_positions: torch.Tensor,
    checkpoint_offsets: torch.Tensor,
    scale: float = 1.0,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    cu_seqlens: torch.Tensor | None = None,
    checkpoint_state_cache: torch.Tensor | None = None,
    checkpoint_slot_ids: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    out, final_state, checkpoint_states = _fused_mul_recurrent_rwkv7_impl(
        r=r,
        w=w,
        k=k,
        v=v,
        kk=kk,
        a=a,
        scale=scale,
        initial_state=initial_state,
        output_final_state=output_final_state,
        cu_seqlens=cu_seqlens,
        checkpoint_positions=checkpoint_positions,
        checkpoint_offsets=checkpoint_offsets,
        output_checkpoint_states=checkpoint_state_cache is None,
        checkpoint_state_cache=checkpoint_state_cache,
        checkpoint_slot_ids=checkpoint_slot_ids,
    )
    return out, final_state, checkpoint_states


def _fused_mul_recurrent_rwkv7_impl(
    r: torch.Tensor,
    w: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    kk: torch.Tensor,
    a: torch.Tensor,
    scale: float = 1.0,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    cu_seqlens: torch.Tensor | None = None,
    checkpoint_positions: torch.Tensor | None = None,
    checkpoint_offsets: torch.Tensor | None = None,
    output_checkpoint_states: bool = False,
    checkpoint_state_cache: torch.Tensor | None = None,
    checkpoint_slot_ids: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    if (
        _rwkv7_fused_recurrent_disabled()
        or not HAS_TRITON
        or r.device.type != "cuda"
        or r.numel() == 0
    ):
        return rwkv7_recurrent_reference_with_checkpoints(
            r=r,
            w=w,
            k=k,
            v=v,
            kk=kk,
            a=a,
            initial_state=initial_state,
            output_final_state=output_final_state,
            cu_seqlens=cu_seqlens,
            checkpoint_positions=checkpoint_positions,
            checkpoint_offsets=checkpoint_offsets,
            output_checkpoint_states=output_checkpoint_states,
            checkpoint_state_cache=checkpoint_state_cache,
            checkpoint_slot_ids=checkpoint_slot_ids,
        )

    if cu_seqlens is not None and r.shape[0] != 1:
        raise ValueError(
            "When `cu_seqlens` is provided, the batch size must be 1."
        )

    B, T, H, K = r.shape
    V = v.shape[-1]
    N = B if cu_seqlens is None else int(cu_seqlens.numel() - 1)
    BK = triton.next_power_of_2(K)
    BV = min(triton.next_power_of_2(V), 64)

    h0 = initial_state
    ht = None
    if output_final_state:
        if initial_state is None:
            h0 = r.new_zeros((N, H, K, V), dtype=torch.float32)
        ht = r.new_empty((N, H, K, V), dtype=torch.float32)

    store_checkpoint_state = output_checkpoint_states or checkpoint_state_cache is not None
    hc = None
    if store_checkpoint_state:
        if checkpoint_positions is None or checkpoint_offsets is None:
            raise ValueError(
                "`checkpoint_positions` and `checkpoint_offsets` are required "
                "when storing checkpoint states."
            )
        if checkpoint_state_cache is not None:
            if checkpoint_slot_ids is None:
                raise ValueError(
                    "`checkpoint_slot_ids` are required when `checkpoint_state_cache` is provided."
                )
            hc = checkpoint_state_cache
        else:
            num_checkpoints = int(checkpoint_offsets[-1].item())
            hc = r.new_empty((num_checkpoints, H, K, V), dtype=torch.float32)

    o = torch.empty_like(v)
    grid = (triton.cdiv(V, BV), N * H)
    fused_recurrent_rwkv7_fwd_kernel[grid](
        r=r.contiguous(),
        w=w.contiguous(),
        k=k.contiguous(),
        v=v.contiguous(),
        kk=kk.contiguous(),
        a=a.contiguous(),
        o=o,
        h0=h0,
        ht=ht,
        checkpoint_positions=checkpoint_positions,
        checkpoint_offsets=checkpoint_offsets,
        checkpoint_slot_ids=checkpoint_slot_ids,
        hc=hc,
        cu_seqlens=cu_seqlens,
        scale=scale,
        T=T,
        B=B,
        H=H,
        K=K,
        V=V,
        BK=BK,
        BV=BV,
        num_warps=4,
        num_stages=3,
    )
    checkpoint_states = None if checkpoint_state_cache is not None else hc
    return o, ht, checkpoint_states
