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

import vllm._custom_ops as custom_ops
from vllm.triton_utils import HAS_TRITON, tl, triton

from .op import exp


def _rwkv7_fused_recurrent_disabled() -> bool:
    return (
        os.getenv("RWKV7_DISABLE_FUSED_RECURRENT") == "1"
        or os.getenv("RWKV7_DISABLE_FUSED_PREFILL") == "1"
    )


def _rwkv7_exact_recurrent_t1_update_enabled() -> bool:
    value = os.getenv("RWKV7_USE_EXACT_RECURRENT_T1_UPDATE", "1")
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _rwkv7_exact_recurrent_t1_output_reduction_enabled() -> bool:
    value = os.getenv("RWKV7_USE_EXACT_RECURRENT_T1_OUTPUT_REDUCTION", "1")
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _rwkv7_exact_recurrent_t1_direct_cache_enabled() -> bool:
    value = os.getenv("RWKV7_USE_EXACT_RECURRENT_T1_DIRECT_CACHE", "1")
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _rwkv7_recurrent_t1_reference(
    recurrent_state: torch.Tensor,
    w: torch.Tensor,
    kk: torch.Tensor,
    a: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    r: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reference for one RWKV7 recurrent decode step.

    The state and all projected terms are FP32 in the model execution path.
    Keeping this implementation next to the fused path makes the dispatch
    guards and parity tests explicit.
    """
    sa = (recurrent_state * (-kk).unsqueeze(-1)).sum(dim=-2)
    new_state = (
        torch.exp(w).unsqueeze(-1) * recurrent_state
        + (kk * a).unsqueeze(-1) * sa.unsqueeze(-2)
        + k.unsqueeze(-1) * v.unsqueeze(-2)
    )
    reduce_out = (new_state * r.unsqueeze(-1)).sum(dim=-2)
    return new_state, reduce_out


if HAS_TRITON:

    @triton.jit
    def _rwkv7_masked_store_kernel(
        cache_ptr,
        values_ptr,
        slot_ids_ptr,
        CACHE_ROW_STRIDE: tl.constexpr,
        VALUES_ROW_STRIDE: tl.constexpr,
        ROW_WIDTH: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        batch_idx = tl.program_id(0)
        block_idx = tl.program_id(1)
        slot_id = tl.load(slot_ids_ptr + batch_idx)
        if slot_id >= 0:
            offsets = block_idx * BLOCK + tl.arange(0, BLOCK)
            mask = offsets < ROW_WIDTH
            values = tl.load(
                values_ptr + batch_idx * VALUES_ROW_STRIDE + offsets, mask=mask
            )
            tl.store(
                cache_ptr + slot_id * CACHE_ROW_STRIDE + offsets, values, mask=mask
            )

    @triton.jit
    def _rwkv7_recurrent_t1_matrix_kernel(
        state_ptr,
        w_ptr,
        kk_ptr,
        a_ptr,
        k_ptr,
        v_ptr,
        r_ptr,
        out_state_ptr,
        out_reduce_ptr,
        H: tl.constexpr,
        D: tl.constexpr,
        V: tl.constexpr,
    ):
        batch_idx = tl.program_id(0)
        head_idx = tl.program_id(1)
        state_idx = batch_idx * H + head_idx
        d_offsets = tl.arange(0, D)[:, None]
        v_offsets = tl.arange(0, V)
        state_base = state_ptr + state_idx * D * V
        scalar_base = state_idx * D
        value_base = state_idx * V

        state = tl.load(state_base + d_offsets * V + v_offsets[None, :]).to(tl.float32)
        kk_val = tl.load(kk_ptr + scalar_base + d_offsets).to(tl.float32)
        sa = tl.sum(state * (-kk_val), axis=0)
        w_val = tl.exp(tl.load(w_ptr + scalar_base + d_offsets).to(tl.float32))
        a_val = tl.load(a_ptr + scalar_base + d_offsets).to(tl.float32)
        k_val = tl.load(k_ptr + scalar_base + d_offsets).to(tl.float32)
        r_val = tl.load(r_ptr + scalar_base + d_offsets).to(tl.float32)
        value = tl.load(v_ptr + value_base + v_offsets)[None, :].to(tl.float32)
        new_state = w_val * state + (kk_val * a_val) * sa + k_val * value

        tl.store(
            out_state_ptr + state_idx * D * V + d_offsets * V + v_offsets[None, :],
            new_state,
        )
        tl.store(
            out_reduce_ptr + value_base + v_offsets,
            tl.sum(new_state * r_val, axis=0),
        )


def rwkv7_masked_store_triton(
    cache: torch.Tensor,
    values: torch.Tensor,
    slot_ids: torch.Tensor,
) -> None:
    """Store valid state-cache rows without remapping graph padding to zero.

    Full CUDA Graph decode pads uniform batches with ``PAD_SLOT_ID=-1``. This
    source-checkout fallback uses a capture-safe Triton scatter instead of
    ``index_select + where + index_copy_``; it avoids reading old recurrent
    state rows and cannot overwrite a valid slot-zero update with a padded row.
    """
    if cache.ndim != values.ndim or cache.shape[1:] != values.shape[1:]:
        raise ValueError("`cache` and `values` must have matching row shapes.")
    if slot_ids.shape != (values.shape[0],):
        raise ValueError("`slot_ids` must have one entry per values row.")
    if slot_ids.dtype not in (torch.int32, torch.int64):
        raise ValueError("`slot_ids` must have an integer dtype.")
    if any(tensor.device != cache.device for tensor in (values, slot_ids)):
        raise ValueError("`cache`, `values`, and `slot_ids` must share a device.")

    if HAS_TRITON and cache.device.type == "cuda":
        # vLLM state-cache views may use padded row strides. The kernel works
        # directly from row strides, without a graph-unsafe contiguous copy.
        row_width = values[0].numel() if values.shape[0] else 0
        if row_width:
            block = 256
            _rwkv7_masked_store_kernel[
                (values.shape[0], triton.cdiv(row_width, block))
            ](
                cache,
                values,
                slot_ids,
                CACHE_ROW_STRIDE=cache.stride(0),
                VALUES_ROW_STRIDE=values.stride(0),
                ROW_WIDTH=row_width,
                BLOCK=block,
            )
            return

    valid = slot_ids >= 0
    if torch.any(valid):
        cache.index_copy_(
            0,
            slot_ids[valid].to(dtype=torch.long),
            values[valid].to(cache.dtype),
        )


def rwkv7_recurrent_t1(
    recurrent_state: torch.Tensor,
    w: torch.Tensor,
    kk: torch.Tensor,
    a: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    r: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fuse the T=1 recurrent update and output reduction.

    The optimized Triton path is enabled by default.  Setting
    ``RWKV7_USE_FUSED_RECURRENT_T1=0`` keeps the normal reference/alternate
    paths available for rollback and A/B benchmarking.
    """
    if recurrent_state.ndim != 4:
        raise ValueError(
            "`recurrent_state` must be rank 4 [B, H, D, V], got "
            f"{recurrent_state.ndim}."
        )

    tensors = (recurrent_state, w, kk, a, k, v, r)
    if (
        not HAS_TRITON
        or os.getenv("RWKV7_USE_FUSED_RECURRENT_T1", "1") != "1"
        or recurrent_state.device.type != "cuda"
        or any(t.dtype != torch.float32 or not t.is_contiguous() for t in tensors)
    ):
        return _rwkv7_recurrent_t1_reference(recurrent_state, w, kk, a, k, v, r)

    B, H, D, V = recurrent_state.shape
    if (
        w.shape != (B, H, D)
        or kk.shape != (B, H, D)
        or a.shape != (B, H, D)
        or k.shape != (B, H, D)
        or r.shape != (B, H, D)
        or v.shape != (B, H, V)
        or D != 64
        or V != 64
    ):
        return _rwkv7_recurrent_t1_reference(recurrent_state, w, kk, a, k, v, r)

    new_state = torch.empty_like(recurrent_state)
    reduce_out = torch.empty((B, H, V), device=v.device, dtype=torch.float32)
    _rwkv7_recurrent_t1_matrix_kernel[(B, H)](
        recurrent_state,
        w,
        kk,
        a,
        k,
        v,
        r,
        new_state,
        reduce_out,
        H=H,
        D=D,
        V=V,
        num_warps=4,
    )
    return new_state, reduce_out


def rwkv7_alt_recurrent_available() -> bool:
    return hasattr(torch.ops, "_C") and hasattr(torch.ops._C, "rwkv7_alt_recurrent")


def rwkv7_recurrent_t1_exact_update_available() -> bool:
    return hasattr(torch.ops, "_C") and hasattr(
        torch.ops._C, "rwkv7_recurrent_t1_exact_update"
    )


def rwkv7_recurrent_t1_exact_output_reduction_available() -> bool:
    return hasattr(torch.ops, "_C") and hasattr(
        torch.ops._C, "rwkv7_reduce_d64_atten_exact"
    )


def rwkv7_recurrent_t1_exact_direct_cache_available() -> bool:
    return hasattr(torch.ops, "_C") and hasattr(
        torch.ops._C, "rwkv7_recurrent_t1_exact_direct_cache"
    )


def rwkv7_recurrent_t1_exact_direct_cache(
    cache: torch.Tensor,
    slot_ids: torch.Tensor,
    w: torch.Tensor,
    kk: torch.Tensor,
    a: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    r: torch.Tensor,
) -> torch.Tensor:
    """Run an exact T=1 recurrence directly on persistent FP32 cache rows.

    This is a narrow Full CUDA Graph decode path: it removes the recurrent
    state gather and store while preserving the ATen FP32 ``sa`` and output
    reduction orders. A padded ``slot_id=-1`` reads slot zero but does not
    write it; its output is ignored by the graph's padded decode lane.
    """
    tensors = (w, kk, a, k, v, r)
    if (
        not _rwkv7_exact_recurrent_t1_direct_cache_enabled()
        or not rwkv7_recurrent_t1_exact_direct_cache_available()
        or cache.device.type != "cuda"
        or cache.ndim != 4
        or cache.shape[-2:] != (64, 64)
        or slot_ids.device != cache.device
        or slot_ids.dtype != torch.long
        or slot_ids.ndim != 1
        or slot_ids.numel() != r.shape[0]
        or any(t.dtype != torch.float32 or not t.is_contiguous() for t in tensors)
        or cache.dtype != torch.float32
        or cache.stride(-1) != 1
        or cache.stride(-2) != 64
        or cache.stride(-3) != 64 * 64
        or cache.stride(0) < cache.shape[1] * 64 * 64
        or not slot_ids.is_contiguous()
    ):
        raise RuntimeError("RWKV7 exact direct-cache recurrent path is unsupported")
    exp_w = torch.exp(w)
    kk_a = kk * a
    return custom_ops.rwkv7_recurrent_t1_exact_direct_cache(
        cache, slot_ids, exp_w, kk, kk_a, k, v, r
    )


def rwkv7_recurrent_t1_exact_output_reduction(
    recurrent_state: torch.Tensor, r: torch.Tensor
) -> torch.Tensor:
    """Exactly reproduce RWKV7's FP32 D=64 output reduction on CUDA.

    This mirrors ATen's observed ``reduce_kernel<128, 4>`` launch and its
    per-thread/shared-memory addition order. It removes the materialized output
    multiply and one launch without changing greedy-decode numerics.
    """
    if (
        not _rwkv7_exact_recurrent_t1_output_reduction_enabled()
        or not rwkv7_recurrent_t1_exact_output_reduction_available()
        or recurrent_state.device.type != "cuda"
        or recurrent_state.ndim != 4
        or recurrent_state.shape[-2:] != (64, 64)
        or r.shape != recurrent_state.shape[:-1]
        or recurrent_state.dtype != torch.float32
        or r.dtype != torch.float32
        or not recurrent_state.is_contiguous()
        or not r.is_contiguous()
    ):
        return (recurrent_state * r.unsqueeze(-1)).sum(dim=-2)
    return custom_ops.rwkv7_reduce_d64_atten_exact(recurrent_state, r)


def rwkv7_recurrent_t1_exact_update(
    recurrent_state: torch.Tensor,
    w: torch.Tensor,
    kk: torch.Tensor,
    a: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    r: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Exactly reproduce the reference T=1 recurrence with fewer state ops.

    ``sa`` remains native ATen because its reduction order is part of the
    greedy-decode numerical contract. The state update and final D=64 output
    reduction use native CUDA only when their separately verified ATen contracts
    are available.
    """
    tensors = (recurrent_state, w, kk, a, k, v, r)
    if (
        not _rwkv7_exact_recurrent_t1_update_enabled()
        or not rwkv7_recurrent_t1_exact_update_available()
        or recurrent_state.device.type != "cuda"
        or recurrent_state.ndim != 4
        or recurrent_state.shape[-2:] != (64, 64)
        or any(t.dtype != torch.float32 or not t.is_contiguous() for t in tensors)
    ):
        return _rwkv7_recurrent_t1_reference(recurrent_state, w, kk, a, k, v, r)

    # Keep these operations separate. The native update kernel uses explicit
    # round-to-nearest mul/add instructions to match this eager materialization
    # bit-for-bit, so recurrent state remains stable over long decode runs.
    sa = (recurrent_state * (-kk).unsqueeze(-1)).sum(dim=-2)
    exp_w = torch.exp(w)
    kk_a = kk * a
    final_state = custom_ops.rwkv7_recurrent_t1_exact_update(
        recurrent_state, exp_w, kk_a, k, v, sa
    )
    recurrent_output = rwkv7_recurrent_t1_exact_output_reduction(final_state, r)
    return final_state, recurrent_output


def rwkv7_alt_recurrent(
    r: torch.Tensor,
    w: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    kk: torch.Tensor,
    a: torch.Tensor,
    initial_state: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not rwkv7_alt_recurrent_available():
        raise RuntimeError("RWKV7 alternate recurrent CUDA op is not available.")
    return custom_ops.rwkv7_alt_recurrent(r, w, k, v, kk, a, initial_state)


@triton.jit
def rwkv7_mix6_fwd_kernel(
    x,
    delta,
    x_r,
    x_w,
    x_k,
    x_v,
    x_a,
    x_g,
    xr,
    xw,
    xk,
    xv,
    xa,
    xg,
    numel,
    hidden_size,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0).to(tl.int64)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < numel
    cols = offsets % hidden_size

    x_vals = tl.load(x + offsets, mask=mask, other=0).to(tl.float32)
    delta_vals = tl.load(delta + offsets, mask=mask, other=0).to(tl.float32)

    x_r_vals = tl.load(x_r + cols, mask=mask, other=0).to(tl.float32)
    x_w_vals = tl.load(x_w + cols, mask=mask, other=0).to(tl.float32)
    x_k_vals = tl.load(x_k + cols, mask=mask, other=0).to(tl.float32)
    x_v_vals = tl.load(x_v + cols, mask=mask, other=0).to(tl.float32)
    x_a_vals = tl.load(x_a + cols, mask=mask, other=0).to(tl.float32)
    x_g_vals = tl.load(x_g + cols, mask=mask, other=0).to(tl.float32)

    tl.store(xr + offsets, x_vals + delta_vals * x_r_vals, mask=mask)
    tl.store(xw + offsets, x_vals + delta_vals * x_w_vals, mask=mask)
    tl.store(xk + offsets, x_vals + delta_vals * x_k_vals, mask=mask)
    tl.store(xv + offsets, x_vals + delta_vals * x_v_vals, mask=mask)
    tl.store(xa + offsets, x_vals + delta_vals * x_a_vals, mask=mask)
    tl.store(xg + offsets, x_vals + delta_vals * x_g_vals, mask=mask)


def rwkv7_mix6_reference(
    hidden_states: torch.Tensor,
    delta: torch.Tensor,
    x_r: torch.Tensor,
    x_w: torch.Tensor,
    x_k: torch.Tensor,
    x_v: torch.Tensor,
    x_a: torch.Tensor,
    x_g: torch.Tensor,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    xr = hidden_states.addcmul(delta, x_r)
    xw = hidden_states.addcmul(delta, x_w)
    xk = hidden_states.addcmul(delta, x_k)
    xv = hidden_states.addcmul(delta, x_v)
    xa = hidden_states.addcmul(delta, x_a)
    xg = hidden_states.addcmul(delta, x_g)
    return xr, xw, xk, xv, xa, xg


def _rwkv7_mix6_use_triton(hidden_states: torch.Tensor) -> bool:
    """Choose the mix6 implementation by token count.

    The fused kernel has a fixed 2048-element block. On the target SM89 GPU,
    small decode batches and the 1280-2048 token launch range are faster with
    PyTorch's pointwise fusion, while larger prefill launches benefit from the
    Triton kernel.
    """
    tokens = hidden_states.shape[0]
    return tokens > 256 and not 1280 <= tokens <= 2048


def rwkv7_mix6(
    hidden_states: torch.Tensor,
    delta: torch.Tensor,
    x_r: torch.Tensor,
    x_w: torch.Tensor,
    x_k: torch.Tensor,
    x_v: torch.Tensor,
    x_a: torch.Tensor,
    x_g: torch.Tensor,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    if hidden_states.shape != delta.shape:
        raise ValueError(
            "`hidden_states` and `delta` must have the same shape, got "
            f"{hidden_states.shape} and {delta.shape}."
        )
    if hidden_states.ndim != 2:
        raise ValueError(f"`hidden_states` must be 2D, got {hidden_states.ndim}.")

    if (
        not HAS_TRITON
        or hidden_states.device.type != "cuda"
        or hidden_states.numel() == 0
        or not hidden_states.is_contiguous()
        or not delta.is_contiguous()
        or not _rwkv7_mix6_use_triton(hidden_states)
    ):
        return rwkv7_mix6_reference(
            hidden_states=hidden_states,
            delta=delta,
            x_r=x_r,
            x_w=x_w,
            x_k=x_k,
            x_v=x_v,
            x_a=x_a,
            x_g=x_g,
        )

    numel = hidden_states.numel()
    hidden_size = hidden_states.shape[-1]
    output_dtype = torch.result_type(hidden_states, x_r)
    xr = torch.empty_like(hidden_states, dtype=output_dtype)
    xw = torch.empty_like(hidden_states, dtype=output_dtype)
    xk = torch.empty_like(hidden_states, dtype=output_dtype)
    xv = torch.empty_like(hidden_states, dtype=output_dtype)
    xa = torch.empty_like(hidden_states, dtype=output_dtype)
    xg = torch.empty_like(hidden_states, dtype=output_dtype)
    block_size = min(2048, triton.next_power_of_2(hidden_size))
    num_warps = 4 if block_size <= 1024 else 8
    grid = (triton.cdiv(numel, block_size),)
    rwkv7_mix6_fwd_kernel[grid](
        x=hidden_states,
        delta=delta,
        x_r=x_r,
        x_w=x_w,
        x_k=x_k,
        x_v=x_v,
        x_a=x_a,
        x_g=x_g,
        xr=xr,
        xw=xw,
        xk=xk,
        xv=xv,
        xa=xa,
        xg=xg,
        numel=numel,
        hidden_size=hidden_size,
        BLOCK_SIZE=block_size,
        num_warps=num_warps,
    )
    return xr, xw, xk, xv, xa, xg


@triton.jit
def rwkv7_kk_pre_fwd_kernel(
    k,
    a,
    k_k,
    k_a,
    k_out,
    kk_out,
    num_rows,
    num_heads,
    head_dim,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    if row >= num_rows:
        return

    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < head_dim
    head_idx = row % num_heads

    row_offset = row * head_dim
    head_offset = head_idx * head_dim

    k_vals = tl.load(k + row_offset + offsets, mask=mask, other=0).to(tl.float32)
    a_vals = tl.load(a + row_offset + offsets, mask=mask, other=0).to(tl.float32)
    k_k_vals = tl.load(k_k + head_offset + offsets, mask=mask, other=0).to(tl.float32)
    k_a_vals = tl.load(k_a + head_offset + offsets, mask=mask, other=0).to(tl.float32)

    kk_raw = k_vals * k_k_vals
    rstd = tl.rsqrt(tl.sum(kk_raw * kk_raw, axis=0) + eps)
    kk_vals = kk_raw * rstd
    k_adj = k_vals * (1 + (a_vals - 1) * k_a_vals)

    tl.store(k_out + row_offset + offsets, k_adj, mask=mask)
    tl.store(kk_out + row_offset + offsets, kk_vals, mask=mask)


def rwkv7_kk_pre_reference(
    k: torch.Tensor,
    k_k: torch.Tensor,
    a: torch.Tensor,
    k_a: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    kk = torch.nn.functional.normalize(k * k_k, dim=-1, p=2.0)
    k_adj = k * (1 + (a - 1) * k_a)
    return k_adj, kk


def rwkv7_kk_pre(
    k: torch.Tensor,
    k_k: torch.Tensor,
    a: torch.Tensor,
    k_a: torch.Tensor,
    *,
    eps: float = 1e-12,
) -> tuple[torch.Tensor, torch.Tensor]:
    if k.shape != a.shape:
        raise ValueError(f"`k` and `a` must match, got {k.shape} and {a.shape}.")
    if k.ndim != 3:
        raise ValueError(f"`k` must be 3D, got {k.ndim}.")
    if k_k.shape != k_a.shape:
        raise ValueError(
            f"`k_k` and `k_a` must match, got {k_k.shape} and {k_a.shape}."
        )
    if k_k.ndim != 2:
        raise ValueError(f"`k_k` must be 2D, got {k_k.ndim}.")
    if k.shape[1:] != k_k.shape:
        raise ValueError(
            "`k_k`/`k_a` must match the head layout of `k`, got "
            f"{k.shape[1:]} and {k_k.shape}."
        )

    if (
        not HAS_TRITON
        or k.device.type != "cuda"
        or k.numel() == 0
        or not k.is_contiguous()
        or not a.is_contiguous()
    ):
        return rwkv7_kk_pre_reference(k=k, k_k=k_k, a=a, k_a=k_a)

    output_dtype = torch.result_type(k, k_k)
    k_out = torch.empty_like(k, dtype=output_dtype)
    kk_out = torch.empty_like(k, dtype=output_dtype)
    num_rows = k.shape[0] * k.shape[1]
    num_heads = k.shape[1]
    head_dim = k.shape[2]
    block_size = triton.next_power_of_2(head_dim)
    num_warps = 4 if block_size <= 64 else 8

    rwkv7_kk_pre_fwd_kernel[(num_rows,)](
        k=k,
        a=a,
        k_k=k_k,
        k_a=k_a,
        k_out=k_out,
        kk_out=kk_out,
        num_rows=num_rows,
        num_heads=num_heads,
        head_dim=head_dim,
        eps=eps,
        BLOCK_SIZE=block_size,
        num_warps=num_warps,
    )
    return k_out, kk_out


@triton.jit
def rwkv7_cast_kk_pre_fwd_kernel(
    r, w, k, a, v, k_k, k_a,
    r_out, w_out, k_out, v_out, kk_out, a_out,
    num_rows, num_heads, head_dim, eps,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    if row >= num_rows:
        return
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < head_dim
    row_offset = row * head_dim
    head_offset = (row % num_heads) * head_dim
    r_vals = tl.load(r + row_offset + offsets, mask=mask, other=0).to(tl.float32)
    w_vals = tl.load(w + row_offset + offsets, mask=mask, other=0).to(tl.float32)
    k_vals = tl.load(k + row_offset + offsets, mask=mask, other=0).to(tl.float32)
    a_vals = tl.load(a + row_offset + offsets, mask=mask, other=0).to(tl.float32)
    v_vals = tl.load(v + row_offset + offsets, mask=mask, other=0).to(tl.float32)
    k_k_vals = tl.load(k_k + head_offset + offsets, mask=mask, other=0).to(tl.float32)
    k_a_vals = tl.load(k_a + head_offset + offsets, mask=mask, other=0).to(tl.float32)
    kk_raw = k_vals * k_k_vals
    kk_vals = kk_raw * tl.rsqrt(tl.sum(kk_raw * kk_raw, axis=0) + eps)
    k_adj = k_vals * (1 + (a_vals - 1) * k_a_vals)
    tl.store(r_out + row_offset + offsets, r_vals, mask=mask)
    tl.store(w_out + row_offset + offsets, w_vals, mask=mask)
    tl.store(k_out + row_offset + offsets, k_adj, mask=mask)
    tl.store(v_out + row_offset + offsets, v_vals, mask=mask)
    tl.store(kk_out + row_offset + offsets, kk_vals, mask=mask)
    tl.store(a_out + row_offset + offsets, a_vals, mask=mask)


def rwkv7_cast_kk_pre(
    r: torch.Tensor,
    w: torch.Tensor,
    k: torch.Tensor,
    a: torch.Tensor,
    v: torch.Tensor,
    k_k: torch.Tensor,
    k_a: torch.Tensor,
    *,
    eps: float = 1e-12,
) -> tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
]:
    """Fuse exact BF16->FP32 casts with RWKV7's k/kk preparation.

    The specialized path is intentionally limited to equal 64-wide R/W/K/A/V
    head layouts. It preserves the existing Triton kk-pre expression and only
    removes materialized FP32 k/a intermediates and separate cast kernels.
    """
    tensors = (r, w, k, a, v)
    if (
        not HAS_TRITON
        or any(x.device.type != "cuda" or x.dtype != torch.bfloat16 for x in tensors)
        or any(x.ndim != 3 or not x.is_contiguous() for x in tensors)
        or r.shape != w.shape or r.shape != k.shape or r.shape != a.shape
        or r.shape != v.shape or r.shape[-1] != 64
        or k_k.shape != k_a.shape or k_k.shape != r.shape[1:]
    ):
        r32, w32, k32, a32, v32 = (x.to(torch.float32) for x in tensors)
        k32, kk = rwkv7_kk_pre(k32, k_k, a32, k_a, eps=eps)
        return r32, w32, k32, v32, kk, a32
    r_out, w_out, k_out, v_out, kk_out, a_out = (
        torch.empty_like(x, dtype=torch.float32) for x in (r, w, k, v, k, a)
    )
    num_rows = r.shape[0] * r.shape[1]
    rwkv7_cast_kk_pre_fwd_kernel[(num_rows,)](
        r, w, k, a, v, k_k, k_a,
        r_out, w_out, k_out, v_out, kk_out, a_out,
        num_rows, r.shape[1], r.shape[2], eps,
        BLOCK_SIZE=64,
        num_warps=4,
    )
    return r_out, w_out, k_out, v_out, kk_out, a_out


@triton.jit
def rwkv7_lnx_rkvres_xg_fwd_kernel(
    recurrent_output,
    r,
    k,
    v,
    r_k,
    weight,
    bias,
    g,
    out,
    num_heads,
    head_dim,
    head_v_dim,
    eps,
    BLOCK_K: tl.constexpr,
    BLOCK_V: tl.constexpr,
):
    row_head = tl.program_id(0).to(tl.int64)
    head_idx = row_head % num_heads
    token_idx = row_head // num_heads

    k_offsets = tl.arange(0, BLOCK_K)
    v_offsets = tl.arange(0, BLOCK_V)
    mask_k = k_offsets < head_dim
    mask_v = v_offsets < head_v_dim

    key_base = row_head * head_dim
    value_base = row_head * head_v_dim
    gate_base = token_idx * num_heads * head_v_dim + head_idx * head_v_dim
    r_k_base = head_idx * head_dim
    affine_base = head_idx * head_v_dim

    x_vals = tl.load(
        recurrent_output + value_base + v_offsets,
        mask=mask_v,
        other=0,
    ).to(tl.float32)
    mean = tl.sum(x_vals, axis=0) / head_v_dim
    centered = tl.where(mask_v, x_vals - mean, 0.0)
    var = tl.sum(centered * centered, axis=0) / head_v_dim
    rstd = tl.rsqrt(var + eps)

    r_vals = tl.load(r + key_base + k_offsets, mask=mask_k, other=0).to(tl.float32)
    k_vals = tl.load(k + key_base + k_offsets, mask=mask_k, other=0).to(tl.float32)
    r_k_vals = tl.load(r_k + r_k_base + k_offsets, mask=mask_k, other=0).to(tl.float32)
    correction_scale = tl.sum(r_vals * k_vals * r_k_vals, axis=0)

    v_vals = tl.load(v + value_base + v_offsets, mask=mask_v, other=0).to(tl.float32)
    weight_vals = tl.load(weight + affine_base + v_offsets, mask=mask_v, other=0).to(
        tl.float32
    )
    bias_vals = tl.load(bias + affine_base + v_offsets, mask=mask_v, other=0).to(
        tl.float32
    )
    g_vals = tl.load(g + gate_base + v_offsets, mask=mask_v, other=0).to(tl.float32)

    y = centered * rstd * weight_vals + bias_vals + correction_scale * v_vals
    tl.store(out + gate_base + v_offsets, y * g_vals, mask=mask_v)


def rwkv7_lnx_rkvres_xg_reference(
    recurrent_output: torch.Tensor,
    r: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    r_k: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    g: torch.Tensor,
    *,
    eps: float,
    output_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    if output_dtype is None:
        output_dtype = g.dtype
    num_heads = recurrent_output.shape[1]
    local_value_dim = recurrent_output.shape[1] * recurrent_output.shape[2]
    output = torch.nn.functional.group_norm(
        recurrent_output.reshape(-1, local_value_dim).to(torch.float32).unsqueeze(-1),
        num_groups=num_heads,
        weight=weight.to(torch.float32),
        bias=bias.to(torch.float32),
        eps=eps,
    ).squeeze(-1)
    correction = ((r * k * r_k.unsqueeze(0)).sum(dim=-1, keepdim=True) * v).reshape(
        -1, local_value_dim
    )
    return ((output + correction) * g.to(torch.float32)).to(output_dtype)


def rwkv7_lnx_rkvres_xg(
    recurrent_output: torch.Tensor,
    r: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    r_k: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    g: torch.Tensor,
    *,
    eps: float,
    output_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    if recurrent_output.ndim != 3:
        raise ValueError(f"`recurrent_output` must be 3D, got {recurrent_output.ndim}.")
    if r.shape != k.shape:
        raise ValueError(f"`r` and `k` must match, got {r.shape} and {k.shape}.")
    if recurrent_output.shape != v.shape:
        raise ValueError(
            "`recurrent_output` and `v` must match, got "
            f"{recurrent_output.shape} and {v.shape}."
        )
    if recurrent_output.shape[:2] != r.shape[:2]:
        raise ValueError(
            "`recurrent_output` and `r` must share token/head dimensions, got "
            f"{recurrent_output.shape[:2]} and {r.shape[:2]}."
        )
    if r_k.shape != r.shape[1:]:
        raise ValueError(f"`r_k` must have shape {r.shape[1:]}, got {r_k.shape}.")

    num_tokens, num_heads, head_v_dim = recurrent_output.shape
    head_dim = r.shape[-1]
    local_value_dim = num_heads * head_v_dim
    if weight.shape != (local_value_dim,) or bias.shape != (local_value_dim,):
        raise ValueError(
            "`weight` and `bias` must match the flattened local value dimension "
            f"{local_value_dim}, got {weight.shape} and {bias.shape}."
        )
    if g.shape != (num_tokens, local_value_dim):
        raise ValueError(
            f"`g` must have shape {(num_tokens, local_value_dim)}, got {g.shape}."
        )

    if output_dtype is None:
        output_dtype = g.dtype

    if (
        not HAS_TRITON
        or recurrent_output.device.type != "cuda"
        or recurrent_output.numel() == 0
        or not recurrent_output.is_contiguous()
        or not r.is_contiguous()
        or not k.is_contiguous()
        or not v.is_contiguous()
        or not r_k.is_contiguous()
        or not weight.is_contiguous()
        or not bias.is_contiguous()
        or not g.is_contiguous()
    ):
        return rwkv7_lnx_rkvres_xg_reference(
            recurrent_output=recurrent_output,
            r=r,
            k=k,
            v=v,
            r_k=r_k,
            weight=weight,
            bias=bias,
            g=g,
            eps=eps,
            output_dtype=output_dtype,
        )

    block_k = triton.next_power_of_2(head_dim)
    block_v = triton.next_power_of_2(head_v_dim)
    if block_k > 1024 or block_v > 1024:
        return rwkv7_lnx_rkvres_xg_reference(
            recurrent_output=recurrent_output,
            r=r,
            k=k,
            v=v,
            r_k=r_k,
            weight=weight,
            bias=bias,
            g=g,
            eps=eps,
            output_dtype=output_dtype,
        )

    out = torch.empty(
        (num_tokens, local_value_dim),
        device=g.device,
        dtype=output_dtype,
    )
    num_warps = 4 if max(block_k, block_v) <= 64 else 8
    rwkv7_lnx_rkvres_xg_fwd_kernel[(num_tokens * num_heads,)](
        recurrent_output=recurrent_output,
        r=r,
        k=k,
        v=v,
        r_k=r_k,
        weight=weight,
        bias=bias,
        g=g,
        out=out,
        num_heads=num_heads,
        head_dim=head_dim,
        head_v_dim=head_v_dim,
        eps=eps,
        BLOCK_K=block_k,
        BLOCK_V=block_v,
        num_warps=num_warps,
    )
    return out


@triton.heuristics(
    {
        "USE_INITIAL_STATE": lambda args: args["h0"] is not None,
        "STORE_FINAL_STATE": lambda args: args["ht"] is not None,
        "STORE_CHECKPOINT_STATE": lambda args: args["hc"] is not None,
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

        b_h = (
            exp(b_w)[:, None] * b_h
            + b_b[:, None] * tl.sum(b_act_a[:, None] * b_h, 0)[None, :]
        )
        b_h += b_k[:, None] * b_v[None, :]
        b_o = tl.sum(b_h * b_r[:, None], 0)

        tl.store(p_o, b_o.to(p_o.dtype.element_ty), mask=mask_v)

        if STORE_CHECKPOINT_STATE:
            should_store = has_checkpoint & (t == checkpoint_pos)
            safe_checkpoint_idx = tl.where(has_checkpoint, checkpoint_idx, 0)
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
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    if r.ndim != 4:
        raise ValueError(f"`r` must be 4D, got {r.ndim}.")
    if cu_seqlens is not None and r.shape[0] != 1:
        raise ValueError("When `cu_seqlens` is provided, the batch size must be 1.")
    if output_checkpoint_states and (
        checkpoint_positions is None or checkpoint_offsets is None
    ):
        raise ValueError(
            "`checkpoint_positions` and `checkpoint_offsets` are required "
            "when `output_checkpoint_states=True`."
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

        if output_checkpoint_states:
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
            sa = (state * (-kk[batch_idx, tensor_token_idx]).unsqueeze(-1)).sum(dim=-2)
            state = (
                torch.exp(w[batch_idx, tensor_token_idx]).unsqueeze(-1) * state
                + (
                    kk[batch_idx, tensor_token_idx] * a[batch_idx, tensor_token_idx]
                ).unsqueeze(-1)
                * sa.unsqueeze(-2)
                + k[batch_idx, tensor_token_idx].unsqueeze(-1)
                * v[batch_idx, tensor_token_idx].unsqueeze(-2)
            )
            out[batch_idx, tensor_token_idx] = (
                (state * r[batch_idx, tensor_token_idx].unsqueeze(-1))
                .sum(dim=-2)
                .to(out.dtype)
            )

            if (
                checkpoint_states is not None
                and checkpoint_idx < checkpoint_end
                and local_token_idx == int(checkpoint_positions[checkpoint_idx].item())
            ):
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
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor]:
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
        output_checkpoint_states=True,
    )
    assert checkpoint_states is not None
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
        )

    if cu_seqlens is not None and r.shape[0] != 1:
        raise ValueError("When `cu_seqlens` is provided, the batch size must be 1.")

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

    hc = None
    if output_checkpoint_states:
        if checkpoint_positions is None or checkpoint_offsets is None:
            raise ValueError(
                "`checkpoint_positions` and `checkpoint_offsets` are required "
                "when `output_checkpoint_states=True`."
            )
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
    return o, ht, hc
