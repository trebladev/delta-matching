"""Packed varlen FP8 producers for block-aligned sequences.

This module is deliberately separate from the fixed-shape producers.  It provides the
layout needed by the native varlen CUDA path: codes are concatenated in sequence order,
while scale rows are padded to one common block count per batch.  The current kernels
require non-empty sequence lengths to be multiples of their quantization block. Empty
segments are retained in the metadata and produce no codes or twin columns.
"""
from __future__ import annotations

import torch

from .core import _quant_native, blocks
from .twins import twin_pi


def _validate_cu(cu: torch.Tensor, total: int, name: str, require_cuda: bool = True) -> torch.Tensor:
    if not isinstance(cu, torch.Tensor) or cu.dim() != 1 or cu.dtype != torch.int32:
        raise TypeError(f"delta_matching: {name} must be a 1-D int32 tensor")
    if require_cuda and not cu.is_cuda:
        raise ValueError(f"delta_matching: {name} must be a CUDA tensor")
    if cu.numel() < 2 or int(cu[0]) != 0 or int(cu[-1]) != total:
        raise ValueError(f"delta_matching: {name} must start at 0 and end at packed length {total}")
    if bool((cu[1:] < cu[:-1]).any()):
        raise ValueError(f"delta_matching: {name} must be nondecreasing")
    return cu


def validate_packed_inputs(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                           cu_seqlens_q: torch.Tensor, cu_seqlens_k: torch.Tensor) -> None:
    """Validate packed shapes and offsets without requiring CUDA.

    This is also used by CPU-only shape tests.  Quantization itself still requires
    CUDA because it delegates to the Triton producers.
    """
    if not all(isinstance(x, torch.Tensor) for x in (q, k, v, cu_seqlens_q, cu_seqlens_k)):
        raise TypeError("delta_matching: packed inputs must be tensors")
    if q.dim() != 3 or k.dim() != 3 or v.dim() != 3:
        raise ValueError("delta_matching: packed q, k, v must be 3-D")
    if k.shape != v.shape or q.shape[2] != k.shape[2]:
        raise ValueError("delta_matching: packed q, k, v have incompatible shapes")
    if q.dtype != k.dtype or q.dtype != v.dtype:
        raise TypeError("delta_matching: packed q, k, v must have one dtype")
    if not (q.device == k.device == v.device):
        raise ValueError("delta_matching: packed q, k, v must be on one device")
    if q.dtype != torch.bfloat16:
        raise TypeError(f"delta_matching: packed q, k, v must be bfloat16 (got {q.dtype})")
    if q.shape[1] == 0 or k.shape[1] == 0 or q.shape[1] % k.shape[1]:
        raise ValueError("delta_matching: q heads must be a positive multiple of k/v heads")
    if q.shape[2] not in (128, 256):
        raise ValueError(f"delta_matching: head_dim must be 128 or 256 (got {q.shape[2]})")
    _validate_cu(cu_seqlens_q, q.shape[0], "cu_seqlens_q", require_cuda=False)
    _validate_cu(cu_seqlens_k, k.shape[0], "cu_seqlens_k", require_cuda=False)
    if cu_seqlens_q.numel() != cu_seqlens_k.numel():
        raise ValueError("delta_matching: cu_seqlens_q and cu_seqlens_k must have the same batch size")


def _pack_one(x: torch.Tensor, cu: torch.Tensor, h: int, D: int, kind: str,
              blk: int, want_twin: bool, used_lengths=None):
    codes, scales, twin = [], [], []
    max_blocks = 0
    for b in range(cu.numel() - 1):
        lo, hi = int(cu[b]), int(cu[b + 1])
        n = hi - lo
        logical_n = n if used_lengths is None else int(used_lengths[b])
        if n < 0 or logical_n < 0 or logical_n > n or (n > 0 and n % blk) or (logical_n > 0 and logical_n % blk):
            raise ValueError(f"delta_matching: {kind} sequence lengths must be nonnegative {blk}-multiples (got {n})")
        if n == 0 or logical_n == 0:
            # FA3 permits empty segments.  There is no producer work or twin
            # tile for such a segment, but its padded scale row is retained.
            scales.append(torch.empty((h, 0), device=x.device, dtype=torch.float32))
            codes.append(torch.zeros((n, h, D), device=x.device, dtype=torch.float8_e4m3fn))
            max_blocks = max(max_blocks, logical_n // blk)
            if want_twin:
                twin.append(torch.zeros((h, D, n), device=x.device, dtype=torch.float8_e4m3fn))
            continue
        xb = x[lo:lo + logical_n].permute(1, 0, 2).contiguous().unsqueeze(0)
        if kind == "q":
            q8, s, _, = _quant_native(xb, xb, xb)[:3]
            # _quant_native quantizes all three operands.  Keep its Q result only.
            c, sc = q8, s
        else:
            # Use the native route so K centering and scale arithmetic stay bitwise
            # identical to the fixed-shape producer.
            _, _, c, sc, _, _, kt = _quant_native(xb, xb, xb)
            if kind == "v":
                _, _, _, _, c, sc, _ = _quant_native(xb, xb, xb)
        c = c.reshape(h, logical_n, D).permute(1, 0, 2).contiguous()
        if logical_n < n:
            c = torch.cat((c, torch.zeros((n - logical_n, h, D), device=x.device, dtype=c.dtype)), dim=0)
        sc = sc.reshape(h, logical_n // blk).contiguous()
        codes.append(c)
        scales.append(sc)
        max_blocks = max(max_blocks, n // blk)
        if want_twin:
            t = twin_pi(c.reshape(h, n, D)).contiguous()
            twin.append(t)
    packed = torch.cat(codes, dim=0) if codes else torch.empty((0, h, D), device=x.device, dtype=torch.float8_e4m3fn)
    # A varlen tile can cover masked keys beyond a sequence's logical end.
    # The forward kernel still reads the complete tile's scale slots while
    # constructing its per-group P/V basis.  Keep those unused slots finite
    # and nonzero so a masked tail cannot create 0/0 in the basis ratios.
    padded = torch.ones((cu.numel() - 1, h, max_blocks), device=x.device, dtype=torch.float32)
    for b, sc in enumerate(scales):
        padded[b, :, :sc.shape[1]] = sc
    tw = torch.cat(twin, dim=2) if want_twin else None
    return packed, padded, tw


def quantize_packed(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                    cu_seqlens_q: torch.Tensor, cu_seqlens_k: torch.Tensor,
                    max_seqlen_q: int | None = None, max_seqlen_k: int | None = None,
                    seqused_q: torch.Tensor | None = None, seqused_k: torch.Tensor | None = None):
    """Quantize packed ``q``, ``k`` and ``v`` for a native varlen launch.

    Inputs use FlashAttention's packed layout ``(total_tokens, heads, head_dim)``.
    When ``seqused_q``/``seqused_k`` are supplied, only each sequence's logical
    prefix is quantized and the physical suffix is filled with zero codes.
    The returned code tensors use the same layout.  Scale tensors are padded to the
    largest block count in the corresponding batch.  ``q_t``/``k_t`` use the kernel's
    pi-ordered twin layout ``(heads, head_dim, total_tokens)``.
    """
    validate_packed_inputs(q, k, v, cu_seqlens_q, cu_seqlens_k)
    if not (q.is_cuda and k.is_cuda and v.is_cuda):
        raise ValueError("delta_matching: packed q, k, v must be CUDA tensors")
    D = q.shape[2]
    _validate_cu(cu_seqlens_q, q.shape[0], "cu_seqlens_q")
    _validate_cu(cu_seqlens_k, k.shape[0], "cu_seqlens_k")
    if (seqused_q is None) != (seqused_k is None):
        raise ValueError("delta_matching: seqused_q and seqused_k must be provided together")
    used_q = used_k = None
    if seqused_q is not None:
        used_q = [int(x) for x in seqused_q.detach().cpu().tolist()]
        used_k = [int(x) for x in seqused_k.detach().cpu().tolist()]
        if len(used_q) != cu_seqlens_q.numel() - 1 or len(used_k) != cu_seqlens_k.numel() - 1:
            raise ValueError("delta_matching: seqused tensors must have one value per batch")
    bm, bn = blocks(D)
    q8, q8s, qt = _pack_one(q, cu_seqlens_q, q.shape[1], D, "q", bm, True, used_q)
    k8, k8s, kt = _pack_one(k, cu_seqlens_k, k.shape[1], D, "k", bn, True, used_k)
    v8, v8s, _ = _pack_one(v, cu_seqlens_k, v.shape[1], D, "v", bn, False, used_k)
    if max_seqlen_q is not None and q8s.size(2) < (int(max_seqlen_q) + bm - 1) // bm:
        q8s = torch.nn.functional.pad(q8s, (0, (int(max_seqlen_q) + bm - 1) // bm - q8s.size(2)), value=1.0)
    if max_seqlen_k is not None and k8s.size(2) < (int(max_seqlen_k) + bn - 1) // bn:
        pad = (int(max_seqlen_k) + bn - 1) // bn - k8s.size(2)
        k8s = torch.nn.functional.pad(k8s, (0, pad), value=1.0)
        v8s = torch.nn.functional.pad(v8s, (0, pad), value=1.0)
    return {"q8": q8, "q8s": q8s, "k8": k8, "k8s": k8s, "v8": v8, "v8s": v8s,
            "q_t": qt, "k_t": kt}


def quantize_dout_packed(dout: torch.Tensor, cu_seqlens_q: torch.Tensor):
    """Quantize packed dO using the Q-side row-block recipe."""
    if dout.dim() != 3 or dout.dtype != torch.bfloat16:
        raise TypeError("delta_matching: packed dO must be bfloat16 [total, heads, d]")
    bm, _ = blocks(dout.shape[-1])
    do8, do8s, dot = _pack_one(
        dout, cu_seqlens_q, dout.shape[1], dout.shape[2], "q", bm, True
    )
    return {"do8": do8, "do8s": do8s, "dot": dot}
