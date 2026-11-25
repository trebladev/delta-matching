"""Route A (Sequence-level Block-Aligned Packing) varlen metadata, packing helpers,
and Triton quantizers / K-centering for Hopper SM90.

Data contract:
  - Each sequence b is padded at its tail to a multiple of ALIGN:
      D == 128 -> ALIGN = 128  (scale blocks: Q/dO = 128, K/V = 64)
      D == 256 -> ALIGN = 128  (scale blocks: Q/dO =  64, K/V = 32)
  - Packed 3-D tensors:
      Q, O, dO: (TotalPaddedTokens, H, D)
      K, V:     (TotalPaddedTokens, Hk, D)
  - Metadata tensors:
      cu_seqlens_padded: int32 [B + 1], cumulative padded token offsets
      seqlens_actual:    int32 [B],     valid token counts (0 <= actual <= padded)
      cu_padded_sq:      int64 [B + 1], cumulative squared padded lengths
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl

from . import producers as _producers
from .constants import _TL_TINY, _TL_INV_E4M3_MAX, _TL_E4M3_MAX
from .core import blocks as _blocks
from .twins import twin_pi

E4M3_MAX = tl.constexpr(448.0)


def varlen_align(head_dim: int) -> int:
    """Return Route-A sequence tail alignment for a given head dimension."""
    if head_dim == 128:
        return 128
    if head_dim == 256:
        return 128
    raise ValueError(f"delta_matching: head_dim must be 128 or 256 (got {head_dim})")


def make_varlen_metadata(
    seqlens_actual: torch.Tensor, align: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build Route-A metadata: (cu_seqlens_padded [int32], seqlens_actual [int32], cu_padded_sq [int64])."""
    if not isinstance(seqlens_actual, torch.Tensor) or seqlens_actual.dtype != torch.int32 or seqlens_actual.dim() != 1:
        raise TypeError("delta_matching: seqlens_actual must be a 1-D int32 tensor")
    if align <= 0:
        raise ValueError(f"delta_matching: align must be positive (got {align})")
    seqlens_actual = seqlens_actual.contiguous()
    if seqlens_actual.numel() > 0 and bool(torch.any(seqlens_actual < 0)):
        raise ValueError("delta_matching: seqlens_actual must be non-negative")
    padded = ((seqlens_actual.to(torch.int64) + (align - 1)) // align) * align
    B = padded.numel()
    dev = seqlens_actual.device
    cu = torch.zeros((B + 1,), device=dev, dtype=torch.int32)
    sq = torch.zeros((B + 1,), device=dev, dtype=torch.int64)
    if B > 0:
        cu_i64 = torch.cumsum(padded, dim=0)
        if int(cu_i64[-1]) > 0x7FFFFFFF:
            raise OverflowError("delta_matching: total padded tokens exceed int32 range")
        cu[1:] = cu_i64.to(torch.int32)
        sq[1:] = torch.cumsum(padded * padded, dim=0)
    return cu, seqlens_actual, sq


def pack_varlen_sequences(q, k, v, seqlens_actual=None, align: int | None = None, layout: str = "bhnd"):
    """Pack per-sequence tensors into Route-A block-aligned 3-D tensors (TotalPaddedTokens, H, D).

    Accepts either:
      - 4-D tensors q, k, v in (B, H, N, D) when layout="bhnd" (default) or (B, N, H, D) when layout="bnhd",
        together with 1-D int32 `seqlens_actual`; OR
      - sequences (list/tuple) of 3-D tensors [(S_b, H, D), ...] (with optional `seqlens_actual`).

    All tail pad regions [actual_len[b], padded_len[b]) are strictly zeroed.
    Returns (qp, kp, vp, cu_seqlens_padded, seqlens_actual, cu_padded_sq).
    """
    if isinstance(q, (list, tuple)):
        if not (isinstance(k, (list, tuple)) and isinstance(v, (list, tuple))):
            raise TypeError("delta_matching: q, k, v must all be tensors or all be lists of 3-D tensors")
        B = len(q)
        if len(k) != B or len(v) != B or B == 0:
            raise ValueError("delta_matching: q, k, v lists must be non-empty and have the same length")
        dev, dtype = q[0].device, q[0].dtype
        H, D = q[0].shape[1], q[0].shape[2]
        Hk = k[0].shape[1]
        lens = [int(t.shape[0]) for t in q]
        for b in range(B):
            if q[b].dim() != 3 or k[b].dim() != 3 or v[b].dim() != 3:
                raise ValueError("delta_matching: each sequence tensor must be 3-D (S_b, H, D)")
            if q[b].shape[1:] != (H, D) or k[b].shape != (lens[b], Hk, D) or v[b].shape != (lens[b], Hk, D):
                raise ValueError("delta_matching: mismatched sequence shapes in list input")
        if seqlens_actual is None:
            actual_t = torch.tensor(lens, device=dev, dtype=torch.int32)
        else:
            actual_t = seqlens_actual.to(device=dev, dtype=torch.int32).contiguous()
            if actual_t.tolist() != lens:
                raise ValueError("delta_matching: seqlens_actual does not match list tensor lengths")
        align_val = varlen_align(D) if align is None else int(align)
        cu, actual, sq = make_varlen_metadata(actual_t, align_val)
        total = int(cu[-1])
        qp = torch.zeros((total, H, D), device=dev, dtype=dtype)
        kp = torch.zeros((total, Hk, D), device=dev, dtype=k[0].dtype)
        vp = torch.zeros((total, Hk, D), device=dev, dtype=v[0].dtype)
        for b, (start, n) in enumerate(zip(cu[:-1].tolist(), lens)):
            if n > 0:
                qp[start:start + n] = q[b]
                kp[start:start + n] = k[b]
                vp[start:start + n] = v[b]
        return qp, kp, vp, cu, actual, sq

    if q.dim() != 4 or k.dim() != 4 or v.dim() != 4:
        raise ValueError("delta_matching: q, k, and v must be 4-D tensors")
    if seqlens_actual is None:
        raise ValueError("delta_matching: seqlens_actual is required for 4-D inputs")
    if layout == "bhnd":
        B, H, N, D = q.shape
        Hk = k.shape[1]
        if k.shape != (B, Hk, N, D) or v.shape != (B, Hk, N, D):
            raise ValueError("delta_matching: q, k, and v must have matching (B, *, N, D) shapes")
    elif layout == "bnhd":
        B, N, H, D = q.shape
        Hk = k.shape[2]
        if k.shape != (B, N, Hk, D) or v.shape != (B, N, Hk, D):
            raise ValueError("delta_matching: q, k, and v must have matching (B, N, *, D) shapes")
    else:
        raise ValueError(f"delta_matching: layout must be 'bhnd' or 'bnhd' (got {layout!r})")

    align_val = varlen_align(D) if align is None else int(align)
    actual_t = seqlens_actual.to(device=q.device, dtype=torch.int32).contiguous()
    if actual_t.numel() != B:
        raise ValueError(f"delta_matching: seqlens_actual length ({actual_t.numel()}) must match batch size ({B})")
    if B > 0 and int(actual_t.max()) > N:
        raise ValueError(f"delta_matching: seqlens_actual max ({int(actual_t.max())}) exceeds N ({N})")
    cu, actual, sq = make_varlen_metadata(actual_t, align_val)
    total = int(cu[-1])
    qp = torch.zeros((total, H, D), device=q.device, dtype=q.dtype)
    kp = torch.zeros((total, Hk, D), device=k.device, dtype=k.dtype)
    vp = torch.zeros((total, Hk, D), device=v.device, dtype=v.dtype)
    for b, (start, n) in enumerate(zip(cu[:-1].tolist(), actual.tolist())):
        if n > 0:
            if layout == "bhnd":
                qp[start:start + n] = q[b, :, :n].transpose(0, 1)
                kp[start:start + n] = k[b, :, :n].transpose(0, 1)
                vp[start:start + n] = v[b, :, :n].transpose(0, 1)
            else:
                qp[start:start + n] = q[b, :n]
                kp[start:start + n] = k[b, :n]
                vp[start:start + n] = v[b, :n]
    return qp, kp, vp, cu, actual, sq


def unpack_varlen_tensor(
    x: torch.Tensor, cu_seqlens_padded: torch.Tensor, seqlens_actual: torch.Tensor
) -> list[torch.Tensor]:
    """Slice a packed (TotalPaddedTokens, H, D) tensor into per-sequence valid views [(actual_b, H, D), ...]."""
    if x.dim() != 3:
        raise ValueError(f"delta_matching: expected 3-D packed tensor (TotalPaddedTokens, H, D), got {x.dim()}-D")
    return [
        x[start:start + n]
        for start, n in zip(cu_seqlens_padded[:-1].tolist(), seqlens_actual.tolist())
    ]


# ----------------------------------------------------------------------------- Triton K-Centering (Varlen)
@triton.jit
def _kmean_varlen_kernel(
    K,        # (TotalPaddedTokens, Hk, D) bf16 with strides (skn, skh, skd)
    CU,       # int32 [B + 1]
    ACTUAL,   # int32 [B]
    KM,       # fp32 [B, Hk, D] contiguous
    Hk,
    skn, skh, skd,
    D: tl.constexpr,
    BD: tl.constexpr,
):
    """Per-(seq, head) FP32 mean of K over ONLY valid tokens [cu[b], cu[b] + actual[b]).
    Accumulates in 64-row chunks in ascending order (bitwise identical to quant._kmean_kernel
    on isolated sequences); stores 0.0 when actual[b] == 0 (no div-by-zero)."""
    pid_bh = tl.program_id(0)
    pid_d = tl.program_id(1)
    b = pid_bh // Hk
    h = pid_bh % Hk
    start = tl.load(CU + b)
    actual = tl.load(ACTUAL + b)
    rd = pid_d * BD + tl.arange(0, BD)
    acc = tl.zeros([BD], dtype=tl.float32)
    if actual > 0:
        n = 0
        while n < actual:
            rn = n + tl.arange(0, 64)
            mask_n = rn < actual
            off = (
                (start.to(tl.int64) + rn.to(tl.int64))[:, None] * skn
                + h.to(tl.int64) * skh
                + rd.to(tl.int64)[None, :] * skd
            )
            x = tl.load(K + off, mask=mask_n[:, None], other=0.0).to(tl.float32)
            acc += tl.sum(x, axis=0)
            n += 64
        mean = acc / actual.to(tl.float32)
    else:
        mean = acc
    tl.store(KM + pid_bh.to(tl.int64) * D + rd, mean)


def kmean_varlen(
    k: torch.Tensor, cu_seqlens_padded: torch.Tensor, seqlens_actual: torch.Tensor
) -> torch.Tensor:
    """Compute per-sequence, per-head FP32 mean of K over valid tokens only.
    k: (TotalPaddedTokens, Hk, D) bf16 CUDA -> KM: (B, Hk, D) fp32 contiguous."""
    if k.dim() != 3:
        raise ValueError(f"delta_matching: packed K must be 3-D (T, Hk, D), got {k.dim()}-D")
    _, Hk, D = k.shape
    B = seqlens_actual.shape[0]
    km = torch.zeros((B, Hk, D), device=k.device, dtype=torch.float32)
    if B == 0 or Hk == 0:
        return km
    BD = 64 if D % 64 == 0 else D
    _kmean_varlen_kernel[(B * Hk, D // BD)](
        k, cu_seqlens_padded, seqlens_actual, km,
        Hk, k.stride(0), k.stride(1), k.stride(2),
        D=D, BD=BD,
    )
    return km


# ----------------------------------------------------------------------------- Triton Varlen Q / K / V Quantizers
@triton.jit
def _quant_varlen_kernel(
    X,        # (T, H, D) bf16 with strides (sxn, sxh, sxd)
    KM,       # (B, H, D) fp32 contiguous (used only when MODE == 1)
    CU,       # int32 [B + 1]
    ACTUAL,   # int32 [B]
    X8,       # (H, T, D) e4m3 contiguous
    X8S,      # (H, T // BLK) fp32 contiguous
    X8T,      # (H, D, T) e4m3 contiguous (plain transpose, used when WANT_T8)
    T, B, H,
    sxn, sxh, sxd,
    SM: tl.constexpr,
    BLK: tl.constexpr,
    D: tl.constexpr,
    MODE: tl.constexpr,
    WANT_T8: tl.constexpr = False,
):
    """Sequence-local block-aligned quantizer matching quant._quant_kernel arithmetic bitwise.
    MODE 0: Q (x * SM), MODE 1: K (x - KM, pad forced back to 0.0), MODE 2: V (plain).
    Pad rows [actual_len[b], padded_len[b]) are strictly forced to 0.0 before absmax/quantization."""
    pid_m = tl.program_id(0)
    pid_h = tl.program_id(1)
    tok0 = pid_m * BLK

    # Locate unique sequence b with CU[b] <= tok0 < CU[b + 1] (handles empty sequences cleanly).
    lo = 0
    hi = B
    while lo + 1 < hi:
        mid = (lo + hi) >> 1
        if tl.load(CU + mid) <= tok0:
            lo = mid
        else:
            hi = mid
    b = lo
    seq_start = tl.load(CU + b)
    seq_actual = tl.load(ACTUAL + b)

    rm = tok0 + tl.arange(0, BLK)
    rd = tl.arange(0, D)
    valid = (rm - seq_start) < seq_actual
    xin = (
        rm.to(tl.int64)[:, None] * sxn
        + pid_h.to(tl.int64) * sxh
        + rd.to(tl.int64)[None, :] * sxd
    )
    x = tl.load(X + xin, mask=valid[:, None], other=0.0).to(tl.float32)

    if MODE == 0:
        x = x * SM
    if MODE == 1:
        km = tl.load(KM + (b.to(tl.int64) * H + pid_h) * D + rd)
        # Explicitly force pad rows back to 0.0 so -km cannot corrupt tail block's fp32 scale.
        x = tl.where(valid[:, None], x - km[None, :], 0.0)

    amax = tl.max(tl.abs(x))
    s = tl.maximum(amax, 1e-30) / E4M3_MAX
    tl.store(X8S + pid_h.to(tl.int64) * (T // BLK) + pid_m, s)
    q = tl.math.div_rn(x, tl.full([1, 1], 1.0, tl.float32) * s)
    q = tl.minimum(tl.maximum(q, -E4M3_MAX), E4M3_MAX)
    q8 = q.to(tl.float8e4nv)
    base = pid_h.to(tl.int64) * T * D + rm.to(tl.int64)[:, None] * D + rd[None, :]
    tl.store(X8 + base, q8)
    if WANT_T8:
        t8_off = pid_h.to(tl.int64) * T * D + rd.to(tl.int64)[:, None] * T + rm[None, :]
        tl.store(X8T + t8_off, tl.trans(q8))


@triton.jit
def _quant_q_varlen_twin_kernel(
    X,        # (T, H, D) bf16 with strides (sxn, sxh, sxd)
    CU,       # int32 [B + 1]
    ACTUAL,   # int32 [B]
    X8,       # (H, T, D) e4m3 contiguous
    X8S,      # (H, T // BLK) fp32 contiguous
    X8T,      # (H, D, T) e4m3 contiguous, pi-ordered
    T, B,
    sxn, sxh, sxd,
    SM: tl.constexpr,
    BLK: tl.constexpr,
    D: tl.constexpr,
    BD: tl.constexpr,
    WANT_T8: tl.constexpr,
    TWIN_MODE: tl.constexpr = 2,
):
    """Fused Q quantizer + pi-ordered twin (used when DM_FUSED_TWIN=1)."""
    pid_m = tl.program_id(0)
    pid_h = tl.program_id(1)
    tok0 = pid_m * BLK

    lo = 0
    hi = B
    while lo + 1 < hi:
        mid = (lo + hi) >> 1
        if tl.load(CU + mid) <= tok0:
            lo = mid
        else:
            hi = mid
    b = lo
    seq_start = tl.load(CU + b)
    seq_actual = tl.load(ACTUAL + b)

    rm_full = tok0 + tl.arange(0, BLK)
    valid_full = (rm_full - seq_start) < seq_actual
    xin_row_full = rm_full.to(tl.int64)[:, None] * sxn + pid_h.to(tl.int64) * sxh

    amax = tl.zeros([1], dtype=tl.float32)
    for c in tl.static_range(D // BD):
        rd = c * BD + tl.arange(0, BD)
        x = tl.load(X + xin_row_full + rd.to(tl.int64)[None, :] * sxd, mask=valid_full[:, None], other=0.0).to(tl.float32)
        x = x * SM
        amax = tl.maximum(amax, tl.max(tl.abs(x)))
    s = tl.maximum(amax, 1e-30) / E4M3_MAX
    tl.store(X8S + pid_h.to(tl.int64) * (T // BLK) + pid_m + tl.arange(0, 1), s)
    sv = tl.reshape(s, (1, 1))
    # Emit [64, BD] chunks so two-store fused twin avoids the ptxas 12.8 [128, 64] hazard.
    for m_sub in tl.static_range(BLK // 64):
        rm = tok0 + m_sub * 64 + tl.arange(0, 64)
        valid = (rm - seq_start) < seq_actual
        xin_row = rm.to(tl.int64)[:, None] * sxn + pid_h.to(tl.int64) * sxh
        base = pid_h.to(tl.int64) * T * D + rm.to(tl.int64)[:, None] * D
        pos = (rm & -16) | ((rm & 6) << 1) | ((rm & 8) >> 2) | (rm & 1)
        for c in tl.static_range(D // BD):
            rd = c * BD + tl.arange(0, BD)
            x = tl.load(X + xin_row + rd.to(tl.int64)[None, :] * sxd, mask=valid[:, None], other=0.0).to(tl.float32)
            x = x * SM
            q = tl.math.div_rn(x, tl.broadcast_to(sv, (64, BD)))
            q = tl.minimum(tl.maximum(q, -E4M3_MAX), E4M3_MAX)
            q8 = q.to(tl.float8e4nv)
            tl.store(X8 + base + rd[None, :], q8)
            if WANT_T8:
                if TWIN_MODE == 1:
                    t8_off = pid_h.to(tl.int64) * T * D + rd.to(tl.int64)[:, None] * T + pos[None, :]
                    tl.store(X8T + t8_off, tl.trans(q8))
                else:
                    t8_off = pid_h.to(tl.int64) * T * D + rd.to(tl.int64)[None, :] * T + pos[:, None]
                    tl.store(X8T + t8_off, q8)


def _check_varlen_meta(
    cu_seqlens_padded: torch.Tensor, seqlens_actual: torch.Tensor, T: int, device: torch.device, blk: int
) -> tuple[torch.Tensor, torch.Tensor, int]:
    if cu_seqlens_padded.dtype != torch.int32 or seqlens_actual.dtype != torch.int32:
        raise TypeError("delta_matching: cu_seqlens_padded and seqlens_actual must be int32")
    if cu_seqlens_padded.dim() != 1 or seqlens_actual.dim() != 1:
        raise ValueError("delta_matching: cu_seqlens_padded and seqlens_actual must be 1-D tensors")
    B = seqlens_actual.shape[0]
    if cu_seqlens_padded.shape[0] != B + 1:
        raise ValueError("delta_matching: cu_seqlens_padded must have shape [B + 1]")
    if cu_seqlens_padded.device != device or seqlens_actual.device != device:
        raise ValueError("delta_matching: metadata tensors must be on the same device as input tensors")
    if T % blk != 0:
        raise ValueError(f"delta_matching: TotalPaddedTokens ({T}) must be a multiple of block size ({blk})")
    return cu_seqlens_padded.contiguous(), seqlens_actual.contiguous(), B


def quant_q_varlen(
    q: torch.Tensor,
    cu_seqlens_padded: torch.Tensor,
    seqlens_actual: torch.Tensor,
    blk: int | None = None,
    want_twin: bool = True,
):
    """Quantize packed Q (T, H, D) bf16 -> (Q8 [H, T, D], Q8s [H, T//blk], Qt [H, D, T] | None)."""
    T, H, D = q.shape
    blk_q = _blocks(D)[0] if blk is None else int(blk)
    cu, actual, B = _check_varlen_meta(cu_seqlens_padded, seqlens_actual, T, q.device, blk_q)
    dev = q.device
    Q8 = torch.empty((H, T, D), device=dev, dtype=torch.float8_e4m3fn)
    Q8s = torch.empty((H, T // blk_q), device=dev, dtype=torch.float32)
    if T == 0:
        Qt = torch.empty((H, D, 0), device=dev, dtype=torch.float8_e4m3fn) if want_twin else None
        return Q8, Q8s, Qt

    sm = D ** -0.5
    fused = want_twin and _producers.FUSED_TWIN
    if fused:
        Qt = torch.empty((H, D, T), device=dev, dtype=torch.float8_e4m3fn)
        _quant_q_varlen_twin_kernel[(T // blk_q, H)](
            q, cu, actual, Q8, Q8s, Qt, T, B,
            q.stride(0), q.stride(1), q.stride(2),
            SM=sm, BLK=blk_q, D=D, BD=64, WANT_T8=True, TWIN_MODE=_producers.TWIN_MODE,
            num_warps=8,
        )
        _producers._selfcheck_twin(("quant_q_varlen", H, T, D, blk_q, _producers.TWIN_MODE), Q8, Qt)
        return Q8, Q8s, Qt

    _quant_varlen_kernel[(T // blk_q, H)](
        q, Q8s, cu, actual, Q8, Q8s, Q8, T, B, H,
        q.stride(0), q.stride(1), q.stride(2),
        SM=sm, BLK=blk_q, D=D, MODE=0, WANT_T8=False,
        num_warps=8,
    )
    Qt = twin_pi(Q8) if want_twin else None
    return Q8, Q8s, Qt


def quant_kv_varlen(
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_padded: torch.Tensor,
    seqlens_actual: torch.Tensor,
    blk_k: int | None = None,
):
    """Center K per sequence over valid tokens only, force pad to 0.0, and quantize K, V.
    k, v: (T, Hk, D) bf16 -> (K8 [Hk, T, D], K8s [Hk, T//blk_k], kTt [Hk, D, T], V8 [Hk, T, D], V8s [Hk, T//blk_k])."""
    T, Hk, D = k.shape
    if v.shape != (T, Hk, D):
        raise ValueError(f"delta_matching: K and V shapes must match (got {k.shape} vs {v.shape})")
    bk = _blocks(D)[1] if blk_k is None else int(blk_k)
    cu, actual, B = _check_varlen_meta(cu_seqlens_padded, seqlens_actual, T, k.device, bk)
    dev = k.device
    K8 = torch.empty((Hk, T, D), device=dev, dtype=torch.float8_e4m3fn)
    K8s = torch.empty((Hk, T // bk), device=dev, dtype=torch.float32)
    kTt = torch.empty((Hk, D, T), device=dev, dtype=torch.float8_e4m3fn)
    V8 = torch.empty((Hk, T, D), device=dev, dtype=torch.float8_e4m3fn)
    V8s = torch.empty((Hk, T // bk), device=dev, dtype=torch.float32)
    if T == 0:
        return K8, K8s, kTt, V8, V8s

    km = kmean_varlen(k, cu, actual)
    sm = D ** -0.5
    _quant_varlen_kernel[(T // bk, Hk)](
        k, km, cu, actual, K8, K8s, kTt, T, B, Hk,
        k.stride(0), k.stride(1), k.stride(2),
        SM=sm, BLK=bk, D=D, MODE=1, WANT_T8=True,
        num_warps=8,
    )
    _quant_varlen_kernel[(T // bk, Hk)](
        v, km, cu, actual, V8, V8s, V8, T, B, Hk,
        v.stride(0), v.stride(1), v.stride(2),
        SM=sm, BLK=bk, D=D, MODE=2, WANT_T8=False,
        num_warps=8,
    )
    return K8, K8s, kTt, V8, V8s


def quant_qkv_varlen(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_padded: torch.Tensor,
    seqlens_actual: torch.Tensor,
    want_qt: bool = True,
    blocks: tuple[int, int] | None = None,
):
    """Quantize packed Route-A Q, K, V tensors in single-launch Triton kernels.
    q: (T, H, D), k/v: (T, Hk, D) bf16 CUDA.
    Returns (Q8, Q8s, K8, K8s, V8, V8s, kTt, Qt)."""
    if q.dim() != 3 or k.dim() != 3 or v.dim() != 3:
        raise ValueError("delta_matching: packed q, k, v must be 3-D tensors (TotalPaddedTokens, H, D)")
    if q.shape[0] != k.shape[0] or q.shape[2] != k.shape[2] or k.shape != v.shape:
        raise ValueError("delta_matching: packed q, k, v must have matching token and head_dim dimensions")
    blk_q, blk_k = blocks if blocks is not None else _blocks(q.shape[-1])
    with torch.cuda.device(q.device):
        Q8, Q8s, Qt = quant_q_varlen(q, cu_seqlens_padded, seqlens_actual, blk=blk_q, want_twin=want_qt)
        K8, K8s, kTt, V8, V8s = quant_kv_varlen(k, v, cu_seqlens_padded, seqlens_actual, blk_k=blk_k)
    return Q8, Q8s, K8, K8s, V8, V8s, kTt, Qt


def quantize_packed_qkv(q, k, v, cu_seqlens_padded, seqlens_actual, D=None):
    """Alias for quant_qkv_varlen preserving compatibility with earlier draft callers."""
    d = q.shape[-1] if D is None else int(D)
    return quant_qkv_varlen(q, k, v, cu_seqlens_padded, seqlens_actual, want_qt=True, blocks=_blocks(d))


# ----------------------------------------------------------------------------- Triton Varlen dO Quantizer (+ stale delta)
@triton.jit
def _doq_varlen_kernel(
    DO,          # bf16 (T, H, D) with strides (sdn, sdh, 1)
    O,           # bf16 (T, H, D) with strides (son, soh, 1) (WANT_DELTA only)
    CU,          # int32 [B + 1]
    ACTUAL,      # int32 [B]
    DO8,         # [H, T, D] e4m3 out
    DO8S,        # [H, T // BM] fp32 out
    DLT,         # [H, T] fp32 out (WANT_DELTA)
    DOT,         # [H, D, T] e4m3 out (WANT_T8), pi-ordered
    T, B,
    sdn, sdh,
    son, soh,
    BM: tl.constexpr,
    D: tl.constexpr,
    BD: tl.constexpr,
    WANT_DELTA: tl.constexpr,
    WANT_T8: tl.constexpr,
    TWIN_MODE: tl.constexpr = 2,
    DELTA_FROM_CODES: tl.constexpr = 0,
):
    """Varlen dO quantizer bitwise matching producers._doq_m17_kernel.
    Pad rows [actual_len[b], padded_len[b]) are strictly zeroed before absmax, quantization, and delta."""
    pid_m = tl.program_id(0)
    pid_h = tl.program_id(1)
    tok0 = pid_m * BM

    lo = 0
    hi = B
    while lo + 1 < hi:
        mid = (lo + hi) >> 1
        if tl.load(CU + mid) <= tok0:
            lo = mid
        else:
            hi = mid
    b = lo
    seq_start = tl.load(CU + b)
    seq_actual = tl.load(ACTUAL + b)

    rm = tok0 + tl.arange(0, BM)
    valid = (rm - seq_start) < seq_actual
    dbase = rm.to(tl.int64)[:, None] * sdn + pid_h.to(tl.int64) * sdh
    obase = rm.to(tl.int64)[:, None] * son + pid_h.to(tl.int64) * soh
    cbase = pid_h.to(tl.int64) * T * D + rm.to(tl.int64)[:, None] * D

    # Pass A: block absmax over valid rows only (pad rows treated as 0.0)
    amax = tl.zeros([1], dtype=tl.float32)
    for c in tl.static_range(D // BD):
        rd = c * BD + tl.arange(0, BD)
        x = tl.load(DO + dbase + rd[None, :], mask=valid[:, None], other=0.0).to(tl.float32)
        amax = tl.maximum(amax, tl.max(tl.abs(x)))
    s = (
        tl.maximum(amax, tl.full([1], _TL_TINY, tl.float32))
        * tl.full([1], _TL_INV_E4M3_MAX, tl.float32)
    )
    tl.store(DO8S + pid_h.to(tl.int64) * (T // BM) + pid_m + tl.arange(0, 1), s)
    sv = tl.reshape(s, (1, 1))

    # Pass B: quantize + code stores (+ stale delta) (+ optional fused twin)
    dlt = tl.zeros([BM], dtype=tl.float32)
    pos = (rm & -16) | ((rm & 6) << 1) | ((rm & 8) >> 2) | (rm & 1)
    for c in tl.static_range(D // BD):
        rd = c * BD + tl.arange(0, BD)
        x = tl.load(DO + dbase + rd[None, :], mask=valid[:, None], other=0.0).to(tl.float32)
        q = tl.math.div_rn(x, tl.broadcast_to(sv, (BM, BD)))
        q = tl.minimum(
            tl.maximum(q, tl.full([1, 1], -_TL_E4M3_MAX, tl.float32)),
            tl.full([1, 1], _TL_E4M3_MAX, tl.float32),
        )
        q8 = q.to(tl.float8e4nv)
        # Store all BM rows so pad rows are actively written as 0x00.
        tl.store(DO8 + cbase + rd[None, :], q8)
        if WANT_T8:
            if TWIN_MODE == 1:
                t_off = pid_h.to(tl.int64) * T * D + rd.to(tl.int64)[:, None] * T + pos[None, :]
                tl.store(DOT + t_off, tl.trans(q8))
            else:
                t_off = pid_h.to(tl.int64) * T * D + rd.to(tl.int64)[None, :] * T + pos[:, None]
                tl.store(DOT + t_off, q8)
        if WANT_DELTA:
            ov = tl.load(O + obase + rd[None, :], mask=valid[:, None], other=0.0).to(tl.float32)
            if DELTA_FROM_CODES:
                dlt += tl.sum(q8.to(tl.float32) * tl.broadcast_to(sv, (BM, BD)) * ov, axis=1)
            else:
                dlt += tl.sum(x * ov, axis=1)
    if WANT_DELTA:
        tl.store(DLT + pid_h.to(tl.int64) * T + rm, dlt)


def doq_varlen(
    dO: torch.Tensor,
    O: torch.Tensor | None,
    cu_seqlens_padded: torch.Tensor,
    seqlens_actual: torch.Tensor,
    blk: int | None = None,
    want_delta: bool = False,
    want_twin: bool = True,
    delta_from_codes: bool = False,
):
    """Quantize packed dO (T, H, D) bf16 -> (dO8 [H, T, D], dO8s [H, T//blk], Dlt [H, T] | None, dOt [H, D, T] | None)."""
    if dO.dim() != 3:
        raise ValueError(f"delta_matching: packed dO must be 3-D (T, H, D), got {dO.dim()}-D")
    if dO.dtype != torch.bfloat16:
        raise TypeError(f"delta_matching: packed dO must be bfloat16 (got {dO.dtype})")
    if dO.stride(-1) != 1:
        dO = dO.contiguous()
    T, H, D = dO.shape
    blk_q = _blocks(D)[0] if blk is None else int(blk)
    cu, actual, B = _check_varlen_meta(cu_seqlens_padded, seqlens_actual, T, dO.device, blk_q)
    if want_delta:
        if O is None or O.shape != (T, H, D) or O.dtype != torch.bfloat16:
            raise ValueError("delta_matching: O must be a matching bf16 (T, H, D) tensor when want_delta=True")
        if O.stride(-1) != 1:
            O = O.contiguous()

    dev = dO.device
    dO8 = torch.empty((H, T, D), device=dev, dtype=torch.float8_e4m3fn)
    dO8s = torch.empty((H, T // blk_q), device=dev, dtype=torch.float32)
    dlt = torch.empty((H, T), device=dev, dtype=torch.float32) if want_delta else dO8s
    if T == 0:
        dOt = torch.empty((H, D, 0), device=dev, dtype=torch.float8_e4m3fn) if want_twin else None
        return dO8, dO8s, (dlt if want_delta else None), dOt

    BD = 64 if D % 64 == 0 else D
    num_warps = 16 if blk_q * BD >= 4096 else 8
    fused = want_twin and _producers.FUSED_TWIN
    dOt = torch.empty((H, D, T), device=dev, dtype=torch.float8_e4m3fn) if fused else dO8
    o_arg = O if want_delta else dO
    with torch.cuda.device(dev):
        _doq_varlen_kernel[(T // blk_q, H)](
            dO, o_arg, cu, actual, dO8, dO8s, dlt, dOt,
            T, B, dO.stride(0), dO.stride(1), o_arg.stride(0), o_arg.stride(1),
            BM=blk_q, D=D, BD=BD, WANT_DELTA=want_delta, WANT_T8=fused,
            TWIN_MODE=_producers.TWIN_MODE,
            DELTA_FROM_CODES=1 if (want_delta and delta_from_codes) else 0,
            num_warps=num_warps, num_stages=2,
        )
        if fused:
            _producers._selfcheck_twin(("doq_varlen", H, T, D, blk_q, want_delta, _producers.TWIN_MODE), dO8, dOt)
        elif want_twin:
            dOt = twin_pi(dO8)
        else:
            dOt = None
    return dO8, dO8s, (dlt if want_delta else None), dOt


def fwd_varlen_fp8(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_padded: torch.Tensor,
    seqlens_actual: torch.Tensor,
    causal: bool = True,
    softmax_scale: float | None = None,
    want_qt: bool = False,
):
    """Run Route-A Triton quantizers + SM90 varlen FP8 forward kernel.

    q: (TotalPaddedTokens, H, D), k/v: (TotalPaddedTokens, Hk, D) bf16 CUDA.
    Returns (O [(T, H, D) bf16], LSE [(H, T) fp32], quant_tuple)."""
    from .core import _ops, _check_hd
    if q.dim() != 3 or k.dim() != 3 or v.dim() != 3:
        raise ValueError("delta_matching: packed q, k, v must be 3-D tensors (TotalPaddedTokens, H, D)")
    if not (q.is_cuda and k.is_cuda and v.is_cuda):
        raise ValueError("delta_matching: packed q, k, v must be CUDA tensors")
    if q.dtype != torch.bfloat16 or k.dtype != torch.bfloat16 or v.dtype != torch.bfloat16:
        raise TypeError("delta_matching: packed q, k, v must be bfloat16")
    T, H, D = q.shape
    Hk = k.shape[1]
    if k.shape != (T, Hk, D) or v.shape != (T, Hk, D):
        raise ValueError("delta_matching: packed q, k, v must have matching token and head_dim dimensions")
    if H % Hk != 0:
        raise ValueError(f"delta_matching: nheads ({H}) must be divisible by nheads_k ({Hk})")
    align = varlen_align(D)
    if T % align != 0:
        raise ValueError(f"delta_matching: TotalPaddedTokens ({T}) must be a multiple of ALIGN ({align}) for D={D}")
    sm = D ** -0.5
    if softmax_scale is not None and abs(float(softmax_scale) - sm) > 1e-6 * sm:
        raise NotImplementedError("delta_matching: custom softmax_scale is not supported (must be D ** -0.5)")
    ops = _ops()
    _check_hd(D, ops)
    cu = cu_seqlens_padded.contiguous()
    actual = seqlens_actual.contiguous()
    with torch.cuda.device(q.device):
        Q8, Q8s, K8, K8s, V8, V8s, kTt, Qt = quant_qkv_varlen(
            q, k, v, cu, actual, want_qt=want_qt, blocks=_blocks(D)
        )
        # View [H, T, D] contiguous as (1, T, H, D) with row_stride=D, head_stride=T*D
        q8_v = Q8.view(1, H, T, D).permute(0, 2, 1, 3)
        k8_v = K8.view(1, Hk, T, D).permute(0, 2, 1, 3)
        v8_v = V8.view(1, Hk, T, D).permute(0, 2, 1, 3)
        O, Lse = ops.fwd_fp8_varlen(
            q8_v, k8_v, v8_v,
            Q8s.view(1, H, -1), K8s.view(1, Hk, -1), V8s.view(1, Hk, -1),
            cu, actual, sm, causal,
        )
    return O.squeeze(0), Lse.squeeze(0), (Q8, Q8s, K8, K8s, V8, V8s, kTt, Qt)


# ----------------------------------------------------------------------------- SM90 Varlen Autograd & Public API
def _check_varlen_attn_inputs(q, k, v, cu_seqlens_padded, seqlens_actual, causal, softmax_scale, mode):
    if not all(isinstance(t, torch.Tensor) for t in (q, k, v)):
        raise TypeError("delta_matching: q, k, v must be torch tensors")
    if q.dim() != 3 or k.dim() != 3 or v.dim() != 3:
        raise ValueError(f"delta_matching: packed q, k, v must be 3-D (T, H, D) (got {q.dim()}-D, {k.dim()}-D, {v.dim()}-D)")
    if not (q.is_cuda and k.is_cuda and v.is_cuda):
        raise ValueError(f"delta_matching: packed q, k, v must be CUDA tensors (got {q.device}, {k.device}, {v.device})")
    if not (q.device == k.device == v.device):
        raise ValueError(f"delta_matching: packed q, k, v must be on one device (got {q.device}, {k.device}, {v.device})")
    if not (q.dtype == k.dtype == v.dtype == torch.bfloat16):
        raise TypeError(f"delta_matching: packed q, k, v must be torch.bfloat16 (got {q.dtype}, {k.dtype}, {v.dtype})")
    if mode not in ("matched", "stale"):
        raise ValueError(f"delta_matching: mode must be 'matched' or 'stale' (got {mode!r})")
    T, H, D = q.shape
    Hk = k.shape[1]
    if k.shape != (T, Hk, D) or v.shape != (T, Hk, D):
        raise ValueError(f"delta_matching: k/v must be (T, Hk, D) matching q (q {tuple(q.shape)}, k {tuple(k.shape)}, v {tuple(v.shape)})")
    if H <= 0 or Hk <= 0 or H % Hk != 0:
        raise ValueError(f"delta_matching: nheads ({H}) must be a positive multiple of nheads_k ({Hk})")
    align = varlen_align(D)
    if T % align != 0:
        raise ValueError(f"delta_matching: TotalPaddedTokens ({T}) must be a multiple of ALIGN ({align}) at head_dim {D}")
    if softmax_scale is not None and abs(float(softmax_scale) - D ** -0.5) > 1e-12:
        raise NotImplementedError(
            f"delta_matching: softmax_scale is fixed at head_dim ** -0.5 = {D ** -0.5!r}; got {softmax_scale!r}"
        )
    major, minor = torch.cuda.get_device_capability(q.device)
    if (major, minor) != (9, 0):
        raise ValueError(f"delta_matching: needs a Hopper GPU (sm90); {q.device} is sm{major}{minor}")
    cu, actual, B = _check_varlen_meta(cu_seqlens_padded, seqlens_actual, T, q.device, align)
    return cu, actual, B


class _DMAttnVarlen(torch.autograd.Function):

    @staticmethod
    def forward(ctx, q, k, v, cu, actual, cu_sq, total_padded_sq, causal, mode):
        with torch.cuda.device(q.device):
            return _DMAttnVarlen._forward(ctx, q, k, v, cu, actual, cu_sq, total_padded_sq, causal, mode)

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, dO):
        with torch.cuda.device(dO.device):
            return _DMAttnVarlen._backward(ctx, dO)

    @staticmethod
    def _forward(ctx, q, k, v, cu, actual, cu_sq, total_padded_sq, causal, mode):
        from .attn import _ops_gt, _refuse_bf16dp
        from .core import _check_hd
        T, H, D = q.shape
        Hk = k.shape[1]
        B = actual.shape[0]
        if T == 0 or B == 0:
            O = torch.zeros_like(q)
            ctx.empty = True
            ctx.dims = (T, H, Hk, D, B)
            return O
        ops = _ops_gt()
        _refuse_bf16dp(ops)
        _check_hd(D, ops)
        sm = D ** -0.5
        Q8, Q8s, K8, K8s, V8, V8s, kTt, Qt = quant_qkv_varlen(
            q, k, v, cu, actual, want_qt=True, blocks=_blocks(D),
        )
        q8_v = Q8.view(1, H, T, D).permute(0, 2, 1, 3)
        k8_v = K8.view(1, Hk, T, D).permute(0, 2, 1, 3)
        v8_v = V8.view(1, Hk, T, D).permute(0, 2, 1, 3)
        O, Lse = ops.fwd_fp8_varlen(
            q8_v, k8_v, v8_v,
            Q8s.view(1, H, -1), K8s.view(1, Hk, -1), V8s.view(1, Hk, -1),
            cu, actual, sm, causal,
        )
        O = O.squeeze(0)
        Lse = Lse.squeeze(0)
        ctx.empty = False
        ctx.save_for_backward(Q8, Q8s, K8, K8s, V8, V8s, kTt, Qt, O, Lse, cu, actual, cu_sq)
        ctx.dims = (T, H, Hk, D, B)
        ctx.total_padded_sq = total_padded_sq
        ctx.causal = causal
        ctx.mode = mode
        return O

    @staticmethod
    def _backward(ctx, dO):
        from .attn import _ops_gt
        from .split import _bwd_heads_per_call, _bwd_varlen_by_head_groups
        T, H, Hk, D, B = ctx.dims
        if ctx.empty:
            dq = torch.zeros((T, H, D), device=dO.device, dtype=dO.dtype)
            dk = torch.zeros((T, Hk, D), device=dO.device, dtype=dO.dtype)
            dv = torch.zeros((T, Hk, D), device=dO.device, dtype=dO.dtype)
            return (dq, dk, dv, None, None, None, None, None, None)

        Q8, Q8s, K8, K8s, V8, V8s, kTt, Qt, O, Lse, cu, actual, cu_sq = ctx.saved_tensors
        ops = _ops_gt()
        sm = D ** -0.5
        if dO.stride(-1) != 1:
            dO = dO.contiguous()
        want_delta = ctx.mode != "matched"
        dO8, dO8s, Dlt, dOt = doq_varlen(
            dO, O if want_delta else None, cu, actual,
            blk=_blocks(D)[0], want_delta=want_delta, want_twin=True,
        )
        total_padded_sq = ctx.total_padded_sq
        if total_padded_sq is None:
            total_padded_sq = int(cu_sq[-1].item())
        n_rep = H // Hk
        hp = _bwd_heads_per_call(B, H, Hk, n_rep, T, cu_padded_sq=cu_sq, total_padded_sq=total_padded_sq)
        per_group_delta = ctx.mode == "matched" and hp < H

        dO8_v = dO8.permute(1, 0, 2)
        Q8_v = Q8.permute(1, 0, 2)
        K8_v = K8.permute(1, 0, 2)
        V8_v = V8.permute(1, 0, 2)
        Lse_c = Lse.contiguous()
        if ctx.mode == "matched" and not per_group_delta:
            Dlt = ops.delta_fp8_varlen(
                dO8_v, Q8_v, K8_v, V8_v, Lse_c, kTt, Q8s, K8s, V8s, dO8s,
                cu, actual, cu_sq, total_padded_sq, ctx.causal,
            )

        bwd_args = (
            dO8_v, Q8_v, K8_v, V8_v, Lse_c,
            None if per_group_delta else Dlt.contiguous(),
            kTt, Qt, dOt, Q8s, K8s, V8s, dO8s,
        )
        if hp >= H:
            dq, dk, dv = ops.bwd_fp8_t_varlen(
                *bwd_args, cu, actual, cu_sq, total_padded_sq, sm, ctx.causal,
            )
        else:
            dq, dk, dv = _bwd_varlen_by_head_groups(
                ops, bwd_args, cu, actual, cu_sq, total_padded_sq, sm,
                H, Hk, n_rep, T, D, hp, matched_delta=per_group_delta, causal=ctx.causal,
            )
        if Hk < H:
            dk = dk.reshape(T, Hk, n_rep, D).sum(2)
            dv = dv.reshape(T, Hk, n_rep, D).sum(2)
        return (dq, dk, dv, None, None, None, None, None, None)


def dm_attn_varlen_func(
    q, k, v, cu_seqlens_padded, seqlens_actual,
    cu_padded_sq=None, causal=True, softmax_scale=None, mode="matched",
    total_padded_sq: int | None = None,
):
    """Block-aligned packed FP8 attention on Hopper SM90 with sequence-local length and causal masks.

    Inputs:
      q: (TotalPaddedTokens, H, D) bf16 CUDA
      k, v: (TotalPaddedTokens, Hk, D) bf16 CUDA (H % Hk == 0)
      cu_seqlens_padded: int32 [B + 1], cumulative padded offsets (each sequence padded to ALIGN)
      seqlens_actual: int32 [B], valid token lengths per sequence
      cu_padded_sq: optional int64 [B + 1], cumulative squared padded lengths (computed if None)
      causal: bool (default True)
      softmax_scale: optional float (must be D ** -0.5 if given)
      mode: "matched" (in-kernel matched delta, default) or "stale" (rowsum(dO * O))
      total_padded_sq: optional host int == int(cu_padded_sq[-1]) to avoid D2H sync (CUDA-Graph safe)
    """
    cu, actual, B = _check_varlen_attn_inputs(
        q, k, v, cu_seqlens_padded, seqlens_actual, causal, softmax_scale, mode,
    )
    if cu_padded_sq is None:
        cu_sq = torch.zeros((B + 1,), device=q.device, dtype=torch.int64)
        if B > 0:
            diffs = (cu[1:] - cu[:-1]).to(torch.int64)
            cu_sq[1:] = torch.cumsum(diffs * diffs, dim=0)
    else:
        if (
            not isinstance(cu_padded_sq, torch.Tensor)
            or cu_padded_sq.dtype != torch.int64
            or cu_padded_sq.dim() != 1
            or cu_padded_sq.shape[0] != B + 1
        ):
            raise ValueError(f"delta_matching: cu_padded_sq must be 1-D int64 of length {B + 1}")
        if cu_padded_sq.device != q.device:
            raise ValueError("delta_matching: cu_padded_sq must be on the same device as inputs")
        cu_sq = cu_padded_sq.contiguous()
    return _DMAttnVarlen.apply(q, k, v, cu, actual, cu_sq, total_padded_sq, bool(causal), mode)


def dm_attn_varlen_backward(
    q, k, v, dout, cu_seqlens_padded, seqlens_actual,
    cu_padded_sq=None, causal=True, softmax_scale=None, mode="matched",
    total_padded_sq: int | None = None,
):
    """Run SM90 varlen forward + backward and return (dq, dk, dv)."""
    with torch.enable_grad():
        qr = q.detach().requires_grad_(True)
        kr = k.detach().requires_grad_(True)
        vr = v.detach().requires_grad_(True)
        out = dm_attn_varlen_func(
            qr, kr, vr, cu_seqlens_padded, seqlens_actual,
            cu_padded_sq=cu_padded_sq, causal=causal, softmax_scale=softmax_scale,
            mode=mode, total_padded_sq=total_padded_sq,
        )
        dq, dk, dv = torch.autograd.grad(out, (qr, kr, vr), grad_outputs=dout, allow_unused=False)
    return dq, dk, dv


def dm_attn_varlen_ref(
    q, k, v, cu_seqlens_padded, seqlens_actual,
    causal=True, softmax_scale=None,
):
    """Per-sequence FP32 reference for Route-A packed varlen attention (for tests & verification)."""
    if q.dim() != 3 or k.dim() != 3 or v.dim() != 3:
        raise ValueError("packed q, k, and v must be 3-D tensors")
    H, Hk = q.shape[1], k.shape[1]
    n_rep = H // Hk
    out = torch.zeros_like(q)
    sm = q.shape[-1] ** -0.5 if softmax_scale is None else softmax_scale
    for start, nvalid in zip(cu_seqlens_padded[:-1].tolist(), seqlens_actual.tolist()):
        n = int(nvalid)
        if n == 0:
            continue
        qs = q[start:start + n].transpose(0, 1).float()
        ks = k[start:start + n].transpose(0, 1).float()
        vs = v[start:start + n].transpose(0, 1).float()
        if n_rep > 1:
            ks = ks.repeat_interleave(n_rep, dim=0)
            vs = vs.repeat_interleave(n_rep, dim=0)
        scores = torch.matmul(qs, ks.transpose(-1, -2)) * sm
        if causal:
            mask = torch.ones((n, n), device=q.device, dtype=torch.bool).triu(1)
            scores = scores.masked_fill(mask, float("-inf"))
        probs = torch.softmax(scores, dim=-1)
        out[start:start + n] = torch.matmul(probs, vs).transpose(0, 1).to(q.dtype)
    return out
