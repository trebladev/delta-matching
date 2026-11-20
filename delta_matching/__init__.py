"""delta_matching: fp8 attention for training with the Delta-Matching backward.

The forward keeps the attention core in fp8 (e4m3 Q, K, V codes with per-block scales, a faithful per-row per-32-key
P basis, PV promoted once per 128-key tile); the backward keeps all five backward matmuls in fp8 and, in the
"matched" mode, forms the softmax-gradient centering term delta from the backward's OWN realized fp8 P / dP pair
(pass 0 of the kernel), so that rowsum(dS) = 0 holds to fp32 round-off inside the kernel.  The "stale" mode is the
conventional delta = rowsum(dO * O), which violates that identity under fp8 rounding.

API (flash-attention layout, tensors (batch, seqlen, nheads, headdim), bf16, CUDA, Hopper):
    from delta_matching import dm_attn_func
    o = dm_attn_func(q, k, v, causal=True, mode="matched")     # GQA: k, v carry nheads_k with nheads % nheads_k == 0
    o = dm_attn_func(q, k, v, causal=True, mode="stale")
Constraints: self-attention (q, k, v share seqlen); head_dim 128 (seqlen % 256 == 0) or 256 (seqlen % 128 == 0); bf16;
softmax_scale = head_dim ** -0.5; mode "matched" needs causal=True.  Violations raise ValueError / TypeError /
NotImplementedError.  `dm_attn_func_bhnd` takes (batch, nheads, seqlen, headdim) and returns (batch, seqlen, nheads,
headdim) (the kernel's native output layout).  `flash_attn_func` has flash-attention 2's signature for one-line ports; it
refuses the arguments the kernel does not implement (dropout, sliding windows, softcap, alibi, returned probabilities);
`deterministic` is accepted and ignored, since every output is deterministic.
"""
from __future__ import annotations

import hashlib as _hashlib
import importlib as _importlib
import os as _os

import torch

# The sm90 entry points and submodules load on first use (PEP 562), so importing a subpackage such as
# delta_matching.sm100 does not import the sm90 producers (and with them Triton).
_LAZY = {"dm_attn_func_bhnd": ".attn", "blocks": ".core", "dm_attn_func_plain": ".core"}
_SUBMODULES = ("attn", "constants", "core", "doq", "producers", "quant", "split", "twins")


def __getattr__(name):
    if name in _LAZY:
        value = getattr(_importlib.import_module(_LAZY[name], __name__), name)
    elif name in _SUBMODULES:
        value = _importlib.import_module("." + name, __name__)
    else:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    globals()[name] = value
    return value

__version__ = "0.1.0"
KERNEL_NUMERICS = {"128": "Delta-Matching-128 v3", "256": "Delta-Matching-256 v1"}
FA_SRC_COMMIT = "0251105a2fb19d2957484b7f023cd8c115286ced"
CUTLASS_COMMIT = "7127592069c2fe01b041e174ba4345ef9b279671"
__all__ = ["dm_attn_func", "dm_attn_func_bhnd", "dm_attn_func_plain", "flash_attn_func", "flash_attn_varlen_func", "blocks", "kernel_version",
           "__version__", "KERNEL_NUMERICS"]


def dm_attn_func(q, k, v, softmax_scale=None, causal=False, mode="matched"):
    """Attention in the flash-attention layout: q (B, N, H, D), k / v (B, N, Hk, D) bf16 on CUDA -> o (B, N, H, D) bf16.
    mode: "matched" (the Delta-Matching backward; needs causal=True) or "stale" (delta = rowsum(dO * O))."""
    for name, t in (("q", q), ("k", k), ("v", v)):
        if not isinstance(t, torch.Tensor) or t.dim() != 4:
            raise ValueError(f"delta_matching: {name} must be a 4-D tensor (batch, seqlen, nheads, headdim)")
    from .attn import dm_attn_func_bhnd
    return dm_attn_func_bhnd(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), causal=causal,
                             softmax_scale=softmax_scale, delta_mode=mode)


def flash_attn_func(q, k, v, dropout_p=0.0, softmax_scale=None, causal=False, window_size=(-1, -1), softcap=0.0,
                    alibi_slopes=None, deterministic=False, return_attn_probs=False, mode="matched"):
    """flash_attn.flash_attn_func's signature for one-line ports; the arguments this kernel does not implement must be at
    their defaults.  `mode` selects the delta ("matched" or "stale")."""
    if dropout_p != 0.0:
        raise NotImplementedError("delta_matching: dropout is not implemented (dropout_p must be 0)")
    if tuple(window_size) != (-1, -1):
        raise NotImplementedError("delta_matching: sliding windows are not implemented (window_size must be (-1, -1))")
    if softcap != 0.0:
        raise NotImplementedError("delta_matching: softcap is not implemented")
    if alibi_slopes is not None:
        raise NotImplementedError("delta_matching: alibi_slopes are not implemented")
    if return_attn_probs:
        raise NotImplementedError("delta_matching: return_attn_probs is not implemented")
    return dm_attn_func(q, k, v, softmax_scale=softmax_scale, causal=causal, mode=mode)


class _VarlenForward(torch.autograd.Function):
    """Packed varlen native forward and stale backward bridge."""

    @staticmethod
    def forward(ctx, q, k, v, cu_q, cu_k, max_q, max_k, causal):
        from .varlen_quant import quantize_packed
        from .core import _ops

        p = quantize_packed(q, k, v, cu_q, cu_k, max_q, max_k)
        o, lse = _ops().fwd_fp8_varlen(
            p["q8"], p["k8"], p["v8"], p["q8s"], p["k8s"], p["v8s"],
            cu_q, cu_k, int(max_q), int(max_k), float(q.shape[-1] ** -0.5), bool(causal)
        )
        ctx.save_for_backward(q, k, v, cu_q, cu_k, p["q8"], p["k8"], p["v8"],
                              p["q8s"], p["k8s"], p["v8s"], p["q_t"], p["k_t"], lse, o)
        ctx.max_q, ctx.max_k, ctx.causal = int(max_q), int(max_k), bool(causal)
        return o

    @staticmethod
    def backward(ctx, grad_o):
        from .varlen_quant import quantize_dout_packed
        from .core import _ops
        (q, k, v, cu_q, cu_k, q8, k8, v8, q8s, k8s, v8s,
         qt, kt, lse, o) = ctx.saved_tensors
        do = quantize_dout_packed(grad_o.contiguous(), cu_q)
        # Stale mode uses the standard row reduction delta = rowsum(dO * O).
        delta = (grad_o.float() * o.float()).sum(dim=-1)
        dq, dk, dv = _ops().bwd_fp8_varlen(
            do["do8"], q8, k8, v8, lse, delta, kt, qt, do["dot"],
            q8s, k8s, v8s, do["do8s"], cu_q, cu_k,
            ctx.max_q, ctx.max_k, float(q.shape[-1] ** -0.5), ctx.causal)
        return dq, dk, dv, None, None, None, None, None


def flash_attn_varlen_func(
    q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k,
    seqused_q=None, seqused_k=None, softmax_scale=None, causal=False, qv=None,
    q_descale=None, k_descale=None, v_descale=None, window_size=(-1, -1),
    attention_chunk=0, softcap=0.0, num_splits=1, pack_gqa=None,
    deterministic=False, sm_margin=0, return_attn_probs=False,
):
    """FA3-shaped packed varlen forward entry point.

    This initial native path supports packed bf16 Q/K/V, causal or full attention,
    and GQA/MQA.  It currently requires every sequence length to be a positive
    quantization-block multiple and is forward-only.  Unsupported FA3 options are
    rejected explicitly until their kernel paths are implemented.
    """
    if not all(isinstance(t, torch.Tensor) for t in (q, k, v, cu_seqlens_q, cu_seqlens_k)):
        raise TypeError("delta_matching: varlen q, k, v and cu_seqlens must be tensors")
    if any(t is not None for t in (qv, q_descale, k_descale, v_descale)):
        raise NotImplementedError("delta_matching: qv and FP8 descale inputs are not implemented for varlen")
    if tuple(window_size) != (-1, -1):
        raise NotImplementedError("delta_matching: sliding windows are not implemented for varlen")
    if attention_chunk != 0 or softcap != 0.0 or num_splits != 1 or pack_gqa not in (None, False):
        raise NotImplementedError("delta_matching: this varlen path does not implement chunking, softcap, splits, or pack_gqa")
    if return_attn_probs:
        raise NotImplementedError("delta_matching: return_attn_probs is not implemented for varlen")
    if seqused_q is not None or seqused_k is not None or sm_margin != 0:
        raise NotImplementedError("delta_matching: seqused and sm_margin are not implemented for varlen")
    if softmax_scale is not None and abs(float(softmax_scale) - q.shape[-1] ** -0.5) > 1e-12:
        raise NotImplementedError("delta_matching: softmax_scale is fixed at head_dim ** -0.5")
    if deterministic not in (False, True):
        raise TypeError("delta_matching: deterministic must be bool")
    if q.dtype != torch.bfloat16 or k.dtype != torch.bfloat16 or v.dtype != torch.bfloat16:
        raise TypeError("delta_matching: packed varlen q, k, v must be torch.bfloat16")
    return _VarlenForward.apply(q, k, v, cu_seqlens_q, cu_seqlens_k, int(max_seqlen_q), int(max_seqlen_k), bool(causal))


def kernel_version():
    """A stamp for training logs: the package version, a content hash of the kernel sources, the pinned flash-attention
    commit and, once the library is loaded, its path and variant flag."""
    try:
        from ._build_info import BUILD_INFO      # stamped by the build: the csrc hash and flags of the compiled library
        info = {"version": __version__, "code_sha": BUILD_INFO["code_sha"], "bf16_dp": BUILD_INFO["bf16_dp"],
                "dbg": BUILD_INFO["dbg"], "fa_src_commit": FA_SRC_COMMIT, "numerics": KERNEL_NUMERICS}
        try:                                     # the LOADED library wins over the stamp (DM_SO may name a variant build)
            from . import core as _core
            info["bf16_dp"] = bool(int(_core._ops().build_bf16dp()))
            info["library"] = _core.loaded_library()
        except Exception:
            pass
        return info
    except ImportError:
        pass
    here = _os.path.dirname(_os.path.abspath(__file__))
    csrc = _os.path.join(_os.path.dirname(here), "csrc")
    h = _hashlib.sha256()
    if _os.path.isdir(csrc):
        for f in sorted(_os.listdir(csrc)):
            if f.endswith((".cpp", ".cu", ".h", ".hpp")):
                h.update(f.encode()); h.update(open(_os.path.join(csrc, f), "rb").read())
    return {"version": __version__, "code_sha": h.hexdigest()[:12] if _os.path.isdir(csrc) else "unknown",
            "fa_src_commit": FA_SRC_COMMIT, "numerics": KERNEL_NUMERICS}
