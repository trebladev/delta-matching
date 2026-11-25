"""Phase 2 tests: SM90 Route-A varlen forward kernel (D=128 & D=256, causal & full, MHA & GQA)."""
import math

import pytest
import torch

import delta_matching as dm
from delta_matching.varlen import (
    varlen_align,
    make_varlen_metadata,
    pack_varlen_sequences,
    fwd_varlen_fp8,
)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")
BF = torch.bfloat16
TOL_O = 8e-2


def _rel(a: torch.Tensor, b: torch.Tensor) -> float:
    return ((a.float() - b.float()).norm() / b.float().norm().clamp_min(1e-30)).item()


def _ref_seq_fp32(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, causal: bool) -> torch.Tensor:
    """Single-sequence FP32 reference: q (N, H, D), k/v (N, Hk, D) -> o (N, H, D)."""
    N, H, D = q.shape
    Hk = k.shape[1]
    qf = q.float().transpose(0, 1)
    kf = k.float().transpose(0, 1).repeat_interleave(H // Hk, dim=0)
    vf = v.float().transpose(0, 1).repeat_interleave(H // Hk, dim=0)
    s = (qf @ kf.transpose(-1, -2)) * (D ** -0.5)
    if causal:
        mask = torch.ones((N, N), device=q.device, dtype=torch.bool).triu(1)
        s = s.masked_fill(mask, float("-inf"))
    return (torch.softmax(s, dim=-1) @ vf).transpose(0, 1)


@pytest.mark.parametrize("D,lens", [
    # D=128 (ALIGN=128, kBlockM=192):
    # - 1, 17, 65, 127: pad to 128; the first kBlockM=192 tile overruns padded_len
    # - 128, 129: padded_len=128 / 256; 128 is a full-tile overrun into seq+1
    # - 191, 192, 193, 200, 255: first/second 192-row tile vs ALIGN=128 tails
    # - 256, 384, 400, 512: multi-tile interior TMA + tail mix
    # - 0: leading, middle, trailing empty sequences
    (128, [0, 1, 17, 65, 127, 128, 129, 191, 192, 193, 200, 255, 0, 256, 384, 400, 512, 0]),
    # D=256 (ALIGN=128, kBlockM=128):
    # - 1, 31, 65, 127, 128, 129, 255, 300, 512 + zeros
    (256, [0, 1, 31, 65, 127, 128, 129, 0, 255, 300, 512, 0]),
])
@pytest.mark.parametrize("causal", [True, False])
@pytest.mark.parametrize("H,Hk", [(4, 4), (8, 2)])
def test_varlen_fwd_matches_per_sequence_fp32_reference(D, lens, causal, H, Hk):
    dev = "cuda"
    g = torch.Generator(device="cpu").manual_seed(D * 1000 + H * 10 + int(causal))
    q_list, k_list, v_list = [], [], []
    for idx, n in enumerate(lens):
        q_list.append(torch.randn(n, H, D, generator=g).to(dev, BF))
        # Large per-sequence K bias ensures sequence isolation & K-centering are tested simultaneously
        k_list.append((torch.randn(n, Hk, D, generator=g) + (idx - 4) * 5.0).to(dev, BF))
        v_list.append(torch.randn(n, Hk, D, generator=g).to(dev, BF))

    qp, kp, vp, cu, actual, _ = pack_varlen_sequences(q_list, k_list, v_list)
    O, LSE, _ = fwd_varlen_fp8(qp, kp, vp, cu, actual, causal=causal)
    assert O.shape == qp.shape and O.dtype == BF
    assert LSE.shape == (H, qp.shape[0]) and LSE.dtype == torch.float32
    assert torch.isfinite(O).all() and torch.isfinite(LSE).all()

    for b, (start, end, n) in enumerate(zip(cu[:-1].tolist(), cu[1:].tolist(), lens)):
        padded = end - start
        if n == 0:
            assert padded == 0
            continue
        # Pad region must be strictly 0.0 in O and 0.0 in LSE
        if n < padded:
            assert torch.count_nonzero(O[start + n:end]) == 0, f"non-zero O in pad region for seq {b}"
            assert torch.count_nonzero(LSE[:, start + n:end]) == 0, f"non-zero LSE in pad region for seq {b}"
        ref_o = _ref_seq_fp32(q_list[b], k_list[b], v_list[b], causal=causal)
        err = _rel(O[start:start + n], ref_o)
        assert math.isfinite(err) and err < TOL_O, (
            f"D={D} causal={causal} H={H}/{Hk} seq={b} len={n}: rel err {err:.4f} >= {TOL_O}"
        )


@pytest.mark.parametrize("D,lens", [
    (128, [128, 256, 0, 128, 384]),
    (256, [128, 384, 0, 256, 512]),
])
@pytest.mark.parametrize("causal", [True, False])
@pytest.mark.parametrize("H,Hk", [(4, 4), (8, 2)])
def test_varlen_fwd_bitwise_matches_dense_when_aligned(D, lens, causal, H, Hk):
    """When every sequence has actual_len == padded_len (multiple of ALIGN), varlen forward
    must match isolated dense dm_attn_func BITWISE."""
    dev = "cuda"
    g = torch.Generator(device="cpu").manual_seed(D * 7 + Hk + int(causal))
    q_list = [torch.randn(n, H, D, generator=g).to(dev, BF) for n in lens]
    k_list = [torch.randn(n, Hk, D, generator=g).to(dev, BF) for n in lens]
    v_list = [torch.randn(n, Hk, D, generator=g).to(dev, BF) for n in lens]

    qp, kp, vp, cu, actual, _ = pack_varlen_sequences(q_list, k_list, v_list)
    O, _, _ = fwd_varlen_fp8(qp, kp, vp, cu, actual, causal=causal)

    for b, (start, end, n) in enumerate(zip(cu[:-1].tolist(), cu[1:].tolist(), lens)):
        if n == 0:
            continue
        ref_dense = dm.dm_attn_func(
            q_list[b].unsqueeze(0), k_list[b].unsqueeze(0), v_list[b].unsqueeze(0),
            causal=causal, mode="stale",
        ).squeeze(0)
        assert torch.equal(O[start:end], ref_dense), (
            f"D={D} causal={causal} H={H}/{Hk} seq={b} len={n}: varlen fwd differs from dense!"
        )


@pytest.mark.parametrize("D", [128, 256])
@pytest.mark.parametrize("causal", [True, False])
def test_varlen_fwd_hd128_tail_never_overwrites_next_sequence(D, causal):
    """Stress adjacent sequences whose kBlockM=192 (D=128) tiles cross ALIGN=128 into the next
    sequence: padded_len=128 (first tile [0, 192)) and longer tails (e.g. 200 pads to 256)."""
    dev = "cuda"
    H, Hk = 8, 2
    # padded_len=128 (65, 128, 100, 1) plus longer sequences whose 192-row tiles still tail
    lens = [65, 128, 100, 1, 200, 200, 255, 193]
    g = torch.Generator(device="cpu").manual_seed(42 + D)
    q_list = [torch.randn(n, H, D, generator=g).to(dev, BF) for n in lens]
    k_list = [torch.randn(n, Hk, D, generator=g).to(dev, BF) for n in lens]
    v_list = [torch.randn(n, Hk, D, generator=g).to(dev, BF) for n in lens]

    qp, kp, vp, cu, actual, _ = pack_varlen_sequences(q_list, k_list, v_list)
    O, LSE, _ = fwd_varlen_fp8(qp, kp, vp, cu, actual, causal=causal)

    # Compare each sequence against running varlen forward on that sequence ALONE (B=1) -> BITWISE!
    for b, (start, end, n) in enumerate(zip(cu[:-1].tolist(), cu[1:].tolist(), lens)):
        padded = end - start
        cu1 = torch.tensor([0, padded], device=dev, dtype=torch.int32)
        act1 = torch.tensor([n], device=dev, dtype=torch.int32)
        O1, LSE1, _ = fwd_varlen_fp8(qp[start:end], kp[start:end], vp[start:end], cu1, act1, causal=causal)
        assert torch.equal(O[start:end], O1), f"seq {b} (len={n}) O corrupted by neighbor!"
        assert torch.equal(LSE[:, start:end], LSE1), f"seq {b} (len={n}) LSE corrupted by neighbor!"


@pytest.mark.parametrize("D", [128, 256])
@pytest.mark.parametrize("causal", [True, False])
def test_varlen_fwd_zero_length_and_empty_batch(D, causal):
    """Zero-length sequences (both TotalPaddedTokens==0 and explicitly padded 0-actual sequence)."""
    dev = "cuda"
    align = varlen_align(D)
    H, Hk = 4, 2

    # All-zero batch (TotalPaddedTokens == 0)
    actual0 = torch.tensor([0, 0, 0], device=dev, dtype=torch.int32)
    cu0, actual0, _ = make_varlen_metadata(actual0, align)
    q0 = torch.empty((0, H, D), device=dev, dtype=BF)
    k0 = torch.empty((0, Hk, D), device=dev, dtype=BF)
    v0 = torch.empty((0, Hk, D), device=dev, dtype=BF)
    O0, LSE0, _ = fwd_varlen_fp8(q0, k0, v0, cu0, actual0, causal=causal)
    assert O0.shape == (0, H, D) and LSE0.shape == (H, 0)

    # Sequence with actual == 0 inside an explicitly padded slot alongside a valid sequence
    cu_pad = torch.tensor([0, align, 2 * align], device=dev, dtype=torch.int32)
    act_pad = torch.tensor([0, 37], device=dev, dtype=torch.int32)
    q = torch.randn(2 * align, H, D, device=dev, dtype=BF)
    k = torch.randn(2 * align, Hk, D, device=dev, dtype=BF)
    v = torch.randn(2 * align, Hk, D, device=dev, dtype=BF)
    O, LSE, _ = fwd_varlen_fp8(q, k, v, cu_pad, act_pad, causal=causal)
    assert torch.isfinite(O).all() and torch.isfinite(LSE).all()
    assert torch.count_nonzero(O[:align]) == 0 and torch.count_nonzero(LSE[:, :align]) == 0
    assert torch.count_nonzero(O[align + 37:]) == 0 and torch.count_nonzero(LSE[:, align + 37:]) == 0
    ref1 = _ref_seq_fp32(q[align:align + 37], k[align:align + 37], v[align:align + 37], causal=causal)
    assert _rel(O[align:align + 37], ref1) < TOL_O
