"""Phase 1 tests: Route-A varlen metadata, packing, K-centering, and Triton quantizers."""
import pytest
import torch

from delta_matching.core import blocks
from delta_matching import producers as _producers
from delta_matching.split import _bwd_heads_per_call
from delta_matching.twins import twin_pi_reference
from delta_matching.varlen import (
    varlen_align,
    make_varlen_metadata,
    pack_varlen_sequences,
    unpack_varlen_tensor,
    kmean_varlen,
    quant_qkv_varlen,
    doq_varlen,
    dm_attn_varlen_ref,
)

BF = torch.bfloat16


def _eq8(a: torch.Tensor, b: torch.Tensor) -> bool:
    return torch.equal(a.view(torch.uint8), b.view(torch.uint8))


def test_varlen_align_is_128_for_both_head_dims():
    assert varlen_align(128) == 128
    assert varlen_align(256) == 128


def test_varlen_metadata_uses_int64_squared_offsets():
    actual = torch.tensor([1, 257, 0], dtype=torch.int32)
    cu, got, sq = make_varlen_metadata(actual, 256)
    assert cu.dtype == torch.int32
    assert sq.dtype == torch.int64
    assert cu.tolist() == [0, 256, 768, 768]
    assert sq.tolist() == [0, 65536, 327680, 327680]
    assert torch.equal(got, actual)


def test_varlen_metadata_prevents_int32_sq_overflow():
    # 65536^2 = 4_294_967_296 > INT32_MAX (2_147_483_647)
    actual = torch.tensor([65536, 32768, 0], dtype=torch.int32)
    cu, _, sq = make_varlen_metadata(actual, 256)
    assert cu.dtype == torch.int32 and sq.dtype == torch.int64
    assert cu.tolist() == [0, 65536, 98304, 98304]
    assert sq.tolist() == [0, 65536 ** 2, 65536 ** 2 + 32768 ** 2, 65536 ** 2 + 32768 ** 2]


def test_varlen_pack_zeroes_tail_all_layouts():
    q = torch.ones(2, 2, 4, 2, dtype=torch.float32)
    k = torch.ones(2, 1, 4, 2, dtype=torch.float32)
    v = torch.ones_like(k)
    actual_in = torch.tensor([1, 3], dtype=torch.int32)
    qp, kp, vp, cu, actual, _ = pack_varlen_sequences(q, k, v, actual_in, align=4, layout="bhnd")
    assert torch.count_nonzero(qp[1:4]) == 0
    assert torch.count_nonzero(kp[1:4]) == 0
    assert torch.count_nonzero(vp[1:4]) == 0
    assert torch.count_nonzero(qp[cu[1] + 3:cu[2]]) == 0

    qp2, kp2, vp2, cu2, actual2, _ = pack_varlen_sequences(
        q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), actual_in, align=4, layout="bnhd"
    )
    assert torch.equal(qp, qp2) and torch.equal(kp, kp2) and torch.equal(vp, vp2)

    q_list = [q[0, :, :1].transpose(0, 1), q[1, :, :3].transpose(0, 1)]
    k_list = [k[0, :, :1].transpose(0, 1), k[1, :, :3].transpose(0, 1)]
    v_list = [v[0, :, :1].transpose(0, 1), v[1, :, :3].transpose(0, 1)]
    qp3, kp3, vp3, cu3, actual3, _ = pack_varlen_sequences(q_list, k_list, v_list, align=4)
    assert torch.equal(qp, qp3) and torch.equal(kp, kp3) and torch.equal(vp, vp3)
    unpacked = unpack_varlen_tensor(qp3, cu3, actual3)
    assert len(unpacked) == 2 and torch.equal(unpacked[0], q_list[0]) and torch.equal(unpacked[1], q_list[1])


def test_varlen_forward_masks_each_sequence():
    q = torch.randn(2, 1, 4, 2)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    qp, kp, vp, cu, actual, _ = pack_varlen_sequences(q, k, v, torch.tensor([2, 4], dtype=torch.int32), align=4)
    out = dm_attn_varlen_ref(qp, kp, vp, cu, actual, causal=True)
    assert torch.count_nonzero(out[2:4]) == 0
    ref = torch.nn.functional.scaled_dot_product_attention(q[0, :, :2], k[0, :, :2], v[0, :, :2], is_causal=True)
    assert torch.allclose(out[:2, 0], ref[0], atol=1e-5, rtol=1e-5)


def test_varlen_backward_zeroes_padding():
    q = torch.randn(2, 1, 4, 2, requires_grad=True)
    k = torch.randn_like(q, requires_grad=True)
    v = torch.randn_like(q, requires_grad=True)
    qp, kp, vp, cu, actual, _ = pack_varlen_sequences(q, k, v, torch.tensor([2, 0], dtype=torch.int32), align=4)
    out = dm_attn_varlen_ref(qp, kp, vp, cu, actual, causal=True)
    dout = torch.randn_like(qp)
    dq, dk, dv = torch.autograd.grad(out, (qp, kp, vp), grad_outputs=dout)
    assert torch.count_nonzero(dq[2:]) == 0
    assert torch.count_nonzero(dk[2:]) == 0
    assert torch.count_nonzero(dv[2:]) == 0


def test_backward_split_accepts_squared_varlen_workspace(monkeypatch):
    monkeypatch.setattr("delta_matching.split._BWD_GROUPS", "auto")
    monkeypatch.setattr("delta_matching.split._BWD_WS_LIMIT", 20.0)
    sq = torch.tensor([0, 16, 32], dtype=torch.int64)
    assert _bwd_heads_per_call(2, 4, 2, 2, 100, sq) == 1


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")
@pytest.mark.parametrize("D,lens", [
    (128, [128, 256, 0, 128]),
    (256, [128, 384, 0, 256]),
])
@pytest.mark.parametrize("H,Hk", [(4, 4), (8, 2)])
def test_varlen_quant_bitwise_matches_dense_isolated_aligned(D, lens, H, Hk):
    """When actual_len == padded_len (multiples of ALIGN), packed varlen quantization
    and doq match dense producers.quant_qkv_m17 / doq_m17 per isolated sequence BITWISE."""
    dev = "cuda"
    g = torch.Generator(device="cpu").manual_seed(D * 10 + H)
    bq, bk = blocks(D)
    q_list, k_list, v_list, do_list, o_list = [], [], [], [], []
    for n in lens:
        q_list.append(torch.randn(n, H, D, generator=g).to(dev, BF))
        # Add large per-sequence offset on K to strongly test per-sequence K-centering
        k_list.append((torch.randn(n, Hk, D, generator=g) + (n % 7 + 3.0)).to(dev, BF))
        v_list.append(torch.randn(n, Hk, D, generator=g).to(dev, BF))
        do_list.append(torch.randn(n, H, D, generator=g).to(dev, BF))
        o_list.append(torch.randn(n, H, D, generator=g).to(dev, BF))

    qp, kp, vp, cu, actual, _ = pack_varlen_sequences(q_list, k_list, v_list)
    dop = torch.cat(do_list, dim=0)
    op = torch.cat(o_list, dim=0)

    Q8, Q8s, K8, K8s, V8, V8s, kTt, Qt = quant_qkv_varlen(qp, kp, vp, cu, actual, want_qt=True)
    dO8, dO8s, Dlt, dOt = doq_varlen(dop, op, cu, actual, blk=bq, want_delta=True, want_twin=True)

    # Check twin_pi alignment across packed sequences
    assert _eq8(Qt, twin_pi_reference(Q8))
    assert _eq8(dOt, twin_pi_reference(dO8))

    for b, (start, end, n) in enumerate(zip(cu[:-1].tolist(), cu[1:].tolist(), lens)):
        if n == 0:
            assert start == end
            continue
        # Run dense producers on isolated sequence b
        qs = q_list[b].transpose(0, 1).unsqueeze(0).contiguous()  # (1, H, n, D)
        ks = k_list[b].transpose(0, 1).unsqueeze(0).contiguous()
        vs = v_list[b].transpose(0, 1).unsqueeze(0).contiguous()
        rQ8, rQ8s, rK8, rK8s, rV8, rV8s, rkTt, rQt = _producers.quant_qkv_m17(
            qs, ks, vs, want_qt=True, blocks=(bq, bk)
        )
        rdO8, rdO8s, rDlt, rdOt = _producers.doq_m17(
            do_list[b].unsqueeze(0).contiguous(),
            o_list[b].unsqueeze(0).contiguous(),
            H, bq, want_delta=True, want_twin=True,
        )
        assert _eq8(Q8[:, start:end], rQ8)
        assert torch.equal(Q8s[:, start // bq:end // bq], rQ8s)
        assert _eq8(Qt[:, :, start:end], rQt)
        assert _eq8(K8[:, start:end], rK8)
        assert torch.equal(K8s[:, start // bk:end // bk], rK8s)
        assert _eq8(kTt[:, :, start:end], rkTt)
        assert _eq8(V8[:, start:end], rV8)
        assert torch.equal(V8s[:, start // bk:end // bk], rV8s)
        assert _eq8(dO8[:, start:end], rdO8)
        assert torch.equal(dO8s[:, start // bq:end // bq], rdO8s)
        assert torch.equal(Dlt[:, start:end], rDlt)
        assert _eq8(dOt[:, :, start:end], rdOt)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")
@pytest.mark.parametrize("D,lens", [
    (128, [0, 1, 13, 129, 255, 0, 300, 512, 0]),
    (256, [0, 1, 31, 65, 127, 0, 193, 256, 0]),
])
@pytest.mark.parametrize("H,Hk", [(4, 4), (8, 2)])
def test_varlen_quant_unaligned_pad_zero_and_isolated_bitwise(D, lens, H, Hk):
    """Unaligned sequences with zeros and injected pad garbage:
      1. Pad regions in Q8/K8/V8/kTt/Qt/dO8/dOt/Dlt are strictly 0.0 (0x00).
      2. K-centering mean uses ONLY [0, actual) and forcing pad back to 0.0 keeps
         the tail K8s scale from being corrupted by -km.
      3. Every sequence slice matches isolated single-sequence quantization bitwise."""
    dev = "cuda"
    align = varlen_align(D)
    bq, bk = blocks(D)
    g = torch.Generator(device="cpu").manual_seed(D * 100 + Hk)

    q_list, k_list, v_list, do_list, o_list = [], [], [], [], []
    for idx, n in enumerate(lens):
        q_list.append(torch.randn(n, H, D, generator=g).to(dev, BF))
        # Large mean + tiny variance: if pad rows were centered to -km instead of 0.0,
        # tail block's K8s scale would explode by ~500x!
        k_list.append((torch.randn(n, Hk, D, generator=g) * 0.02 + (idx + 1) * 10.0).to(dev, BF))
        v_list.append(torch.randn(n, Hk, D, generator=g).to(dev, BF))
        do_list.append(torch.randn(n, H, D, generator=g).to(dev, BF))
        o_list.append(torch.randn(n, H, D, generator=g).to(dev, BF))

    qp, kp, vp, cu, actual, _ = pack_varlen_sequences(q_list, k_list, v_list)
    total = int(cu[-1])
    dop = torch.zeros((total, H, D), device=dev, dtype=BF)
    op = torch.zeros((total, H, D), device=dev, dtype=BF)
    for b, (start, n) in enumerate(zip(cu[:-1].tolist(), lens)):
        if n > 0:
            dop[start:start + n] = do_list[b]
            op[start:start + n] = o_list[b]

    # Inject NaN / extreme garbage into the pad slots of copies to prove kernels
    # strictly mask/zero the pad region before and during quantization!
    qp_dirty, kp_dirty, vp_dirty = qp.clone(), kp.clone(), vp.clone()
    dop_dirty, op_dirty = dop.clone(), op.clone()
    for start, end, n in zip(cu[:-1].tolist(), cu[1:].tolist(), lens):
        if n < (end - start):
            qp_dirty[start + n:end] = float("nan")
            kp_dirty[start + n:end] = float("nan")
            vp_dirty[start + n:end] = 1e4
            dop_dirty[start + n:end] = float("nan")
            op_dirty[start + n:end] = float("nan")

    Q8, Q8s, K8, K8s, V8, V8s, kTt, Qt = quant_qkv_varlen(qp_dirty, kp_dirty, vp_dirty, cu, actual, want_qt=True)
    dO8, dO8s, Dlt, dOt = doq_varlen(dop_dirty, op_dirty, cu, actual, blk=bq, want_delta=True, want_twin=True)
    km = kmean_varlen(kp_dirty, cu, actual)

    assert torch.isfinite(km).all()
    assert torch.isfinite(Q8s).all() and torch.isfinite(K8s).all() and torch.isfinite(V8s).all()
    assert torch.isfinite(dO8s).all() and torch.isfinite(Dlt).all()
    assert _eq8(Qt, twin_pi_reference(Q8))
    assert _eq8(dOt, twin_pi_reference(dO8))

    for b, (start, end, n) in enumerate(zip(cu[:-1].tolist(), cu[1:].tolist(), lens)):
        padded = end - start
        if n == 0:
            assert torch.equal(km[b], torch.zeros_like(km[b]))
            assert padded == 0
            continue
        assert padded % align == 0 and n <= padded

        # 1. Pad region must be strictly 0x00 in all codes and 0.0 in Dlt
        if n < padded:
            assert torch.count_nonzero(Q8[:, start + n:end].view(torch.uint8)) == 0
            assert torch.count_nonzero(K8[:, start + n:end].view(torch.uint8)) == 0
            assert torch.count_nonzero(V8[:, start + n:end].view(torch.uint8)) == 0
            assert torch.count_nonzero(kTt[:, :, start + n:end].view(torch.uint8)) == 0
            assert torch.count_nonzero(dO8[:, start + n:end].view(torch.uint8)) == 0
            assert torch.count_nonzero(Dlt[:, start + n:end]) == 0

            # 2. Tail block K scale must NOT be corrupted by -km (true scale ~ 0.1/448 << 10/448)
            tail_blk = (start + n - 1) // bk
            assert float(K8s[:, tail_blk].max()) < 0.5 / 448.0, (
                f"tail K8s scale {float(K8s[:, tail_blk].max())} looks corrupted by unzeroed pad -km!"
            )

            # Completely padded blocks must have the floor scale 1e-30 / 448.0
            first_empty_blk = ((n + bk - 1) // bk)
            for blk_i in range(first_empty_blk, padded // bk):
                assert torch.equal(
                    K8s[:, start // bk + blk_i],
                    torch.full((Hk,), 1e-30 / 448.0, device=dev, dtype=torch.float32),
                )

        # 3. Bitwise comparison against isolated single-sequence quantization
        cu1 = torch.tensor([0, padded], device=dev, dtype=torch.int32)
        act1 = torch.tensor([n], device=dev, dtype=torch.int32)
        iQ8, iQ8s, iK8, iK8s, iV8, iV8s, ikTt, iQt = quant_qkv_varlen(
            qp[start:end], kp[start:end], vp[start:end], cu1, act1, want_qt=True
        )
        idO8, idO8s, iDlt, idOt = doq_varlen(
            dop[start:end], op[start:end], cu1, act1, blk=bq, want_delta=True, want_twin=True
        )
        assert _eq8(Q8[:, start:end], iQ8) and torch.equal(Q8s[:, start // bq:end // bq], iQ8s)
        assert _eq8(Qt[:, :, start:end], iQt)
        assert _eq8(K8[:, start:end], iK8) and torch.equal(K8s[:, start // bk:end // bk], iK8s)
        assert _eq8(kTt[:, :, start:end], ikTt)
        assert _eq8(V8[:, start:end], iV8) and torch.equal(V8s[:, start // bk:end // bk], iV8s)
        assert _eq8(dO8[:, start:end], idO8) and torch.equal(dO8s[:, start // bq:end // bq], idO8s)
        assert torch.equal(Dlt[:, start:end], iDlt) and _eq8(dOt[:, :, start:end], idOt)

        # Also check Q, V, dO, Dlt against dense producers on the zero-padded isolated sequence!
        qs = qp[start:end].transpose(0, 1).unsqueeze(0).contiguous()
        vs = vp[start:end].transpose(0, 1).unsqueeze(0).contiguous()
        dQ8, dQ8s, dQt = _producers.quant_q_m17(qs, blk=bq, want_twin=True)
        _, _, _, dV8, dV8s = _producers.quant_kv_m17(vs, vs, blk_k=bk)
        ddO8, ddO8s, dDlt, ddOt = _producers.doq_m17(
            dop[start:end].unsqueeze(0).contiguous(),
            op[start:end].unsqueeze(0).contiguous(),
            H, bq, want_delta=True, want_twin=True,
        )
        assert _eq8(Q8[:, start:end], dQ8) and torch.equal(Q8s[:, start // bq:end // bq], dQ8s)
        assert _eq8(Qt[:, :, start:end], dQt)
        assert _eq8(V8[:, start:end], dV8) and torch.equal(V8s[:, start // bk:end // bk], dV8s)
        assert _eq8(dO8[:, start:end], ddO8) and torch.equal(dO8s[:, start // bq:end // bq], ddO8s)
        assert torch.equal(Dlt[:, start:end], dDlt) and _eq8(dOt[:, :, start:end], ddOt)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")
@pytest.mark.parametrize("D", [128, 256])
def test_varlen_quant_zero_length_and_empty_batch(D):
    """Edge cases:
      - All sequences have seqlens_actual == 0 (TotalPaddedTokens == 0).
      - A sequence with seqlens_actual == 0 but explicitly allocated padded_len == ALIGN."""
    dev = "cuda"
    align = varlen_align(D)
    bq, bk = blocks(D)
    H, Hk = 4, 2

    # Case 1: TotalPaddedTokens == 0
    actual0 = torch.tensor([0, 0], device=dev, dtype=torch.int32)
    cu0, actual0, sq0 = make_varlen_metadata(actual0, align)
    assert int(cu0[-1]) == 0 and int(sq0[-1]) == 0
    q0 = torch.empty((0, H, D), device=dev, dtype=BF)
    k0 = torch.empty((0, Hk, D), device=dev, dtype=BF)
    v0 = torch.empty((0, Hk, D), device=dev, dtype=BF)
    Q8, Q8s, K8, K8s, V8, V8s, kTt, Qt = quant_qkv_varlen(q0, k0, v0, cu0, actual0)
    dO8, dO8s, Dlt, dOt = doq_varlen(q0, q0, cu0, actual0, blk=bq, want_delta=True)
    assert Q8.shape == (H, 0, D) and Q8s.shape == (H, 0) and Qt.shape == (H, D, 0)
    assert K8.shape == (Hk, 0, D) and K8s.shape == (Hk, 0) and kTt.shape == (Hk, D, 0)
    assert dO8.shape == (H, 0, D) and Dlt.shape == (H, 0) and dOt.shape == (H, D, 0)

    # Case 2: actual == 0 with non-zero padded slot [0, align) containing NaN
    cu_pad = torch.tensor([0, align], device=dev, dtype=torch.int32)
    act_pad = torch.tensor([0], device=dev, dtype=torch.int32)
    q_nan = torch.full((align, H, D), float("nan"), device=dev, dtype=BF)
    k_nan = torch.full((align, Hk, D), float("nan"), device=dev, dtype=BF)
    v_nan = torch.full((align, Hk, D), float("nan"), device=dev, dtype=BF)
    Q8, Q8s, K8, K8s, V8, V8s, kTt, Qt = quant_qkv_varlen(q_nan, k_nan, v_nan, cu_pad, act_pad)
    dO8, dO8s, Dlt, dOt = doq_varlen(q_nan, q_nan, cu_pad, act_pad, blk=bq, want_delta=True)
    km = kmean_varlen(k_nan, cu_pad, act_pad)
    assert torch.equal(km, torch.zeros_like(km))
    for t in (Q8, K8, V8, kTt, Qt, dO8, dOt):
        assert torch.count_nonzero(t.view(torch.uint8)) == 0
    assert torch.count_nonzero(Dlt) == 0
    assert torch.isfinite(Q8s).all() and torch.isfinite(K8s).all() and torch.isfinite(V8s).all() and torch.isfinite(dO8s).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")
@pytest.mark.parametrize("D", [128, 256])
def test_varlen_quant_fused_twin_mode_matches_twin_pi(D, monkeypatch):
    """DM_FUSED_TWIN=1 (TWIN_MODE=2) produces bitwise identical Q/dO codes, scales, and twins."""
    dev = "cuda"
    H, Hk = 4, 2
    bq, _ = blocks(D)
    lens = [17, 0, varlen_align(D) + 5]
    q_list = [torch.randn(n, H, D, device=dev, dtype=BF) for n in lens]
    k_list = [torch.randn(n, Hk, D, device=dev, dtype=BF) for n in lens]
    v_list = [torch.randn(n, Hk, D, device=dev, dtype=BF) for n in lens]
    qp, kp, vp, cu, actual, _ = pack_varlen_sequences(q_list, k_list, v_list)
    dop = torch.randn_like(qp)
    op = torch.randn_like(qp)

    monkeypatch.setattr(_producers, "FUSED_TWIN", False)
    rQ8, rQ8s, _, _, _, _, _, rQt = quant_qkv_varlen(qp, kp, vp, cu, actual, want_qt=True)
    rdO8, rdO8s, rDlt, rdOt = doq_varlen(dop, op, cu, actual, blk=bq, want_delta=True, want_twin=True)

    monkeypatch.setattr(_producers, "FUSED_TWIN", True)
    monkeypatch.setattr(_producers, "TWIN_MODE", 2)
    fQ8, fQ8s, _, _, _, _, _, fQt = quant_qkv_varlen(qp, kp, vp, cu, actual, want_qt=True)
    fdO8, fdO8s, fDlt, fdOt = doq_varlen(dop, op, cu, actual, blk=bq, want_delta=True, want_twin=True)

    assert _eq8(rQ8, fQ8) and torch.equal(rQ8s, fQ8s) and _eq8(rQt, fQt)
    assert _eq8(rdO8, fdO8) and torch.equal(rdO8s, fdO8s) and torch.equal(rDlt, fDlt) and _eq8(rdOt, fdOt)
