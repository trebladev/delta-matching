import pytest
import torch

from delta_matching import (
    blocks,
    dm_attn_func,
    dm_attn_varlen_func,
    dm_attn_varlen_backward,
    make_varlen_metadata,
    pack_varlen_sequences,
    unpack_varlen_tensor,
    quant_qkv_varlen,
    doq_varlen,
    varlen_align,
)
from delta_matching.core import _ops
from delta_matching.varlen import dm_attn_varlen_ref

BF = torch.bfloat16
pytestmark = pytest.mark.skipif(
    not (torch.cuda.is_available() and torch.cuda.get_device_capability() == (9, 0)),
    reason="needs a Hopper (SM90) CUDA device",
)


# Relative-norm bounds against fp32 attention (matching tests/test_dm_attn.py):
# e4m3 quantization floor on Gaussian inputs is O/dV ~0.045, dQ/dK ~0.065; 2x gives clean margin.
TOL = {"O": 8e-2, "dV": 8e-2, "dQ": 1.2e-1, "dK": 1.2e-1}


def _rel_l2(x: torch.Tensor, ref: torch.Tensor) -> float:
    xf = x.detach().float()
    rf = ref.detach().float()
    denom = float(rf.norm())
    if denom == 0.0:
        return float((xf - rf).abs().max())
    return float((xf - rf).norm() / denom)


def _ref_varlen_fwd_bwd(q, k, v, dout, cu, actual, causal):
    qr = q.detach().float().requires_grad_(True)
    kr = k.detach().float().requires_grad_(True)
    vr = v.detach().float().requires_grad_(True)
    H, Hk = q.shape[1], k.shape[1]
    n_rep = H // Hk
    sm = q.shape[-1] ** -0.5
    outs = []
    for start, end, nvalid in zip(cu[:-1].tolist(), cu[1:].tolist(), actual.tolist()):
        n = int(nvalid)
        padded = end - start
        if n > 0:
            qs = qr[start:start + n].transpose(0, 1)
            ks = kr[start:start + n].transpose(0, 1)
            vs = vr[start:start + n].transpose(0, 1)
            if n_rep > 1:
                ks = ks.repeat_interleave(n_rep, dim=0)
                vs = vs.repeat_interleave(n_rep, dim=0)
            scores = torch.matmul(qs, ks.transpose(-1, -2)) * sm
            if causal:
                mask = torch.ones((n, n), device=q.device, dtype=torch.bool).triu(1)
                scores = scores.masked_fill(mask, float("-inf"))
            probs = torch.softmax(scores, dim=-1)
            o_valid = torch.matmul(probs, vs).transpose(0, 1)
            outs.append(o_valid)
        if padded > n:
            outs.append(torch.zeros((padded - n, H, q.shape[2]), device=q.device, dtype=torch.float32))
    out = torch.cat(outs, dim=0) if outs else torch.zeros_like(qr)
    dq, dk, dv = torch.autograd.grad(out, (qr, kr, vr), grad_outputs=dout.float())
    return out, dq, dk, dv


@pytest.mark.parametrize("D,lens", [
    (128, [128, 256, 0, 128]),
    (256, [128, 384, 0, 256]),
])
@pytest.mark.parametrize("H,Hk", [(4, 4), (8, 2)])
@pytest.mark.parametrize("causal,mode", [(True, "matched"), (True, "stale"), (False, "stale")])
def test_varlen_bwd_bitwise_matches_dense_aligned(D, lens, H, Hk, causal, mode):
    """When actual_len == padded_len (multiples of ALIGN), packed varlen forward+backward
    must match isolated dense dm_attn_func BITWISE on O, dQ, dK, dV (even with 0-len seqs)."""
    dev = "cuda"
    g = torch.Generator(device="cpu").manual_seed(D * 100 + H * 10 + int(causal) * 3 + (mode == "matched"))
    q_list, k_list, v_list, do_list = [], [], [], []
    for i, n in enumerate(lens):
        q_list.append(torch.randn(n, H, D, generator=g).to(dev, BF))
        k_list.append((torch.randn(n, Hk, D, generator=g) + (i + 1) * 1.5).to(dev, BF))
        v_list.append(torch.randn(n, Hk, D, generator=g).to(dev, BF))
        do_list.append(torch.randn(n, H, D, generator=g).to(dev, BF))

    qp, kp, vp, cu, actual, sq = pack_varlen_sequences(q_list, k_list, v_list)
    dop = torch.cat(do_list, dim=0)
    qp_v = qp.detach().clone().requires_grad_(True)
    kp_v = kp.detach().clone().requires_grad_(True)
    vp_v = vp.detach().clone().requires_grad_(True)
    o_v = dm_attn_varlen_func(qp_v, kp_v, vp_v, cu, actual, cu_padded_sq=sq, causal=causal, mode=mode)
    dq_v, dk_v, dv_v = torch.autograd.grad(o_v, (qp_v, kp_v, vp_v), grad_outputs=dop)

    o_slices = unpack_varlen_tensor(o_v, cu, actual)
    dq_slices = unpack_varlen_tensor(dq_v, cu, actual)
    dk_slices = unpack_varlen_tensor(dk_v, cu, actual)
    dv_slices = unpack_varlen_tensor(dv_v, cu, actual)

    for b, n in enumerate(lens):
        if n == 0:
            assert o_slices[b].shape[0] == 0
            continue
        qb = q_list[b].unsqueeze(0).detach().clone().requires_grad_(True)
        kb = k_list[b].unsqueeze(0).detach().clone().requires_grad_(True)
        vb = v_list[b].unsqueeze(0).detach().clone().requires_grad_(True)
        ob = dm_attn_func(qb, kb, vb, causal=causal, mode=mode)
        dqb, dkb, dvb = torch.autograd.grad(ob, (qb, kb, vb), grad_outputs=do_list[b].unsqueeze(0))
        assert torch.equal(o_slices[b], ob.squeeze(0)), f"O mismatch seq {b} (n={n})"
        assert torch.equal(dq_slices[b], dqb.squeeze(0)), f"dQ mismatch seq {b} (n={n})"
        assert torch.equal(dk_slices[b], dkb.squeeze(0)), f"dK mismatch seq {b} (n={n})"
        assert torch.equal(dv_slices[b], dvb.squeeze(0)), f"dV mismatch seq {b} (n={n})"


@pytest.mark.parametrize("D,lens", [
    (128, [1, 13, 64, 65, 127, 128, 129, 255, 256, 300, 513]),
    (128, [0, 0, 17, 0, 257, 0]),
    (256, [1, 17, 32, 33, 63, 64, 65, 127, 128, 130, 300]),
    (256, [0, 31, 0, 0, 193, 0]),
])
@pytest.mark.parametrize("H,Hk", [(4, 4), (8, 2)])
@pytest.mark.parametrize("causal,mode", [
    (True, "matched"),
    (True, "stale"),
    (False, "matched"),
    (False, "stale"),
])
def test_varlen_bwd_accuracy_unaligned_and_pad_zeros(D, lens, H, Hk, causal, mode):
    """Unaligned + extreme lengths + consecutive zeros: O/dQ/dK/dV match FP32 ref accurately,
    and every pad slot is strictly 0.0 with zero NaN/Inf."""
    dev = "cuda"
    g = torch.Generator(device="cpu").manual_seed(D * 37 + len(lens) * 13 + H * 5 + int(causal) * 2 + (mode == "matched"))
    q_list = [torch.randn(n, H, D, generator=g).to(dev, BF) for n in lens]
    k_list = [(torch.randn(n, Hk, D, generator=g) + (i % 5 - 2.0)).to(dev, BF) for i, n in enumerate(lens)]
    v_list = [torch.randn(n, Hk, D, generator=g).to(dev, BF) for n in lens]
    do_list = [torch.randn(n, H, D, generator=g).to(dev, BF) for n in lens]

    qp, kp, vp, cu, actual, sq = pack_varlen_sequences(q_list, k_list, v_list)
    dop = torch.zeros_like(qp)
    for b, (start, n) in enumerate(zip(cu[:-1].tolist(), lens)):
        if n > 0:
            dop[start:start + n] = do_list[b]

    qp_v = qp.detach().clone().requires_grad_(True)
    kp_v = kp.detach().clone().requires_grad_(True)
    vp_v = vp.detach().clone().requires_grad_(True)
    o_v = dm_attn_varlen_func(qp_v, kp_v, vp_v, cu, actual, cu_padded_sq=sq, causal=causal, mode=mode)
    dq_v, dk_v, dv_v = torch.autograd.grad(o_v, (qp_v, kp_v, vp_v), grad_outputs=dop)

    for name, t in (("O", o_v), ("dQ", dq_v), ("dK", dk_v), ("dV", dv_v)):
        assert torch.isfinite(t).all(), f"{name} has NaN/Inf"

    o_ref, dq_ref, dk_ref, dv_ref = _ref_varlen_fwd_bwd(qp, kp, vp, dop, cu, actual, causal)

    for b, (start, end, n) in enumerate(zip(cu[:-1].tolist(), cu[1:].tolist(), lens)):
        # Strict zero pad slots
        if end > start + n:
            assert torch.count_nonzero(o_v[start + n:end]) == 0, f"O pad nonzero seq {b}"
            assert torch.count_nonzero(dq_v[start + n:end]) == 0, f"dQ pad nonzero seq {b}"
            assert torch.count_nonzero(dk_v[start + n:end]) == 0, f"dK pad nonzero seq {b}"
            assert torch.count_nonzero(dv_v[start + n:end]) == 0, f"dV pad nonzero seq {b}"
        if n == 0:
            continue
        eo = _rel_l2(o_v[start:start + n], o_ref[start:start + n])
        edq = _rel_l2(dq_v[start:start + n], dq_ref[start:start + n])
        edk = _rel_l2(dk_v[start:start + n], dk_ref[start:start + n])
        edv = _rel_l2(dv_v[start:start + n], dv_ref[start:start + n])
        assert eo < TOL["O"], f"O rel L2 too high ({eo:.4f}) seq {b} n={n}"
        assert edq < TOL["dQ"], f"dQ rel L2 too high ({edq:.4f}) seq {b} n={n}"
        assert edk < TOL["dK"], f"dK rel L2 too high ({edk:.4f}) seq {b} n={n}"
        assert edv < TOL["dV"], f"dV rel L2 too high ({edv:.4f}) seq {b} n={n}"


@pytest.mark.parametrize("D", [128, 256])
@pytest.mark.parametrize("causal", [True, False])
def test_varlen_bwd_sequence_isolation_and_pad_poison_immunity(D, causal):
    """Perturbing seq 1 by +1000 or poisoning tail pad slots with extreme values/NaN
    must leave other sequences' O/dQ/dK/dV BITWISE identical."""
    dev = "cuda"
    H, Hk = 4, 2
    lens = [137, 291, 73]
    g = torch.Generator(device="cpu").manual_seed(D * 999 + int(causal))
    q_list = [torch.randn(n, H, D, generator=g).to(dev, BF) for n in lens]
    k_list = [torch.randn(n, Hk, D, generator=g).to(dev, BF) for n in lens]
    v_list = [torch.randn(n, Hk, D, generator=g).to(dev, BF) for n in lens]
    do_list = [torch.randn(n, H, D, generator=g).to(dev, BF) for n in lens]

    qp, kp, vp, cu, actual, sq = pack_varlen_sequences(q_list, k_list, v_list)
    dop = torch.zeros_like(qp)
    for b, (start, n) in enumerate(zip(cu[:-1].tolist(), lens)):
        dop[start:start + n] = do_list[b]

    def run(q_, k_, v_, do_):
        q1 = q_.detach().clone().requires_grad_(True)
        k1 = k_.detach().clone().requires_grad_(True)
        v1 = v_.detach().clone().requires_grad_(True)
        o1 = dm_attn_varlen_func(q1, k1, v1, cu, actual, cu_padded_sq=sq, causal=causal, mode="matched")
        dq1, dk1, dv1 = torch.autograd.grad(o1, (q1, k1, v1), grad_outputs=do_)
        return o1, dq1, dk1, dv1

    o_clean, dq_clean, dk_clean, dv_clean = run(qp, kp, vp, dop)

    # 1) Perturb sequence 1 massively and poison ALL pad slots with NaN / 1e4
    qp2, kp2, vp2, dop2 = qp.clone(), kp.clone(), vp.clone(), dop.clone()
    s1, e1 = int(cu[1]), int(cu[2])
    qp2[s1:s1 + lens[1]] += 1000.0
    kp2[s1:s1 + lens[1]] -= 1000.0
    vp2[s1:s1 + lens[1]] += 500.0
    dop2[s1:s1 + lens[1]] -= 500.0
    for start, end, n in zip(cu[:-1].tolist(), cu[1:].tolist(), lens):
        if end > start + n:
            qp2[start + n:end] = float("nan")
            kp2[start + n:end] = 1e4
            vp2[start + n:end] = -1e4
            dop2[start + n:end] = float("nan")

    o_pert, dq_pert, dk_pert, dv_pert = run(qp2, kp2, vp2, dop2)

    for b in (0, 2):
        st, en = int(cu[b]), int(cu[b + 1])
        assert torch.equal(o_clean[st:en], o_pert[st:en]), f"O leaked into seq {b}"
        assert torch.equal(dq_clean[st:en], dq_pert[st:en]), f"dQ leaked into seq {b}"
        assert torch.equal(dk_clean[st:en], dk_pert[st:en]), f"dK leaked into seq {b}"
        assert torch.equal(dv_clean[st:en], dv_pert[st:en]), f"dV leaked into seq {b}"


@pytest.mark.parametrize("D", [128, 256])
@pytest.mark.parametrize("mode", ["matched", "stale"])
@pytest.mark.parametrize("group_mode,streams", [("kv", 1), ("head", 1), ("head", 2)])
def test_varlen_bwd_head_group_split_bitwise(monkeypatch, D, mode, group_mode, streams):
    """Splitting varlen backward across KV groups or individual heads (with 1 or 2 streams)
    must be BITWISE identical to unsplit execution."""
    dev = "cuda"
    H, Hk = 8, 2
    lens = [193, 0, 311, 65]
    g = torch.Generator(device="cpu").manual_seed(D * 53 + (mode == "matched"))
    q_list = [torch.randn(n, H, D, generator=g).to(dev, BF) for n in lens]
    k_list = [torch.randn(n, Hk, D, generator=g).to(dev, BF) for n in lens]
    v_list = [torch.randn(n, Hk, D, generator=g).to(dev, BF) for n in lens]
    do_list = [torch.randn(n, H, D, generator=g).to(dev, BF) for n in lens]

    qp, kp, vp, cu, actual, sq = pack_varlen_sequences(q_list, k_list, v_list)
    dop = torch.zeros_like(qp)
    for b, (start, n) in enumerate(zip(cu[:-1].tolist(), lens)):
        if n > 0:
            dop[start:start + n] = do_list[b]

    monkeypatch.setattr("delta_matching.split._BWD_GROUPS", "off")
    monkeypatch.setattr("delta_matching.split._BWD_STREAMS", 1)
    dq0, dk0, dv0 = dm_attn_varlen_backward(qp, kp, vp, dop, cu, actual, cu_padded_sq=sq, causal=True, mode=mode)

    monkeypatch.setattr("delta_matching.split._BWD_GROUPS", group_mode)
    monkeypatch.setattr("delta_matching.split._BWD_STREAMS", streams)
    dq1, dk1, dv1 = dm_attn_varlen_backward(qp, kp, vp, dop, cu, actual, cu_padded_sq=sq, causal=True, mode=mode)

    assert torch.equal(dq0, dq1), f"dQ split mismatch ({group_mode}, streams={streams})"
    assert torch.equal(dk0, dk1), f"dK split mismatch ({group_mode}, streams={streams})"
    assert torch.equal(dv0, dv1), f"dV split mismatch ({group_mode}, streams={streams})"


@pytest.mark.parametrize("D", [128, 256])
def test_varlen_bwd_long_multi_window_and_determinism(D):
    """Long unaligned sequences spanning > 2048 tokens exercise Pass-1 dV promotion windows
    and Pass-2 dQ promotion windows, plus bitwise determinism across runs."""
    dev = "cuda"
    H, Hk = 4, 2
    lens = [2113, 777, 133]
    g = torch.Generator(device="cpu").manual_seed(D * 777)
    q_list = [torch.randn(n, H, D, generator=g).to(dev, BF) for n in lens]
    k_list = [torch.randn(n, Hk, D, generator=g).to(dev, BF) for n in lens]
    v_list = [torch.randn(n, Hk, D, generator=g).to(dev, BF) for n in lens]
    do_list = [torch.randn(n, H, D, generator=g).to(dev, BF) for n in lens]

    qp, kp, vp, cu, actual, sq = pack_varlen_sequences(q_list, k_list, v_list)
    dop = torch.zeros_like(qp)
    for b, (start, n) in enumerate(zip(cu[:-1].tolist(), lens)):
        dop[start:start + n] = do_list[b]
    total_sq = int(sq[-1].item())

    dq1, dk1, dv1 = dm_attn_varlen_backward(
        qp, kp, vp, dop, cu, actual, cu_padded_sq=sq, causal=True, mode="matched", total_padded_sq=total_sq,
    )
    dq2, dk2, dv2 = dm_attn_varlen_backward(
        qp, kp, vp, dop, cu, actual, cu_padded_sq=sq, causal=True, mode="matched", total_padded_sq=total_sq,
    )
    assert torch.equal(dq1, dq2) and torch.equal(dk1, dk2) and torch.equal(dv1, dv2)

    o_ref, dq_ref, dk_ref, dv_ref = _ref_varlen_fwd_bwd(qp, kp, vp, dop, cu, actual, causal=True)
    for b, (start, end, n) in enumerate(zip(cu[:-1].tolist(), cu[1:].tolist(), lens)):
        assert torch.count_nonzero(dq1[start + n:end]) == 0
        assert torch.count_nonzero(dk1[start + n:end]) == 0
        assert torch.count_nonzero(dv1[start + n:end]) == 0
        assert _rel_l2(dq1[start:start + n], dq_ref[start:start + n]) < TOL["dQ"]
        assert _rel_l2(dk1[start:start + n], dk_ref[start:start + n]) < TOL["dK"]
        assert _rel_l2(dv1[start:start + n], dv_ref[start:start + n]) < TOL["dV"]


@pytest.mark.parametrize("D", [128, 256])
def test_varlen_bwd_edge_cases_and_cudagraph(D):
    """All-zero batch, expanded stride-0 grad_output (out.sum().backward()), and CUDA Graph capture/replay."""
    dev = "cuda"
    H, Hk = 4, 4

    # 1) All-zero batch (T == 0)
    cu0, actual0, sq0 = make_varlen_metadata(torch.tensor([0, 0, 0], device=dev, dtype=torch.int32), varlen_align(D))
    q0 = torch.zeros((0, H, D), device=dev, dtype=BF, requires_grad=True)
    k0 = torch.zeros((0, Hk, D), device=dev, dtype=BF, requires_grad=True)
    v0 = torch.zeros((0, Hk, D), device=dev, dtype=BF, requires_grad=True)
    o0 = dm_attn_varlen_func(q0, k0, v0, cu0, actual0, cu_padded_sq=sq0, causal=True, total_padded_sq=0)
    assert o0.shape == (0, H, D)
    o0.sum().backward()
    assert q0.grad.shape == (0, H, D) and k0.grad.shape == (0, Hk, D) and v0.grad.shape == (0, Hk, D)

    # 2) Expanded stride-0 dO via o.sum().backward()
    g = torch.Generator(device="cpu").manual_seed(D + 1234)
    lens = [97, 205]
    q_list = [torch.randn(n, H, D, generator=g).to(dev, BF) for n in lens]
    k_list = [torch.randn(n, Hk, D, generator=g).to(dev, BF) for n in lens]
    v_list = [torch.randn(n, Hk, D, generator=g).to(dev, BF) for n in lens]
    qp, kp, vp, cu, actual, sq = pack_varlen_sequences(q_list, k_list, v_list)
    total_sq = int(sq[-1].item())
    q_a = qp.clone().requires_grad_(True)
    k_a = kp.clone().requires_grad_(True)
    v_a = vp.clone().requires_grad_(True)
    o_a = dm_attn_varlen_func(q_a, k_a, v_a, cu, actual, cu_padded_sq=sq, causal=True, total_padded_sq=total_sq)
    o_a.sum().backward()
    assert torch.isfinite(q_a.grad).all() and torch.isfinite(k_a.grad).all() and torch.isfinite(v_a.grad).all()

    # 3) CUDA Graph capture & replay with explicit total_padded_sq (zero D2H sync)
    q_s = qp.clone().requires_grad_(True)
    k_s = kp.clone().requires_grad_(True)
    v_s = vp.clone().requires_grad_(True)
    do_s = torch.randn_like(qp)

    # Warmup on side stream before capture
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(2):
            o_w = dm_attn_varlen_func(q_s, k_s, v_s, cu, actual, cu_padded_sq=sq, causal=True, total_padded_sq=total_sq)
            torch.autograd.grad(o_w, (q_s, k_s, v_s), grad_outputs=do_s)
    torch.cuda.current_stream().wait_stream(s)

    o_eager = dm_attn_varlen_func(q_s, k_s, v_s, cu, actual, cu_padded_sq=sq, causal=True, total_padded_sq=total_sq)
    dq_e, dk_e, dv_e = torch.autograd.grad(o_eager, (q_s, k_s, v_s), grad_outputs=do_s)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        o_g = dm_attn_varlen_func(q_s, k_s, v_s, cu, actual, cu_padded_sq=sq, causal=True, total_padded_sq=total_sq)
        dq_g, dk_g, dv_g = torch.autograd.grad(o_g, (q_s, k_s, v_s), grad_outputs=do_s)
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(o_g, o_eager)
    assert torch.equal(dq_g, dq_e)
    assert torch.equal(dk_g, dk_e)
    assert torch.equal(dv_g, dv_e)
