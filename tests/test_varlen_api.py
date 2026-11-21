import pytest
import torch
import os
from pathlib import Path

from delta_matching import flash_attn_varlen_func


@pytest.mark.skipif(
    not torch.cuda.is_available()
    or (not os.environ.get("DM_SO") and not list(Path(__file__).parents[1].glob("delta_matching/_C*.so"))),
    reason="needs a CUDA GPU and built extension",
)
@pytest.mark.parametrize("explicit_used", [False, True])
def test_varlen_causal_ragged_is_finite(explicit_used):
    """Regression for the H200 mixed-mask NaN at Q=100, K=70."""
    q = torch.randn(100, 2, 256, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    k = torch.randn(70, 1, 256, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    v = torch.randn_like(k, requires_grad=True)
    cu_q = torch.tensor([0, 100], device="cuda", dtype=torch.int32)
    cu_k = torch.tensor([0, 70], device="cuda", dtype=torch.int32)
    kwargs = {}
    if explicit_used:
        kwargs.update(
            seqused_q=torch.tensor([90], device="cuda", dtype=torch.int32),
            seqused_k=torch.tensor([55], device="cuda", dtype=torch.int32),
        )
    out = flash_attn_varlen_func(q, k, v, cu_q, cu_k, 100, 70, causal=True, **kwargs)
    out.float().sum().backward()
    assert torch.isfinite(out).all()
    assert torch.isfinite(q.grad).all()
    assert torch.isfinite(k.grad).all()
    assert torch.isfinite(v.grad).all()


def test_varlen_options_rejected_without_cuda():
    q = torch.empty(1, 1, 128, dtype=torch.bfloat16)
    cu = torch.tensor([0, 1], dtype=torch.int32)
    with pytest.raises((ValueError, TypeError, NotImplementedError, RuntimeError)):
        flash_attn_varlen_func(q, q, q, cu, cu, 1, 1, window_size=(1, -1))


def test_varlen_seqused_requires_a_pair():
    q = torch.empty(1, 1, 128, dtype=torch.bfloat16)
    cu = torch.tensor([0, 1], dtype=torch.int32)
    used = torch.tensor([1], dtype=torch.int32)
    with pytest.raises(ValueError, match="provided together"):
        flash_attn_varlen_func(q, q, q, cu, cu, 1, 1, seqused_q=used)
