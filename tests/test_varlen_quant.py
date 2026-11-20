import pytest
import torch

from delta_matching.varlen_quant import validate_packed_inputs


def _inputs(dtype=torch.bfloat16):
    q = torch.empty(12, 4, 128, dtype=dtype)
    k = torch.empty(10, 2, 128, dtype=dtype)
    v = torch.empty_like(k)
    cq = torch.tensor([0, 4, 12], dtype=torch.int32)
    ck = torch.tensor([0, 6, 10], dtype=torch.int32)
    return q, k, v, cq, ck


def test_validate_packed_gqa_and_different_lengths():
    validate_packed_inputs(*_inputs())


@pytest.mark.parametrize("bad", ["dtype", "heads", "offsets"])
def test_validate_packed_rejects_bad_shapes(bad):
    q, k, v, cq, ck = _inputs()
    if bad == "dtype":
        v = v.float()
    elif bad == "heads":
        k = torch.empty(10, 3, 128, dtype=torch.bfloat16)
        v = torch.empty_like(k)
    else:
        cq = torch.tensor([0, 8, 7], dtype=torch.int32)
    with pytest.raises((TypeError, ValueError)):
        validate_packed_inputs(q, k, v, cq, ck)
