import pytest
import torch

from delta_matching import flash_attn_varlen_func


def test_varlen_options_rejected_without_cuda():
    q = torch.empty(1, 1, 128, dtype=torch.bfloat16)
    cu = torch.tensor([0, 1], dtype=torch.int32)
    with pytest.raises((ValueError, TypeError, NotImplementedError, RuntimeError)):
        flash_attn_varlen_func(q, q, q, cu, cu, 1, 1, window_size=(1, -1))
