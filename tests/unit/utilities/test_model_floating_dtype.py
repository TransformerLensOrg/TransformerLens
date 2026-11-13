"""``model_floating_dtype`` reads past quantizer-style integer storage."""

import torch
from torch import nn

from transformer_lens.utilities.multi_gpu import model_floating_dtype


class _IntFirst(nn.Module):
    def __init__(self, with_float: bool):
        super().__init__()
        self.scale_idx = nn.Parameter(torch.zeros(2, dtype=torch.int64), requires_grad=False)
        if with_float:
            self.w = nn.Parameter(torch.zeros(2, dtype=torch.float16))


def test_skips_integer_parameters():
    assert model_floating_dtype(_IntFirst(with_float=True)) is torch.float16


def test_none_without_floating_parameters():
    assert model_floating_dtype(_IntFirst(with_float=False)) is None
    assert model_floating_dtype(object()) is None
