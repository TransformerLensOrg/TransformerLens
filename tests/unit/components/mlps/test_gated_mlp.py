from typing import Any, Dict

import pytest
import torch
import torch.nn as nn

from transformer_lens.components import GatedMLP, LayerNorm
from transformer_lens.utils import solu


@pytest.fixture
def cfg() -> Dict[str, Any]:
    return {
        "n_layers": 12,
        "n_ctx": 1024,
        "d_head": 64,
        "d_model": 128,
        "d_mlp": 256,
        "dtype": torch.float32,
        "act_fn": "solu_ln",
        "normalization_type": "LN",
        "load_in_4bit": False,
    }


def test_initialization(cfg: Dict[str, Any]):
    model = GatedMLP(cfg)
    assert isinstance(model.W_in, nn.Parameter)
    assert isinstance(model.W_gate, nn.Parameter)
    assert isinstance(model.W_out, nn.Parameter)
    assert isinstance(model.b_in, nn.Parameter)
    assert isinstance(model.b_out, nn.Parameter)
    assert model.act_fn == solu
    assert isinstance(model.ln, LayerNorm)


def test_forward(cfg: Dict[str, Any]):
    model = GatedMLP(cfg)
    x = torch.randn(2, 10, cfg["d_model"])
    output = model(x)
    assert output.shape == (2, 10, cfg["d_model"])
