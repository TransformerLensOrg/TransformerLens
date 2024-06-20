from typing import Any, Callable, Dict

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from transformer_lens.components import GatedMLP, LayerNorm
from transformer_lens.utils import gelu_fast, gelu_new, solu


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


@pytest.mark.parametrize(
    "act_fn_name, expected_act_fn",
    [
        ("relu", F.relu),
        ("gelu", F.gelu),
        ("silu", F.silu),
        ("gelu_new", gelu_new),
        ("gelu_fast", gelu_fast),
    ],
)
def test_activation_functions(act_fn_name: str, expected_act_fn: Callable[..., torch.Tensor]):
    cfg = {
        "n_layers": 12,
        "n_ctx": 1024,
        "d_head": 64,
        "d_model": 128,
        "d_mlp": 256,
        "dtype": torch.float32,
        "act_fn": act_fn_name,
        "normalization_type": "LN",
        "load_in_4bit": False,
    }

    model = GatedMLP(cfg)
    assert model.act_fn == expected_act_fn
