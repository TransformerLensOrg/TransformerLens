from typing import Any, Dict

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

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


def test_forward_matches_reference_equation():
    """Numeric equivalence vs a hand-rolled gated-MLP reference (issue #264).

    Closes the original ask in the thread: build an "equivalent gated MLP in
    pytorch" and confirm the component matches it under ``torch.allclose``.
    Uses ``silu`` so the LN-activation branch is not exercised — that keeps the
    reference equation to the documented form.
    """
    cfg: Dict[str, Any] = {
        "n_layers": 1,
        "n_ctx": 16,
        "d_head": 32,
        "d_model": 64,
        "d_mlp": 128,
        "dtype": torch.float32,
        "act_fn": "silu",
        "normalization_type": None,
        "load_in_4bit": False,
    }
    torch.manual_seed(0)
    model = GatedMLP(cfg).eval()
    # Randomize the params so the test isn't run against zero-bias defaults.
    for p in model.parameters():
        torch.nn.init.normal_(p, std=0.02)

    x = torch.randn(2, 5, cfg["d_model"])
    actual = model(x)

    # Reference: mlp_out = (silu(x @ W_gate) * (x @ W_in) + b_in) @ W_out + b_out.
    # GatedMLP uses F.linear with .T.contiguous() to match HF accumulation order;
    # mirror that here so the two compute graphs are bitwise comparable in fp32.
    pre_act = F.linear(x, model.W_gate.T.contiguous())
    pre_linear = F.linear(x, model.W_in.T.contiguous())
    post_act = F.silu(pre_act) * pre_linear + model.b_in
    expected = F.linear(post_act, model.W_out.T.contiguous(), model.b_out)

    assert torch.allclose(actual, expected, atol=1e-6)
