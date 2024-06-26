from typing import Any, Callable, Dict
import pytest

import torch
import torch.nn.functional as F

from transformer_lens.components.mlps.can_be_used_as_mlp import CanBeUsedAsMLP
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
    CanBeUsedAsMLP(cfg)

def test_initialization_fails_without_d_mlp(cfg: Dict[str, Any]):
    cfg["d_mlp"] = None
    pytest.raises(ValueError)
    CanBeUsedAsMLP(cfg)
    

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
def test_select_activation_function_selects_correct_function(act_fn_name: str, expected_act_fn: Callable[..., torch.Tensor]):
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

    model = CanBeUsedAsMLP(cfg)
    model.select_activation_function()
    assert model.act_fn == expected_act_fn