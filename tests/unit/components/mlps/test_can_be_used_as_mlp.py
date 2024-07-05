from typing import Any, Dict

import pytest
import torch

from transformer_lens.components import LayerNorm, LayerNormPre
from transformer_lens.components.mlps.can_be_used_as_mlp import CanBeUsedAsMLP
from transformer_lens.hook_points import HookPoint
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
    CanBeUsedAsMLP(cfg)


def test_initialization_fails_without_d_mlp(cfg: Dict[str, Any]):
    cfg["d_mlp"] = None
    pytest.raises(ValueError)
    CanBeUsedAsMLP(cfg)


def test_select_activation_function_selects_function():
    cfg = {
        "n_layers": 12,
        "n_ctx": 1024,
        "d_head": 64,
        "d_model": 128,
        "d_mlp": 256,
        "dtype": torch.float32,
        "act_fn": "silu",
        "normalization_type": "LN",
        "load_in_4bit": False,
    }

    model = CanBeUsedAsMLP(cfg)
    model.select_activation_function()
    assert model.act_fn is not None


def test_select_activation_function_with_layer_norm():
    cfg = {
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

    model = CanBeUsedAsMLP(cfg)
    model.select_activation_function()
    assert model.act_fn == solu
    assert isinstance(model.hook_mid, HookPoint)
    assert isinstance(model.ln, LayerNorm)


def test_select_activation_function_with_layer_norm_pre():
    cfg = {
        "n_layers": 12,
        "n_ctx": 1024,
        "d_head": 64,
        "d_model": 128,
        "d_mlp": 256,
        "dtype": torch.float32,
        "act_fn": "solu_ln",
        "normalization_type": "LNPre",
        "load_in_4bit": False,
    }

    model = CanBeUsedAsMLP(cfg)
    model.select_activation_function()
    assert model.act_fn == solu
    assert isinstance(model.hook_mid, HookPoint)
    assert isinstance(model.ln, LayerNormPre)
