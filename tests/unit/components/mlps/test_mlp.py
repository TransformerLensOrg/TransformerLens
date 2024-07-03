from typing import Any, Dict
import pytest

import torch

from transformer_lens.components import LayerNorm, LayerNormPre
from transformer_lens.hook_points import HookPoint
from transformer_lens.components.mlps.mlp import MLP
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
    MLP(cfg)

def test_forward_without_layer_norm(cfg: Dict[str, Any]):
    cfg["act_fn"] = "solu"

    model = MLP(cfg)
    model.select_activation_function()
    # assert model.hook_mid is None
    # assert not hasattr(model, "ln")
    
    input = torch.full((1, 1, 128), 0.085)
    print(input.shape)
    
    result = model(input)
    
    print("result" + str(result))
    assert result[0][0][1] == 0.85

def test_forward_with_layer_norm(cfg: Dict[str, Any]):
    model = MLP(cfg)
    model.select_activation_function()
    assert isinstance(model.hook_mid, HookPoint)
    assert isinstance(model.ln, LayerNorm)

    input = torch.full((1, 1, 128), 0.85)
    result = model(input)