from transformers.utils import is_bitsandbytes_available
from transformer_lens.HookedTransformerConfig import HookedTransformerConfig
from transformer_lens.factories.mlp_factory import (
    MLPFactory,
)
from transformer_lens.components.mlps.mlp import MLP
from transformer_lens.components.mlps.gated_mlp import GatedMLP
from transformer_lens.components.mlps.gated_mlp_4bit import GatedMLP4Bit

def test_create_mlp_basic():
    
    config = HookedTransformerConfig.unwrap({
        "n_layers": 12,
        "n_ctx": 1024,
        "d_head": 64,
        "d_model": 128,
        "act_fn": "solu",
    })
    mlp = MLPFactory.create_mlp(config)
    assert isinstance(mlp, MLP)

def test_create_mlp_gated():
    
    config = HookedTransformerConfig.unwrap({
        "n_layers": 12,
        "n_ctx": 1024,
        "d_head": 64,
        "d_model": 128,
        "act_fn": "solu",
        "gated_mlp": True,
    })
    mlp = MLPFactory.create_mlp(config)
    assert isinstance(mlp, GatedMLP)

def test_create_mlp_gated_4bit():
    
    if is_bitsandbytes_available():
        config = HookedTransformerConfig.unwrap({
            "n_layers": 12,
            "n_ctx": 1024,
            "d_head": 64,
            "d_model": 128,
            "act_fn": "solu",
            "gated_mlp": True,
            "load_in_4bit": True,
        })
        mlp = MLPFactory.create_mlp(config)
        assert isinstance(mlp, GatedMLP4Bit)

def test_create_moe():
    
    if is_bitsandbytes_available():
        config = HookedTransformerConfig.unwrap({
            "n_layers": 12,
            "n_ctx": 1024,
            "d_head": 64,
            "d_model": 128,
            "act_fn": "solu",
            "gated_mlp": True,
            "num_experts": 32,
        })
        mlp = MLPFactory.create_mlp(config)
        assert isinstance(mlp, GatedMLP4Bit)