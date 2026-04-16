import pytest
from transformers.utils import is_bitsandbytes_available

from transformer_lens.components.mlps.gated_mlp import GatedMLP
from transformer_lens.components.mlps.gated_mlp_4bit import GatedMLP4Bit
from transformer_lens.components.mlps.mlp import MLP
from transformer_lens.config import HookedTransformerConfig
from transformer_lens.factories.mlp_factory import MLPFactory


def test_create_mlp_basic():
    config = HookedTransformerConfig.unwrap(
        {
            "n_layers": 12,
            "n_ctx": 1024,
            "d_head": 64,
            "d_model": 128,
            "act_fn": "solu",
        }
    )
    mlp = MLPFactory.create_mlp(config)
    assert isinstance(mlp, MLP)


def test_create_mlp_gated():
    config = HookedTransformerConfig.unwrap(
        {
            "n_layers": 12,
            "n_ctx": 1024,
            "d_head": 64,
            "d_model": 128,
            "act_fn": "solu",
            "gated_mlp": True,
        }
    )
    mlp = MLPFactory.create_mlp(config)
    assert isinstance(mlp, GatedMLP)


@pytest.mark.skipif(
    not is_bitsandbytes_available(),
    reason="4 bit not available on current architecture",
)
def test_create_mlp_gated_4bit():
    config = HookedTransformerConfig.unwrap(
        {
            "n_layers": 12,
            "n_ctx": 1024,
            "d_head": 64,
            "d_model": 128,
            "act_fn": "solu",
            "gated_mlp": True,
            "load_in_4bit": True,
        }
    )
    mlp = MLPFactory.create_mlp(config)
    assert isinstance(mlp, GatedMLP4Bit)


def test_create_moe():
    from transformer_lens.components.mlps.moe import MoE

    config = HookedTransformerConfig.unwrap(
        {
            "n_layers": 12,
            "n_ctx": 1024,
            "d_head": 64,
            "d_model": 128,
            "act_fn": "solu",
            "gated_mlp": True,
            "num_experts": 32,
            "experts_per_token": 2,
        }
    )
    mlp = MLPFactory.create_mlp(config)
    assert isinstance(mlp, MoE), f"Expected MoE, got {type(mlp)}"


def test_create_gpt_oss_moe():
    """MLPFactory should return GptOssMoE for GPT-OSS architecture."""
    from transformer_lens.components.mlps.gpt_oss_moe import GptOssMoE

    config = HookedTransformerConfig.unwrap(
        {
            "n_layers": 1,
            "n_ctx": 64,
            "d_head": 8,
            "d_model": 32,
            "d_mlp": 16,
            "act_fn": "silu",
            "gated_mlp": True,
            "num_experts": 2,
            "experts_per_token": 1,
            "original_architecture": "GptOssForCausalLM",
        }
    )
    mlp = MLPFactory.create_mlp(config)
    assert isinstance(mlp, GptOssMoE), f"Expected GptOssMoE, got {type(mlp)}"
