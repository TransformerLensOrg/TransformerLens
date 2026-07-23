"""Unit tests for DreamArchitectureAdapter.

Dream is Qwen2.5-shaped but bidirectional (diffusion): attention must stay
delegated to HF, generation phases are excluded, and the v5 rope shim must
restore the 'default' ROPE_INIT_FUNCTIONS entry the remote code looks up.
"""
from typing import Any

import pytest
import torch

from tests.unit.model_bridge.supported_architectures.helpers import make_bridge_cfg
from transformer_lens.config import TransformerBridgeConfig
from transformer_lens.factories.architecture_adapter_factory import (
    SUPPORTED_ARCHITECTURES,
)
from transformer_lens.model_bridge.generalized_components import (
    AttentionBridge,
    BlockBridge,
    EmbeddingBridge,
    GatedMLPBridge,
    RMSNormalizationBridge,
    UnembeddingBridge,
)
from transformer_lens.model_bridge.supported_architectures.dream import (
    DreamArchitectureAdapter,
    _v4_default_rope_parameters,
)


def _make_cfg(**overrides: Any) -> TransformerBridgeConfig:
    """Dream tiny test config (GQA + biased QKV like Qwen2.5)."""
    overrides.setdefault("n_key_value_heads", 4)
    return make_bridge_cfg("DreamModel", **overrides)


@pytest.fixture
def adapter() -> DreamArchitectureAdapter:
    return DreamArchitectureAdapter(_make_cfg())


class TestDreamAdapterPhases:
    def test_diffusion_phases_and_no_generation(self, adapter):
        """Iterative denoising: no autoregressive generation, no P4."""
        assert adapter.applicable_phases == [1, 2, 3, 4]
        assert adapter.supports_generation is False


class TestDreamComponentMapping:
    def test_qwen2_shaped_top_level(self, adapter):
        mapping = adapter.component_mapping
        assert isinstance(mapping["embed"], EmbeddingBridge)
        assert mapping["embed"].name == "model.embed_tokens"
        assert mapping["rotary_emb"].name == "model.rotary_emb"
        assert isinstance(mapping["ln_final"], RMSNormalizationBridge)
        assert mapping["ln_final"].name == "model.norm"
        assert isinstance(mapping["unembed"], UnembeddingBridge)
        assert mapping["unembed"].name == "lm_head"

    def test_attention_is_delegated(self, adapter):
        """Bidirectional attention: the bridge must not reimpose causal masking."""
        blocks = adapter.component_mapping["blocks"]
        assert isinstance(blocks, BlockBridge)
        attn = blocks.submodules["attn"]
        assert type(attn) is AttentionBridge
        assert attn.maintain_native_attention is True
        assert attn.submodules["q"].name == "q_proj"
        assert attn.submodules["o"].name == "o_proj"

    def test_gated_mlp(self, adapter):
        mlp = adapter.component_mapping["blocks"].submodules["mlp"]
        assert isinstance(mlp, GatedMLPBridge)
        assert mlp.submodules["gate"].name == "gate_proj"


class TestDreamRopeShim:
    def test_prepare_loading_registers_default_rope(self, adapter):
        from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS

        ROPE_INIT_FUNCTIONS.pop("default", None)
        adapter.prepare_loading("Dream-org/Dream-v0-Instruct-7B", {})
        assert "default" in ROPE_INIT_FUNCTIONS

    def test_v4_rope_matches_reference_formula(self):
        class Cfg:
            rope_theta = 10000.0
            hidden_size = 64
            num_attention_heads = 4

        inv_freq, scaling = _v4_default_rope_parameters(Cfg(), device="cpu")
        dim = 16
        expected = 1.0 / (10000.0 ** (torch.arange(0, dim, 2, dtype=torch.int64).float() / dim))
        torch.testing.assert_close(inv_freq, expected)
        assert scaling == 1.0


def test_factory_registration():
    assert SUPPORTED_ARCHITECTURES["DreamModel"] is DreamArchitectureAdapter
