"""Unit tests for GiddArchitectureAdapter.

Uniform-noise diffusion: bidirectional softcap attention delegates,
QK norms are optional (config.use_qk_norm), ScaledLinear projections
rule out folding, and generation is the model's own diffusion sampler.
"""
from typing import Any

import pytest

from tests.unit.model_bridge.supported_architectures.helpers import make_bridge_cfg
from transformer_lens.config import TransformerBridgeConfig
from transformer_lens.factories.architecture_adapter_factory import (
    SUPPORTED_ARCHITECTURES,
)
from transformer_lens.model_bridge.generalized_components import (
    AttentionBridge,
    MLPBridge,
)
from transformer_lens.model_bridge.supported_architectures.gidd import (
    GiddArchitectureAdapter,
)


def _make_cfg(**overrides: Any) -> TransformerBridgeConfig:
    return make_bridge_cfg("GiddForDiffusionLM", **overrides)


@pytest.fixture
def adapter() -> GiddArchitectureAdapter:
    return GiddArchitectureAdapter(_make_cfg())


class TestGiddPhases:
    def test_diffusion_treatment(self, adapter):
        assert adapter.applicable_phases == [1, 2, 3, 4]
        assert adapter.supports_generation is False
        assert adapter.supports_fold_ln is False


class TestGiddMapping:
    def test_attention_delegated_with_optional_qk_norms(self, adapter):
        attn = adapter.component_mapping["blocks"].submodules["attn"]
        assert type(attn) is AttentionBridge
        assert attn.maintain_native_attention is True
        assert attn.submodules["q_norm"].optional is True
        assert attn.submodules["k_norm"].optional is True

    def test_gidd_norm_names(self, adapter):
        blocks = adapter.component_mapping["blocks"]
        assert blocks.submodules["ln1"].name == "attn_layernorm"
        assert blocks.submodules["ln2"].name == "mlp_layernorm"

    def test_ungated_mlp(self, adapter):
        mlp = adapter.component_mapping["blocks"].submodules["mlp"]
        assert isinstance(mlp, MLPBridge)
        assert mlp.submodules["in"].name == "up_proj"

    def test_no_rotary_module_entry(self, adapter):
        """Rotary lives as a model-level buffer, not a module."""
        assert "rotary_emb" not in adapter.component_mapping


def test_factory_registration():
    assert SUPPORTED_ARCHITECTURES["GiddForDiffusionLM"] is GiddArchitectureAdapter


def test_restore_frequencies_recomputes_buffer():
    """The non-persistent rotary table materializes as garbage under v5 meta-device
    loading; restore_frequencies must overwrite it with the config-derived table."""
    import sys
    from types import SimpleNamespace

    import torch

    from transformer_lens.model_bridge.supported_architectures.gidd import (
        restore_frequencies,
    )

    expected = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    sys.modules["_fake_gidd_module"] = SimpleNamespace(
        compute_basic_frequencies=lambda **kw: expected.clone()
    )

    class _Inner:
        __module__ = "_fake_gidd_module"

        def __init__(self):
            self.frequencies = torch.full((3, 4), float("nan"))

    inner = _Inner()
    hf_model = SimpleNamespace(
        model=inner,
        config=SimpleNamespace(
            rope_theta=10000.0, hidden_size=8, num_attention_heads=2, max_position_embeddings=3
        ),
    )
    assert restore_frequencies(hf_model) is True
    assert torch.equal(inner.frequencies, expected)
    del sys.modules["_fake_gidd_module"]


def test_restore_frequencies_no_op_without_buffer():
    """Models lacking the buffer (or the remote helper) are left untouched."""
    from types import SimpleNamespace

    from transformer_lens.model_bridge.supported_architectures.gidd import (
        restore_frequencies,
    )

    assert restore_frequencies(SimpleNamespace(model=SimpleNamespace(), config=None)) is False
