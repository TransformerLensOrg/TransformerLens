"""Unit tests for the OlmoHybridArchitectureAdapter.

Download-free: synthetic configs and structural assertions only.
"""

from types import SimpleNamespace

import pytest

from tests.unit.model_bridge.supported_architectures.helpers import make_bridge_cfg
from transformer_lens.config import TransformerBridgeConfig
from transformer_lens.factories.architecture_adapter_factory import (
    SUPPORTED_ARCHITECTURES,
)
from transformer_lens.model_bridge.supported_architectures.olmo_hybrid import (
    OlmoHybridArchitectureAdapter,
)


def _make_cfg() -> TransformerBridgeConfig:
    return make_bridge_cfg(
        "OlmoHybridForCausalLM",
        d_model=64,
        d_head=16,
        n_layers=4,
        n_ctx=128,
        n_heads=4,
        d_mlp=128,
        d_vocab=512,
        n_key_value_heads=2,
        default_prepend_bos=True,
    )


@pytest.fixture(scope="class")
def adapter() -> OlmoHybridArchitectureAdapter:
    return OlmoHybridArchitectureAdapter(_make_cfg())


class TestOlmoHybridComponentMapping:
    def test_per_layer_type_optionality(self, adapter):
        """Full-attention layers are OLMo2 post-norm (no input_layernorm,
        with post_feedforward_layernorm); linear layers are pre-norm."""
        blocks = adapter.component_mapping["blocks"].submodules
        assert blocks["ln1"].optional is True
        assert blocks["ln1"].name == "input_layernorm"
        assert blocks["ln2_post"].optional is True
        assert blocks["attn"].optional is True
        assert blocks["attn"].maintain_native_attention is True
        assert blocks["linear_attn"].optional is True

    def test_stateful(self, adapter):
        assert adapter.cfg.is_stateful is True
        assert adapter.supports_fold_ln is False


class TestOlmoHybridRegistration:
    def test_factory_lookup(self):
        assert SUPPORTED_ARCHITECTURES["OlmoHybridForCausalLM"] is OlmoHybridArchitectureAdapter

    def test_model_type_detection(self):
        from transformer_lens.model_bridge.sources.transformers import (
            determine_architecture_from_hf_config,
        )

        cfg = SimpleNamespace(model_type="olmo_hybrid", architectures=[])
        assert determine_architecture_from_hf_config(cfg) == "OlmoHybridForCausalLM"


class TestOlmoHybridPerLayerContributionAliases:
    """set_original_component selects norm keys and aliases per layer type:
    full-attention layers are post-norm (contribution = norm output),
    linear-attention layers are pre-norm (contribution = raw output). #1648."""

    def _bind(self, adapter, hf_layer):
        import copy

        block = copy.deepcopy(adapter.component_mapping["blocks"])
        block.set_original_component(hf_layer)
        return block

    def test_full_attention_layer_uses_post_norm_outputs(self, adapter) -> None:
        import torch.nn as nn

        hf_layer = nn.Module()
        hf_layer.post_feedforward_layernorm = nn.LayerNorm(8)
        block = self._bind(adapter, hf_layer)
        assert block.hook_aliases["hook_attn_out"] == "ln1_post.hook_out"
        assert block.hook_aliases["hook_mlp_out"] == "ln2_post.hook_out"
        assert block.hook_aliases["hook_resid_mid"] == "mlp.hook_in"

    def test_full_attention_layer_rekeys_post_attention_norm(self, adapter) -> None:
        """post_attention_layernorm follows attention here, so it must not sit in
        the ln2 slot that tools read as the norm feeding the MLP."""
        import torch.nn as nn

        hf_layer = nn.Module()
        hf_layer.post_feedforward_layernorm = nn.LayerNorm(8)
        block = self._bind(adapter, hf_layer)
        assert "ln2" not in block.submodules
        assert block.submodules["ln1_post"].name == "post_attention_layernorm"

    def test_linear_attention_layer_uses_raw_outputs(self, adapter) -> None:
        import torch.nn as nn

        block = self._bind(adapter, nn.Module())
        assert block.hook_aliases["hook_attn_out"] == "linear_attn.hook_out"
        assert block.hook_aliases["hook_mlp_out"] == "mlp.hook_out"
        assert block.hook_aliases["hook_resid_mid"] == "ln2.hook_in"
        assert block.submodules["ln2"].name == "post_attention_layernorm"
