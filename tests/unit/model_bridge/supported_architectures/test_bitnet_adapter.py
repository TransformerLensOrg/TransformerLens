"""Unit tests for the BitNetArchitectureAdapter.

Download-free: synthetic configs and structural assertions only.
"""

from types import SimpleNamespace

import pytest

from tests.unit.model_bridge.supported_architectures.helpers import make_bridge_cfg
from transformer_lens.config import TransformerBridgeConfig
from transformer_lens.factories.architecture_adapter_factory import (
    SUPPORTED_ARCHITECTURES,
)
from transformer_lens.model_bridge.generalized_components import GatedMLPBridge
from transformer_lens.model_bridge.supported_architectures.bitnet import (
    BitNetArchitectureAdapter,
    _BitNetAttentionBridge,
)


def _make_cfg() -> TransformerBridgeConfig:
    return make_bridge_cfg(
        "BitNetForCausalLM",
        d_model=64,
        d_head=16,
        n_layers=2,
        n_ctx=128,
        n_heads=4,
        d_mlp=128,
        d_vocab=512,
        n_key_value_heads=2,
        default_prepend_bos=True,
    )


@pytest.fixture(scope="class")
def adapter() -> BitNetArchitectureAdapter:
    return BitNetArchitectureAdapter(_make_cfg())


class TestBitNetComponentMapping:
    def test_attention_uses_sub_norm_bridge(self, adapter):
        """BitNet's distinguishing feature: RMSNorms before both output
        projections, applied via the adapter-local attention bridge and the
        delegated HF MLP."""
        submodules = adapter.component_mapping["blocks"].submodules
        assert isinstance(submodules["attn"], _BitNetAttentionBridge)
        assert isinstance(submodules["mlp"], GatedMLPBridge)

    def test_sub_norms_disable_folding(self, adapter):
        assert adapter.supports_fold_ln is False

    def test_attention_reconstruction_applies_sub_norm(self, adapter):
        """The generic reconstruction skips attn_sub_norm; the subclass's
        pre-output-projection seam must apply it."""
        import torch

        attn = adapter.component_mapping["blocks"].submodules["attn"]

        class _FakeNorm(torch.nn.Module):
            def forward(self, x):
                return x * 2.0

        class _FakeAttn(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.attn_sub_norm = _FakeNorm()

        attn._modules["_original_component"] = _FakeAttn()
        x = torch.ones(1, 2, 4)
        out = attn._pre_output_projection(x)
        assert torch.equal(out, x * 2.0)


class TestBitNetRegistration:
    def test_factory_lookup(self):
        assert SUPPORTED_ARCHITECTURES["BitNetForCausalLM"] is BitNetArchitectureAdapter

    def test_model_type_detection(self):
        from transformer_lens.model_bridge.sources.transformers import (
            determine_architecture_from_hf_config,
        )

        cfg = SimpleNamespace(model_type="bitnet", architectures=[])
        assert determine_architecture_from_hf_config(cfg) == "BitNetForCausalLM"
