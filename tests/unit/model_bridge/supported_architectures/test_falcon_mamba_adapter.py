"""Unit tests for the FalconMambaArchitectureAdapter.

Download-free: synthetic configs and structural assertions only.
"""

from types import SimpleNamespace

import pytest

from transformer_lens.config import TransformerBridgeConfig
from transformer_lens.factories.architecture_adapter_factory import (
    SUPPORTED_ARCHITECTURES,
)
from transformer_lens.model_bridge.supported_architectures.falcon_mamba import (
    FalconMambaArchitectureAdapter,
)
from transformer_lens.model_bridge.supported_architectures.mamba import (
    MambaArchitectureAdapter,
)


def _make_cfg() -> TransformerBridgeConfig:
    return TransformerBridgeConfig(
        d_model=64,
        d_head=16,
        n_layers=2,
        n_ctx=128,
        n_heads=4,
        d_mlp=128,
        d_vocab=512,
        architecture="FalconMambaForCausalLM",
    )


@pytest.fixture(scope="class")
def adapter() -> FalconMambaArchitectureAdapter:
    return FalconMambaArchitectureAdapter(_make_cfg())


class TestFalconMambaAdapter:
    def test_inherits_mamba_mapping(self, adapter):
        """FalconMamba's module tree is identical to Mamba-1; only the mixer
        math differs (parameter-free B/C/dt RMS handled in SSMMixerBridge)."""
        assert isinstance(adapter, MambaArchitectureAdapter)
        mapping = adapter.component_mapping
        assert mapping["embed"].name == "backbone.embeddings"
        assert mapping["blocks"].name == "backbone.layers"
        assert mapping["ln_final"].name == "backbone.norm_f"
        mixer = mapping["blocks"].submodules["mixer"]
        assert set(mixer.submodules) == {"in_proj", "conv1d", "x_proj", "dt_proj", "out_proj"}

    def test_stateful_flag(self, adapter):
        assert adapter.cfg.is_stateful is True


class TestFalconMambaRegistration:
    def test_factory_lookup(self):
        assert SUPPORTED_ARCHITECTURES["FalconMambaForCausalLM"] is FalconMambaArchitectureAdapter

    def test_model_type_detection(self):
        from transformer_lens.model_bridge.sources.transformers import (
            determine_architecture_from_hf_config,
        )

        cfg = SimpleNamespace(model_type="falcon_mamba", architectures=[])
        assert determine_architecture_from_hf_config(cfg) == "FalconMambaForCausalLM"
