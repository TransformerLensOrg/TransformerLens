"""Unit tests for RwkvArchitectureAdapter.

WKV linear-attention RNN: time-mix/channel-mix mixers delegate with their
projections hookable, the layer-0 pre_ln is optional, the block alias set
is mixer-flavored (no attention), and rescale_every is zeroed at load so
fp32 weights stay canonical.
"""
from typing import Any

import pytest

from tests.unit.model_bridge.supported_architectures.helpers import make_bridge_cfg
from transformer_lens.config import TransformerBridgeConfig
from transformer_lens.factories.architecture_adapter_factory import (
    SUPPORTED_ARCHITECTURES,
)
from transformer_lens.model_bridge.generalized_components import MLPBridge
from transformer_lens.model_bridge.generalized_components.base import (
    GeneralizedComponent,
)
from transformer_lens.model_bridge.supported_architectures.rwkv import (
    RwkvArchitectureAdapter,
    _RwkvBlockBridge,
)


def _make_cfg(**overrides: Any) -> TransformerBridgeConfig:
    return make_bridge_cfg("RwkvForCausalLM", **overrides)


@pytest.fixture
def adapter() -> RwkvArchitectureAdapter:
    return RwkvArchitectureAdapter(_make_cfg())


class TestRwkvPhases:
    def test_recurrent_treatment(self, adapter):
        """State-kwarg generation isn't wired; WKV has no fold target."""
        assert adapter.applicable_phases == [1, 2, 3]
        assert adapter.supports_generation is False
        assert adapter.supports_fold_ln is False


class TestRwkvMapping:
    def test_mixer_flavored_aliases(self, adapter):
        blocks = adapter.component_mapping["blocks"]
        assert isinstance(blocks, _RwkvBlockBridge)
        assert "hook_time_mix_out" in blocks.hook_aliases
        assert "hook_attn_out" not in blocks.hook_aliases

    def test_time_and_channel_mix_delegated(self, adapter):
        blocks = adapter.component_mapping["blocks"]
        tm = blocks.submodules["time_mix"]
        cm = blocks.submodules["channel_mix"]
        assert type(tm) is GeneralizedComponent and tm.name == "attention"
        assert tm.submodules["receptance"].name == "receptance"
        assert isinstance(cm, MLPBridge) and cm.name == "feed_forward"
        assert cm.submodules["in"].name == "key"
        assert cm.submodules["out"].name == "value"

    def test_layer0_pre_ln_optional(self, adapter):
        pre = adapter.component_mapping["blocks"].submodules["pre_ln"]
        assert pre.optional is True

    def test_use_cache_forced_off_at_load(self, adapter):
        """In-place recurrent state writes break autograd under backward hooks."""

        class Cfg:
            use_cache = True

        kwargs = {"config": Cfg()}
        adapter.prepare_loading("RWKV/rwkv-4-169m-pile", kwargs)
        assert kwargs["config"].use_cache is False

    def test_rwkv_paths(self, adapter):
        mapping = adapter.component_mapping
        assert mapping["embed"].name == "rwkv.embeddings"
        assert mapping["ln_final"].name == "rwkv.ln_out"
        assert mapping["unembed"].name == "head"


def test_factory_registration():
    assert SUPPORTED_ARCHITECTURES["RwkvForCausalLM"] is RwkvArchitectureAdapter
