"""Unit tests for the AudioFlamingo3 and MusicFlamingo adapters.

Both are pure subclasses of the Qwen2-Audio adapter: identical module
names (model.audio_tower / model.multi_modal_projector /
model.language_model.*), so the tests pin the inheritance and the mapped paths.
"""
from typing import Any

import pytest

from tests.unit.model_bridge.supported_architectures.helpers import make_bridge_cfg
from transformer_lens.config import TransformerBridgeConfig
from transformer_lens.factories.architecture_adapter_factory import (
    SUPPORTED_ARCHITECTURES,
)
from transformer_lens.model_bridge.generalized_components.base import (
    GeneralizedComponent,
)
from transformer_lens.model_bridge.supported_architectures.audio_flamingo3 import (
    AudioFlamingo3ArchitectureAdapter,
)
from transformer_lens.model_bridge.supported_architectures.music_flamingo import (
    MusicFlamingoArchitectureAdapter,
)
from transformer_lens.model_bridge.supported_architectures.qwen2_audio import (
    Qwen2AudioArchitectureAdapter,
)


def _make_cfg(arch: str, **overrides: Any) -> TransformerBridgeConfig:
    overrides.setdefault("n_key_value_heads", 4)
    return make_bridge_cfg(arch, **overrides)


@pytest.fixture
def af3() -> AudioFlamingo3ArchitectureAdapter:
    return AudioFlamingo3ArchitectureAdapter(_make_cfg("AudioFlamingo3ForConditionalGeneration"))


class TestAudioFlamingo3:
    def test_is_qwen2_audio_layout(self, af3):
        assert isinstance(af3, Qwen2AudioArchitectureAdapter)
        assert af3.cfg.is_multimodal is True
        mapping = af3.component_mapping
        assert type(mapping["audio_encoder"]) is GeneralizedComponent
        assert mapping["audio_encoder"].name == "model.audio_tower"
        assert mapping["audio_projector"].name == "model.multi_modal_projector"
        assert mapping["embed"].name == "model.language_model.embed_tokens"
        assert mapping["unembed"].name == "lm_head"


class TestMusicFlamingo:
    def test_is_af3_layout(self):
        adapter = MusicFlamingoArchitectureAdapter(
            _make_cfg("MusicFlamingoForConditionalGeneration")
        )
        assert isinstance(adapter, AudioFlamingo3ArchitectureAdapter)
        assert adapter.component_mapping["audio_encoder"].name == "model.audio_tower"


def test_factory_registration():
    assert (
        SUPPORTED_ARCHITECTURES["AudioFlamingo3ForConditionalGeneration"]
        is AudioFlamingo3ArchitectureAdapter
    )
    assert (
        SUPPORTED_ARCHITECTURES["MusicFlamingoForConditionalGeneration"]
        is MusicFlamingoArchitectureAdapter
    )
