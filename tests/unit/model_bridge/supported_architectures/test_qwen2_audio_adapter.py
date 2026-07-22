"""Unit tests for the Qwen2AudioArchitectureAdapter.

Download-free: synthetic configs and structural assertions only.
"""

from types import SimpleNamespace

import pytest

from tests.unit.model_bridge.supported_architectures.helpers import make_bridge_cfg
from transformer_lens.config import TransformerBridgeConfig
from transformer_lens.factories.architecture_adapter_factory import (
    SUPPORTED_ARCHITECTURES,
)
from transformer_lens.model_bridge.generalized_components.base import (
    GeneralizedComponent,
)
from transformer_lens.model_bridge.supported_architectures.qwen2_audio import (
    Qwen2AudioArchitectureAdapter,
)


def _make_cfg() -> TransformerBridgeConfig:
    return make_bridge_cfg(
        "Qwen2AudioForConditionalGeneration",
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
def adapter() -> Qwen2AudioArchitectureAdapter:
    return Qwen2AudioArchitectureAdapter(_make_cfg())


class TestQwen2AudioComponentMapping:
    def test_audio_components_wrapped_opaquely(self, adapter):
        mapping = adapter.component_mapping
        audio = mapping["audio_encoder"]
        assert isinstance(audio, GeneralizedComponent)
        assert audio.name == "model.audio_tower"
        proj = mapping["audio_projector"]
        assert proj.name == "model.multi_modal_projector"

    def test_text_model_paths_are_nested_with_lm_head(self, adapter):
        """transformers >= 5.13 nests the audio tower, projector and text
        decoder under ``model.*`` and lifts ``lm_head`` to the top level."""
        mapping = adapter.component_mapping
        assert mapping["embed"].name == "model.language_model.embed_tokens"
        assert mapping["blocks"].name == "model.language_model.layers"
        assert mapping["ln_final"].name == "model.language_model.norm"
        assert mapping["unembed"].name == "lm_head"

    def test_qwen2_text_flags(self, adapter):
        assert adapter.cfg.normalization_type == "RMS"
        assert adapter.cfg.gated_mlp is True
        assert adapter.cfg.default_prepend_bos is False
        assert adapter.cfg.n_key_value_heads == 2


class TestQwen2AudioRegistration:
    def test_factory_lookup(self):
        assert (
            SUPPORTED_ARCHITECTURES["Qwen2AudioForConditionalGeneration"]
            is Qwen2AudioArchitectureAdapter
        )

    def test_loads_via_seq2seq_but_classifies_causal(self):
        """AutoModelForSeq2SeqLM carries the HF mapping, but the model is an
        audio-conditioned text decoder — classification must stay causal_lm so
        the bridge does not apply encoder-decoder semantics."""
        from transformers import AutoModelForSeq2SeqLM

        from transformer_lens.model_bridge.sources.transformers import (
            get_hf_model_class_for_architecture,
        )
        from transformer_lens.utilities.architectures import classify_architecture

        assert (
            get_hf_model_class_for_architecture("Qwen2AudioForConditionalGeneration")
            is AutoModelForSeq2SeqLM
        )
        assert classify_architecture("Qwen2AudioForConditionalGeneration") == "causal_lm"

    def test_model_type_detection(self):
        from transformer_lens.model_bridge.sources.transformers import (
            determine_architecture_from_hf_config,
        )

        cfg = SimpleNamespace(model_type="qwen2_audio", architectures=[])
        assert determine_architecture_from_hf_config(cfg) == "Qwen2AudioForConditionalGeneration"
