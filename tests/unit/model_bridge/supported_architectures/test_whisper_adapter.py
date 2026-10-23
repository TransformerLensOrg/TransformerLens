"""Unit tests for the WhisperArchitectureAdapter.

Download-free: tiny synthetic configs and structural assertions only.
"""

from types import SimpleNamespace

import pytest
import torch

from transformer_lens.config import TransformerBridgeConfig
from transformer_lens.factories.architecture_adapter_factory import (
    SUPPORTED_ARCHITECTURES,
)
from transformer_lens.model_bridge.generalized_components import (
    EmbeddingBridge,
    NormalizationBridge,
    PosEmbedBridge,
    UnembeddingBridge,
)
from transformer_lens.model_bridge.generalized_components.base import (
    GeneralizedComponent,
)
from transformer_lens.model_bridge.sources._bridge_builder import (
    build_bridge_config_from_hf,
    skip_tokenizer_for_modality,
)
from transformer_lens.model_bridge.supported_architectures.blenderbot import (
    BlenderbotArchitectureAdapter,
)
from transformer_lens.model_bridge.supported_architectures.whisper import (
    WhisperArchitectureAdapter,
)

ARCH = "WhisperForConditionalGeneration"


def _base_cfg(encoder_layers: int = 4, decoder_layers: int = 2) -> TransformerBridgeConfig:
    cfg = TransformerBridgeConfig(
        d_model=64,
        d_head=16,
        n_layers=encoder_layers,
        n_ctx=448,
        n_heads=4,
        d_mlp=256,
        d_vocab=512,
        architecture=ARCH,
    )
    cfg.encoder_layers = encoder_layers
    cfg.decoder_layers = decoder_layers
    cfg.encoder_attention_heads = 4
    cfg.decoder_attention_heads = 4
    cfg.encoder_ffn_dim = 256
    cfg.decoder_ffn_dim = 256
    return cfg


@pytest.fixture(scope="class")
def adapter() -> WhisperArchitectureAdapter:
    return WhisperArchitectureAdapter(_base_cfg())


class TestWhisperConfig:
    def test_asymmetric_stacks_follow_decoder(self, adapter):
        """large-v3-turbo (32/4) and distil-whisper (n/2) are asymmetric."""
        assert adapter.cfg.n_layers == 2

    def test_audio_seq2seq_flags(self, adapter):
        assert adapter.cfg.is_audio_model is True
        assert adapter.cfg.is_audio_seq2seq is True
        assert adapter.cfg.is_multimodal is False

    def test_declares_input_features(self, adapter):
        assert adapter.audio_input_name == "input_features"

    def test_generation_and_weight_processing_disabled(self, adapter):
        assert adapter.supports_generation is False
        assert adapter.supports_fold_ln is False
        assert adapter.supports_center_writing_weights is False

    def test_scale_embedding_not_forced_on(self, adapter):
        assert not adapter.cfg.scale_embedding

    def test_rejects_asymmetric_heads(self):
        cfg = _base_cfg()
        cfg.decoder_attention_heads = 8
        with pytest.raises(ValueError, match="attention heads"):
            WhisperArchitectureAdapter(cfg)

    def test_rejects_asymmetric_ffn(self):
        cfg = _base_cfg()
        cfg.decoder_ffn_dim = 512
        with pytest.raises(ValueError, match="FFN dims"):
            WhisperArchitectureAdapter(cfg)

    def test_n_ctx_from_max_target_positions(self):
        """With no other context field, n_ctx is the decoder's 448, not the 2048 fallback."""
        from transformers import WhisperConfig

        hf_cfg = WhisperConfig(
            vocab_size=128,
            num_mel_bins=16,
            d_model=32,
            encoder_layers=2,
            decoder_layers=2,
            encoder_attention_heads=4,
            decoder_attention_heads=4,
            max_target_positions=448,
        )
        cfg = build_bridge_config_from_hf(hf_cfg, ARCH, "whisper-unit", torch.float32)
        assert cfg.n_ctx == 448


class TestWhisperComponentMapping:
    def test_conv_front_end_replaces_token_embedding(self, adapter):
        mapping = adapter.component_mapping
        assert "embed" not in mapping
        assert "embed_ln" not in mapping
        assert type(mapping["conv1"]) is GeneralizedComponent
        assert mapping["conv1"].name == "model.encoder.conv1"
        assert type(mapping["conv2"]) is GeneralizedComponent
        assert mapping["conv2"].name == "model.encoder.conv2"
        assert isinstance(mapping["pos_embed"], PosEmbedBridge)
        assert mapping["pos_embed"].name == "model.encoder.embed_positions"

    def test_encoder_front_end_runs_first(self, adapter):
        keys = list(adapter.component_mapping)
        assert keys[:4] == ["conv1", "conv2", "pos_embed", "encoder_blocks"]

    def test_channel_first_hooks_declare_position_axis(self, adapter):
        mapping = adapter.component_mapping
        for name in ("conv1", "conv2"):
            assert mapping[name].hook_in.pos_dim == 2
            assert mapping[name].hook_out.pos_dim == 2
        # The sinusoidal table is read whole as [n_frames, d_model], no batch dim.
        assert mapping["pos_embed"].hook_out.pos_dim == 0

    def test_decoder_side_and_stack_norms(self, adapter):
        mapping = adapter.component_mapping
        assert isinstance(mapping["decoder_embed"], EmbeddingBridge)
        assert mapping["decoder_embed"].name == "model.decoder.embed_tokens"
        assert isinstance(mapping["decoder_pos_embed"], PosEmbedBridge)
        assert isinstance(mapping["encoder_ln_final"], NormalizationBridge)
        assert mapping["encoder_ln_final"].name == "model.encoder.layer_norm"
        assert mapping["decoder_ln_final"].name == "model.decoder.layer_norm"

    def test_unembed_is_proj_out(self, adapter):
        unembed = adapter.component_mapping["unembed"]
        assert isinstance(unembed, UnembeddingBridge)
        assert unembed.name == "proj_out"

    def test_blocks_match_bart_family_layout(self, adapter):
        """Block layout is inherited unchanged; only the front end and head differ."""
        reference = BlenderbotArchitectureAdapter(_base_cfg()).component_mapping
        mapping = adapter.component_mapping
        for stack in ("encoder_blocks", "decoder_blocks"):
            assert mapping[stack].name == reference[stack].name
            assert set(mapping[stack].submodules) == set(reference[stack].submodules)
        assert set(mapping["decoder_blocks"].submodules) == {
            "ln1",
            "self_attn",
            "ln2",
            "cross_attn",
            "ln3",
            "mlp",
        }


class TestWhisperRegistration:
    def test_factory_lookup(self):
        assert SUPPORTED_ARCHITECTURES[ARCH] is WhisperArchitectureAdapter

    def test_model_type_detection(self):
        from transformer_lens.model_bridge.sources.transformers import (
            determine_architecture_from_hf_config,
        )

        cfg = SimpleNamespace(model_type="whisper", architectures=[])
        assert determine_architecture_from_hf_config(cfg) == ARCH

    def test_loads_via_speech_seq2seq_auto_class(self):
        from transformers import AutoModelForSpeechSeq2Seq

        from transformer_lens.model_bridge.sources.transformers.helpers import (
            get_hf_model_class_for_architecture,
        )

        assert get_hf_model_class_for_architecture(ARCH) is AutoModelForSpeechSeq2Seq

    def test_classified_as_seq2seq_not_audio_text(self):
        from transformer_lens.utilities.architectures import (
            AUDIO_SEQ2SEQ_ARCHITECTURES,
            AUDIO_TEXT_ARCHITECTURES,
            SEQ2SEQ_ARCHITECTURES,
            classify_architecture,
        )

        assert ARCH in AUDIO_SEQ2SEQ_ARCHITECTURES
        assert ARCH not in AUDIO_TEXT_ARCHITECTURES
        assert ARCH not in SEQ2SEQ_ARCHITECTURES
        assert classify_architecture(ARCH) == "seq2seq"

    def test_tokenizer_kept_despite_audio_flag(self, adapter):
        assert skip_tokenizer_for_modality(adapter.cfg) is False
