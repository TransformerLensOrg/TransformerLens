"""Unit tests for ASTArchitectureAdapter."""

import pytest
from transformers import ASTConfig, ASTForAudioClassification

from transformer_lens.config.transformer_bridge_config import TransformerBridgeConfig
from transformer_lens.model_bridge import TransformerBridge
from transformer_lens.model_bridge.supported_architectures.ast import (
    ASTArchitectureAdapter,
)


@pytest.fixture(autouse=True)
def prevent_internet_processor_download(monkeypatch):
    """Prevent V3 boot from attempting to download a feature extractor named 'ast' from the Hub."""
    monkeypatch.setattr(
        "transformers.AutoFeatureExtractor.from_pretrained", lambda *args, **kwargs: None
    )


@pytest.fixture(scope="module")
def hf_config():
    """Setup a tiny HF AST config for instantaneous local testing."""
    cfg = ASTConfig(
        hidden_size=12,
        num_hidden_layers=2,
        num_attention_heads=2,
        intermediate_size=24,
        patch_size=16,
        num_labels=2,
    )
    # explicitly define the architecture list for the local config
    cfg.architectures = ["ASTForAudioClassification"]
    return cfg


@pytest.fixture(scope="module")
def tl_config():
    """Setup a barebones TransformerBridgeConfig for adapter instantiation test."""
    return TransformerBridgeConfig(
        d_model=12,
        n_heads=2,
        d_head=6,
        n_layers=2,
        n_ctx=1024,
        d_vocab=2,
        architecture="ASTForAudioClassification",
    )


class TestASTAdapterConfig:
    """Anti-drift config assertions."""

    def test_audio_flags(self, tl_config):
        adapter = ASTArchitectureAdapter(tl_config)
        assert adapter.cfg.is_audio_model is True
        assert adapter.cfg.normalization_type == "LN"
        assert ASTArchitectureAdapter.supports_generation is False


class TestASTComponentMapping:
    """Verify hierarchical mapping names and prefix rebinding."""

    def test_classification_prefix_rebinding(self, hf_config, tl_config):
        model = ASTForAudioClassification(hf_config)

        adapter = ASTArchitectureAdapter(tl_config)
        adapter.prepare_model(model)

        assert adapter.component_mapping["blocks"].name == "audio_spectrogram_transformer.layers"
        assert adapter.component_mapping["unembed"].name == "classifier.dense"
        assert adapter.cfg.d_vocab_out == 2


class TestASTBoot:
    """Verify the real load path routes to AutoModelForAudioClassification."""

    def test_ast_boot_transformers_load_weights_false(self):
        # load_weights=False prevents downloading massive model binaries during unit tests
        tl_model = TransformerBridge.boot_transformers(
            "MIT/ast-finetuned-audioset-10-10-0.4593", load_weights=False
        )

        # strict assertions to guarantee the classification path fired
        assert type(tl_model.original_model).__name__ == "ASTForAudioClassification"
        assert "unembed" in tl_model.adapter.component_mapping
