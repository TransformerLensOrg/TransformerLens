"""Tests for audio model start_at_layer guard."""

from types import SimpleNamespace

import pytest
import torch

from transformer_lens.config import TransformerBridgeConfig
from transformer_lens.model_bridge import TransformerBridge


class _NoParameterDriver:
    def supports(self, name: str) -> bool:
        return False


class TestAudioStartAtLayerGuard:
    """Test that start_at_layer raises NotImplementedError for audio models."""

    @pytest.fixture
    def audio_bridge_cfg(self) -> TransformerBridgeConfig:
        """Create a config with is_audio_model=True."""
        return TransformerBridgeConfig(
            d_model=768,
            d_head=64,
            n_layers=12,
            n_ctx=1024,
            d_vocab=1000,
            d_mlp=3072,
            n_heads=12,
            is_audio_model=True,
            architecture="HubertModel",
        )

    @pytest.fixture
    def non_audio_bridge_cfg(self) -> TransformerBridgeConfig:
        """Create a config with is_audio_model=False."""
        return TransformerBridgeConfig(
            d_model=768,
            d_head=64,
            n_layers=12,
            n_ctx=1024,
            d_vocab=1000,
            d_mlp=3072,
            n_heads=12,
            is_audio_model=False,
            architecture="GPT2LMHeadModel",
        )

    @pytest.fixture
    def audio_bridge(self, audio_bridge_cfg: TransformerBridgeConfig) -> TransformerBridge:
        bridge = TransformerBridge.__new__(TransformerBridge)
        bridge.cfg = audio_bridge_cfg
        bridge._hook_registry = {}
        bridge._hook_registry_initialized = True
        bridge.compatibility_mode = False
        bridge.adapter = SimpleNamespace(supports_hf_output_attentions=False)
        bridge._driver = _NoParameterDriver()
        return bridge

    @pytest.mark.parametrize("runner", ["forward", "run_with_cache", "run_with_hooks"])
    def test_audio_model_start_at_layer_raises_specific_error(
        self,
        audio_bridge: TransformerBridge,
        runner: str,
    ) -> None:
        """Test that start_at_layer on audio model raises via public execution helpers."""
        resid = torch.zeros(1, 8, audio_bridge.cfg.d_model)

        with pytest.raises(
            NotImplementedError,
            match="not supported for audio models",
        ):
            getattr(audio_bridge, runner)(resid, start_at_layer=2)

    def test_non_audio_model_start_at_layer_raises_blocks_error(
        self,
        non_audio_bridge_cfg: TransformerBridgeConfig,
    ) -> None:
        """Test that start_at_layer on non-audio model raises the generic blocks error."""
        bridge = TransformerBridge.__new__(TransformerBridge)
        bridge.cfg = non_audio_bridge_cfg

        resid = torch.zeros(1, 8, non_audio_bridge_cfg.d_model)

        with pytest.raises(NotImplementedError, match="requires a 'blocks' stack"):
            bridge.forward(resid, start_at_layer=2)

    def test_audio_error_message_mentions_convolutional(
        self,
        audio_bridge: TransformerBridge,
    ) -> None:
        """Test that the audio-specific error mentions convolutional feature extraction."""
        resid = torch.zeros(1, 8, audio_bridge.cfg.d_model)

        with pytest.raises(NotImplementedError, match="convolutional feature"):
            audio_bridge.forward(resid, start_at_layer=2)

    def test_audio_error_message_mentions_residual_injection(
        self,
        audio_bridge: TransformerBridge,
    ) -> None:
        """Test that the audio-specific error mentions residual-stream injection."""
        resid = torch.zeros(1, 8, audio_bridge.cfg.d_model)

        with pytest.raises(NotImplementedError, match="residual-stream injection"):
            audio_bridge.forward(resid, start_at_layer=2)
