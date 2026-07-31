"""Tests for audio model start_at_layer guard."""

import pytest
import torch

from transformer_lens.config import TransformerBridgeConfig
from transformer_lens.model_bridge import TransformerBridge


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

    def test_audio_model_start_at_layer_raises_specific_error(self, audio_bridge_cfg):
        """Test that start_at_layer on audio model raises audio-specific NotImplementedError."""
        bridge = TransformerBridge.__new__(TransformerBridge)
        bridge.cfg = audio_bridge_cfg

        resid = torch.zeros(1, 8, audio_bridge_cfg.d_model)

        with pytest.raises(NotImplementedError, match="not supported for audio models"):
            bridge.forward(resid, start_at_layer=2)

    def test_non_audio_model_start_at_layer_raises_blocks_error(self, non_audio_bridge_cfg):
        """Test that start_at_layer on non-audio model raises the generic blocks error."""
        bridge = TransformerBridge.__new__(TransformerBridge)
        bridge.cfg = non_audio_bridge_cfg

        resid = torch.zeros(1, 8, non_audio_bridge_cfg.d_model)

        with pytest.raises(NotImplementedError, match="requires a 'blocks' stack"):
            bridge.forward(resid, start_at_layer=2)

    def test_audio_error_message_mentions_convolutional(self, audio_bridge_cfg):
        """Test that the audio-specific error mentions convolutional feature extraction."""
        bridge = TransformerBridge.__new__(TransformerBridge)
        bridge.cfg = audio_bridge_cfg

        resid = torch.zeros(1, 8, audio_bridge_cfg.d_model)

        with pytest.raises(NotImplementedError, match="convolutional feature"):
            bridge.forward(resid, start_at_layer=2)

    def test_audio_error_message_mentions_residual_injection(self, audio_bridge_cfg):
        """Test that the audio-specific error mentions residual-stream injection."""
        bridge = TransformerBridge.__new__(TransformerBridge)
        bridge.cfg = audio_bridge_cfg

        resid = torch.zeros(1, 8, audio_bridge_cfg.d_model)

        with pytest.raises(NotImplementedError, match="residual-stream injection"):
            bridge.forward(resid, start_at_layer=2)
