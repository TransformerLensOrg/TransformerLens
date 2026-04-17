"""Tests for the get_params_util module."""

from unittest.mock import Mock

import pytest
import torch

from transformer_lens.model_bridge.get_params_util import get_bridge_params


class TestGetBridgeParams:
    """Test cases for the get_bridge_params utility function."""

    def test_get_bridge_params_basic_structure(self):
        """Test that get_bridge_params returns the expected parameter structure."""
        # Create a mock bridge with basic components
        mock_bridge = self._create_mock_bridge()

        params = get_bridge_params(mock_bridge)

        # Check that we get the expected parameter keys
        expected_keys = ["embed.W_E", "pos_embed.W_pos", "unembed.W_U"]

        # Add attention and MLP keys for each layer
        for layer_idx in range(mock_bridge.cfg.n_layers):
            expected_keys.extend(
                [
                    f"blocks.{layer_idx}.attn.W_Q",
                    f"blocks.{layer_idx}.attn.W_K",
                    f"blocks.{layer_idx}.attn.W_V",
                    f"blocks.{layer_idx}.attn.W_O",
                    f"blocks.{layer_idx}.attn.b_Q",
                    f"blocks.{layer_idx}.attn.b_K",
                    f"blocks.{layer_idx}.attn.b_V",
                    f"blocks.{layer_idx}.attn.b_O",
                    f"blocks.{layer_idx}.mlp.W_in",
                    f"blocks.{layer_idx}.mlp.W_out",
                    f"blocks.{layer_idx}.mlp.b_in",
                    f"blocks.{layer_idx}.mlp.b_out",
                ]
            )

        for key in expected_keys:
            assert key in params, f"Missing parameter key: {key}"
            assert isinstance(params[key], torch.Tensor), f"Parameter {key} should be a tensor"

    def test_get_bridge_params_missing_components(self):
        """Test that get_bridge_params handles missing components gracefully."""
        # Create a mock bridge with missing components
        mock_bridge = self._create_mock_bridge_with_missing_components()

        params = get_bridge_params(mock_bridge)

        # Should still return all expected keys, but with zero tensors for missing components
        assert "embed.W_E" in params
        assert "pos_embed.W_pos" in params
        assert "unembed.W_U" in params

        # Check that missing components return zero tensors
        assert torch.allclose(params["embed.W_E"], torch.zeros(1000, 768))
        assert torch.allclose(params["pos_embed.W_pos"], torch.zeros(1024, 768))

    def test_get_bridge_params_attention_reshaping(self):
        """Test that attention weights are properly reshaped."""
        mock_bridge = self._create_mock_bridge()

        params = get_bridge_params(mock_bridge)

        # Check attention weight shapes
        for layer_idx in range(mock_bridge.cfg.n_layers):
            w_q = params[f"blocks.{layer_idx}.attn.W_Q"]
            w_k = params[f"blocks.{layer_idx}.attn.W_K"]
            w_v = params[f"blocks.{layer_idx}.attn.W_V"]
            w_o = params[f"blocks.{layer_idx}.attn.W_O"]

            # Should be reshaped to [n_heads, d_model, d_head] format
            expected_shape = (12, 768, 64)  # n_heads=12, d_model=768, d_head=64
            assert w_q.shape == expected_shape
            assert w_k.shape == expected_shape
            assert w_v.shape == expected_shape

            # Output should be [n_heads, d_head, d_model]
            expected_o_shape = (12, 64, 768)
            assert w_o.shape == expected_o_shape

    def test_get_bridge_params_bias_handling(self):
        """Test that biases are handled correctly, including None biases."""
        mock_bridge = self._create_mock_bridge_with_none_biases()

        params = get_bridge_params(mock_bridge)

        # Check that None biases are replaced with zero tensors
        for layer_idx in range(mock_bridge.cfg.n_layers):
            b_q = params[f"blocks.{layer_idx}.attn.b_Q"]
            b_k = params[f"blocks.{layer_idx}.attn.b_K"]
            b_v = params[f"blocks.{layer_idx}.attn.b_V"]
            b_o = params[f"blocks.{layer_idx}.attn.b_O"]

            # Should be zero tensors for None biases
            assert torch.allclose(b_q, torch.zeros(12, 64))
            assert torch.allclose(b_k, torch.zeros(12, 64))
            assert torch.allclose(b_v, torch.zeros(12, 64))
            assert torch.allclose(b_o, torch.zeros(768))

    def test_get_bridge_params_config_mismatch_error(self):
        """Test that config mismatch raises appropriate error."""
        mock_bridge = self._create_mock_bridge_with_config_mismatch()

        with pytest.raises(ValueError, match="Configuration mismatch"):
            get_bridge_params(mock_bridge)

    def test_get_bridge_params_gate_weights(self):
        """Test that gate weights are included when present."""
        mock_bridge = self._create_mock_bridge_with_gate_weights()

        params = get_bridge_params(mock_bridge)

        # Check that gate weights are included
        for layer_idx in range(mock_bridge.cfg.n_layers):
            gate_key = f"blocks.{layer_idx}.mlp.W_gate"
            gate_bias_key = f"blocks.{layer_idx}.mlp.b_gate"

            assert gate_key in params
            assert gate_bias_key in params
            assert isinstance(params[gate_key], torch.Tensor)
            assert isinstance(params[gate_bias_key], torch.Tensor)

    def _create_mock_bridge(self):
        """Create a mock bridge with all standard components."""
        mock_bridge = Mock()
        mock_bridge.cfg = Mock()
        mock_bridge.cfg.n_layers = 2
        mock_bridge.cfg.d_model = 768
        mock_bridge.cfg.n_heads = 12
        mock_bridge.cfg.d_head = 64
        mock_bridge.cfg.d_vocab = 1000
        mock_bridge.cfg.n_ctx = 1024
        mock_bridge.cfg.d_mlp = 3072
        mock_bridge.cfg.device = torch.device("cpu")

        # Mock embedding
        mock_bridge.embed = Mock()
        mock_bridge.embed.weight = torch.randn(1000, 768)

        # Mock positional embedding
        mock_bridge.pos_embed = Mock()
        mock_bridge.pos_embed.weight = torch.randn(1024, 768)

        # Mock unembedding
        mock_bridge.unembed = Mock()
        mock_bridge.unembed.weight = torch.randn(1000, 768)

        # Mock blocks
        mock_bridge.blocks = []
        for layer_idx in range(2):
            block = self._create_mock_block()
            mock_bridge.blocks.append(block)

        return mock_bridge

    def _create_mock_bridge_with_missing_components(self):
        """Create a mock bridge with missing components."""
        mock_bridge = Mock()
        mock_bridge.cfg = Mock()
        mock_bridge.cfg.n_layers = 1
        mock_bridge.cfg.d_model = 768
        mock_bridge.cfg.n_heads = 12
        mock_bridge.cfg.d_head = 64
        mock_bridge.cfg.d_vocab = 1000
        mock_bridge.cfg.n_ctx = 1024
        mock_bridge.cfg.d_mlp = 3072
        mock_bridge.cfg.device = torch.device("cpu")

        # Missing embed and pos_embed
        mock_bridge.embed = None
        mock_bridge.pos_embed = None

        # Mock unembedding
        mock_bridge.unembed = Mock()
        mock_bridge.unembed.weight = torch.randn(1000, 768)

        # Mock blocks
        mock_bridge.blocks = []
        for layer_idx in range(1):
            block = self._create_mock_block()
            mock_bridge.blocks.append(block)

        return mock_bridge

    def _create_mock_bridge_with_none_biases(self):
        """Create a mock bridge with None biases."""
        mock_bridge = self._create_mock_bridge()

        # Set all biases to None
        for block in mock_bridge.blocks:
            block.attn.q.bias = None
            block.attn.k.bias = None
            block.attn.v.bias = None
            block.attn.o.bias = None
            setattr(block.mlp, "in", Mock())
            getattr(block.mlp, "in").bias = None
            block.mlp.out.bias = None

        return mock_bridge

    def _create_mock_bridge_with_config_mismatch(self):
        """Create a mock bridge with config mismatch."""
        mock_bridge = Mock()
        mock_bridge.cfg = Mock()
        mock_bridge.cfg.n_layers = 3  # Config says 3 layers
        mock_bridge.cfg.d_model = 768
        mock_bridge.cfg.n_heads = 12
        mock_bridge.cfg.d_head = 64
        mock_bridge.cfg.d_vocab = 1000
        mock_bridge.cfg.n_ctx = 1024
        mock_bridge.cfg.d_mlp = 3072
        mock_bridge.cfg.device = torch.device("cpu")

        # But only provide 1 block
        mock_bridge.blocks = [self._create_mock_block()]

        return mock_bridge

    def _create_mock_bridge_with_gate_weights(self):
        """Create a mock bridge with gate weights."""
        mock_bridge = self._create_mock_bridge()

        # Add gate weights to MLP
        for block in mock_bridge.blocks:
            block.mlp.gate = Mock()
            block.mlp.gate.weight = torch.randn(3072, 768)
            block.mlp.gate.bias = torch.randn(3072)

        return mock_bridge

    def _create_mock_block(self):
        """Create a mock transformer block."""
        block = Mock()

        # Mock attention
        block.attn = Mock()
        block.attn.q = Mock()
        block.attn.q.weight = torch.randn(768, 768)
        block.attn.q.bias = torch.randn(768)

        block.attn.k = Mock()
        block.attn.k.weight = torch.randn(768, 768)
        block.attn.k.bias = torch.randn(768)

        block.attn.v = Mock()
        block.attn.v.weight = torch.randn(768, 768)
        block.attn.v.bias = torch.randn(768)

        block.attn.o = Mock()
        block.attn.o.weight = torch.randn(768, 768)
        block.attn.o.bias = torch.randn(768)

        # Mock MLP
        block.mlp = Mock()
        setattr(block.mlp, "in", Mock())
        getattr(block.mlp, "in").weight = torch.randn(768, 3072)
        getattr(block.mlp, "in").bias = torch.randn(3072)

        block.mlp.out = Mock()
        block.mlp.out.weight = torch.randn(3072, 768)
        block.mlp.out.bias = torch.randn(768)

        return block
