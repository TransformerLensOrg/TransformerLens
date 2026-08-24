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

            # Shape alone cannot catch Q/K/V being read from the wrong
            # projection: pin the VALUES against the TL-layout properties the
            # mock block exposes.
            block = mock_bridge.blocks[layer_idx]
            n_heads, d_model, d_head = 12, 768, 64
            assert w_q.shape == (n_heads, d_model, d_head)
            assert w_o.shape == (n_heads, d_head, d_model)
            torch.testing.assert_close(w_q, block.attn.W_Q)
            torch.testing.assert_close(w_o, block.attn.W_O)
            # Negative control: Q and K must be distinguishable in this fixture,
            # or reading either one would satisfy the assertions above.
            assert not torch.equal(w_q, w_k)
            assert not torch.equal(w_k, w_v)

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
            block.attn.b_Q = None
            block.attn.b_K = None
            block.attn.b_V = None
            block.attn.b_O = None
            block.mlp.b_in = None
            block.mlp.b_out = None

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
            block.mlp.W_gate = torch.randn(768, 3072)
            block.mlp.b_gate = torch.randn(3072)

        return mock_bridge

    def _create_mock_block(self):
        """Create a mock transformer block exposing TL-layout weight properties."""
        block = Mock()

        # Mock attention (TL-layout properties, as the component bridges expose)
        block.attn = Mock()
        block.attn.W_Q = torch.randn(12, 768, 64)
        block.attn.W_K = torch.randn(12, 768, 64)
        block.attn.W_V = torch.randn(12, 768, 64)
        block.attn.W_O = torch.randn(12, 64, 768)
        block.attn.b_Q = torch.randn(12, 64)
        block.attn.b_K = torch.randn(12, 64)
        block.attn.b_V = torch.randn(12, 64)
        block.attn.b_O = torch.randn(768)

        # Mock MLP
        block.mlp = Mock()
        block.mlp.W_in = torch.randn(768, 3072)
        block.mlp.W_out = torch.randn(3072, 768)
        block.mlp.b_in = torch.randn(3072)
        block.mlp.b_out = torch.randn(768)

        return block


class TestGQAExpansion:
    """Grouped K/V must be expanded to n_heads (legacy HT convention)."""

    def _make_gqa_bridge(self):
        mock_bridge = Mock()
        mock_bridge.cfg = Mock()
        mock_bridge.cfg.n_layers = 1
        mock_bridge.cfg.d_model = 64
        mock_bridge.cfg.n_heads = 4
        mock_bridge.cfg.d_head = 16
        mock_bridge.cfg.d_vocab = 100
        mock_bridge.cfg.n_ctx = 32
        mock_bridge.cfg.d_mlp = 128
        mock_bridge.cfg.device = torch.device("cpu")

        mock_bridge.embed = Mock()
        mock_bridge.embed.weight = torch.randn(100, 64)
        mock_bridge.pos_embed = Mock()
        mock_bridge.pos_embed.weight = torch.randn(32, 64)
        mock_bridge.unembed = Mock()
        mock_bridge.unembed.weight = torch.randn(100, 64)

        block = Mock()
        block.attn = Mock()
        block.attn.W_Q = torch.randn(4, 64, 16)
        block.attn.W_K = torch.randn(2, 64, 16)  # grouped: n_kv_heads=2
        block.attn.W_V = torch.randn(2, 64, 16)
        block.attn.W_O = torch.randn(4, 16, 64)
        block.attn.b_Q = torch.randn(4, 16)
        block.attn.b_K = torch.randn(2, 16)
        block.attn.b_V = torch.randn(2, 16)
        block.attn.b_O = torch.randn(64)
        block.mlp = Mock()
        block.mlp.W_in = torch.randn(64, 128)
        block.mlp.W_out = torch.randn(128, 64)
        block.mlp.b_in = torch.randn(128)
        block.mlp.b_out = torch.randn(64)
        mock_bridge.blocks = [block]
        return mock_bridge

    def test_grouped_kv_expanded_to_n_heads(self):
        bridge = self._make_gqa_bridge()
        params = get_bridge_params(bridge)

        w_k = params["blocks.0.attn.W_K"]
        w_v = params["blocks.0.attn.W_V"]
        assert w_k.shape == (4, 64, 16)
        assert w_v.shape == (4, 64, 16)
        # repeat_interleave semantics: heads 0,1 share kv head 0; heads 2,3 share kv head 1
        assert torch.equal(w_k[0], w_k[1])
        assert torch.equal(w_k[0], bridge.blocks[0].attn.W_K[0])
        assert torch.equal(w_k[2], bridge.blocks[0].attn.W_K[1])

    def test_grouped_biases_expanded(self):
        bridge = self._make_gqa_bridge()
        params = get_bridge_params(bridge)
        b_k = params["blocks.0.attn.b_K"]
        assert b_k.shape == (4, 16)
        assert params["blocks.0.attn.b_V"].shape == (4, 16)
        # Pairing must be repeat_interleave (blocked), not tiling: heads 0,1
        # share kv head 0 and heads 2,3 share kv head 1.
        grouped = bridge.blocks[0].attn.b_K
        assert torch.equal(b_k[0], b_k[1])
        assert torch.equal(b_k[0], grouped[0])
        assert torch.equal(b_k[2], grouped[1])
        assert torch.equal(params["blocks.0.attn.b_Q"], bridge.blocks[0].attn.b_Q)
