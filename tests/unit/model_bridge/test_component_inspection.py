"""Unit tests for bridge component access and properties.

Tests that TransformerBridge exposes components correctly through its own API,
not just through the underlying HuggingFace model. Uses distilgpt2 (CI-cached).
"""

import pytest
import torch

from transformer_lens.model_bridge.bridge import TransformerBridge


@pytest.fixture(scope="module")
def bridge():
    """Create a TransformerBridge for testing."""
    return TransformerBridge.boot_transformers("distilgpt2", device="cpu")


@pytest.fixture(scope="module")
def bridge_compat():
    """Create a TransformerBridge with compatibility mode for weight property tests."""
    b = TransformerBridge.boot_transformers("distilgpt2", device="cpu")
    b.enable_compatibility_mode()
    return b


class TestBridgeComponentAccess:
    """Test that bridge exposes components through its own API."""

    def test_blocks_accessible_and_indexed(self, bridge):
        """Bridge blocks should be accessible by index."""
        assert hasattr(bridge, "blocks"), "Bridge should have blocks attribute"
        assert len(bridge.blocks) == bridge.cfg.n_layers
        block_0 = bridge.blocks[0]
        assert block_0 is not None

    def test_block_has_attn_and_mlp(self, bridge):
        """Each block should have attention and MLP subcomponents."""
        block = bridge.blocks[0]
        assert hasattr(block, "attn"), "Block should have attn"
        assert hasattr(block, "mlp"), "Block should have mlp"
        assert hasattr(block, "ln1"), "Block should have ln1"
        assert hasattr(block, "ln2"), "Block should have ln2"

    def test_embed_accessible(self, bridge):
        """Token embedding should be accessible."""
        assert hasattr(bridge, "embed"), "Bridge should have embed"

    def test_unembed_accessible(self, bridge):
        """Unembedding should be accessible."""
        assert hasattr(bridge, "unembed"), "Bridge should have unembed"

    def test_ln_final_accessible(self, bridge):
        """Final layer norm should be accessible."""
        assert hasattr(bridge, "ln_final"), "Bridge should have ln_final"

    def test_cfg_accessible(self, bridge):
        """Config should be accessible with expected fields."""
        cfg = bridge.cfg
        assert cfg.n_layers > 0
        assert cfg.n_heads > 0
        assert cfg.d_model > 0
        assert cfg.d_vocab > 0

    def test_tokenizer_accessible(self, bridge):
        """Tokenizer should be accessible."""
        assert bridge.tokenizer is not None
        tokens = bridge.to_tokens("Hello world")
        assert tokens.shape[0] == 1  # batch dim
        assert tokens.shape[1] > 0  # seq dim


class TestBridgeForwardPass:
    """Test that bridge produces valid outputs."""

    def test_forward_returns_logits(self, bridge):
        """Forward pass should return logits tensor."""
        with torch.no_grad():
            logits = bridge("Hello world", return_type="logits")
        assert logits.shape == (1, 3, bridge.cfg.d_vocab)  # "Hello world" = 3 tokens with BOS
        assert not torch.isnan(logits).any()
        assert not torch.isinf(logits).any()

    def test_forward_returns_loss(self, bridge):
        """Forward pass should return reasonable loss."""
        with torch.no_grad():
            loss = bridge("The cat sat on the mat", return_type="loss")
        assert loss.ndim == 0  # scalar
        assert 0 < loss.item() < 15

    def test_run_with_cache_returns_activations(self, bridge):
        """run_with_cache should return non-empty cache."""
        with torch.no_grad():
            _, cache = bridge.run_with_cache("Hello")
        assert len(cache) > 0
        # Should have block-level hooks
        block_keys = [k for k in cache.keys() if "blocks.0" in k]
        assert len(block_keys) > 0


class TestBridgeWeightProperties:
    """Test weight property accessors on bridge with compatibility mode."""

    def test_W_Q_shape(self, bridge_compat):
        """W_Q should have shape [n_layers, n_heads, d_model, d_head]."""
        W_Q = bridge_compat.W_Q
        cfg = bridge_compat.cfg
        assert W_Q.shape == (cfg.n_layers, cfg.n_heads, cfg.d_model, cfg.d_head)

    def test_W_K_shape(self, bridge_compat):
        """W_K should have shape [n_layers, n_heads, d_model, d_head]."""
        W_K = bridge_compat.W_K
        cfg = bridge_compat.cfg
        assert W_K.shape == (cfg.n_layers, cfg.n_heads, cfg.d_model, cfg.d_head)

    def test_W_V_shape(self, bridge_compat):
        """W_V should have shape [n_layers, n_heads, d_model, d_head]."""
        W_V = bridge_compat.W_V
        cfg = bridge_compat.cfg
        assert W_V.shape == (cfg.n_layers, cfg.n_heads, cfg.d_model, cfg.d_head)

    def test_W_O_shape(self, bridge_compat):
        """W_O should have shape [n_layers, n_heads, d_head, d_model]."""
        W_O = bridge_compat.W_O
        cfg = bridge_compat.cfg
        assert W_O.shape == (cfg.n_layers, cfg.n_heads, cfg.d_head, cfg.d_model)

    def test_QK_factored_matrix(self, bridge_compat):
        """QK property should return a functional FactoredMatrix."""
        QK = bridge_compat.QK
        assert QK is not None
        # FactoredMatrix should have A and B with correct shapes
        cfg = bridge_compat.cfg
        assert QK.A.shape == (cfg.n_layers, cfg.n_heads, cfg.d_model, cfg.d_head)
        assert QK.B.shape == (cfg.n_layers, cfg.n_heads, cfg.d_head, cfg.d_model)
        # Should be computable (not contain NaN)
        assert not torch.isnan(QK.A).any()
        assert not torch.isnan(QK.B).any()

    def test_OV_factored_matrix(self, bridge_compat):
        """OV property should return a functional FactoredMatrix."""
        OV = bridge_compat.OV
        assert OV is not None
        cfg = bridge_compat.cfg
        assert OV.A.shape == (cfg.n_layers, cfg.n_heads, cfg.d_model, cfg.d_head)
        assert OV.B.shape == (cfg.n_layers, cfg.n_heads, cfg.d_head, cfg.d_model)
        assert not torch.isnan(OV.A).any()
        assert not torch.isnan(OV.B).any()
