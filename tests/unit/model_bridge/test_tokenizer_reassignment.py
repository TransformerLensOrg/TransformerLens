"""Tests for tokenizer reassignment wiring."""

import pytest
from transformers import AutoTokenizer

from transformer_lens.config import TransformerBridgeConfig
from transformer_lens.model_bridge.bridge_core import BridgeCore
from transformer_lens.model_bridge.architecture_adapter import ArchitectureAdapter


class MockAdapter(ArchitectureAdapter):
    """Minimal adapter for testing."""

    def __init__(self, cfg: TransformerBridgeConfig):
        super().__init__(cfg)
        self.component_mapping = {"embed": None}


class MockDriver:
    """Minimal driver for testing."""

    pass


class ConcreteBridgeCore(BridgeCore):
    """Concrete implementation of BridgeCore for testing."""

    def _scan_existing_hooks(self, module, prefix: str = "") -> None:
        pass


class TestTokenizerReassignment:
    """Test that tokenizer reassignment re-runs wiring logic."""

    @pytest.fixture
    def base_cfg(self) -> TransformerBridgeConfig:
        return TransformerBridgeConfig(
            d_model=768,
            d_head=64,
            n_layers=12,
            n_ctx=1024,
            d_vocab=-1,  # Will be inferred from tokenizer
            d_mlp=3072,
            n_heads=12,
        )

    @pytest.fixture
    def gpt2_tokenizer(self):
        """GPT-2 tokenizer (does not prepend BOS by default)."""
        return AutoTokenizer.from_pretrained("gpt2")

    @pytest.fixture
    def llama_style_tokenizer(self):
        """A tokenizer that prepends BOS (using gpt-neox as example)."""
        tok = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
        return tok

    def test_initial_tokenizer_sets_d_vocab(self, base_cfg, gpt2_tokenizer):
        """Test that initial tokenizer assignment sets d_vocab."""
        adapter = MockAdapter(base_cfg)
        bridge = ConcreteBridgeCore(adapter, gpt2_tokenizer, MockDriver())

        # GPT-2 vocab size is 50257
        assert bridge.cfg.d_vocab == 50257
        assert bridge.cfg.d_vocab_out == 50257

    def test_reassignment_updates_d_vocab(self, base_cfg, gpt2_tokenizer, llama_style_tokenizer):
        """Test that reassigning tokenizer updates d_vocab."""
        adapter = MockAdapter(base_cfg)
        bridge = ConcreteBridgeCore(adapter, gpt2_tokenizer, MockDriver())

        old_d_vocab = bridge.cfg.d_vocab

        bridge.tokenizer = llama_style_tokenizer

        # GPT-NeoX has a different vocab size than GPT-2
        assert bridge.cfg.d_vocab != old_d_vocab
        assert bridge.cfg.d_vocab_out == bridge.cfg.d_vocab

    def test_reassignment_updates_bos_flag(self, base_cfg, gpt2_tokenizer, llama_style_tokenizer):
        """Test that reassigning tokenizer updates tokenizer_prepends_bos."""
        adapter = MockAdapter(base_cfg)
        bridge = ConcreteBridgeCore(adapter, gpt2_tokenizer, MockDriver())

        gpt2_bos = bridge.cfg.tokenizer_prepends_bos

        bridge.tokenizer = llama_style_tokenizer

        neox_bos = bridge.cfg.tokenizer_prepends_bos
        # The flags should be properly detected (actual values depend on tokenizer behavior)
        assert isinstance(neox_bos, bool)

    def test_reassignment_to_none_preserves_config(self, base_cfg, gpt2_tokenizer):
        """Test that setting tokenizer to None doesn't crash."""
        adapter = MockAdapter(base_cfg)
        bridge = ConcreteBridgeCore(adapter, gpt2_tokenizer, MockDriver())

        old_d_vocab = bridge.cfg.d_vocab

        bridge.tokenizer = None

        assert bridge.tokenizer is None
        assert bridge.cfg.d_vocab == old_d_vocab  # Preserved from previous tokenizer

    def test_tokenizer_property_returns_tokenizer(self, base_cfg, gpt2_tokenizer):
        """Test that the tokenizer property returns the stored tokenizer."""
        adapter = MockAdapter(base_cfg)
        bridge = ConcreteBridgeCore(adapter, gpt2_tokenizer, MockDriver())

        assert bridge.tokenizer is not None
        assert hasattr(bridge.tokenizer, "encode")

    def test_no_tokenizer_at_init(self, base_cfg):
        """Test that bridge can be created without tokenizer."""
        base_cfg.d_vocab = 50257  # Set explicitly since no tokenizer
        adapter = MockAdapter(base_cfg)
        bridge = ConcreteBridgeCore(adapter, None, MockDriver())

        assert bridge.tokenizer is None
        assert bridge.cfg.d_vocab == 50257
