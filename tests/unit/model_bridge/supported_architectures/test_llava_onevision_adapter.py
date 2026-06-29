"""Unit tests for LlavaOnevisionArchitectureAdapter.

LlavaOnevisionArchitectureAdapter inherits its config, component mapping, and
weight conversions from LlavaArchitectureAdapter (covered by test_llava_adapter.py).
This suite pins the subclass contract and the prepare_model weight-tying override.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from transformer_lens.config.transformer_bridge_config import TransformerBridgeConfig
from transformer_lens.model_bridge.supported_architectures.llava import (
    LlavaArchitectureAdapter,
)
from transformer_lens.model_bridge.supported_architectures.llava_onevision import (
    LlavaOnevisionArchitectureAdapter,
)

# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------


def _make_cfg(vision_model_type: str = "clip_vision_model") -> TransformerBridgeConfig:
    """Return a minimal TransformerBridgeConfig for LLaVA-OneVision tests."""
    cfg = TransformerBridgeConfig(
        d_model=64,
        d_head=16,
        n_heads=4,
        n_layers=2,
        n_ctx=512,
        d_vocab=1000,
        architecture="LlavaOnevisionForConditionalGeneration",
    )
    cfg.vision_config = SimpleNamespace(
        model_type=vision_model_type,
        hidden_size=256,
        num_hidden_layers=4,
        num_attention_heads=4,
    )
    return cfg


@pytest.fixture
def adapter() -> LlavaOnevisionArchitectureAdapter:
    return LlavaOnevisionArchitectureAdapter(_make_cfg())


# ---------------------------------------------------------------------------
# Inheritance tests
# ---------------------------------------------------------------------------


class TestLlavaOnevisionInheritance:
    """LlavaOnevisionArchitectureAdapter must be a LlavaArchitectureAdapter subclass."""

    def test_subclass_of_llava(self) -> None:
        assert issubclass(LlavaOnevisionArchitectureAdapter, LlavaArchitectureAdapter)

    def test_instance_of_llava(self, adapter: LlavaOnevisionArchitectureAdapter) -> None:
        assert isinstance(adapter, LlavaArchitectureAdapter)


# ---------------------------------------------------------------------------
# prepare_model weight-tying tests
# ---------------------------------------------------------------------------


class TestLlavaOnevisionPrepareModel:
    """prepare_model fixes weight tying when text_config and top-level config disagree."""

    def _make_hf_model(
        self, tie_word_embeddings_text: bool, tie_word_embeddings_top: bool
    ) -> MagicMock:
        """Build a minimal mock HF model for prepare_model testing."""
        embed = MagicMock()
        embed.weight = "original_weight"

        language_model = MagicMock()
        language_model.embed_tokens = embed

        model = MagicMock()
        model.language_model = language_model

        lm_head = MagicMock()
        lm_head.weight = "random_weight"

        text_config = SimpleNamespace(tie_word_embeddings=tie_word_embeddings_text)
        config = SimpleNamespace(
            tie_word_embeddings=tie_word_embeddings_top,
            text_config=text_config,
        )

        hf_model = MagicMock()
        hf_model.model = model
        hf_model.lm_head = lm_head
        hf_model.config = config
        return hf_model

    def test_ties_weights_when_text_config_says_tied_but_top_level_says_not(
        self, adapter: LlavaOnevisionArchitectureAdapter
    ) -> None:
        """lm_head.weight should be set to embed.weight when text_config disagrees."""
        hf_model = self._make_hf_model(tie_word_embeddings_text=True, tie_word_embeddings_top=False)
        adapter.prepare_model(hf_model)
        assert hf_model.lm_head.weight == "original_weight"

    def test_no_tying_when_both_agree_tied(
        self, adapter: LlavaOnevisionArchitectureAdapter
    ) -> None:
        """No weight override when top-level config already says tied."""
        hf_model = self._make_hf_model(tie_word_embeddings_text=True, tie_word_embeddings_top=True)
        original_weight = hf_model.lm_head.weight
        adapter.prepare_model(hf_model)
        assert hf_model.lm_head.weight == original_weight

    def test_no_tying_when_text_config_says_not_tied(
        self, adapter: LlavaOnevisionArchitectureAdapter
    ) -> None:
        """No weight override when text_config says not tied."""
        hf_model = self._make_hf_model(
            tie_word_embeddings_text=False, tie_word_embeddings_top=False
        )
        original_weight = hf_model.lm_head.weight
        adapter.prepare_model(hf_model)
        assert hf_model.lm_head.weight == original_weight

    def test_no_op_when_no_lm_head(self, adapter: LlavaOnevisionArchitectureAdapter) -> None:
        """prepare_model is a no-op when lm_head is absent."""
        hf_model = MagicMock(spec=[])  # no attributes at all
        adapter.prepare_model(hf_model)  # must not raise
