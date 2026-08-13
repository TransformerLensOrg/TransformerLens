"""Unit tests for the target gate on mask-derived ``position_ids`` injection.

``TransformerBridge.forward`` derives ``position_ids`` from ``attention_mask``
so left-padded input gets the right absolute positions (see #1609). That kwarg
is only safe for models that both accept it and do not derive positions
themselves, so the injection is gated the same way ``output_attentions`` is in
``run_with_cache``. The gate is exercised directly here with stand-in modules:
the models it exists to protect (fixed-signature remote code, mRoPE) are either
never loaded in CI or belong to the integration tier.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Optional

import torch
import torch.nn as nn

from transformer_lens.model_bridge import TransformerBridge

# Unbound so it can run against a stand-in that owns only ``_driver``; the gate
# reads nothing else off the bridge.
gate = TransformerBridge._accepts_derived_position_ids


def _bridge_over(model: Optional[nn.Module]) -> Any:
    return SimpleNamespace(_driver=SimpleNamespace(underlying_model=model))


class _FixedSignature(nn.Module):
    """Mirrors ``LLaDAModelLM.forward``: no ``position_ids``, no ``**kwargs``."""

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        use_cache: bool = False,
    ) -> torch.Tensor:
        return input_ids


class _AcceptsPositionIds(nn.Module):
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return input_ids


class _AcceptsKwargs(nn.Module):
    def forward(self, input_ids: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        return input_ids


class _MaskConsumingEmbedding(nn.Embedding):
    """Mirrors ``OPTLearnedPositionalEmbedding``: positions come from the mask."""

    def forward(  # type: ignore[override]
        self,
        attention_mask: torch.Tensor,
        past_key_values_length: int = 0,
        position_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if position_ids is None:
            position_ids = (attention_mask.cumsum(1) * attention_mask - 1).long()
        return super().forward(position_ids)


class _OwnsPositions(_AcceptsPositionIds):
    """mRoPE models compute a 3-D index here, but only while position_ids is None."""

    def get_rope_index(self, *args: Any, **kwargs: Any) -> None:
        return None


class TestSignatureGate:
    def test_refuses_model_that_cannot_take_position_ids(self) -> None:
        """Injecting into a fixed-signature remote-code forward raises TypeError
        where the model previously returned logits."""
        assert gate(_bridge_over(_FixedSignature())) is False

    def test_allows_explicit_position_ids_parameter(self) -> None:
        assert gate(_bridge_over(_AcceptsPositionIds())) is True

    def test_allows_var_keyword_forward(self) -> None:
        """**kwargs forwards pass the kwarg through to the inner model."""
        assert gate(_bridge_over(_AcceptsKwargs())) is True


class TestOwnsPositionsGate:
    def test_refuses_model_defining_get_rope_index(self) -> None:
        assert gate(_bridge_over(_OwnsPositions())) is False

    def test_refuses_wrapper_whose_inner_model_owns_positions(self) -> None:
        """get_rope_index lives on the inner text model, while original_model is
        usually the ForConditionalGeneration wrapper around it."""
        wrapper = _AcceptsPositionIds()
        wrapper.model = _OwnsPositions()
        assert gate(_bridge_over(wrapper)) is False

    def test_refuses_wrapper_whose_language_model_owns_positions(self) -> None:
        wrapper = _AcceptsPositionIds()
        wrapper.language_model = _OwnsPositions()
        assert gate(_bridge_over(wrapper)) is False

    def test_refuses_mrope_section_in_config(self) -> None:
        """Config-level backstop: the section list is what makes positions 3-D."""
        model = _AcceptsPositionIds()
        model.config = SimpleNamespace(  # type: ignore[assignment]
            rope_scaling={"mrope_section": [1, 1, 2], "rope_type": "default"}
        )
        assert gate(_bridge_over(model)) is False

    def test_refuses_mrope_section_in_text_config(self) -> None:
        """Multimodal configs nest the text model's rope_scaling one level down."""
        model = _AcceptsPositionIds()
        model.config = SimpleNamespace(  # type: ignore[assignment]
            rope_scaling=None,
            text_config=SimpleNamespace(rope_scaling={"mrope_section": [1, 1, 2]}),
        )
        assert gate(_bridge_over(model)) is False

    def test_refuses_mask_consuming_positional_embedding(self) -> None:
        """OPT's OPTLearnedPositionalEmbedding takes the mask and derives its own
        positions, including its own convention for the padded slots."""
        model = _AcceptsPositionIds()
        model.embed_positions = _MaskConsumingEmbedding(8, 4)
        assert gate(_bridge_over(model)) is False

    def test_ordinary_embeddings_do_not_trip_the_scan(self) -> None:
        """nn.Embedding takes only indices, so the common case stays injectable."""
        model = _AcceptsPositionIds()
        model.wte = nn.Embedding(8, 4)
        model.wpe = nn.Embedding(8, 4)
        assert gate(_bridge_over(model)) is True

    def test_plain_rope_scaling_is_not_treated_as_mrope(self) -> None:
        """Only mrope_section means a multi-stream index; yarn/linear do not."""
        model = _AcceptsPositionIds()
        model.config = SimpleNamespace(  # type: ignore[assignment]
            rope_scaling={"rope_type": "yarn", "factor": 8.0}
        )
        assert gate(_bridge_over(model)) is True


class TestDriverAndCaching:
    def test_refuses_driver_without_a_local_module(self) -> None:
        """vLLM/Inspect expose no module to introspect and own positions internally."""
        assert gate(_bridge_over(None)) is False

    def test_recomputes_when_the_underlying_model_is_swapped(self) -> None:
        """Weight processing replaces original_model, so a cached verdict keyed on
        the old module must not survive."""
        bridge = _bridge_over(_AcceptsPositionIds())
        assert gate(bridge) is True
        bridge._driver.underlying_model = _FixedSignature()
        assert gate(bridge) is False
