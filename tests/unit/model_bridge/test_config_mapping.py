"""Unit tests for map_default_transformer_lens_config field precedence.

Style mirrors tests/unit/model_bridge/test_output_logits_contract.py: plain
SimpleNamespace HF-config stand-ins, no network access.
"""

from types import SimpleNamespace

from transformer_lens.model_bridge.sources.transformers import (
    map_default_transformer_lens_config,
)


def _hf_config(**overrides: object) -> SimpleNamespace:
    values: dict[str, object] = {
        "hidden_size": 16,
        "num_attention_heads": 4,
        "num_hidden_layers": 2,
        "vocab_size": 32,
        "max_position_embeddings": 64,
        "intermediate_size": 32,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_explicit_head_dim_wins_over_derived() -> None:
    """head_dim need not equal hidden_size // num_attention_heads (16 // 4 = 4)."""
    mapped = map_default_transformer_lens_config(_hf_config(head_dim=8))
    assert mapped.d_head == 8


def test_d_head_falls_back_to_derived_when_head_dim_absent() -> None:
    mapped = map_default_transformer_lens_config(_hf_config())
    assert mapped.d_head == 16 // 4


def test_d_head_falls_back_to_derived_when_head_dim_is_none() -> None:
    """HF configs may carry head_dim=None; treat it the same as absent."""
    mapped = map_default_transformer_lens_config(_hf_config(head_dim=None))
    assert mapped.d_head == 16 // 4
