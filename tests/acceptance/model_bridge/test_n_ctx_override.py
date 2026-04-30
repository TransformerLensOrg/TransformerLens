"""Tests for the n_ctx override parameter on TransformerBridge.boot_transformers().

Uses load_weights=False so we can verify config plumbing without fighting HF's
weight-loading checks. Models with learned positional embeddings (e.g. GPT-2)
cannot have their n_ctx reduced at weight-load time — only rotary models can
freely resize. These tests verify the config is written correctly; users are
responsible for choosing n_ctx values their model supports.
"""

import logging

import pytest

from transformer_lens.model_bridge import TransformerBridge


def test_n_ctx_override_writes_to_correct_hf_field():
    """For GPT-2 the field is n_positions — overriding n_ctx should update it."""
    bridge = TransformerBridge.boot_transformers(
        "gpt2", device="cpu", n_ctx=256, load_weights=False
    )
    assert bridge.cfg.n_ctx == 256
    assert bridge.original_model.config.n_positions == 256


def test_n_ctx_default_uses_model_max():
    """Without an override, cfg.n_ctx reflects the HF config's value."""
    bridge = TransformerBridge.boot_transformers("gpt2", device="cpu", load_weights=False)
    # GPT-2's n_positions default is 1024
    assert bridge.cfg.n_ctx == 1024


def test_n_ctx_warns_when_above_default(caplog):
    """Overriding n_ctx above the model default should emit a logging.warning."""
    with caplog.at_level(logging.WARNING):
        TransformerBridge.boot_transformers("gpt2", device="cpu", n_ctx=2048, load_weights=False)
    assert any(
        "larger than the model's default context length" in rec.message for rec in caplog.records
    )


def test_n_ctx_combined_with_hf_config_overrides():
    """Explicit n_ctx should take precedence over hf_config_overrides for that field."""
    bridge = TransformerBridge.boot_transformers(
        "gpt2",
        device="cpu",
        n_ctx=256,
        hf_config_overrides={"n_positions": 512},  # should be overridden by n_ctx=256
        load_weights=False,
    )
    assert bridge.cfg.n_ctx == 256


# --- Coverage for code-review items #2, #4, #5, #7 ---


def test_n_ctx_zero_raises_value_error():
    """#2: n_ctx must be positive; zero should raise ValueError."""
    with pytest.raises(ValueError, match="positive integer"):
        TransformerBridge.boot_transformers("gpt2", device="cpu", n_ctx=0, load_weights=False)


def test_n_ctx_negative_raises_value_error():
    """#2: n_ctx must be positive; negative should raise ValueError."""
    with pytest.raises(ValueError, match="positive integer"):
        TransformerBridge.boot_transformers("gpt2", device="cpu", n_ctx=-1, load_weights=False)


def test_n_ctx_conflict_with_hf_config_overrides_warns(caplog):
    """#4: When both n_ctx and the same hf_config_overrides field are set with different values,
    a warning should be emitted explaining that n_ctx wins."""
    with caplog.at_level(logging.WARNING):
        TransformerBridge.boot_transformers(
            "gpt2",
            device="cpu",
            n_ctx=256,
            hf_config_overrides={"n_positions": 512},
            load_weights=False,
        )
    assert any(
        "Both n_ctx=256 and hf_config_overrides['n_positions']" in rec.message
        and "takes precedence" in rec.message
        for rec in caplog.records
    )


def test_n_ctx_no_conflict_when_values_match(caplog):
    """#4: If n_ctx and hf_config_overrides agree on the value, no conflict warning is emitted."""
    with caplog.at_level(logging.WARNING):
        TransformerBridge.boot_transformers(
            "gpt2",
            device="cpu",
            n_ctx=256,
            hf_config_overrides={"n_positions": 256},  # same as n_ctx
            load_weights=False,
        )
    assert not any("takes precedence" in rec.message for rec in caplog.records)


def test_n_ctx_shrink_with_load_weights_gives_clear_error():
    """#5: Shrinking a learned-pos-embed model's n_ctx at weight-load time should raise
    with a message explaining the cause and suggesting alternatives."""
    with pytest.raises(RuntimeError) as exc_info:
        TransformerBridge.boot_transformers("gpt2", device="cpu", n_ctx=256, load_weights=True)
    err = str(exc_info.value)
    assert "n_ctx=256" in err
    assert "learned positional embeddings" in err or "load_weights=False" in err


def test_n_ctx_override_verified_on_loaded_model():
    """#7: After load, the override should be visible on hf_model.config so users
    can trust that the longer/shorter context is actually in effect."""
    bridge = TransformerBridge.boot_transformers(
        "gpt2", device="cpu", n_ctx=2048, load_weights=False
    )
    # The override persisted through model construction
    assert bridge.original_model.config.n_positions == 2048
