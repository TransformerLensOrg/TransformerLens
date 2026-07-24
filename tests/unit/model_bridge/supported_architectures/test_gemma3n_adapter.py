"""Unit tests for the Gemma 3n text-only architecture adapter."""

import pytest

from transformer_lens.config.transformer_bridge_config import TransformerBridgeConfig
from transformer_lens.factories.architecture_adapter_factory import (
    ArchitectureAdapterFactory,
)
from transformer_lens.model_bridge.generalized_components import (
    AltUpBlockBridge,
    EmbeddingBridge,
    LinearBridge,
    UnembeddingBridge,
)
from transformer_lens.model_bridge.supported_architectures.gemma3n import (
    Gemma3nArchitectureAdapter,
)

ARCH = "Gemma3nForConditionalGeneration"


def _adapter():
    cfg = TransformerBridgeConfig(
        d_model=2048,
        d_head=256,
        n_heads=8,
        n_layers=30,
        n_ctx=4096,
        d_vocab=262400,
        n_key_value_heads=2,
        architecture=ARCH,
    )
    return ArchitectureAdapterFactory.select_architecture_adapter(cfg)


def test_missing_required_library_raises_actionable_error():
    """A missing multimodal dep surfaces a clear error, not a deep HF import failure."""

    class _FakeMissing(Gemma3nArchitectureAdapter):
        required_libraries = ["definitely_not_installed_xyz"]
        required_libraries_group = "custom-group"

    cfg = TransformerBridgeConfig(
        d_model=2048,
        d_head=256,
        n_heads=8,
        n_layers=30,
        n_ctx=4096,
        d_vocab=262400,
        n_key_value_heads=2,
        architecture=ARCH,
    )
    # The error names the missing lib and the adapter's declared dependency group (not a
    # hardcoded one).
    with pytest.raises(ImportError, match=r"definitely_not_installed_xyz.*custom-group"):
        _FakeMissing(cfg)


def test_config_flags():
    a = _adapter()
    # Text-only for now; AltUp/PLE topology is not fold-safe.
    assert a.cfg.is_multimodal is False
    assert a.supports_fold_ln is False
    assert a.weight_processing_conversions == {}
    assert a.cfg.normalization_type == "RMS"
    assert a.cfg.rmsnorm_uses_offset is True


def test_text_path_nested_under_language_model():
    m = _adapter().component_mapping
    assert m["embed"].name == "model.language_model.embed_tokens"
    assert m["blocks"].name == "model.language_model.layers"
    assert m["ln_final"].name == "model.language_model.norm"
    assert m["unembed"].name == "lm_head"
    assert isinstance(m["embed"], EmbeddingBridge)
    assert isinstance(m["blocks"], AltUpBlockBridge)
    assert isinstance(m["unembed"], UnembeddingBridge)
    # Vision/audio are referenced but not bridged (text-only adapter for now).
    assert "vision_encoder" not in m and "audio_encoder" not in m


def test_altup_block_decomposition():
    blocks = _adapter().component_mapping["blocks"]
    assert blocks.altup_active_idx == 0
    # AltUp/LAuReL/PLE submodules present alongside attn + mlp + the five norms.
    for name in (
        "altup",
        "laurel",
        "per_layer_input_gate",
        "per_layer_projection",
        "self_attn",
        "mlp",
    ):
        assert name in blocks.submodules
    for norm in (
        "input_layernorm",
        "post_attention_layernorm",
        "pre_feedforward_layernorm",
        "post_feedforward_layernorm",
        "post_per_layer_input_norm",
    ):
        assert norm in blocks.submodules


def test_kv_shared_submodules_are_optional():
    """The last num_kv_shared_layers layers drop their own k/v proj + norms."""
    attn = _adapter().component_mapping["blocks"].submodules["self_attn"]
    assert attn.submodules["q"].optional is False
    assert attn.submodules["o"].optional is False
    for shared in ("k", "v", "k_norm", "v_norm"):
        assert attn.submodules[shared].optional is True
    assert isinstance(attn.submodules["q"], LinearBridge)
