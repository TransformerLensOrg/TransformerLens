"""Unit tests for the Gemma 4 text-only architecture adapter."""

from transformer_lens.config.transformer_bridge_config import TransformerBridgeConfig
from transformer_lens.factories.architecture_adapter_factory import (
    ArchitectureAdapterFactory,
)
from transformer_lens.model_bridge.generalized_components import (
    DelegatedAttentionBlockBridge,
    EmbeddingBridge,
    LinearBridge,
    RotaryEmbeddingBridge,
    UnembeddingBridge,
)

ARCH = "Gemma4ForConditionalGeneration"


def _adapter():
    # Dimensions follow google/gemma-4-E2B's text_config.
    cfg = TransformerBridgeConfig(
        d_model=1536,
        d_head=256,
        n_heads=8,
        n_layers=35,
        n_ctx=131072,
        d_vocab=262144,
        n_key_value_heads=1,
        architecture=ARCH,
    )
    return ArchitectureAdapterFactory.select_architecture_adapter(cfg)


def test_config_flags():
    a = _adapter()
    # Text-only; PLE / layer_scalar / MoE residual topology is not fold-safe.
    assert a.cfg.is_multimodal is False
    assert a.supports_fold_ln is False
    assert a.weight_processing_conversions == {}
    assert a.cfg.normalization_type == "RMS"
    # Gemma4RMSNorm scales by weight directly — no (1 + weight) offset, unlike Gemma 1-3.
    assert a.cfg.rmsnorm_uses_offset is False
    assert a.cfg.positional_embedding_type == "rotary"
    assert a.applicable_phases == [1, 2, 4]


def test_text_path_nested_under_language_model():
    m = _adapter().component_mapping
    assert m["embed"].name == "model.language_model.embed_tokens"
    assert m["rotary_emb"].name == "model.language_model.rotary_emb"
    assert m["blocks"].name == "model.language_model.layers"
    assert m["ln_final"].name == "model.language_model.norm"
    assert m["unembed"].name == "lm_head"
    assert isinstance(m["embed"], EmbeddingBridge)
    assert isinstance(m["rotary_emb"], RotaryEmbeddingBridge)
    assert isinstance(m["blocks"], DelegatedAttentionBlockBridge)
    assert isinstance(m["unembed"], UnembeddingBridge)
    # Vision/audio towers are referenced-but-unbridged.
    assert "vision_encoder" not in m and "audio_encoder" not in m


def test_block_decomposition():
    blocks = _adapter().component_mapping["blocks"]
    for name in ("attn", "mlp"):
        assert name in blocks.submodules
    # Sandwich norms (same shape as Gemma 2/3) under canonical keys.
    for norm in ("ln1", "ln1_post", "ln2", "ln2_post"):
        assert norm in blocks.submodules
        assert blocks.submodules[norm].optional is False


def test_split_qkv_fork_aliases_absent():
    """Attention is delegated wholesale to HF; per-layer structure is heterogeneous
    (KV-shared layers have no k/v projections), so the split-qkv fork aliases
    do not apply."""
    blocks = _adapter().component_mapping["blocks"]
    for alias in ("hook_q_input", "hook_k_input", "hook_v_input", "hook_attn_in"):
        assert alias not in blocks.hook_aliases
    # The single-stream residual aliases remain, redirected through the sandwich norms.
    assert blocks.hook_aliases["hook_resid_mid"] == "ln2.hook_in"
    assert blocks.hook_aliases["hook_attn_out"] == "ln1_post.hook_out"
    assert blocks.hook_aliases["hook_mlp_out"] == "ln2_post.hook_out"


def test_kv_shared_and_k_eq_v_submodules_are_optional():
    """KV-shared layers (E2B/E4B) drop k/v proj + norms; K==V global-attention
    layers (31B / 26B-A4B) drop v_proj."""
    attn = _adapter().component_mapping["blocks"].submodules["attn"]
    assert attn.submodules["q"].optional is False
    assert attn.submodules["o"].optional is False
    assert attn.submodules["q_norm"].optional is False
    for shared in ("k", "v", "k_norm", "v_norm"):
        assert attn.submodules[shared].optional is True
    assert isinstance(attn.submodules["q"], LinearBridge)


def test_per_layer_embedding_submodules_are_optional():
    """PLE modules exist only when hidden_size_per_layer_input > 0 (E2B/E4B)."""
    blocks = _adapter().component_mapping["blocks"]
    for name in (
        "per_layer_input_gate",
        "per_layer_projection",
        "post_per_layer_input_norm",
    ):
        assert blocks.submodules[name].optional is True


def test_moe_submodules_are_optional():
    """MoE branch exists only when enable_moe_block (26B-A4B)."""
    blocks = _adapter().component_mapping["blocks"]
    for name in (
        "router",
        "experts",
        "pre_feedforward_layernorm_2",
        "post_feedforward_layernorm_1",
        "post_feedforward_layernorm_2",
    ):
        assert blocks.submodules[name].optional is True


def test_gated_mlp_decomposition():
    mlp = _adapter().component_mapping["blocks"].submodules["mlp"]
    assert mlp.submodules["gate"].name == "gate_proj"
    assert mlp.submodules["in"].name == "up_proj"
    assert mlp.submodules["out"].name == "down_proj"
