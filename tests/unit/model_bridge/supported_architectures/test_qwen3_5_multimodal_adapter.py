"""Unit tests for the Qwen3.5 multimodal (vision-language) architecture adapter."""

from types import SimpleNamespace

import pytest
import torch

from transformer_lens.config.transformer_bridge_config import TransformerBridgeConfig
from transformer_lens.factories.architecture_adapter_factory import (
    SUPPORTED_ARCHITECTURES,
    ArchitectureAdapterFactory,
)
from transformer_lens.model_bridge.generalized_components import (
    BlockBridge,
    EmbeddingBridge,
    GatedMLPBridge,
    LinearBridge,
    Qwen3_5VisionBlockBridge,
    Qwen3_5VisionEncoderBridge,
    RMSNormalizationBridge,
    RotaryEmbeddingBridge,
    UnembeddingBridge,
    VisionProjectionBridge,
)
from transformer_lens.model_bridge.generalized_components.gated_delta_net import (
    GatedDeltaNetBridge,
)
from transformer_lens.model_bridge.generalized_components.position_embeddings_attention import (
    PositionEmbeddingsAttentionBridge,
)
from transformer_lens.model_bridge.supported_architectures.qwen3_5_multimodal import (
    Qwen3_5MultimodalArchitectureAdapter,
)
from transformer_lens.tools.model_registry import (
    CANONICAL_AUTHORS_BY_ARCH,
    HF_SUPPORTED_ARCHITECTURES,
)

ARCH = "Qwen3_5ForConditionalGeneration"


def _make_qwen3_5_mm_cfg(with_vision_config: bool = True, **overrides):
    """Create a TransformerBridgeConfig mirroring a Qwen3.5 multimodal checkpoint."""
    defaults = dict(
        d_model=16,
        d_head=256,
        n_heads=4,
        n_layers=2,
        n_ctx=4096,
        d_vocab=248320,
        n_key_value_heads=2,
        architecture=ARCH,
    )
    defaults.update(overrides)
    cfg = TransformerBridgeConfig(**defaults)
    if with_vision_config:
        # Qwen vision config uses depth/num_heads (not num_hidden_layers/num_attention_heads).
        cfg.vision_config = SimpleNamespace(hidden_size=16, depth=2, num_heads=4)
    return cfg


class TestQwen3_5MultimodalRegistration:
    """The adapter must be wired into all the registration surfaces."""

    def test_architecture_in_supported_architectures(self):
        assert ARCH in SUPPORTED_ARCHITECTURES

    def test_architecture_maps_to_correct_adapter(self):
        assert SUPPORTED_ARCHITECTURES[ARCH] is Qwen3_5MultimodalArchitectureAdapter

    def test_factory_selects_correct_adapter(self):
        adapter = ArchitectureAdapterFactory.select_architecture_adapter(_make_qwen3_5_mm_cfg())
        assert isinstance(adapter, Qwen3_5MultimodalArchitectureAdapter)

    def test_in_hf_supported_and_canonical_authors(self):
        assert ARCH in HF_SUPPORTED_ARCHITECTURES
        assert CANONICAL_AUTHORS_BY_ARCH.get(ARCH) == ["Qwen"]


class TestQwen3_5MultimodalAdapterConfig:
    """Config flags set by the adapter."""

    @pytest.fixture(scope="class")
    def adapter(self):
        return Qwen3_5MultimodalArchitectureAdapter(_make_qwen3_5_mm_cfg())

    def test_is_multimodal(self, adapter):
        assert adapter.cfg.is_multimodal is True

    def test_gated_q_proj(self, adapter):
        assert adapter.cfg.gated_q_proj is True

    def test_hybrid_disables_fold_ln(self, adapter):
        # Hybrid (GatedDeltaNet) variants cannot fold LN.
        assert adapter.supports_fold_ln is False

    def test_no_weight_processing_conversions(self, adapter):
        # Hybrid path uses preprocess_weights, not declarative conversions.
        assert adapter.weight_processing_conversions == {}

    def test_uses_rms_norm_and_rotary(self, adapter):
        assert adapter.cfg.uses_rms_norm is True
        assert adapter.cfg.normalization_type == "RMS"
        assert adapter.cfg.positional_embedding_type == "rotary"

    def test_vision_config_extracted(self, adapter):
        assert adapter.cfg.vision_hidden_size == 16
        assert adapter.cfg.vision_num_layers == 2
        assert adapter.cfg.vision_num_heads == 4


class TestQwen3_5MultimodalComponentMapping:
    """Top-level slots, HF paths, and bridge types (refactor-drift guard)."""

    @pytest.fixture(scope="class")
    def adapter(self):
        return Qwen3_5MultimodalArchitectureAdapter(_make_qwen3_5_mm_cfg())

    def test_has_vision_and_language_components(self, adapter):
        for name in (
            "vision_encoder",
            "vision_projector",
            "embed",
            "rotary_emb",
            "blocks",
            "ln_final",
            "unembed",
        ):
            assert name in adapter.component_mapping

    def test_vision_paths(self, adapter):
        assert adapter.component_mapping["vision_encoder"].name == "model.visual"
        # The Qwen3.5 merger is the vision->text projector.
        assert adapter.component_mapping["vision_projector"].name == "model.visual.merger"

    def test_language_paths_nested_under_language_model(self, adapter):
        assert adapter.component_mapping["embed"].name == "model.language_model.embed_tokens"
        assert adapter.component_mapping["rotary_emb"].name == "model.language_model.rotary_emb"
        assert adapter.component_mapping["blocks"].name == "model.language_model.layers"
        assert adapter.component_mapping["ln_final"].name == "model.language_model.norm"
        # lm_head stays top-level (not nested).
        assert adapter.component_mapping["unembed"].name == "lm_head"

    def test_component_types(self, adapter):
        m = adapter.component_mapping
        assert isinstance(m["vision_encoder"], Qwen3_5VisionEncoderBridge)
        assert isinstance(m["vision_projector"], VisionProjectionBridge)
        assert isinstance(m["embed"], EmbeddingBridge)
        assert isinstance(m["rotary_emb"], RotaryEmbeddingBridge)
        assert isinstance(m["blocks"], BlockBridge)
        assert isinstance(m["ln_final"], RMSNormalizationBridge)
        assert isinstance(m["unembed"], UnembeddingBridge)


class TestQwen3_5MultimodalLanguageBlocks:
    """Hybrid language blocks: optional full attention + GatedDeltaNet linear attention."""

    @pytest.fixture(scope="class")
    def blocks(self):
        return Qwen3_5MultimodalArchitectureAdapter(_make_qwen3_5_mm_cfg()).component_mapping[
            "blocks"
        ]

    def test_hybrid_block_submodules(self, blocks):
        for name in ("ln1", "ln2", "attn", "mlp", "linear_attn"):
            assert name in blocks.submodules, f"missing block submodule '{name}'"

    def test_linear_attn_is_gated_delta_net(self, blocks):
        assert isinstance(blocks.submodules["linear_attn"], GatedDeltaNetBridge)

    def test_attention_is_optional_with_qk_norm(self, blocks):
        attn = blocks.submodules["attn"]
        assert isinstance(attn, PositionEmbeddingsAttentionBridge)
        assert attn.optional is True  # hybrid: not every layer has full attention
        for sub in ("q", "k", "v", "o", "q_norm", "k_norm"):
            assert sub in attn.submodules

    def test_mlp_is_gated(self, blocks):
        assert isinstance(blocks.submodules["mlp"], GatedMLPBridge)


class TestQwen3_5VisionBridge:
    """The decomposed vision tower mapping (model.visual)."""

    @pytest.fixture(scope="class")
    def vision(self):
        return Qwen3_5MultimodalArchitectureAdapter(_make_qwen3_5_mm_cfg()).component_mapping[
            "vision_encoder"
        ]

    def test_vision_encoder_submodules(self, vision):
        for name in ("patch_embed", "pos_embed", "blocks"):
            assert name in vision.submodules

    def test_vision_blocks_decomposed(self, vision):
        blocks = vision.submodules["blocks"]
        assert isinstance(blocks, Qwen3_5VisionBlockBridge)
        for name in ("norm1", "norm2", "attn", "mlp"):
            assert name in blocks.submodules

    def test_vision_attn_and_mlp_submodules(self, vision):
        blocks = vision.submodules["blocks"]
        assert {"qkv", "proj"} <= set(blocks.submodules["attn"].submodules)
        assert {"linear_fc1", "linear_fc2"} <= set(blocks.submodules["mlp"].submodules)
        for comp in (
            blocks.submodules["attn"].submodules["qkv"],
            blocks.submodules["attn"].submodules["proj"],
            blocks.submodules["mlp"].submodules["linear_fc1"],
            blocks.submodules["mlp"].submodules["linear_fc2"],
        ):
            assert isinstance(comp, LinearBridge)


class TestQwen3_5MultimodalPreprocessWeights:
    """Gated q_proj slicing applies under the nested language-model path."""

    def test_gated_q_proj_query_half_is_sliced(self):
        adapter = Qwen3_5MultimodalArchitectureAdapter(_make_qwen3_5_mm_cfg())
        n_heads, d_head = adapter.cfg.n_heads, adapter.cfg.d_head
        hidden = adapter.cfg.d_model
        key = "model.language_model.layers.1.self_attn.q_proj.weight"
        # Per head: rows [query(d_head), gate(d_head)] -> 2*d_head wide.
        full = torch.randn(n_heads * d_head * 2, hidden)
        out = adapter.preprocess_weights({key: full.clone()})
        assert out[key].shape == (n_heads * d_head, hidden)
        # Recovered rows must be the query half of each head.
        expected = full.view(n_heads, d_head * 2, hidden)[:, :d_head, :].reshape(
            n_heads * d_head, hidden
        )
        assert torch.equal(out[key], expected)
