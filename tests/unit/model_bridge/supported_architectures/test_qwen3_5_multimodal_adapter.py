"""Unit tests for the Qwen3.5 multimodal (vision-language) architecture adapter.

Scoped to what is multimodal-specific. The hybrid text-block internals are built by the
shared Qwen3 base and are covered by test_qwen3_5_adapter.py / test_qwen3_adapter.py; the
end-to-end correctness of the wiring is covered by the verify suite + the integration test.
"""

from types import SimpleNamespace

import torch

from transformer_lens.config.transformer_bridge_config import TransformerBridgeConfig
from transformer_lens.factories.architecture_adapter_factory import (
    SUPPORTED_ARCHITECTURES,
    ArchitectureAdapterFactory,
)
from transformer_lens.model_bridge.generalized_components import (
    LinearBridge,
    Qwen3_5VisionBlockBridge,
    Qwen3_5VisionEncoderBridge,
    VisionProjectionBridge,
)
from transformer_lens.model_bridge.supported_architectures.qwen3_5_multimodal import (
    Qwen3_5MultimodalArchitectureAdapter,
)
from transformer_lens.tools.model_registry import (
    CANONICAL_AUTHORS_BY_ARCH,
    HF_SUPPORTED_ARCHITECTURES,
)

ARCH = "Qwen3_5ForConditionalGeneration"


def _make_cfg(**overrides):
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
    # Qwen vision config uses depth/num_heads (not num_hidden_layers/num_attention_heads).
    cfg.vision_config = SimpleNamespace(hidden_size=16, depth=2, num_heads=4)
    return cfg


def test_registered_and_selected_as_multimodal_adapter():
    assert SUPPORTED_ARCHITECTURES[ARCH] is Qwen3_5MultimodalArchitectureAdapter
    adapter = ArchitectureAdapterFactory.select_architecture_adapter(_make_cfg())
    assert isinstance(adapter, Qwen3_5MultimodalArchitectureAdapter)


def test_registry_invariants():
    assert ARCH in HF_SUPPORTED_ARCHITECTURES
    assert CANONICAL_AUTHORS_BY_ARCH.get(ARCH) == ["Qwen"]


class TestConfig:
    def test_multimodal_and_gated_q_proj(self):
        adapter = Qwen3_5MultimodalArchitectureAdapter(_make_cfg())
        assert adapter.cfg.is_multimodal is True
        assert adapter.cfg.gated_q_proj is True

    def test_hybrid_mode(self):
        # hybrid=True path: no LN folding, no declarative weight conversions.
        adapter = Qwen3_5MultimodalArchitectureAdapter(_make_cfg())
        assert adapter.supports_fold_ln is False
        assert adapter.weight_processing_conversions == {}

    def test_vision_config_extracted(self):
        adapter = Qwen3_5MultimodalArchitectureAdapter(_make_cfg())
        assert adapter.cfg.vision_hidden_size == 16
        assert adapter.cfg.vision_num_layers == 2
        assert adapter.cfg.vision_num_heads == 4


class TestComponentWiring:
    """The wiring unique to this adapter: vision tower + LM nested under model.language_model."""

    def test_vision_paths_and_types(self):
        m = Qwen3_5MultimodalArchitectureAdapter(_make_cfg()).component_mapping
        assert isinstance(m["vision_encoder"], Qwen3_5VisionEncoderBridge)
        assert m["vision_encoder"].name == "model.visual"
        # The Qwen3.5 merger is the vision->text projector.
        assert isinstance(m["vision_projector"], VisionProjectionBridge)
        assert m["vision_projector"].name == "model.visual.merger"

    def test_language_model_nested_paths(self):
        m = Qwen3_5MultimodalArchitectureAdapter(_make_cfg()).component_mapping
        assert m["embed"].name == "model.language_model.embed_tokens"
        assert m["rotary_emb"].name == "model.language_model.rotary_emb"
        assert m["blocks"].name == "model.language_model.layers"
        assert m["ln_final"].name == "model.language_model.norm"
        assert m["unembed"].name == "lm_head"  # stays top-level


class TestVisionDecomposition:
    """The decomposed vision tower is this adapter's novel contribution."""

    def test_vision_tower_decomposed(self):
        vision = Qwen3_5MultimodalArchitectureAdapter(_make_cfg()).component_mapping[
            "vision_encoder"
        ]
        assert set(vision.submodules) >= {"patch_embed", "pos_embed", "blocks"}
        block = vision.submodules["blocks"]
        assert isinstance(block, Qwen3_5VisionBlockBridge)
        assert set(block.submodules) >= {"norm1", "norm2", "attn", "mlp"}
        # Fused-qkv attention + 2-layer MLP, all hookable linear projections.
        assert {"qkv", "proj"} <= set(block.submodules["attn"].submodules)
        assert {"linear_fc1", "linear_fc2"} <= set(block.submodules["mlp"].submodules)
        for leaf in ("attn.qkv", "attn.proj", "mlp.linear_fc1", "mlp.linear_fc2"):
            comp, sub = leaf.split(".")
            assert isinstance(block.submodules[comp].submodules[sub], LinearBridge)


def test_gated_q_proj_query_half_is_sliced_under_nested_path():
    """preprocess_weights slices the query half from the 2x-wide gated q_proj, matching the
    nested model.language_model.* key."""
    adapter = Qwen3_5MultimodalArchitectureAdapter(_make_cfg())
    n_heads, d_head, hidden = adapter.cfg.n_heads, adapter.cfg.d_head, adapter.cfg.d_model
    key = "model.language_model.layers.1.self_attn.q_proj.weight"
    # Per head: rows [query(d_head), gate(d_head)] -> 2*d_head wide.
    full = torch.randn(n_heads * d_head * 2, hidden)
    out = adapter.preprocess_weights({key: full.clone()})
    assert out[key].shape == (n_heads * d_head, hidden)
    expected = full.view(n_heads, d_head * 2, hidden)[:, :d_head, :].reshape(
        n_heads * d_head, hidden
    )
    assert torch.equal(out[key], expected)
