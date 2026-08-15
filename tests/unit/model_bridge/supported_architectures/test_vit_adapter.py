"""Unit tests for ViTArchitectureAdapter.

Modeled on tests/unit/model_bridge/supported_architectures/test_hubert_adapter.py
(the closest existing analog: HubertArchitectureAdapter also sets a modality flag
— is_audio_model there, is_visual_model here — and rebinds a prefix + injects a
head in prepare_model()).

Tests cover:
- Component mapping structure for a bare ViTModel/DeiTModel (default, before
  prepare_model() sees the real HF model)
- Block/attn/mlp submodule mapping and hook-alias overrides
- Weight conversion key set and rearrange patterns (weights + biases)
- Anti-drift config flags (is_visual_model, positional_embedding_type,
  normalization_type, supports_fold_ln, supports_generation)
- prepare_model(): bare model / ViTForImageClassification / DeiTForImageClassification
  prefix rebinding + classifier injection, and the DeiTForImageClassificationWithTeacher
  guard rail

NOTE (transformers ViT/DeiT flattening refactor): current transformers removed
the separate `ViTEncoder`/`ViTSelfAttention`/`ViTSelfOutput`/`ViTIntermediate`/
`ViTOutput` wrapper modules. Blocks now live directly at `<prefix>.layers`
(not `<prefix>.encoder.layer`); `ViTAttention`/`DeiTAttention` own `q_proj`/
`k_proj`/`v_proj`/`o_proj` directly (not nested `attention.query`/`attention.key`/
`attention.value`/`output.dense`); and `ViTLayer.mlp` is already a flat `ViTMLP`
with `fc1`/`fc2` (not `intermediate.dense`/`output.dense`). All the HF-path
assertions below reflect that.

NOT covered here (needs a real HF model + real forward pass — see the
integration test at tests/integration/model_bridge/test_vit_adapter.py instead):
- Numeric parity with HF
- Hook firing / cache shapes
- n_ctx propagation: This is cross-checked against a real checkpoint
  (google/vit-base-patch16-224 -> n_ctx == 197) in the integration test.
"""

from types import SimpleNamespace

import pytest

from transformer_lens.config.transformer_bridge_config import TransformerBridgeConfig
from transformer_lens.conversion_utils.conversion_steps import RearrangeTensorConversion
from transformer_lens.conversion_utils.param_processing_conversion import (
    ParamProcessingConversion,
)
from transformer_lens.model_bridge.generalized_components import (
    AttentionBridge,
    BlockBridge,
    LinearBridge,
    MLPBridge,
    NormalizationBridge,
)
from transformer_lens.model_bridge.generalized_components.vision_classifier_head import (
    VisionClassifierHeadBridge,
)
from transformer_lens.model_bridge.generalized_components.vision_embeddings import (
    VisionEmbeddingsBridge,
)
from transformer_lens.model_bridge.supported_architectures.vit import (
    ViTArchitectureAdapter,
)

# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------


def _make_cfg(
    n_heads: int = 12,
    d_model: int = 768,
    n_layers: int = 12,
    d_vocab: int = 1000,
    n_ctx: int = 197,
    **overrides,
) -> TransformerBridgeConfig:
    """Return a minimal TransformerBridgeConfig for ViT/DeiT adapter tests.

    n_ctx=197 is the real value for a 224x224 image / patch16 model
    (14*14 patches + 1 CLS token) — used as the *input* cfg here so tests can
    tell apart "the adapter left this alone" from "the adapter zeroed it".
    """
    cfg = TransformerBridgeConfig(
        d_model=d_model,
        d_head=d_model // n_heads,
        n_heads=n_heads,
        n_layers=n_layers,
        n_ctx=n_ctx,
        d_vocab=d_vocab,
        architecture="ViTForImageClassification",
    )
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg


@pytest.fixture()
def adapter() -> ViTArchitectureAdapter:
    """Fresh adapter per test — prepare_model() mutates component_mapping in place."""
    return ViTArchitectureAdapter(_make_cfg())


# ---------------------------------------------------------------------------
# Component mapping — bare ViTModel/DeiTModel (default, pre prepare_model())
# ---------------------------------------------------------------------------


class TestViTComponentMappingBare:
    """Default mapping assumes no prefix and no classifier (matches __init__)."""

    def test_top_level_keys(self, adapter: ViTArchitectureAdapter) -> None:
        assert set(adapter.component_mapping.keys()) == {"embed", "blocks", "ln_final"}

    def test_no_unembed_by_default(self, adapter: ViTArchitectureAdapter) -> None:
        assert "unembed" not in adapter.component_mapping

    def test_bridge_types(self, adapter: ViTArchitectureAdapter) -> None:
        mapping = adapter.component_mapping
        assert isinstance(mapping["embed"], VisionEmbeddingsBridge)
        assert isinstance(mapping["blocks"], BlockBridge)
        assert isinstance(mapping["ln_final"], NormalizationBridge)

    def test_top_level_hf_paths(self, adapter: ViTArchitectureAdapter) -> None:
        mapping = adapter.component_mapping
        assert mapping["embed"].name == "embeddings"
        # No more ViTEncoder wrapper — blocks sit directly on `.layers`.
        assert mapping["blocks"].name == "layers"
        assert mapping["ln_final"].name == "layernorm"

    def test_block_submodule_keys(self, adapter: ViTArchitectureAdapter) -> None:
        assert set(adapter.component_mapping["blocks"].submodules.keys()) == {
            "ln1",
            "ln2",
            "attn",
            "mlp",
        }

    def test_block_submodule_types(self, adapter: ViTArchitectureAdapter) -> None:
        blocks = adapter.component_mapping["blocks"]
        assert isinstance(blocks.submodules["ln1"], NormalizationBridge)
        assert isinstance(blocks.submodules["ln2"], NormalizationBridge)
        assert isinstance(blocks.submodules["attn"], AttentionBridge)
        assert isinstance(blocks.submodules["mlp"], MLPBridge)

    def test_block_submodule_hf_paths(self, adapter: ViTArchitectureAdapter) -> None:
        blocks = adapter.component_mapping["blocks"]
        assert blocks.submodules["ln1"].name == "layernorm_before"
        assert blocks.submodules["ln2"].name == "layernorm_after"
        assert blocks.submodules["attn"].name == "attention"
        assert blocks.submodules["mlp"].name == "mlp"

    def test_attn_submodule_keys(self, adapter: ViTArchitectureAdapter) -> None:
        attn = adapter.component_mapping["blocks"].submodules["attn"]
        assert set(attn.submodules.keys()) == {"q", "k", "v", "o"}

    def test_attn_qkvo_hf_paths(self, adapter: ViTArchitectureAdapter) -> None:
        """Paths here are relative to the already-resolved "attn" bridge module
        (block.attention). ViTAttention/DeiTAttention now owns q_proj/k_proj/
        v_proj/o_proj directly (flattened, no nested self-attention + separate
        output.dense module), so no "attention." prefix."""
        attn = adapter.component_mapping["blocks"].submodules["attn"]
        assert attn.submodules["q"].name == "q_proj"
        assert attn.submodules["k"].name == "k_proj"
        assert attn.submodules["v"].name == "v_proj"
        assert attn.submodules["o"].name == "o_proj"

    def test_attn_submodules_are_linear_bridges(self, adapter: ViTArchitectureAdapter) -> None:
        attn = adapter.component_mapping["blocks"].submodules["attn"]
        for sub in attn.submodules.values():
            assert isinstance(sub, LinearBridge)

    def test_mlp_submodule_hf_paths(self, adapter: ViTArchitectureAdapter) -> None:
        """Paths here are relative to the already-resolved "mlp" bridge module
        (block.mlp). ViTLayer.mlp is a real ViTMLP now (fc1/fc2 directly) — no
        more intermediate/output split, so no wrapper shim is needed."""
        mlp = adapter.component_mapping["blocks"].submodules["mlp"]
        assert mlp.submodules["in"].name == "fc1"
        assert mlp.submodules["out"].name == "fc2"

    def test_mlp_hook_alias_overrides(self, adapter: ViTArchitectureAdapter) -> None:
        aliases = adapter.component_mapping["blocks"].hook_aliases
        assert aliases.get("hook_mlp_out") == "mlp.out.hook_out"
        assert aliases.get("hook_mlp_in") == "mlp.in.hook_in"


# ---------------------------------------------------------------------------
# Anti-drift config flags
# ---------------------------------------------------------------------------


class TestViTAdapterConfig:
    """Flags that must not silently regress."""

    def test_normalization_type_is_ln(self, adapter: ViTArchitectureAdapter) -> None:
        assert adapter.cfg.normalization_type == "LN"

    def test_positional_embedding_type_is_standard(self, adapter: ViTArchitectureAdapter) -> None:
        assert adapter.cfg.positional_embedding_type == "standard"

    def test_is_visual_model_true(self, adapter: ViTArchitectureAdapter) -> None:
        """Only vision adapters set this flag; text/audio adapters must not."""
        assert adapter.cfg.is_visual_model is True

    def test_is_audio_model_not_set_true(self, adapter: ViTArchitectureAdapter) -> None:
        assert getattr(adapter.cfg, "is_audio_model", False) is False

    def test_final_rms_is_false(self, adapter: ViTArchitectureAdapter) -> None:
        assert adapter.cfg.final_rms is False

    def test_gated_mlp_is_false(self, adapter: ViTArchitectureAdapter) -> None:
        assert adapter.cfg.gated_mlp is False

    def test_attn_only_is_false(self, adapter: ViTArchitectureAdapter) -> None:
        assert adapter.cfg.attn_only is False

    def test_supports_generation_is_false(self) -> None:
        assert ViTArchitectureAdapter.supports_generation is False

    def test_supports_fold_ln_true_pre_ln(self, adapter: ViTArchitectureAdapter) -> None:
        """ViT/DeiT are pre-LN — unlike BertArchitectureAdapter's post-LN False."""
        assert adapter.supports_fold_ln is True


# ---------------------------------------------------------------------------
# Weight processing conversions
# ---------------------------------------------------------------------------


class TestViTWeightConversions:
    def test_exact_conversion_key_set(self, adapter: ViTArchitectureAdapter) -> None:
        assert set(adapter.weight_processing_conversions.keys()) == {
            "blocks.{i}.attn.q.weight",
            "blocks.{i}.attn.k.weight",
            "blocks.{i}.attn.v.weight",
            "blocks.{i}.attn.q.bias",
            "blocks.{i}.attn.k.bias",
            "blocks.{i}.attn.v.bias",
            "blocks.{i}.attn.o.weight",
        }

    def test_qkv_weight_pattern(self, adapter: ViTArchitectureAdapter) -> None:
        for slot in ("q", "k", "v"):
            conv = adapter.weight_processing_conversions[f"blocks.{{i}}.attn.{slot}.weight"]
            assert isinstance(conv, ParamProcessingConversion)
            assert isinstance(conv.tensor_conversion, RearrangeTensorConversion)
            assert conv.tensor_conversion.pattern == "(h d_head) d_model -> h d_model d_head"

    def test_qkv_bias_pattern(self, adapter: ViTArchitectureAdapter) -> None:
        for slot in ("q", "k", "v"):
            conv = adapter.weight_processing_conversions[f"blocks.{{i}}.attn.{slot}.bias"]
            assert isinstance(conv, ParamProcessingConversion)
            assert isinstance(conv.tensor_conversion, RearrangeTensorConversion)
            assert conv.tensor_conversion.pattern == "(h d_head) -> h d_head"

    def test_o_weight_pattern(self, adapter: ViTArchitectureAdapter) -> None:
        conv = adapter.weight_processing_conversions["blocks.{i}.attn.o.weight"]
        assert isinstance(conv, ParamProcessingConversion)
        assert isinstance(conv.tensor_conversion, RearrangeTensorConversion)
        assert conv.tensor_conversion.pattern == "d_model (h d_head) -> h d_head d_model"

    def test_qkv_weight_head_axis(self, adapter: ViTArchitectureAdapter) -> None:
        for slot in ("q", "k", "v"):
            conv = adapter.weight_processing_conversions[f"blocks.{{i}}.attn.{slot}.weight"]
            assert conv.tensor_conversion.axes_lengths["h"] == 12

    def test_qkv_bias_head_axis(self, adapter: ViTArchitectureAdapter) -> None:
        for slot in ("q", "k", "v"):
            conv = adapter.weight_processing_conversions[f"blocks.{{i}}.attn.{slot}.bias"]
            assert conv.tensor_conversion.axes_lengths["h"] == 12


# ---------------------------------------------------------------------------
# prepare_model() — prefix rebinding + classifier injection
# ---------------------------------------------------------------------------


class TestViTPrepareModel:
    """prepare_model() detects bare/ViT/DeiT wrappers and rebuilds the mapping."""

    def _bare_model(self) -> object:
        """No 'vit'/'deit'/'classifier' attribute — mimics bare ViTModel/DeiTModel."""
        return SimpleNamespace()

    def _bare_model_with_pooler(self) -> object:
        return SimpleNamespace(pooler=SimpleNamespace(dense=SimpleNamespace()))

    def _vit_for_classification(self) -> object:
        return SimpleNamespace(vit=SimpleNamespace(), classifier=SimpleNamespace())

    def _deit_for_classification(self) -> object:
        return SimpleNamespace(deit=SimpleNamespace(), classifier=SimpleNamespace())

    def _deit_with_teacher(self) -> object:
        """Dual cls_classifier + distillation_classifier — explicitly unsupported."""
        return SimpleNamespace(
            deit=SimpleNamespace(),
            cls_classifier=SimpleNamespace(),
            distillation_classifier=SimpleNamespace(),
        )

    # -- bare model --------------------------------------------------------

    def test_bare_model_keeps_no_prefix(self, adapter: ViTArchitectureAdapter) -> None:
        adapter.prepare_model(self._bare_model())
        assert adapter.component_mapping["embed"].name == "embeddings"
        assert adapter.component_mapping["blocks"].name == "layers"
        assert adapter.component_mapping["ln_final"].name == "layernorm"

    def test_bare_model_has_no_unembed(self, adapter: ViTArchitectureAdapter) -> None:
        adapter.prepare_model(self._bare_model())
        assert "unembed" not in adapter.component_mapping

    def test_bare_model_maps_pooler_without_root_name_collision(
        self, adapter: ViTArchitectureAdapter
    ) -> None:
        adapter.prepare_model(self._bare_model_with_pooler())
        assert adapter.component_mapping["pooler"].name == "pooler.dense"

    def test_bare_model_does_not_require_encoder_attribute(
        self, adapter: ViTArchitectureAdapter
    ) -> None:
        """Regression test: prepare_model() must not assume a `.encoder`
        wrapper module exists on the HF model. Current transformers removed
        it — blocks live directly on `<prefix>.layers` — so a minimal stub
        with no `.encoder` attribute at all must still work."""
        hf_model = SimpleNamespace()
        assert not hasattr(hf_model, "encoder")
        adapter.prepare_model(hf_model)  # must not raise AttributeError

    # -- ViTForImageClassification -----------------------------------------

    def test_vit_classification_adds_vit_prefix(self, adapter: ViTArchitectureAdapter) -> None:
        adapter.prepare_model(self._vit_for_classification())
        assert adapter.component_mapping["embed"].name == "vit.embeddings"
        assert adapter.component_mapping["blocks"].name == "vit.layers"
        assert adapter.component_mapping["ln_final"].name == "vit.layernorm"

    def test_vit_classification_adds_unembed(self, adapter: ViTArchitectureAdapter) -> None:
        adapter.prepare_model(self._vit_for_classification())
        assert "unembed" in adapter.component_mapping
        assert isinstance(adapter.component_mapping["unembed"], VisionClassifierHeadBridge)

    def test_classifier_is_never_prefixed(self, adapter: ViTArchitectureAdapter) -> None:
        """classifier is always a top-level attr on the *ForImageClassification wrapper."""
        adapter.prepare_model(self._vit_for_classification())
        assert adapter.component_mapping["unembed"].name == "classifier"

    # -- DeiTForImageClassification ------------------------------------------

    def test_deit_classification_adds_deit_prefix(self, adapter: ViTArchitectureAdapter) -> None:
        adapter.prepare_model(self._deit_for_classification())
        assert adapter.component_mapping["embed"].name == "deit.embeddings"
        assert adapter.component_mapping["blocks"].name == "deit.layers"
        assert adapter.component_mapping["ln_final"].name == "deit.layernorm"

    def test_deit_classification_adds_unembed(self, adapter: ViTArchitectureAdapter) -> None:
        adapter.prepare_model(self._deit_for_classification())
        assert adapter.component_mapping["unembed"].name == "classifier"

    # -- DeiTForImageClassificationWithTeacher guard rail --------------------

    def test_deit_with_teacher_raises(self, adapter: ViTArchitectureAdapter) -> None:
        with pytest.raises(NotImplementedError):
            adapter.prepare_model(self._deit_with_teacher())

    def test_deit_with_teacher_message_mentions_reason(
        self, adapter: ViTArchitectureAdapter
    ) -> None:
        with pytest.raises(NotImplementedError, match="distillation_classifier"):
            adapter.prepare_model(self._deit_with_teacher())
