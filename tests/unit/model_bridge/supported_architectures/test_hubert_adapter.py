"""Unit tests for HubertArchitectureAdapter.

Tests cover:
- Component mapping structure (bridge types and HF module names)
- Audio-specific components: AudioFeatureExtractorBridge, ConvPosEmbedBridge
- No embed/pos_embed/unembed in default (bare HubertModel) mapping
- Weight conversion key set and rearrange patterns (weights + biases)
- Post-LN vs pre-LN: supports_fold_ln tied to do_stable_layer_norm
- prepare_model(): HubertForCTC prefix rebinding and unembed injection
- prepare_loading(): do_stable_layer_norm propagation from HF config
- Anti-drift config flags
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
    AudioFeatureExtractorBridge,
    BlockBridge,
    ConvPosEmbedBridge,
    LinearBridge,
    MLPBridge,
    NormalizationBridge,
    UnembeddingBridge,
)
from transformer_lens.model_bridge.generalized_components.base import GeneralizedComponent
from transformer_lens.model_bridge.supported_architectures.hubert import (
    HubertArchitectureAdapter,
)

# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------


def _make_cfg(
    n_heads: int = 12,
    d_model: int = 768,
    n_layers: int = 12,
    d_vocab: int = 32,
    n_ctx: int = 512,
    do_stable_layer_norm: bool = False,
    **overrides,
) -> TransformerBridgeConfig:
    """Return a minimal TransformerBridgeConfig for HuBERT adapter tests."""
    cfg = TransformerBridgeConfig(
        d_model=d_model,
        d_head=d_model // n_heads,
        n_heads=n_heads,
        n_layers=n_layers,
        n_ctx=n_ctx,
        d_vocab=d_vocab,
        architecture="HubertModel",
    )
    cfg.do_stable_layer_norm = do_stable_layer_norm
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg


@pytest.fixture(scope="module")
def adapter() -> HubertArchitectureAdapter:
    """Default bare HubertModel adapter (post-LN, no HubertForCTC prefix)."""
    return HubertArchitectureAdapter(_make_cfg())


# ---------------------------------------------------------------------------
# Component mapping — bare HubertModel (default)
# ---------------------------------------------------------------------------


class TestHubertComponentMapping:
    """Component mapping has the correct slots, bridge types, and HF module paths."""

    def test_top_level_keys_bare_model(self, adapter: HubertArchitectureAdapter) -> None:
        """Bare HubertModel has no unembed — only audio-specific + encoder components."""
        assert set(adapter.component_mapping.keys()) == {
            "audio_feature_extractor",
            "feat_proj",
            "conv_pos_embed",
            "embed_ln",
            "blocks",
        }

    def test_bridge_types(self, adapter: HubertArchitectureAdapter) -> None:
        mapping = adapter.component_mapping
        assert isinstance(mapping["audio_feature_extractor"], AudioFeatureExtractorBridge)
        assert isinstance(mapping["feat_proj"], GeneralizedComponent)
        assert isinstance(mapping["conv_pos_embed"], ConvPosEmbedBridge)
        assert isinstance(mapping["embed_ln"], NormalizationBridge)
        assert isinstance(mapping["blocks"], BlockBridge)

    def test_top_level_hf_paths(self, adapter: HubertArchitectureAdapter) -> None:
        mapping = adapter.component_mapping
        assert mapping["audio_feature_extractor"].name == "feature_extractor"
        assert mapping["feat_proj"].name == "feature_projection"
        assert mapping["conv_pos_embed"].name == "encoder.pos_conv_embed"
        assert mapping["embed_ln"].name == "encoder.layer_norm"
        assert mapping["blocks"].name == "encoder.layers"

    def test_block_submodule_keys(self, adapter: HubertArchitectureAdapter) -> None:
        assert set(adapter.component_mapping["blocks"].submodules.keys()) == {
            "ln1",
            "ln2",
            "attn",
            "mlp",
        }

    def test_block_submodule_types(self, adapter: HubertArchitectureAdapter) -> None:
        blocks = adapter.component_mapping["blocks"]
        assert isinstance(blocks.submodules["ln1"], NormalizationBridge)
        assert isinstance(blocks.submodules["ln2"], NormalizationBridge)
        assert isinstance(blocks.submodules["attn"], AttentionBridge)
        assert isinstance(blocks.submodules["mlp"], MLPBridge)

    def test_block_submodule_hf_paths(self, adapter: HubertArchitectureAdapter) -> None:
        blocks = adapter.component_mapping["blocks"]
        assert blocks.submodules["ln1"].name == "layer_norm"
        assert blocks.submodules["ln2"].name == "final_layer_norm"
        assert blocks.submodules["attn"].name == "attention"
        assert blocks.submodules["mlp"].name == "feed_forward"

    def test_attn_submodule_keys(self, adapter: HubertArchitectureAdapter) -> None:
        attn = adapter.component_mapping["blocks"].submodules["attn"]
        assert set(attn.submodules.keys()) == {"q", "k", "v", "o"}

    def test_attn_qkvo_hf_paths(self, adapter: HubertArchitectureAdapter) -> None:
        attn = adapter.component_mapping["blocks"].submodules["attn"]
        assert attn.submodules["q"].name == "q_proj"
        assert attn.submodules["k"].name == "k_proj"
        assert attn.submodules["v"].name == "v_proj"
        assert attn.submodules["o"].name == "out_proj"

    def test_attn_submodules_are_linear_bridges(self, adapter: HubertArchitectureAdapter) -> None:
        attn = adapter.component_mapping["blocks"].submodules["attn"]
        for sub in attn.submodules.values():
            assert isinstance(sub, LinearBridge)

    def test_mlp_submodule_hf_paths(self, adapter: HubertArchitectureAdapter) -> None:
        """HuBERT uses intermediate_dense / output_dense (not intermediate.dense / output.dense)."""
        mlp = adapter.component_mapping["blocks"].submodules["mlp"]
        assert mlp.submodules["in"].name == "intermediate_dense"
        assert mlp.submodules["out"].name == "output_dense"

    def test_mlp_hook_alias_overrides(self, adapter: HubertArchitectureAdapter) -> None:
        """Virtual MLP module redirects hooks to the actual linear output hook."""
        aliases = adapter.component_mapping["blocks"].hook_aliases
        assert aliases.get("hook_mlp_out") == "mlp.out.hook_out"
        assert aliases.get("hook_mlp_in") == "mlp.in.hook_in"


# ---------------------------------------------------------------------------
# Anti-drift config flags
# ---------------------------------------------------------------------------


class TestHubertAdapterConfig:
    """Anti-drift config flags that must not silently regress."""

    def test_positional_embedding_type_is_conv(self, adapter: HubertArchitectureAdapter) -> None:
        """Anti-drift: HuBERT is the only adapter using 'conv' — not standard/rotary/alibi."""
        assert adapter.cfg.positional_embedding_type == "conv"

    def test_is_audio_model_true(self, adapter: HubertArchitectureAdapter) -> None:
        """Anti-drift: only audio adapters set this flag; text adapters must not."""
        assert adapter.cfg.is_audio_model is True

    def test_supports_generation_is_false(self) -> None:
        """HuBERT is an audio encoder — generation is not supported."""
        assert HubertArchitectureAdapter.supports_generation is False

    def test_supports_fold_ln_false_post_ln(self, adapter: HubertArchitectureAdapter) -> None:
        """Post-LN (do_stable_layer_norm=False): fold_ln would apply to the wrong sublayer."""
        assert adapter.supports_fold_ln is False

    def test_supports_fold_ln_true_pre_ln(self) -> None:
        """Pre-LN (do_stable_layer_norm=True): fold_ln is safe."""
        adapter = HubertArchitectureAdapter(_make_cfg(do_stable_layer_norm=True))
        assert adapter.supports_fold_ln is True


# ---------------------------------------------------------------------------
# Weight processing conversions
# ---------------------------------------------------------------------------


class TestHubertWeightConversions:
    """weight_processing_conversions matches BERT's seven-key set (weight + biases for QKVO)."""

    def test_exact_conversion_key_set(self, adapter: HubertArchitectureAdapter) -> None:
        assert set(adapter.weight_processing_conversions.keys()) == {
            "blocks.{i}.attn.q.weight",
            "blocks.{i}.attn.k.weight",
            "blocks.{i}.attn.v.weight",
            "blocks.{i}.attn.q.bias",
            "blocks.{i}.attn.k.bias",
            "blocks.{i}.attn.v.bias",
            "blocks.{i}.attn.o.weight",
        }

    def test_qkv_weight_pattern(self, adapter: HubertArchitectureAdapter) -> None:
        for slot in ("q", "k", "v"):
            conv = adapter.weight_processing_conversions[f"blocks.{{i}}.attn.{slot}.weight"]
            assert isinstance(conv, ParamProcessingConversion)
            assert isinstance(conv.tensor_conversion, RearrangeTensorConversion)
            assert conv.tensor_conversion.pattern == "(h d_head) d_model -> h d_model d_head"

    def test_qkv_bias_pattern(self, adapter: HubertArchitectureAdapter) -> None:
        for slot in ("q", "k", "v"):
            conv = adapter.weight_processing_conversions[f"blocks.{{i}}.attn.{slot}.bias"]
            assert isinstance(conv, ParamProcessingConversion)
            assert isinstance(conv.tensor_conversion, RearrangeTensorConversion)
            assert conv.tensor_conversion.pattern == "(h d_head) -> h d_head"

    def test_o_weight_pattern(self, adapter: HubertArchitectureAdapter) -> None:
        conv = adapter.weight_processing_conversions["blocks.{i}.attn.o.weight"]
        assert isinstance(conv, ParamProcessingConversion)
        assert isinstance(conv.tensor_conversion, RearrangeTensorConversion)
        assert conv.tensor_conversion.pattern == "d_model (h d_head) -> h d_head d_model"

    def test_qkv_weight_head_axis(self, adapter: HubertArchitectureAdapter) -> None:
        """h axis in weight conversions must match n_heads=12."""
        for slot in ("q", "k", "v"):
            conv = adapter.weight_processing_conversions[f"blocks.{{i}}.attn.{slot}.weight"]
            assert conv.tensor_conversion.axes_lengths["h"] == 12

    def test_qkv_bias_head_axis(self, adapter: HubertArchitectureAdapter) -> None:
        """h axis in bias conversions must match n_heads=12."""
        for slot in ("q", "k", "v"):
            conv = adapter.weight_processing_conversions[f"blocks.{{i}}.attn.{slot}.bias"]
            assert conv.tensor_conversion.axes_lengths["h"] == 12

    def test_no_norm_conversion_keys(self, adapter: HubertArchitectureAdapter) -> None:
        """LayerNorm does not need head-splitting."""
        assert not any("ln" in k for k in adapter.weight_processing_conversions)


# ---------------------------------------------------------------------------
# prepare_model() — HubertForCTC prefix rebinding
# ---------------------------------------------------------------------------


class TestHubertPrepareModel:
    """prepare_model() detects HubertForCTC and rebinds all component paths to 'hubert.' prefix."""

    def _bare_model(self) -> object:
        """Namespace with no 'hubert' attribute — mimics bare HubertModel."""
        return SimpleNamespace()

    def _ctc_model(self) -> object:
        """Namespace with 'hubert' attribute — mimics HubertForCTC."""
        return SimpleNamespace(hubert=SimpleNamespace(), lm_head=SimpleNamespace())

    def test_bare_model_keeps_no_prefix(self) -> None:
        adapter = HubertArchitectureAdapter(_make_cfg())
        adapter.prepare_model(self._bare_model())
        assert adapter.component_mapping["audio_feature_extractor"].name == "feature_extractor"
        assert adapter.component_mapping["blocks"].name == "encoder.layers"

    def test_ctc_model_adds_hubert_prefix_to_feature_extractor(self) -> None:
        adapter = HubertArchitectureAdapter(_make_cfg())
        adapter.prepare_model(self._ctc_model())
        assert adapter.component_mapping["audio_feature_extractor"].name == "hubert.feature_extractor"

    def test_ctc_model_adds_hubert_prefix_to_blocks(self) -> None:
        adapter = HubertArchitectureAdapter(_make_cfg())
        adapter.prepare_model(self._ctc_model())
        assert adapter.component_mapping["blocks"].name == "hubert.encoder.layers"

    def test_ctc_model_adds_hubert_prefix_to_embed_ln(self) -> None:
        adapter = HubertArchitectureAdapter(_make_cfg())
        adapter.prepare_model(self._ctc_model())
        assert adapter.component_mapping["embed_ln"].name == "hubert.encoder.layer_norm"

    def test_ctc_model_adds_hubert_prefix_to_conv_pos_embed(self) -> None:
        adapter = HubertArchitectureAdapter(_make_cfg())
        adapter.prepare_model(self._ctc_model())
        assert adapter.component_mapping["conv_pos_embed"].name == "hubert.encoder.pos_conv_embed"

    def test_ctc_model_adds_unembed(self) -> None:
        adapter = HubertArchitectureAdapter(_make_cfg())
        adapter.prepare_model(self._ctc_model())
        assert "unembed" in adapter.component_mapping
        assert isinstance(adapter.component_mapping["unembed"], UnembeddingBridge)
        assert adapter.component_mapping["unembed"].name == "lm_head"

    def test_bare_model_has_no_unembed_after_prepare(self) -> None:
        adapter = HubertArchitectureAdapter(_make_cfg())
        adapter.prepare_model(self._bare_model())
        assert "unembed" not in adapter.component_mapping


# ---------------------------------------------------------------------------
# prepare_loading() — HF config propagation
# ---------------------------------------------------------------------------


class TestHubertPrepareLoading:
    """prepare_loading() propagates do_stable_layer_norm from the HF config object."""

    def _model_kwargs_with(self, do_stable_layer_norm: bool) -> dict:
        hf_config = SimpleNamespace(do_stable_layer_norm=do_stable_layer_norm)
        return {"config": hf_config}

    def test_stable_layer_norm_true_propagates(self) -> None:
        adapter = HubertArchitectureAdapter(_make_cfg())
        adapter.prepare_loading("dummy-model", self._model_kwargs_with(True))
        assert adapter.supports_fold_ln is True
        assert adapter.cfg.do_stable_layer_norm is True

    def test_stable_layer_norm_false_propagates(self) -> None:
        adapter = HubertArchitectureAdapter(_make_cfg(do_stable_layer_norm=True))
        adapter.prepare_loading("dummy-model", self._model_kwargs_with(False))
        assert adapter.supports_fold_ln is False
        assert adapter.cfg.do_stable_layer_norm is False

    def test_no_config_in_model_kwargs_is_noop(self) -> None:
        """Missing 'config' key must not raise."""
        adapter = HubertArchitectureAdapter(_make_cfg())
        adapter.prepare_loading("dummy-model", {})  # must not raise
