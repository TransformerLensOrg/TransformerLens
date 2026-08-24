"""Unit tests for BertArchitectureAdapter.

Tests cover:
- Component mapping structure (bridge types and HF module names)
- Learned positional embeddings (pos_embed present; no rotary_emb)
- Weight conversion key set and rearrange patterns (weights + biases)
- Post-LN architecture: supports_fold_ln must remain False
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
    BlockBridge,
    EmbeddingBridge,
    LinearBridge,
    MLPBridge,
    NormalizationBridge,
    PosEmbedBridge,
    UnembeddingBridge,
)
from transformer_lens.model_bridge.supported_architectures.bert import (
    BertArchitectureAdapter,
)

# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------


def _make_cfg(
    n_heads: int = 12,
    d_model: int = 768,
    n_layers: int = 12,
    d_vocab: int = 30522,
    n_ctx: int = 512,
    **overrides,
) -> TransformerBridgeConfig:
    """Return a minimal TransformerBridgeConfig for BERT adapter tests."""
    cfg = TransformerBridgeConfig(
        d_model=d_model,
        d_head=d_model // n_heads,
        n_heads=n_heads,
        n_layers=n_layers,
        n_ctx=n_ctx,
        d_vocab=d_vocab,
        architecture="BertForMaskedLM",
    )
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg


@pytest.fixture(scope="module")
def adapter() -> BertArchitectureAdapter:
    return BertArchitectureAdapter(_make_cfg())


# ---------------------------------------------------------------------------
# Component mapping
# ---------------------------------------------------------------------------


class TestBertComponentMapping:
    """Component mapping has the correct slots, bridge types, and HF module paths."""

    def test_top_level_keys(self, adapter: BertArchitectureAdapter) -> None:
        assert set(adapter.component_mapping.keys()) == {
            "embed",
            "token_type_embed",
            "pos_embed",
            "embed_ln",
            "blocks",
            "mlm_head",
            "ln_final",
            "unembed",
        }

    def test_bridge_types(self, adapter: BertArchitectureAdapter) -> None:
        mapping = adapter.component_mapping
        assert isinstance(mapping["embed"], EmbeddingBridge)
        assert isinstance(mapping["token_type_embed"], EmbeddingBridge)
        assert isinstance(mapping["pos_embed"], PosEmbedBridge)
        assert isinstance(mapping["embed_ln"], NormalizationBridge)
        assert isinstance(mapping["blocks"], BlockBridge)
        assert isinstance(mapping["mlm_head"], LinearBridge)
        assert isinstance(mapping["ln_final"], NormalizationBridge)
        assert isinstance(mapping["unembed"], UnembeddingBridge)

    def test_top_level_hf_paths(self, adapter: BertArchitectureAdapter) -> None:
        mapping = adapter.component_mapping
        assert mapping["embed"].name == "bert.embeddings.word_embeddings"
        assert mapping["token_type_embed"].name == "bert.embeddings.token_type_embeddings"
        assert mapping["pos_embed"].name == "bert.embeddings.position_embeddings"
        assert mapping["embed_ln"].name == "bert.embeddings.LayerNorm"
        assert mapping["blocks"].name == "bert.encoder.layer"
        assert mapping["mlm_head"].name == "cls.predictions.transform.dense"
        assert mapping["ln_final"].name == "cls.predictions.transform.LayerNorm"
        assert mapping["unembed"].name == "cls.predictions.decoder"

    def test_token_type_embedding_is_cached_with_hf_output(self) -> None:
        import torch
        from transformers import BertForMaskedLM

        from transformer_lens.model_bridge.sources import build_bridge_from_module

        hf_model = BertForMaskedLM.from_pretrained("bert-base-cased").eval()
        input_ids = torch.tensor([[101, 7592, 102, 2088, 102]])
        token_type_ids = torch.tensor([[0, 0, 0, 1, 1]])
        with torch.no_grad():
            expected = hf_model.bert.embeddings.token_type_embeddings(token_type_ids).clone()

        bridge = build_bridge_from_module(
            hf_model,
            "BertForMaskedLM",
            hf_config=hf_model.config,
            dtype=torch.float32,
            device="cpu",
            model_name="bert-base-cased",
        )
        with torch.no_grad():
            _, cache = bridge.run_with_cache(
                input_ids,
                token_type_ids=token_type_ids,
                names_filter=["token_type_embed.hook_out"],
            )

        assert "token_type_embed.hook_out" in cache
        torch.testing.assert_close(cache["token_type_embed.hook_out"], expected)

    def test_block_submodule_keys(self, adapter: BertArchitectureAdapter) -> None:
        assert set(adapter.component_mapping["blocks"].submodules.keys()) == {
            "ln1",
            "ln2",
            "attn",
            "mlp",
        }

    def test_block_submodule_types(self, adapter: BertArchitectureAdapter) -> None:
        blocks = adapter.component_mapping["blocks"]
        assert isinstance(blocks.submodules["ln1"], NormalizationBridge)
        assert isinstance(blocks.submodules["ln2"], NormalizationBridge)
        assert isinstance(blocks.submodules["attn"], AttentionBridge)
        assert isinstance(blocks.submodules["mlp"], MLPBridge)

    def test_block_submodule_hf_paths(self, adapter: BertArchitectureAdapter) -> None:
        blocks = adapter.component_mapping["blocks"]
        assert blocks.submodules["ln1"].name == "attention.output.LayerNorm"
        assert blocks.submodules["ln2"].name == "output.LayerNorm"
        assert blocks.submodules["attn"].name == "attention"
        assert blocks.submodules["mlp"].name is None

    def test_attn_submodule_keys(self, adapter: BertArchitectureAdapter) -> None:
        attn = adapter.component_mapping["blocks"].submodules["attn"]
        assert set(attn.submodules.keys()) == {"q", "k", "v", "o"}

    def test_attn_qkvo_hf_paths(self, adapter: BertArchitectureAdapter) -> None:
        attn = adapter.component_mapping["blocks"].submodules["attn"]
        assert attn.submodules["q"].name == "self.query"
        assert attn.submodules["k"].name == "self.key"
        assert attn.submodules["v"].name == "self.value"
        assert attn.submodules["o"].name == "output.dense"

    def test_attn_submodules_are_linear_bridges(self, adapter: BertArchitectureAdapter) -> None:
        attn = adapter.component_mapping["blocks"].submodules["attn"]
        for sub in attn.submodules.values():
            assert isinstance(sub, LinearBridge)

    def test_mlp_submodule_hf_paths(self, adapter: BertArchitectureAdapter) -> None:
        mlp = adapter.component_mapping["blocks"].submodules["mlp"]
        assert mlp.submodules["in"].name == "intermediate.dense"
        assert mlp.submodules["out"].name == "output.dense"


class TestBertTaskHeadMappings:
    def test_nsp_only_model_uses_hooked_encoder_names(self) -> None:
        adapter = BertArchitectureAdapter(_make_cfg())
        hf_model = SimpleNamespace(
            bert=SimpleNamespace(pooler=object()),
            cls=SimpleNamespace(seq_relationship=object()),
        )

        adapter.prepare_model(hf_model)

        assert adapter.components["pooler"].name == "bert.pooler.dense"
        assert adapter.components["unembed"].name == "cls.seq_relationship"
        assert "mlm_head" not in adapter.components
        assert "ln_final" not in adapter.components

    def test_combined_mlm_nsp_model_registers_both_heads(self) -> None:
        adapter = BertArchitectureAdapter(_make_cfg())
        hf_model = SimpleNamespace(
            bert=SimpleNamespace(pooler=object()),
            cls=SimpleNamespace(predictions=object(), seq_relationship=object()),
        )

        adapter.prepare_model(hf_model)

        assert adapter.components["pooler"].name == "bert.pooler.dense"
        assert adapter.components["mlm_head"].name == "cls.predictions.transform.dense"
        assert adapter.components["nsp_head"].name == "cls.seq_relationship"
        assert adapter.components["unembed"].name == "cls.predictions.decoder"


# ---------------------------------------------------------------------------
# Anti-drift config flags
# ---------------------------------------------------------------------------


class TestBertAdapterConfig:
    """Anti-drift flags that must not silently regress."""

    def test_normalization_type_is_ln(self, adapter: BertArchitectureAdapter) -> None:
        assert adapter.cfg.normalization_type == "LN"

    def test_final_rms_is_false(self, adapter: BertArchitectureAdapter) -> None:
        assert adapter.cfg.final_rms is False

    def test_supports_fold_ln_is_false(self, adapter: BertArchitectureAdapter) -> None:
        """BERT uses post-LN: fold_ln assumes pre-LN and produces wrong results if enabled."""
        assert adapter.supports_fold_ln is False

    def test_supports_generation_is_false(self) -> None:
        """BERT is an encoder-only model — generation is not supported."""
        assert BertArchitectureAdapter.supports_generation is False


# ---------------------------------------------------------------------------
# Weight processing conversions
# ---------------------------------------------------------------------------


class TestBertWeightConversions:
    """weight_processing_conversions has exactly the expected QKV weight+bias and O weight keys."""

    def test_exact_conversion_key_set(self, adapter: BertArchitectureAdapter) -> None:
        assert set(adapter.weight_processing_conversions.keys()) == {
            "blocks.{i}.attn.q.weight",
            "blocks.{i}.attn.k.weight",
            "blocks.{i}.attn.v.weight",
            "blocks.{i}.attn.q.bias",
            "blocks.{i}.attn.k.bias",
            "blocks.{i}.attn.v.bias",
            "blocks.{i}.attn.o.weight",
        }

    def test_qkv_weight_pattern(self, adapter: BertArchitectureAdapter) -> None:
        """'(h d_head) d_model -> h d_model d_head' splits heads for Q/K/V weights."""
        for slot in ("q", "k", "v"):
            conv = adapter.weight_processing_conversions[f"blocks.{{i}}.attn.{slot}.weight"]
            assert isinstance(conv, ParamProcessingConversion)
            assert isinstance(conv.tensor_conversion, RearrangeTensorConversion)
            assert conv.tensor_conversion.pattern == "(h d_head) d_model -> h d_model d_head"

    def test_qkv_bias_pattern(self, adapter: BertArchitectureAdapter) -> None:
        """'(h d_head) -> h d_head' splits heads for Q/K/V biases."""
        for slot in ("q", "k", "v"):
            conv = adapter.weight_processing_conversions[f"blocks.{{i}}.attn.{slot}.bias"]
            assert isinstance(conv, ParamProcessingConversion)
            assert isinstance(conv.tensor_conversion, RearrangeTensorConversion)
            assert conv.tensor_conversion.pattern == "(h d_head) -> h d_head"

    def test_o_weight_pattern(self, adapter: BertArchitectureAdapter) -> None:
        """'d_model (h d_head) -> h d_head d_model' for output projection."""
        conv = adapter.weight_processing_conversions["blocks.{i}.attn.o.weight"]
        assert isinstance(conv, ParamProcessingConversion)
        assert isinstance(conv.tensor_conversion, RearrangeTensorConversion)
        assert conv.tensor_conversion.pattern == "d_model (h d_head) -> h d_head d_model"

    def test_qkv_weight_head_axis(self, adapter: BertArchitectureAdapter) -> None:
        """h axis in weight conversions matches n_heads=12."""
        for slot in ("q", "k", "v"):
            conv = adapter.weight_processing_conversions[f"blocks.{{i}}.attn.{slot}.weight"]
            assert conv.tensor_conversion.axes_lengths["h"] == 12

    def test_qkv_bias_head_axis(self, adapter: BertArchitectureAdapter) -> None:
        """h axis in bias conversions matches n_heads=12."""
        for slot in ("q", "k", "v"):
            conv = adapter.weight_processing_conversions[f"blocks.{{i}}.attn.{slot}.bias"]
            assert conv.tensor_conversion.axes_lengths["h"] == 12


class TestBertLMHeadAlias:
    def test_bert_lm_head_model_maps_to_bert_adapter(self) -> None:
        """BertLMHeadModel (decoder-style BERT with a causal LM head) shares
        BertForMaskedLM's module tree, so the factory reuses the adapter."""
        from transformer_lens.factories.architecture_adapter_factory import (
            SUPPORTED_ARCHITECTURES,
        )

        assert SUPPORTED_ARCHITECTURES["BertLMHeadModel"] is BertArchitectureAdapter
