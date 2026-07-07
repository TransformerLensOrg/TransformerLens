"""Unit tests for BloomArchitectureAdapter."""

from types import SimpleNamespace
from typing import Any

import pytest
import torch
import torch.nn as nn

from transformer_lens.config import TransformerBridgeConfig
from transformer_lens.conversion_utils.conversion_steps import RearrangeTensorConversion
from transformer_lens.conversion_utils.param_processing_conversion import (
    ParamProcessingConversion,
)
from transformer_lens.model_bridge.generalized_components import (
    BloomAttentionBridge,
    BloomBlockBridge,
    BloomMLPBridge,
    EmbeddingBridge,
    NormalizationBridge,
    UnembeddingBridge,
)
from transformer_lens.model_bridge.supported_architectures.bloom import (
    BloomArchitectureAdapter,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_cfg(
    n_heads: int = 8,
    d_model: int = 64,
    n_layers: int = 2,
    d_vocab: int = 100,
    n_ctx: int = 128,
) -> TransformerBridgeConfig:
    return TransformerBridgeConfig(
        d_model=d_model,
        d_head=d_model // n_heads,
        n_layers=n_layers,
        n_ctx=n_ctx,
        n_heads=n_heads,
        d_vocab=d_vocab,
        architecture="BloomForCausalLM",
    )


@pytest.fixture
def cfg() -> TransformerBridgeConfig:
    return _make_cfg()


@pytest.fixture
def adapter(cfg: TransformerBridgeConfig) -> BloomArchitectureAdapter:
    return BloomArchitectureAdapter(cfg)


def _make_qkv_component(d_model: int) -> Any:
    ns = SimpleNamespace()
    ns.query_key_value = nn.Linear(d_model, 3 * d_model, bias=True)
    return ns


# ---------------------------------------------------------------------------
# Config attribute tests
# ---------------------------------------------------------------------------


class TestBloomAdapterConfig:
    def test_positional_embedding_type(self, adapter: BloomArchitectureAdapter) -> None:
        assert adapter.cfg.positional_embedding_type == "alibi"

    def test_default_prepend_bos(self, adapter: BloomArchitectureAdapter) -> None:
        assert adapter.cfg.default_prepend_bos is False


# ---------------------------------------------------------------------------
# Component mapping tests
# ---------------------------------------------------------------------------


class TestBloomAdapterComponentMapping:
    @staticmethod
    def _mapping(adapter: BloomArchitectureAdapter) -> dict[str, Any]:
        mapping = adapter.component_mapping
        assert mapping is not None
        return mapping

    def test_embed_type_and_name(self, adapter: BloomArchitectureAdapter) -> None:
        mapping = self._mapping(adapter)
        assert isinstance(mapping["embed"], EmbeddingBridge)
        assert mapping["embed"].name == "transformer.word_embeddings"

    def test_embed_ln_type_and_name(self, adapter: BloomArchitectureAdapter) -> None:
        mapping = self._mapping(adapter)
        assert isinstance(mapping["embed_ln"], NormalizationBridge)
        assert mapping["embed_ln"].name == "transformer.word_embeddings_layernorm"

    def test_blocks_type_and_name(self, adapter: BloomArchitectureAdapter) -> None:
        mapping = self._mapping(adapter)
        assert isinstance(mapping["blocks"], BloomBlockBridge)
        assert mapping["blocks"].name == "transformer.h"

    def test_ln_final_type_and_name(self, adapter: BloomArchitectureAdapter) -> None:
        mapping = self._mapping(adapter)
        assert isinstance(mapping["ln_final"], NormalizationBridge)
        assert mapping["ln_final"].name == "transformer.ln_f"

    def test_unembed_type_and_name(self, adapter: BloomArchitectureAdapter) -> None:
        mapping = self._mapping(adapter)
        assert isinstance(mapping["unembed"], UnembeddingBridge)
        assert mapping["unembed"].name == "lm_head"

    def test_ln1_type_and_name(self, adapter: BloomArchitectureAdapter) -> None:
        blocks = self._mapping(adapter)["blocks"]
        assert isinstance(blocks.submodules["ln1"], NormalizationBridge)
        assert blocks.submodules["ln1"].name == "input_layernorm"

    def test_ln2_type_and_name(self, adapter: BloomArchitectureAdapter) -> None:
        blocks = self._mapping(adapter)["blocks"]
        assert isinstance(blocks.submodules["ln2"], NormalizationBridge)
        assert blocks.submodules["ln2"].name == "post_attention_layernorm"

    def test_attn_type_and_name(self, adapter: BloomArchitectureAdapter) -> None:
        blocks = self._mapping(adapter)["blocks"]
        assert isinstance(blocks.submodules["attn"], BloomAttentionBridge)
        assert blocks.submodules["attn"].name == "self_attention"

    def test_attn_qkv_name(self, adapter: BloomArchitectureAdapter) -> None:
        blocks = self._mapping(adapter)["blocks"]
        assert blocks.submodules["attn"].submodules["qkv"].name == "query_key_value"

    def test_attn_o_name(self, adapter: BloomArchitectureAdapter) -> None:
        blocks = self._mapping(adapter)["blocks"]
        assert blocks.submodules["attn"].submodules["o"].name == "dense"

    def test_mlp_type_and_name(self, adapter: BloomArchitectureAdapter) -> None:
        blocks = self._mapping(adapter)["blocks"]
        assert isinstance(blocks.submodules["mlp"], BloomMLPBridge)
        assert blocks.submodules["mlp"].name == "mlp"

    def test_mlp_in_name(self, adapter: BloomArchitectureAdapter) -> None:
        blocks = self._mapping(adapter)["blocks"]
        assert blocks.submodules["mlp"].submodules["in"].name == "dense_h_to_4h"

    def test_mlp_out_name(self, adapter: BloomArchitectureAdapter) -> None:
        blocks = self._mapping(adapter)["blocks"]
        assert blocks.submodules["mlp"].submodules["out"].name == "dense_4h_to_h"


# ---------------------------------------------------------------------------
# Weight conversion tests
# ---------------------------------------------------------------------------


class TestBloomWeightConversions:
    def test_four_conversion_keys(self, adapter: BloomArchitectureAdapter) -> None:
        convs = adapter.weight_processing_conversions
        assert convs is not None
        assert len(convs) == 4

    def test_qkvo_keys_present(self, adapter: BloomArchitectureAdapter) -> None:
        convs = adapter.weight_processing_conversions
        assert convs is not None

        for key in [
            "blocks.{i}.attn.q",
            "blocks.{i}.attn.k",
            "blocks.{i}.attn.v",
            "blocks.{i}.attn.o",
        ]:
            assert key in convs

    def test_q_rearrange_pattern(self, adapter: BloomArchitectureAdapter) -> None:
        convs = adapter.weight_processing_conversions
        assert convs is not None

        conv = convs["blocks.{i}.attn.q"]

        assert isinstance(conv, ParamProcessingConversion)
        assert isinstance(conv.tensor_conversion, RearrangeTensorConversion)

        assert conv.tensor_conversion.pattern == "(n h) m -> n m h"

    def test_o_rearrange_pattern(self, adapter: BloomArchitectureAdapter) -> None:
        convs = adapter.weight_processing_conversions
        assert convs is not None

        conv = convs["blocks.{i}.attn.o"]

        assert isinstance(conv, ParamProcessingConversion)
        assert isinstance(conv.tensor_conversion, RearrangeTensorConversion)

        assert conv.tensor_conversion.pattern == "m (n h) -> n h m"

    def test_q_rearrange_n_equals_n_heads(
        self,
        adapter: BloomArchitectureAdapter,
    ) -> None:
        convs = adapter.weight_processing_conversions
        assert convs is not None

        conv = convs["blocks.{i}.attn.q"]

        assert isinstance(conv, ParamProcessingConversion)
        assert isinstance(conv.tensor_conversion, RearrangeTensorConversion)

        assert conv.tensor_conversion.axes_lengths["n"] == adapter.cfg.n_heads


# ---------------------------------------------------------------------------
# split_qkv_matrix tests
# ---------------------------------------------------------------------------


class TestBloomSplitQKV:
    def _adapter(self) -> BloomArchitectureAdapter:
        return BloomArchitectureAdapter(_make_cfg())

    def test_returns_three_linears(self) -> None:
        adapter = self._adapter()

        component = _make_qkv_component(64)

        q, k, v = adapter.split_qkv_matrix(component)

        assert isinstance(q, nn.Linear)
        assert isinstance(k, nn.Linear)
        assert isinstance(v, nn.Linear)

    def test_output_shapes(self) -> None:
        adapter = self._adapter()

        component = _make_qkv_component(64)

        q, k, v = adapter.split_qkv_matrix(component)

        assert q.weight.shape == (64, 64)
        assert k.weight.shape == (64, 64)
        assert v.weight.shape == (64, 64)

    def test_biases_present(self) -> None:
        adapter = self._adapter()

        component = _make_qkv_component(64)

        q, k, v = adapter.split_qkv_matrix(component)

        assert q.bias is not None
        assert k.bias is not None
        assert v.bias is not None

    def test_interleaved_split_correctness(self) -> None:
        """
        Bloom stores QKV interleaved:
        [Q0,K0,V0,Q1,K1,V1,...]
        """

        d_model = 12
        n_heads = 3
        d_head = 4

        cfg = _make_cfg(
            n_heads=n_heads,
            d_model=d_model,
        )

        adapter = BloomArchitectureAdapter(cfg)

        component = _make_qkv_component(d_model)

        W = torch.zeros(3 * d_model, d_model)

        for h in range(n_heads):
            start = h * 3 * d_head

            W[start : start + d_head] = 1.0
            W[start + d_head : start + 2 * d_head] = 2.0
            W[start + 2 * d_head : start + 3 * d_head] = 3.0

        component.query_key_value.weight = nn.Parameter(W)

        bias = torch.zeros(3 * d_model)

        for h in range(n_heads):
            start = h * 3 * d_head

            bias[start : start + d_head] = 1.0
            bias[start + d_head : start + 2 * d_head] = 2.0
            bias[start + 2 * d_head : start + 3 * d_head] = 3.0

        component.query_key_value.bias = nn.Parameter(bias)

        q, k, v = adapter.split_qkv_matrix(component)

        assert torch.all(q.weight == 1.0)
        assert torch.all(k.weight == 2.0)
        assert torch.all(v.weight == 3.0)

        assert torch.all(q.bias == 1.0)
        assert torch.all(k.bias == 2.0)
        assert torch.all(v.bias == 3.0)


# ---------------------------------------------------------------------------
# Factory registration tests
# ---------------------------------------------------------------------------


class TestBloomFactoryRegistration:
    def test_factory_key(self) -> None:
        from transformer_lens.factories.architecture_adapter_factory import (
            SUPPORTED_ARCHITECTURES,
        )

        assert "BloomForCausalLM" in SUPPORTED_ARCHITECTURES

    def test_headless_bloom_model_alias(self) -> None:
        """Old headless checkpoints (bigscience-small-testing, norbloom) carry
        architectures=['BloomModel']; they load as BloomForCausalLM with tied
        embeddings, so the alias reuses the same adapter."""
        from transformer_lens.factories.architecture_adapter_factory import (
            SUPPORTED_ARCHITECTURES,
        )

        assert SUPPORTED_ARCHITECTURES["BloomModel"] is BloomArchitectureAdapter

    def test_factory_returns_bloom_adapter(self) -> None:
        from transformer_lens.factories.architecture_adapter_factory import (
            ArchitectureAdapterFactory,
        )

        cfg = _make_cfg()

        adapter = ArchitectureAdapterFactory.select_architecture_adapter(cfg)

        assert isinstance(adapter, BloomArchitectureAdapter)

    def test_import_from_init(self) -> None:
        from transformer_lens.model_bridge.supported_architectures import (
            BloomArchitectureAdapter as FromInit,
        )

        assert FromInit is BloomArchitectureAdapter
