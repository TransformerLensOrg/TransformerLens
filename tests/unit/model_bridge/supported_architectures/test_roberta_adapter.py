"""Unit tests for RobertaArchitectureAdapter."""

from types import SimpleNamespace
from typing import Any

import pytest
from torch import nn

from transformer_lens.config.transformer_bridge_config import TransformerBridgeConfig
from transformer_lens.model_bridge.generalized_components import (
    BlockBridge,
    EmbeddingBridge,
)
from transformer_lens.model_bridge.supported_architectures.bert import (
    BertArchitectureAdapter,
)
from transformer_lens.model_bridge.supported_architectures.roberta import (
    RobertaArchitectureAdapter,
)


def _make_cfg(
    architecture: str = "RobertaForMaskedLM",
    n_ctx: int = 514,
    **overrides: Any,
) -> TransformerBridgeConfig:
    """n_ctx mimics HF's max_position_embeddings."""
    cfg = TransformerBridgeConfig(
        d_model=768,
        d_head=64,
        n_heads=12,
        n_layers=12,
        n_ctx=n_ctx,
        d_vocab=50265,
        architecture=architecture,
    )
    for name, value in overrides.items():
        setattr(cfg, name, value)
    return cfg


def _layout(component: Any) -> dict[str, Any]:
    """Nested (bridge type, HF path, native-autograd flag) layout, compared against BERT's."""
    return {
        key: (
            type(sub).__name__,
            sub.name,
            getattr(sub, "use_native_layernorm_autograd", None),
            _layout(sub),
        )
        for key, sub in component.submodules.items()
    }


def _hf_model_with_table(rows: int) -> SimpleNamespace:
    """Stand-in RobertaForMaskedLM tree: no pooler, no cls heads, a real position table."""
    return SimpleNamespace(
        roberta=SimpleNamespace(
            pooler=None,
            embeddings=SimpleNamespace(position_embeddings=nn.Embedding(rows, 8, padding_idx=1)),
        ),
        lm_head=SimpleNamespace(),
    )


@pytest.fixture(scope="module")
def adapter() -> RobertaArchitectureAdapter:
    return RobertaArchitectureAdapter(_make_cfg())


@pytest.fixture(scope="module")
def bert_adapter() -> BertArchitectureAdapter:
    return BertArchitectureAdapter(_make_cfg(architecture="BertForMaskedLM", n_ctx=512))


class TestRobertaComponentMapping:
    def test_top_level_keys_match_bert(
        self, adapter: RobertaArchitectureAdapter, bert_adapter: BertArchitectureAdapter
    ) -> None:
        assert set(adapter.component_mapping) == set(bert_adapter.component_mapping)

    def test_top_level_bridge_types_match_bert(
        self, adapter: RobertaArchitectureAdapter, bert_adapter: BertArchitectureAdapter
    ) -> None:
        for key, component in adapter.component_mapping.items():
            bert_component = bert_adapter.component_mapping[key]
            assert type(component) is type(bert_component), key
            assert getattr(component, "use_native_layernorm_autograd", None) == getattr(
                bert_component, "use_native_layernorm_autograd", None
            ), key

    def test_top_level_hf_paths(self, adapter: RobertaArchitectureAdapter) -> None:
        mapping = adapter.component_mapping
        assert mapping["embed"].name == "roberta.embeddings.word_embeddings"
        assert mapping["token_type_embed"].name == "roberta.embeddings.token_type_embeddings"
        assert mapping["pos_embed"].name == "roberta.embeddings.position_embeddings"
        assert mapping["embed_ln"].name == "roberta.embeddings.LayerNorm"
        assert mapping["blocks"].name == "roberta.encoder.layer"
        assert mapping["mlm_head"].name == "lm_head.dense"
        assert mapping["ln_final"].name == "lm_head.layer_norm"
        assert mapping["unembed"].name == "lm_head.decoder"

    def test_keeps_token_type_embedding(self, adapter: RobertaArchitectureAdapter) -> None:
        token_type_embed = adapter.component_mapping["token_type_embed"]
        assert isinstance(token_type_embed, EmbeddingBridge)

    def test_block_layout_matches_bert(
        self, adapter: RobertaArchitectureAdapter, bert_adapter: BertArchitectureAdapter
    ) -> None:
        blocks = adapter.component_mapping["blocks"]
        assert isinstance(blocks, BlockBridge)
        assert _layout(blocks) == _layout(bert_adapter.component_mapping["blocks"])

    def test_preserves_bert_block_hook_aliases(self, adapter: RobertaArchitectureAdapter) -> None:
        blocks = adapter.component_mapping["blocks"]
        assert blocks.hook_aliases["hook_mlp_in"] == "mlp.in.hook_in"
        assert blocks.hook_aliases["hook_mlp_out"] == "mlp.out.hook_out"


class TestRobertaAdapterConfig:
    @pytest.mark.parametrize("rows", [514, 1026])
    def test_context_length_is_read_off_the_position_table(self, rows: int) -> None:
        adapter = RobertaArchitectureAdapter(_make_cfg(n_ctx=rows))
        hf_model = _hf_model_with_table(rows)

        adapter.prepare_model(hf_model)
        adapter.prepare_model(hf_model)

        assert adapter.cfg.n_ctx == rows - 2

    def test_inherits_bert_post_ln_behavior(self, adapter: RobertaArchitectureAdapter) -> None:
        assert adapter.cfg.normalization_type == "LN"
        assert adapter.cfg.positional_embedding_type == "standard"
        assert adapter.cfg.final_rms is False
        assert adapter.supports_fold_ln is False

    def test_inherits_bert_weight_conversions(
        self, adapter: RobertaArchitectureAdapter, bert_adapter: BertArchitectureAdapter
    ) -> None:
        assert set(adapter.weight_processing_conversions) == set(
            bert_adapter.weight_processing_conversions
        )

    @pytest.mark.parametrize("prepend_bos", [True, False])
    def test_does_not_override_tokenizer_policy(self, prepend_bos: bool) -> None:
        adapter = RobertaArchitectureAdapter(
            _make_cfg(default_prepend_bos=prepend_bos, default_padding_side="left")
        )
        assert adapter.cfg.default_prepend_bos is prepend_bos
        assert adapter.cfg.default_padding_side == "left"

    def test_generation_is_not_supported(self) -> None:
        assert RobertaArchitectureAdapter.supports_generation is False


class TestRobertaPrepareModel:
    def test_prepare_model_keeps_the_mapping(self) -> None:
        adapter = RobertaArchitectureAdapter(_make_cfg())
        mapping_before = dict(adapter.components)

        adapter.prepare_model(_hf_model_with_table(514))

        assert adapter.components == mapping_before


class TestRobertaFactoryRouting:
    def test_roberta_for_masked_lm_maps_to_roberta_adapter(self) -> None:
        from transformer_lens.factories.architecture_adapter_factory import (
            SUPPORTED_ARCHITECTURES,
        )

        assert SUPPORTED_ARCHITECTURES["RobertaForMaskedLM"] is RobertaArchitectureAdapter
