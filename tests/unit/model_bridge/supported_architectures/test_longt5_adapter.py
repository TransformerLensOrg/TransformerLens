"""Unit tests for the LongT5ArchitectureAdapter.

Download-free: synthetic configs and structural assertions only.
"""

from types import SimpleNamespace

import pytest

from tests.unit.model_bridge.supported_architectures.helpers import make_bridge_cfg
from transformer_lens.config import TransformerBridgeConfig
from transformer_lens.factories.architecture_adapter_factory import (
    SUPPORTED_ARCHITECTURES,
)
from transformer_lens.model_bridge.supported_architectures.longt5 import (
    LongT5ArchitectureAdapter,
)


def _make_cfg(encoder_attention_type: str = "transient-global") -> TransformerBridgeConfig:
    cfg = make_bridge_cfg(
        "LongT5ForConditionalGeneration",
        d_model=64,
        d_head=16,
        n_layers=2,
        n_ctx=128,
        n_heads=4,
        d_mlp=128,
        d_vocab=512,
        default_prepend_bos=True,
    )
    cfg.encoder_attention_type = encoder_attention_type
    cfg.is_gated_act = True
    return cfg


class TestLongT5EncoderAttention:
    @pytest.mark.parametrize(
        "attn_type,attr",
        [("local", "LocalSelfAttention"), ("transient-global", "TransientGlobalSelfAttention")],
    )
    def test_encoder_attention_module_selected(self, attn_type, attr):
        adapter = LongT5ArchitectureAdapter(_make_cfg(attn_type))
        attn = adapter.component_mapping["encoder_blocks"].submodules["attn"]
        assert attn.name == f"layer.0.{attr}"
        # Block-wise position bias shapes rule out generic reconstruction.
        assert attn.maintain_native_attention is True
        pos = adapter.component_mapping["pos_embed"]
        assert pos.name == f"encoder.block.0.layer.0.{attr}.relative_attention_bias"

    def test_decoder_inherited_from_t5(self):
        adapter = LongT5ArchitectureAdapter(_make_cfg())
        dec = adapter.component_mapping["decoder_blocks"].submodules
        assert dec["self_attn"].name == "layer.0.SelfAttention"
        assert dec["cross_attn"].name == "layer.1.EncDecAttention"


class TestLongT5Registration:
    def test_factory_lookup(self):
        assert (
            SUPPORTED_ARCHITECTURES["LongT5ForConditionalGeneration"] is LongT5ArchitectureAdapter
        )

    def test_model_type_detection(self):
        from transformer_lens.model_bridge.sources.transformers import (
            determine_architecture_from_hf_config,
        )

        cfg = SimpleNamespace(model_type="longt5", architectures=[])
        assert determine_architecture_from_hf_config(cfg) == "LongT5ForConditionalGeneration"

    def test_seq2seq_loader_class(self):
        from transformer_lens.utilities.architectures import SEQ2SEQ_ARCHITECTURES

        assert "LongT5ForConditionalGeneration" in SEQ2SEQ_ARCHITECTURES
