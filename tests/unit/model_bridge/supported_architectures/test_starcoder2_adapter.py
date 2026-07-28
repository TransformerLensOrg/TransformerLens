"""Unit tests for the Starcoder2ArchitectureAdapter.

Download-free: synthetic configs and structural assertions only.
"""

from types import SimpleNamespace

import pytest

from tests.unit.model_bridge.supported_architectures.helpers import make_bridge_cfg
from transformer_lens.config import TransformerBridgeConfig
from transformer_lens.factories.architecture_adapter_factory import (
    SUPPORTED_ARCHITECTURES,
)
from transformer_lens.model_bridge.generalized_components import (
    BlockBridge,
    EmbeddingBridge,
    GatedMLPBridge,
    LinearBridge,
    MLPBridge,
    NormalizationBridge,
    PositionEmbeddingsAttentionBridge,
    RMSNormalizationBridge,
    RotaryEmbeddingBridge,
    UnembeddingBridge,
)
from transformer_lens.model_bridge.supported_architectures.starcoder2 import (
    Starcoder2ArchitectureAdapter,
)


def _make_cfg(n_key_value_heads: int | None = 2) -> TransformerBridgeConfig:
    return make_bridge_cfg(
        "Starcoder2ForCausalLM",
        d_model=64,
        d_head=16,
        n_layers=2,
        n_ctx=128,
        n_heads=4,
        d_mlp=256,
        d_vocab=512,
        n_key_value_heads=n_key_value_heads,
        default_prepend_bos=True,
    )


@pytest.fixture(scope="class")
def adapter() -> Starcoder2ArchitectureAdapter:
    return Starcoder2ArchitectureAdapter(_make_cfg())


class TestStarcoder2AdapterConfig:
    def test_config_flags(self, adapter):
        assert adapter.cfg.normalization_type == "LN"
        assert adapter.cfg.positional_embedding_type == "rotary"
        assert adapter.cfg.final_rms is False
        assert adapter.cfg.gated_mlp is False
        assert adapter.cfg.n_key_value_heads == 2

    def test_qkv_bias_conversions_use_kv_head_count(self, adapter):
        """StarCoder2 biases every q/k/v; compat-mode weight access needs the
        K/V bias reshapes split by the kv-head count (GQA), not n_heads."""
        conv = adapter.weight_processing_conversions
        for key in ("blocks.{i}.attn.k.bias", "blocks.{i}.attn.v.bias"):
            assert key in conv, f"missing {key}"
            assert conv[key].tensor_conversion.axes_lengths["h"] == adapter.cfg.n_key_value_heads

    def test_conversion_key_set(self, adapter):
        """q/k/v get both a weight and a per-head bias conversion; o's bias stays
        [d_model], so it must not gain a per-head reshape."""
        assert set(adapter.weight_processing_conversions) == {
            "blocks.{i}.attn.q.weight",
            "blocks.{i}.attn.k.weight",
            "blocks.{i}.attn.v.weight",
            "blocks.{i}.attn.o.weight",
            "blocks.{i}.attn.q.bias",
            "blocks.{i}.attn.k.bias",
            "blocks.{i}.attn.v.bias",
        }

    def test_missing_kv_heads_falls_back_to_n_heads(self):
        """MHA checkpoints omit n_key_value_heads; the bias reshapes must still
        be emitted, split by n_heads."""
        adapter = Starcoder2ArchitectureAdapter(_make_cfg(n_key_value_heads=None))
        conv = adapter.weight_processing_conversions
        assert conv["blocks.{i}.attn.k.bias"].tensor_conversion.axes_lengths["h"] == 4


class TestStarcoder2ComponentMapping:
    def test_top_level_mapping(self, adapter):
        mapping = adapter.component_mapping
        assert isinstance(mapping["embed"], EmbeddingBridge)
        assert isinstance(mapping["rotary_emb"], RotaryEmbeddingBridge)
        assert isinstance(mapping["blocks"], BlockBridge)
        assert isinstance(mapping["unembed"], UnembeddingBridge)
        assert mapping["embed"].name == "model.embed_tokens"
        assert mapping["blocks"].name == "model.layers"
        assert mapping["unembed"].name == "lm_head"

    def test_attention_is_separate_qkvo(self, adapter):
        """Separate q/k/v/o projections — unlike GPTBigCode's fused c_attn."""
        attn = adapter.component_mapping["blocks"].submodules["attn"]
        assert isinstance(attn, PositionEmbeddingsAttentionBridge)
        assert attn.name == "self_attn"
        expected = {"q": "q_proj", "k": "k_proj", "v": "v_proj", "o": "o_proj"}
        assert set(attn.submodules) == set(expected)
        for key, hf_name in expected.items():
            assert isinstance(attn.submodules[key], LinearBridge)
            assert attn.submodules[key].name == hf_name

    def test_norms_are_plain_layernorm(self, adapter):
        """StarCoder2 uses nn.LayerNorm despite its llama-like shape."""
        submodules = adapter.component_mapping["blocks"].submodules
        for key, hf_name in (("ln1", "input_layernorm"), ("ln2", "post_attention_layernorm")):
            assert isinstance(submodules[key], NormalizationBridge)
            assert not isinstance(submodules[key], RMSNormalizationBridge)
            assert submodules[key].name == hf_name
        assert adapter.component_mapping["ln_final"].name == "model.norm"

    def test_mlp_is_plain_c_fc_c_proj(self, adapter):
        mlp = adapter.component_mapping["blocks"].submodules["mlp"]
        assert isinstance(mlp, MLPBridge)
        assert not isinstance(mlp, GatedMLPBridge)
        assert mlp.submodules["in"].name == "c_fc"
        assert mlp.submodules["out"].name == "c_proj"


class TestStarcoder2Registration:
    def test_factory_lookup(self):
        assert SUPPORTED_ARCHITECTURES["Starcoder2ForCausalLM"] is Starcoder2ArchitectureAdapter

    def test_model_type_detection(self):
        from transformer_lens.model_bridge.sources.transformers import (
            determine_architecture_from_hf_config,
        )

        cfg = SimpleNamespace(model_type="starcoder2", architectures=[])
        assert determine_architecture_from_hf_config(cfg) == "Starcoder2ForCausalLM"
