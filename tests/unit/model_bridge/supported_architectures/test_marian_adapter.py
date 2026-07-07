"""Unit tests for the MarianArchitectureAdapter.

These tests are download-free: they use tiny synthetic configs and assert the
adapter's structural contract rather than loading real checkpoints.
"""

from typing import Any

import pytest

from transformer_lens.config import TransformerBridgeConfig
from transformer_lens.factories.architecture_adapter_factory import (
    SUPPORTED_ARCHITECTURES,
)
from transformer_lens.model_bridge.generalized_components import (
    AttentionBridge,
    BlockBridge,
    EmbeddingBridge,
    LinearBridge,
    PosEmbedBridge,
    SymbolicBridge,
    UnembeddingBridge,
)
from transformer_lens.model_bridge.sources.transformers import (
    determine_architecture_from_hf_config,
)
from transformer_lens.model_bridge.supported_architectures.marian import (
    MarianArchitectureAdapter,
)


def _base_cfg() -> TransformerBridgeConfig:
    cfg = TransformerBridgeConfig(
        d_model=64,
        d_head=16,
        n_layers=2,
        n_ctx=128,
        n_heads=4,
        d_mlp=256,
        d_vocab=512,
        architecture="MarianMTModel",
    )
    cfg.encoder_layers = 2
    cfg.decoder_layers = 2
    cfg.encoder_attention_heads = 4
    cfg.decoder_attention_heads = 4
    cfg.encoder_ffn_dim = 256
    cfg.decoder_ffn_dim = 256
    return cfg


@pytest.fixture(scope="class")
def adapter() -> MarianArchitectureAdapter:
    return MarianArchitectureAdapter(_base_cfg())


def _mapping(adapter: MarianArchitectureAdapter) -> dict[str, Any]:
    mapping = adapter.component_mapping
    assert mapping is not None
    return mapping


def _encoder_block(adapter: MarianArchitectureAdapter) -> BlockBridge:
    block = _mapping(adapter)["encoder_blocks"]
    assert isinstance(block, BlockBridge)
    return block


def _decoder_block(adapter: MarianArchitectureAdapter) -> BlockBridge:
    block = _mapping(adapter)["decoder_blocks"]
    assert isinstance(block, BlockBridge)
    return block


class TestMarianAdapterConfig:
    def test_post_ln_processing_guards(self, adapter: MarianArchitectureAdapter) -> None:
        assert adapter.cfg.normalization_type == "LN"
        assert adapter.cfg.positional_embedding_type == "standard"
        assert adapter.cfg.gated_mlp is False
        assert adapter.cfg.attn_only is False
        assert adapter.supports_fold_ln is False
        assert adapter.supports_center_writing_weights is False
        assert adapter.weight_processing_conversions == {}

    def test_cfg_uses_symmetric_marian_dimensions(self, adapter: MarianArchitectureAdapter) -> None:
        assert adapter.cfg.n_layers == 2
        assert adapter.cfg.n_heads == 4
        assert adapter.cfg.d_head == 16
        assert adapter.cfg.d_mlp == 256

    def test_scale_embedding_defaults_true(self, adapter: MarianArchitectureAdapter) -> None:
        # opus-mt checkpoints scale token embeddings by sqrt(d_model) in the HF
        # forward; the adapter surfaces the flag for TL-side consumers.
        assert adapter.cfg.scale_embedding is True

    def test_scale_embedding_respects_config_value(self) -> None:
        cfg = _base_cfg()
        cfg.scale_embedding = False
        adapter = MarianArchitectureAdapter(cfg)
        assert adapter.cfg.scale_embedding is False


class TestMarianComponentMapping:
    # Unlike Bart, Marian has no layernorm_embedding on either side.
    EXPECTED_TOP_LEVEL_KEYS = {
        "embed",
        "pos_embed",
        "encoder_blocks",
        "decoder_embed",
        "decoder_pos_embed",
        "decoder_blocks",
        "unembed",
    }

    def test_top_level_keys_are_expected(self, adapter: MarianArchitectureAdapter) -> None:
        assert set(_mapping(adapter)) == self.EXPECTED_TOP_LEVEL_KEYS

    def test_top_level_component_types(self, adapter: MarianArchitectureAdapter) -> None:
        mapping = _mapping(adapter)
        assert isinstance(mapping["embed"], EmbeddingBridge)
        assert isinstance(mapping["pos_embed"], PosEmbedBridge)
        assert isinstance(mapping["encoder_blocks"], BlockBridge)
        assert isinstance(mapping["decoder_embed"], EmbeddingBridge)
        assert isinstance(mapping["decoder_pos_embed"], PosEmbedBridge)
        assert isinstance(mapping["decoder_blocks"], BlockBridge)
        assert isinstance(mapping["unembed"], UnembeddingBridge)

    def test_top_level_paths(self, adapter: MarianArchitectureAdapter) -> None:
        mapping = _mapping(adapter)
        assert mapping["embed"].name == "model.encoder.embed_tokens"
        assert mapping["pos_embed"].name == "model.encoder.embed_positions"
        assert mapping["encoder_blocks"].name == "model.encoder.layers"
        assert mapping["decoder_embed"].name == "model.decoder.embed_tokens"
        assert mapping["decoder_pos_embed"].name == "model.decoder.embed_positions"
        assert mapping["decoder_blocks"].name == "model.decoder.layers"
        assert mapping["unembed"].name == "lm_head"


class TestMarianEncoderBlock:
    def test_encoder_block_submodules(self, adapter: MarianArchitectureAdapter) -> None:
        assert set(_encoder_block(adapter).submodules) == {"attn", "ln1", "ln2", "mlp"}

    def test_encoder_self_attention(self, adapter: MarianArchitectureAdapter) -> None:
        attn = _encoder_block(adapter).submodules["attn"]
        assert isinstance(attn, AttentionBridge)
        assert attn.name == "self_attn"
        assert getattr(attn, "is_cross_attention", False) is False
        for name, expected_path in (
            ("q", "q_proj"),
            ("k", "k_proj"),
            ("v", "v_proj"),
            ("o", "out_proj"),
        ):
            submodule = attn.submodules[name]
            assert isinstance(submodule, LinearBridge)
            assert submodule.name == expected_path

    def test_encoder_layer_norm_paths(self, adapter: MarianArchitectureAdapter) -> None:
        submodules = _encoder_block(adapter).submodules
        assert submodules["ln1"].name == "self_attn_layer_norm"
        assert submodules["ln2"].name == "final_layer_norm"
        assert submodules["ln1"].use_native_layernorm_autograd is True
        assert submodules["ln2"].use_native_layernorm_autograd is True

    def test_encoder_mlp_uses_symbolic_fc1_fc2(self, adapter: MarianArchitectureAdapter) -> None:
        mlp = _encoder_block(adapter).submodules["mlp"]
        assert isinstance(mlp, SymbolicBridge)
        assert set(mlp.submodules) == {"in", "out"}
        assert mlp.submodules["in"].name == "fc1"
        assert mlp.submodules["out"].name == "fc2"


class TestMarianDecoderBlock:
    def test_decoder_block_submodules(self, adapter: MarianArchitectureAdapter) -> None:
        assert set(_decoder_block(adapter).submodules) == {
            "self_attn",
            "ln1",
            "cross_attn",
            "ln2",
            "ln3",
            "mlp",
        }

    def test_decoder_cross_attention(self, adapter: MarianArchitectureAdapter) -> None:
        cross_attn = _decoder_block(adapter).submodules["cross_attn"]
        assert isinstance(cross_attn, AttentionBridge)
        assert cross_attn.name == "encoder_attn"
        assert cross_attn.is_cross_attention is True

    def test_decoder_layer_norm_paths(self, adapter: MarianArchitectureAdapter) -> None:
        submodules = _decoder_block(adapter).submodules
        assert submodules["ln1"].name == "self_attn_layer_norm"
        assert submodules["ln2"].name == "encoder_attn_layer_norm"
        assert submodules["ln3"].name == "final_layer_norm"

    def test_decoder_self_attention_hook_aliases(self, adapter: MarianArchitectureAdapter) -> None:
        aliases = _decoder_block(adapter).hook_aliases
        assert aliases["hook_attn_in"] == "self_attn.hook_attn_in"
        assert aliases["hook_attn_out"] == "self_attn.hook_out"
        assert aliases["hook_q_input"] == "self_attn.hook_q_input"
        assert aliases["hook_k_input"] == "self_attn.hook_k_input"
        assert aliases["hook_v_input"] == "self_attn.hook_v_input"


class TestMarianRegistration:
    def test_factory_lookup_returns_adapter_class(self) -> None:
        assert SUPPORTED_ARCHITECTURES["MarianMTModel"] is MarianArchitectureAdapter

    def test_marian_model_type_detection(self) -> None:
        from transformers import MarianConfig

        cfg = MarianConfig(
            d_model=64,
            encoder_layers=2,
            decoder_layers=2,
            encoder_attention_heads=4,
            decoder_attention_heads=4,
            encoder_ffn_dim=256,
            decoder_ffn_dim=256,
        )
        assert determine_architecture_from_hf_config(cfg) == "MarianMTModel"


class TestMarianSymmetryGuards:
    def test_rejects_asymmetric_layers(self) -> None:
        cfg = _base_cfg()
        cfg.decoder_layers = 3
        with pytest.raises(ValueError, match="encoder_layers=2, decoder_layers=3"):
            MarianArchitectureAdapter(cfg)

    def test_rejects_asymmetric_attention_heads(self) -> None:
        cfg = _base_cfg()
        cfg.decoder_attention_heads = 8
        with pytest.raises(
            ValueError, match="encoder_attention_heads=4, decoder_attention_heads=8"
        ):
            MarianArchitectureAdapter(cfg)

    def test_rejects_asymmetric_ffn_dims(self) -> None:
        cfg = _base_cfg()
        cfg.decoder_ffn_dim = 512
        with pytest.raises(ValueError, match="encoder_ffn_dim=256, decoder_ffn_dim=512"):
            MarianArchitectureAdapter(cfg)
