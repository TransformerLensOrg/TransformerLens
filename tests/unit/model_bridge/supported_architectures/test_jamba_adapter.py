"""Unit tests for JambaArchitectureAdapter without model downloads."""

import pytest

from transformer_lens.config import TransformerBridgeConfig
from transformer_lens.factories.architecture_adapter_factory import (
    SUPPORTED_ARCHITECTURES,
    ArchitectureAdapterFactory,
)
from transformer_lens.model_bridge.generalized_components import (
    AttentionBridge,
    BlockBridge,
    DepthwiseConv1DBridge,
    EmbeddingBridge,
    GatedMLPBridge,
    LinearBridge,
    MoEBridge,
    RMSNormalizationBridge,
    SSMMixerBridge,
    UnembeddingBridge,
)
from transformer_lens.model_bridge.supported_architectures.jamba import (
    JambaArchitectureAdapter,
)


def _make_cfg(
    *,
    d_model: int = 64,
    n_layers: int = 4,
    layers_block_type: list[str] | None = None,
    mamba_expand: int | None = 2,
    mamba_d_state: int | None = 16,
    mamba_d_conv: int | None = 4,
    mamba_dt_rank: int | None = 8,
    num_experts: int | None = 1,
    n_key_value_heads: int | None = 2,
) -> TransformerBridgeConfig:
    cfg = TransformerBridgeConfig(
        d_model=d_model,
        d_head=d_model // 4,
        n_layers=n_layers,
        n_ctx=128,
        n_heads=4,
        d_vocab=256,
        d_mlp=128,
        n_key_value_heads=n_key_value_heads,
        architecture="JambaForCausalLM",
    )
    if layers_block_type is not None:
        setattr(cfg, "layers_block_type", layers_block_type)
    if mamba_expand is not None:
        setattr(cfg, "mamba_expand", mamba_expand)
    if mamba_d_state is not None:
        setattr(cfg, "mamba_d_state", mamba_d_state)
    if mamba_d_conv is not None:
        setattr(cfg, "mamba_d_conv", mamba_d_conv)
    if mamba_dt_rank is not None:
        setattr(cfg, "mamba_dt_rank", mamba_dt_rank)
    if num_experts is not None:
        setattr(cfg, "num_experts", num_experts)
    return cfg


@pytest.fixture(scope="class")
def adapter() -> JambaArchitectureAdapter:
    return JambaArchitectureAdapter(
        _make_cfg(layers_block_type=["mamba", "mamba", "attention", "mamba"], num_experts=1)
    )


class TestJambaAdapterConfig:
    def test_architecture_flags(self, adapter: JambaArchitectureAdapter) -> None:
        assert adapter.cfg.normalization_type == "RMS"
        assert adapter.cfg.uses_rms_norm is True
        assert adapter.cfg.positional_embedding_type == "none"
        assert adapter.cfg.gated_mlp is True
        assert adapter.cfg.attn_only is False
        assert adapter.cfg.final_rms is True
        assert adapter.cfg.default_prepend_bos is True

    def test_uses_standard_kv_cache_path(self, adapter: JambaArchitectureAdapter) -> None:
        assert adapter.cfg.is_stateful is False

    def test_layers_block_type_propagated(self, adapter: JambaArchitectureAdapter) -> None:
        assert getattr(adapter.cfg, "layers_block_type") == [
            "mamba",
            "mamba",
            "attention",
            "mamba",
        ]

    def test_hf_layer_types_normalized_to_tl_names(self) -> None:
        adapter = JambaArchitectureAdapter(
            _make_cfg(layers_block_type=["linear_attention", "full_attention"])
        )
        assert getattr(adapter.cfg, "layers_block_type") == ["mamba", "attention"]

    def test_mamba_dimensions_derived_from_hf_config(
        self, adapter: JambaArchitectureAdapter
    ) -> None:
        assert getattr(adapter.cfg, "mamba_expand") == 2
        assert getattr(adapter.cfg, "intermediate_size") == 2 * 64
        assert getattr(adapter.cfg, "state_size") == 16
        assert getattr(adapter.cfg, "conv_kernel") == 4
        assert getattr(adapter.cfg, "mamba_dt_rank") == 8

    def test_gqa_n_key_value_heads(self, adapter: JambaArchitectureAdapter) -> None:
        assert adapter.cfg.n_key_value_heads == 2

    def test_verification_phases_and_weight_processing(
        self, adapter: JambaArchitectureAdapter
    ) -> None:
        assert adapter.applicable_phases == [1, 2, 3, 4]
        assert adapter.weight_processing_conversions == {}
        assert adapter.supports_fold_ln is False


class TestJambaTopLevelComponents:
    def test_types_and_hf_paths(self, adapter: JambaArchitectureAdapter) -> None:
        mapping = adapter.get_component_mapping()

        assert set(mapping) == {"embed", "blocks", "ln_final", "unembed"}
        assert isinstance(mapping["embed"], EmbeddingBridge)
        assert isinstance(mapping["blocks"], BlockBridge)
        assert isinstance(mapping["ln_final"], RMSNormalizationBridge)
        assert isinstance(mapping["unembed"], UnembeddingBridge)
        assert mapping["embed"].name == "model.embed_tokens"
        assert mapping["blocks"].name == "model.layers"
        assert mapping["ln_final"].name == "model.final_layernorm"
        assert mapping["unembed"].name == "lm_head"
        assert "rotary_emb" not in mapping


class TestJambaBlockComponents:
    @pytest.fixture(scope="class")
    def blocks(self, adapter: JambaArchitectureAdapter) -> BlockBridge:
        component = adapter.get_component_mapping()["blocks"]
        assert isinstance(component, BlockBridge)
        return component

    def test_block_submodule_keys(self, blocks: BlockBridge) -> None:
        assert set(blocks.submodules) == {"ln1", "ln2", "attn", "mixer", "mlp"}

    def test_norm_paths(self, blocks: BlockBridge) -> None:
        assert isinstance(blocks.submodules["ln1"], RMSNormalizationBridge)
        assert isinstance(blocks.submodules["ln2"], RMSNormalizationBridge)
        assert blocks.submodules["ln1"].name == "input_layernorm"
        assert blocks.submodules["ln2"].name == "pre_ff_layernorm"

    def test_attn_optional_native(self, blocks: BlockBridge) -> None:
        attn = blocks.submodules["attn"]
        assert isinstance(attn, AttentionBridge)
        assert attn.optional is True
        assert attn.maintain_native_attention is True
        assert attn.requires_position_embeddings is False
        assert attn.requires_attention_mask is True
        assert attn.name == "self_attn"
        assert set(attn.submodules) == {"q", "k", "v", "o"}

    def test_mixer_is_optional_ssm_mixer_bridge(self, blocks: BlockBridge) -> None:
        mixer = blocks.submodules["mixer"]
        assert isinstance(mixer, SSMMixerBridge)
        assert mixer.optional is True
        assert mixer.name == "mamba"

    def test_mixer_mamba1_submodules(self, blocks: BlockBridge) -> None:
        mixer = blocks.submodules["mixer"]
        assert isinstance(mixer, SSMMixerBridge)
        assert set(mixer.submodules) == {
            "in_proj",
            "conv1d",
            "x_proj",
            "dt_proj",
            "out_proj",
            "dt_layernorm",
            "b_layernorm",
            "c_layernorm",
        }
        assert isinstance(mixer.submodules["in_proj"], LinearBridge)
        assert isinstance(mixer.submodules["conv1d"], DepthwiseConv1DBridge)
        assert isinstance(mixer.submodules["x_proj"], LinearBridge)
        assert isinstance(mixer.submodules["dt_proj"], LinearBridge)
        assert isinstance(mixer.submodules["out_proj"], LinearBridge)
        for name in ("dt_layernorm", "b_layernorm", "c_layernorm"):
            assert isinstance(mixer.submodules[name], RMSNormalizationBridge)
            assert mixer.submodules[name].optional is True

    def test_dense_mlp_when_num_experts_one(self, blocks: BlockBridge) -> None:
        mlp = blocks.submodules["mlp"]
        assert isinstance(mlp, GatedMLPBridge)
        assert mlp.name == "feed_forward"
        assert set(mlp.submodules) == {"gate", "in", "out"}
        assert mlp.submodules["gate"].name == "gate_proj"
        assert mlp.submodules["in"].name == "up_proj"
        assert mlp.submodules["out"].name == "down_proj"


class TestJambaMoEMapping:
    def test_moe_bridge_when_num_experts_gt_one(self) -> None:
        adapter = JambaArchitectureAdapter(
            _make_cfg(layers_block_type=["mamba", "attention"], num_experts=4)
        )
        mlp = adapter.get_component_mapping()["blocks"].submodules["mlp"]
        assert isinstance(mlp, MoEBridge)
        assert mlp.name == "feed_forward"
        assert set(mlp.submodules) == {"gate", "in", "out", "router"}
        assert mlp.submodules["router"].name == "router"
        assert mlp.submodules["router"].optional is True
        assert mlp.submodules["gate"].optional is True


class TestJambaFactoryRegistration:
    def test_factory_selects_jamba_adapter(self) -> None:
        adapter = ArchitectureAdapterFactory.select_architecture_adapter(
            _make_cfg(layers_block_type=["mamba"] * 4)
        )
        assert isinstance(adapter, JambaArchitectureAdapter)

    def test_architecture_key_present(self) -> None:
        assert "JambaForCausalLM" in SUPPORTED_ARCHITECTURES
        assert SUPPORTED_ARCHITECTURES["JambaForCausalLM"] is JambaArchitectureAdapter


class TestJambaModelRegistry:
    def test_canonical_author_is_ai21labs(self) -> None:
        from transformer_lens.tools.model_registry import CANONICAL_AUTHORS_BY_ARCH

        assert CANONICAL_AUTHORS_BY_ARCH.get("JambaForCausalLM") == ["ai21labs"]

    def test_listed_in_hf_supported_architectures(self) -> None:
        from transformer_lens.tools.model_registry import HF_SUPPORTED_ARCHITECTURES

        assert "JambaForCausalLM" in HF_SUPPORTED_ARCHITECTURES
