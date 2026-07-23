"""Unit tests for Zamba2ArchitectureAdapter without model downloads."""

import pytest

from transformer_lens.config import TransformerBridgeConfig
from transformer_lens.factories.architecture_adapter_factory import (
    ArchitectureAdapterFactory,
)
from transformer_lens.model_bridge.generalized_components import (
    DepthwiseConv1DBridge,
    EmbeddingBridge,
    GatedRMSNormBridge,
    LinearBridge,
    RMSNormalizationBridge,
    SSM2MixerBridge,
    SSMBlockBridge,
    UnembeddingBridge,
)
from transformer_lens.model_bridge.supported_architectures.zamba2 import (
    Zamba2ArchitectureAdapter,
)


def _make_cfg(
    *,
    d_model: int = 64,
    layers_block_type: list[str] | None = None,
    mamba_expand: int | None = 3,
    mamba_ngroups: int | None = 4,
    mamba_d_state: int | None = 8,
    num_mem_blocks: int | None = 2,
    use_shared_attention_adapter: bool | None = True,
) -> TransformerBridgeConfig:
    cfg = TransformerBridgeConfig(
        d_model=d_model,
        d_head=8,
        n_layers=4,
        n_ctx=128,
        n_heads=8,
        d_vocab=256,
        architecture="Zamba2ForCausalLM",
    )
    if layers_block_type is not None:
        setattr(cfg, "layers_block_type", layers_block_type)
    if mamba_expand is not None:
        setattr(cfg, "mamba_expand", mamba_expand)
    if mamba_ngroups is not None:
        setattr(cfg, "mamba_ngroups", mamba_ngroups)
    if mamba_d_state is not None:
        setattr(cfg, "mamba_d_state", mamba_d_state)
    if num_mem_blocks is not None:
        setattr(cfg, "num_mem_blocks", num_mem_blocks)
    if use_shared_attention_adapter is not None:
        setattr(cfg, "use_shared_attention_adapter", use_shared_attention_adapter)
    return cfg


@pytest.fixture(scope="class")
def adapter() -> Zamba2ArchitectureAdapter:
    return Zamba2ArchitectureAdapter(
        _make_cfg(layers_block_type=["mamba", "hybrid", "mamba", "hybrid"])
    )


class TestZamba2AdapterConfig:
    def test_architecture_flags(self, adapter: Zamba2ArchitectureAdapter) -> None:
        assert adapter.cfg.normalization_type == "RMS"
        assert adapter.cfg.uses_rms_norm is True
        assert adapter.cfg.positional_embedding_type == "none"
        assert adapter.cfg.gated_mlp is False
        assert adapter.cfg.attn_only is False
        assert adapter.cfg.final_rms is True

    def test_uses_standard_kv_cache_path(self, adapter: Zamba2ArchitectureAdapter) -> None:
        assert adapter.cfg.is_stateful is False

    def test_hybrid_metadata_is_propagated(self, adapter: Zamba2ArchitectureAdapter) -> None:
        assert getattr(adapter.cfg, "layers_block_type") == [
            "linear_attention",
            "hybrid",
            "linear_attention",
            "hybrid",
        ]
        assert getattr(adapter.cfg, "num_mem_blocks") == 2
        assert getattr(adapter.cfg, "use_shared_attention_adapter") is True

    def test_layer_type_metadata_is_copied(self) -> None:
        layer_types = ["mamba", "hybrid"]
        adapter = Zamba2ArchitectureAdapter(_make_cfg(layers_block_type=layer_types))

        # HF "mamba" normalizes to the canonical name; "hybrid" passes through.
        assert getattr(adapter.cfg, "layers_block_type") == ["linear_attention", "hybrid"]
        assert getattr(adapter.cfg, "layers_block_type") is not layer_types

    def test_mamba_dimensions_are_derived_from_hf_config(
        self, adapter: Zamba2ArchitectureAdapter
    ) -> None:
        assert getattr(adapter.cfg, "mamba_intermediate_size") == 3 * 64
        assert getattr(adapter.cfg, "conv_dim") == 3 * 64 + 2 * 4 * 8

    def test_zamba2_specific_defaults(self) -> None:
        cfg = _make_cfg(
            layers_block_type=None,
            mamba_expand=None,
            mamba_ngroups=None,
            mamba_d_state=None,
            num_mem_blocks=None,
            use_shared_attention_adapter=None,
        )
        adapter = Zamba2ArchitectureAdapter(cfg)

        assert getattr(adapter.cfg, "layers_block_type") == []
        assert getattr(adapter.cfg, "num_mem_blocks") == 1
        assert getattr(adapter.cfg, "use_shared_attention_adapter") is False
        assert getattr(adapter.cfg, "mamba_intermediate_size") == 2 * 64
        assert getattr(adapter.cfg, "conv_dim") == 2 * 64 + 2 * 1 * 64

    def test_verification_phases_and_weight_processing(
        self, adapter: Zamba2ArchitectureAdapter
    ) -> None:
        assert adapter.applicable_phases == [1, 2, 3, 4]
        assert adapter.weight_processing_conversions == {}


class TestZamba2TopLevelComponents:
    def test_types_and_hf_paths(self, adapter: Zamba2ArchitectureAdapter) -> None:
        mapping = adapter.get_component_mapping()

        assert set(mapping) == {"embed", "blocks", "ln_final", "unembed"}
        assert isinstance(mapping["embed"], EmbeddingBridge)
        assert isinstance(mapping["blocks"], SSMBlockBridge)
        assert isinstance(mapping["ln_final"], RMSNormalizationBridge)
        assert isinstance(mapping["unembed"], UnembeddingBridge)
        assert mapping["embed"].name == "model.embed_tokens"
        assert mapping["blocks"].name == "model.layers"
        assert mapping["ln_final"].name == "model.final_layernorm"
        assert mapping["unembed"].name == "lm_head"


class TestZamba2BlockComponents:
    @pytest.fixture(scope="class")
    def blocks(self, adapter: Zamba2ArchitectureAdapter) -> SSMBlockBridge:
        component = adapter.get_component_mapping()["blocks"]
        assert isinstance(component, SSMBlockBridge)
        return component

    def test_block_submodules_are_optional(self, blocks: SSMBlockBridge) -> None:
        assert set(blocks.submodules) == {"norm", "mixer"}
        assert isinstance(blocks.submodules["norm"], RMSNormalizationBridge)
        assert isinstance(blocks.submodules["mixer"], SSM2MixerBridge)
        assert blocks.submodules["norm"].name == "input_layernorm"
        assert blocks.submodules["mixer"].name == "mamba"
        assert blocks.submodules["norm"].optional is True
        assert blocks.submodules["mixer"].optional is True

    def test_mixer_submodule_types_and_paths(self, blocks: SSMBlockBridge) -> None:
        mixer = blocks.submodules["mixer"]
        assert isinstance(mixer, SSM2MixerBridge)
        assert set(mixer.submodules) == {"in_proj", "conv1d", "inner_norm", "out_proj"}

        assert isinstance(mixer.submodules["in_proj"], LinearBridge)
        assert isinstance(mixer.submodules["conv1d"], DepthwiseConv1DBridge)
        assert isinstance(mixer.submodules["inner_norm"], GatedRMSNormBridge)
        assert isinstance(mixer.submodules["out_proj"], LinearBridge)
        assert mixer.submodules["in_proj"].name == "in_proj"
        assert mixer.submodules["conv1d"].name == "conv1d"
        assert mixer.submodules["inner_norm"].name == "norm"
        assert mixer.submodules["out_proj"].name == "out_proj"

    def test_mixer_submodules_are_optional(self, blocks: SSMBlockBridge) -> None:
        mixer = blocks.submodules["mixer"]
        assert isinstance(mixer, SSM2MixerBridge)
        assert all(component.optional is True for component in mixer.submodules.values())


class TestZamba2FactoryRegistration:
    def test_factory_selects_zamba2_adapter(self) -> None:
        adapter = ArchitectureAdapterFactory.select_architecture_adapter(
            _make_cfg(layers_block_type=["mamba"] * 4)
        )

        assert isinstance(adapter, Zamba2ArchitectureAdapter)


def test_setup_component_testing_tolerates_missing_attn_template() -> None:
    """HF model exposes rotary_emb but the bridge maps no attn submodule
    (shared attention is fully delegated) — setup must no-op, not raise."""
    from types import SimpleNamespace

    adapter = Zamba2ArchitectureAdapter(_make_cfg())
    hf_model = SimpleNamespace(model=SimpleNamespace(rotary_emb=object(), layers=[]))
    adapter.setup_component_testing(hf_model)
