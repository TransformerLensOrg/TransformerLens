"""Unit tests for the GLM-MoE-DSA architecture adapter."""

from transformer_lens.config import TransformerBridgeConfig
from transformer_lens.factories.architecture_adapter_factory import (
    SUPPORTED_ARCHITECTURES,
    ArchitectureAdapterFactory,
)
from transformer_lens.model_bridge.generalized_components import (
    MLABlockBridge,
    MoEBridge,
)
from transformer_lens.model_bridge.generalized_components.glm_moe_dsa_attention import (
    GlmMoeDsaAttentionBridge,
)
from transformer_lens.model_bridge.supported_architectures.glm_moe_dsa import (
    GlmMoeDsaArchitectureAdapter,
)
from transformer_lens.tools.model_registry import (
    CANONICAL_AUTHORS_BY_ARCH,
    HF_SUPPORTED_ARCHITECTURES,
)


def make_cfg() -> TransformerBridgeConfig:
    return TransformerBridgeConfig(
        d_model=64,
        d_head=16,
        n_layers=3,
        n_ctx=32,
        n_heads=4,
        d_vocab=128,
        architecture="GlmMoeDsaForCausalLM",
    )


class TestGlmMoeDsaAdapter:
    def test_factory_selects_adapter(self) -> None:
        adapter = ArchitectureAdapterFactory.select_architecture_adapter(make_cfg())

        assert isinstance(adapter, GlmMoeDsaArchitectureAdapter)

    def test_registry_wiring(self) -> None:
        assert SUPPORTED_ARCHITECTURES["GlmMoeDsaForCausalLM"] is GlmMoeDsaArchitectureAdapter
        assert "GlmMoeDsaForCausalLM" in HF_SUPPORTED_ARCHITECTURES
        assert CANONICAL_AUTHORS_BY_ARCH["GlmMoeDsaForCausalLM"] == ["zai-org"]

    def test_config_flags(self) -> None:
        adapter = GlmMoeDsaArchitectureAdapter(make_cfg())

        assert adapter.cfg.normalization_type == "RMS"
        assert adapter.cfg.positional_embedding_type == "rotary"
        assert adapter.cfg.final_rms is True
        assert adapter.cfg.gated_mlp is True
        assert adapter.cfg.attn_implementation == "eager"
        assert adapter.cfg.default_prepend_bos is False
        assert adapter.supports_fold_ln is False

    def test_component_mapping_uses_dsa_attention_and_moe(self) -> None:
        adapter = GlmMoeDsaArchitectureAdapter(make_cfg())
        blocks = adapter.component_mapping["blocks"]

        assert isinstance(blocks, MLABlockBridge)
        assert isinstance(blocks.submodules["attn"], GlmMoeDsaAttentionBridge)
        assert isinstance(blocks.submodules["mlp"], MoEBridge)
        assert "shared_experts" in blocks.submodules["mlp"].submodules
