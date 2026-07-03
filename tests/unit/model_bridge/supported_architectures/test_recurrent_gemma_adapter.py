"""Unit tests for the RecurrentGemmaArchitectureAdapter — no model downloads.

RecurrentGemma (Griffin) is a hybrid of RG-LRU recurrent layers and local
sliding-window attention layers, wrapped whole-layer (residual hooks only) like
LFM2. These tests pin the architecture-specific quirks:

- Gemma-family numerics: RMSNorm ``(1.0 + weight)`` offset, final-logit soft cap.
- Gated MLP, RMSNorm, ``final_rms``.
- The heterogeneous ``block_types`` pattern is exposed as ``layers_block_type``.
- Whole-layer block bridge advertising only residual-stream hooks (no attn/MLP
  substructure, since recurrent layers have none).
"""

import pytest

from transformer_lens.config import TransformerBridgeConfig
from transformer_lens.model_bridge.generalized_components import (
    EmbeddingBridge,
    RMSNormalizationBridge,
    UnembeddingBridge,
)
from transformer_lens.model_bridge.supported_architectures.recurrent_gemma import (
    RecurrentGemmaArchitectureAdapter,
    RecurrentGemmaBlockBridge,
)


@pytest.fixture(scope="class")
def cfg() -> TransformerBridgeConfig:
    bridge_cfg = TransformerBridgeConfig(
        d_model=64,
        d_head=16,
        n_layers=6,
        n_ctx=128,
        n_heads=4,
        n_key_value_heads=1,
        d_vocab=256,
        d_mlp=128,
        architecture="RecurrentGemmaForCausalLM",
    )
    # HF RecurrentGemmaConfig fields the adapter reads off the raw config.
    bridge_cfg.block_types = ["recurrent", "recurrent", "attention"]
    bridge_cfg.rms_norm_eps = 1e-6
    bridge_cfg.logits_soft_cap = 30.0
    return bridge_cfg


@pytest.fixture(scope="class")
def adapter(cfg: TransformerBridgeConfig) -> RecurrentGemmaArchitectureAdapter:
    return RecurrentGemmaArchitectureAdapter(cfg)


class TestRecurrentGemmaAdapterConfig:
    def test_norm_config_is_gemma_rmsnorm(
        self, adapter: RecurrentGemmaArchitectureAdapter
    ) -> None:
        assert adapter.cfg.normalization_type == "RMS"
        assert adapter.cfg.uses_rms_norm is True
        assert adapter.cfg.final_rms is True
        # Gemma RMSNorm applies (1.0 + weight); see HF PR #29402.
        assert adapter.cfg.rmsnorm_uses_offset is True
        assert adapter.cfg.eps == 1e-6

    def test_gated_mlp_and_rotary(self, adapter: RecurrentGemmaArchitectureAdapter) -> None:
        assert adapter.cfg.gated_mlp is True
        assert adapter.cfg.attn_only is False
        assert adapter.cfg.positional_embedding_type == "rotary"

    def test_final_logit_soft_cap(self, adapter: RecurrentGemmaArchitectureAdapter) -> None:
        assert adapter.cfg.output_logits_soft_cap == 30.0

    def test_default_prepend_bos_is_false(
        self, adapter: RecurrentGemmaArchitectureAdapter
    ) -> None:
        assert adapter.cfg.default_prepend_bos is False

    def test_block_types_exposed_for_analysis(
        self, adapter: RecurrentGemmaArchitectureAdapter
    ) -> None:
        expected = ["recurrent", "recurrent", "attention"]
        assert adapter.cfg.block_types == expected
        # Normalized to the canonical name used by Nemotron-H / Granite tooling.
        assert adapter.cfg.layers_block_type == expected

    def test_applicable_phases_is_generation_only(
        self, adapter: RecurrentGemmaArchitectureAdapter
    ) -> None:
        assert adapter.applicable_phases == [4]


class TestRecurrentGemmaComponentMapping:
    def test_has_residual_only_top_level_mapping(
        self, adapter: RecurrentGemmaArchitectureAdapter
    ) -> None:
        mapping = adapter.component_mapping
        assert mapping is not None
        assert set(mapping) == {"embed", "blocks", "ln_final", "unembed"}

    def test_component_types(self, adapter: RecurrentGemmaArchitectureAdapter) -> None:
        mapping = adapter.component_mapping
        assert isinstance(mapping["embed"], EmbeddingBridge)
        assert isinstance(mapping["blocks"], RecurrentGemmaBlockBridge)
        assert isinstance(mapping["ln_final"], RMSNormalizationBridge)
        assert isinstance(mapping["unembed"], UnembeddingBridge)

    def test_hf_module_paths(self, adapter: RecurrentGemmaArchitectureAdapter) -> None:
        mapping = adapter.component_mapping
        assert mapping["embed"].name == "model.embed_tokens"
        assert mapping["blocks"].name == "model.layers"
        # RecurrentGemma names the decoder-final norm `final_norm`, not `norm`.
        assert mapping["ln_final"].name == "model.final_norm"
        assert mapping["unembed"].name == "lm_head"

    def test_blocks_only_advertise_supported_residual_aliases(
        self, adapter: RecurrentGemmaArchitectureAdapter
    ) -> None:
        blocks = adapter.component_mapping["blocks"]
        assert blocks.hook_aliases == {
            "hook_resid_pre": "hook_in",
            "hook_resid_post": "hook_out",
        }
        assert blocks.submodules == {}


class TestRecurrentGemmaFactoryRegistration:
    def test_registered_in_factory(self) -> None:
        from transformer_lens.factories.architecture_adapter_factory import (
            SUPPORTED_ARCHITECTURES,
        )

        assert (
            SUPPORTED_ARCHITECTURES["RecurrentGemmaForCausalLM"]
            is RecurrentGemmaArchitectureAdapter
        )
