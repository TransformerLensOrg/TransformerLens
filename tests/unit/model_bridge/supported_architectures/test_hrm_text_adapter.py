"""Unit tests for HrmTextArchitectureAdapter.

HRM-Text has L_blocks and H_blocks instead of a single blocks list,
and uses parameterless RMSNorm which cannot be folded.
"""

import pytest

from transformer_lens.config.transformer_bridge_config import TransformerBridgeConfig


def _make_cfg(**overrides):
    defaults = dict(
        d_model=256,
        d_head=64,
        n_heads=4,
        n_layers=4,
        n_ctx=128,
        d_vocab=512,
        d_mlp=128,
        architecture="HrmTextForCausalLM",
    )
    defaults.update(overrides)
    return TransformerBridgeConfig(**defaults)


@pytest.fixture(scope="module")
def adapter():
    from transformer_lens.model_bridge.supported_architectures.hrm_text import (
        HrmTextArchitectureAdapter,
    )

    return HrmTextArchitectureAdapter(_make_cfg())


class TestHrmTextComponentMapping:
    def test_component_mapping_keys(self, adapter):
        assert set(adapter.component_mapping.keys()) == {
            "embed",
            "rotary_emb",
            "L_blocks",
            "H_blocks",
            "L_ln_final",
            "H_ln_final",
            "unembed",
        }

    def test_embed_path(self, adapter):
        assert adapter.component_mapping["embed"].name == "model.embed_tokens"

    def test_rotary_emb_path(self, adapter):
        assert adapter.component_mapping["rotary_emb"].name == "model.rotary_emb"

    def test_L_blocks_path(self, adapter):
        assert adapter.component_mapping["L_blocks"].name == "model.L_module.layers"

    def test_H_blocks_path(self, adapter):
        assert adapter.component_mapping["H_blocks"].name == "model.H_module.layers"

    def test_L_ln_final_path(self, adapter):
        assert adapter.component_mapping["L_ln_final"].name == "model.L_module.final_norm"

    def test_H_ln_final_path(self, adapter):
        assert adapter.component_mapping["H_ln_final"].name == "model.H_module.final_norm"

    def test_unembed_path(self, adapter):
        assert adapter.component_mapping["unembed"].name == "lm_head"

    def test_L_block_submodules_keys(self, adapter):
        submodules = adapter.component_mapping["L_blocks"].submodules
        assert set(submodules.keys()) == {"ln1", "ln2", "mlp", "attn"}

    def test_H_block_submodules_keys(self, adapter):
        submodules = adapter.component_mapping["H_blocks"].submodules
        assert set(submodules.keys()) == {"ln1", "ln2", "mlp", "attn"}

    def test_attn_submodule_keys(self, adapter):
        attn = adapter.component_mapping["L_blocks"].submodules["attn"]
        assert set(attn.submodules.keys()) == {"q", "k", "v", "o", "gate"}

    def test_attn_q_path(self, adapter):
        attn = adapter.component_mapping["L_blocks"].submodules["attn"]
        assert attn.submodules["q"].name == "q_proj"

    def test_attn_gate_path(self, adapter):
        attn = adapter.component_mapping["L_blocks"].submodules["attn"]
        assert attn.submodules["gate"].name == "gate_proj"

    def test_mlp_submodule_keys(self, adapter):
        mlp = adapter.component_mapping["L_blocks"].submodules["mlp"]
        assert set(mlp.submodules.keys()) == {"gate", "in", "out"}

    def test_mlp_gate_path(self, adapter):
        mlp = adapter.component_mapping["L_blocks"].submodules["mlp"]
        assert mlp.submodules["gate"].name == "gate_proj"

    def test_mlp_in_path(self, adapter):
        mlp = adapter.component_mapping["L_blocks"].submodules["mlp"]
        assert mlp.submodules["in"].name == "up_proj"

    def test_mlp_out_path(self, adapter):
        mlp = adapter.component_mapping["L_blocks"].submodules["mlp"]
        assert mlp.submodules["out"].name == "down_proj"


class TestHrmTextConfigAttributes:
    def test_normalization_type(self, adapter):
        assert adapter.cfg.normalization_type == "RMS"

    def test_positional_embedding_type(self, adapter):
        assert adapter.cfg.positional_embedding_type == "rotary"

    def test_final_rms(self, adapter):
        assert adapter.cfg.final_rms is True

    def test_gated_mlp(self, adapter):
        assert adapter.cfg.gated_mlp is True

    def test_attn_only(self, adapter):
        assert adapter.cfg.attn_only is False

    def test_uses_rms_norm(self, adapter):
        assert adapter.cfg.uses_rms_norm is True

    def test_supports_fold_ln(self, adapter):
        assert adapter.supports_fold_ln is False

    def test_supports_center_writing_weights(self, adapter):
        assert adapter.supports_center_writing_weights is False

    def test_applicable_phases(self, adapter):
        assert adapter.applicable_phases == [1, 2, 3]


class TestHrmTextBlockTypes:
    def test_L_blocks_is_block_bridge(self, adapter):
        from transformer_lens.model_bridge.generalized_components import BlockBridge

        assert isinstance(adapter.component_mapping["L_blocks"], BlockBridge)

    def test_H_blocks_is_block_bridge(self, adapter):
        from transformer_lens.model_bridge.generalized_components import BlockBridge

        assert isinstance(adapter.component_mapping["H_blocks"], BlockBridge)

    def test_attn_is_position_embeddings_attention(self, adapter):
        from transformer_lens.model_bridge.generalized_components import (
            PositionEmbeddingsAttentionBridge,
        )

        attn = adapter.component_mapping["L_blocks"].submodules["attn"]
        assert isinstance(attn, PositionEmbeddingsAttentionBridge)

    def test_ln1_is_rms_norm(self, adapter):
        from transformer_lens.model_bridge.generalized_components import (
            RMSNormalizationBridge,
        )

        ln1 = adapter.component_mapping["L_blocks"].submodules["ln1"]
        assert isinstance(ln1, RMSNormalizationBridge)

    def test_mlp_is_gated_mlp_bridge(self, adapter):
        from transformer_lens.model_bridge.generalized_components import GatedMLPBridge

        mlp = adapter.component_mapping["L_blocks"].submodules["mlp"]
        assert isinstance(mlp, GatedMLPBridge)

    def test_embed_is_embedding_bridge(self, adapter):
        from transformer_lens.model_bridge.generalized_components import EmbeddingBridge

        assert isinstance(adapter.component_mapping["embed"], EmbeddingBridge)

    def test_unembed_is_unembedding_bridge(self, adapter):
        from transformer_lens.model_bridge.generalized_components import (
            UnembeddingBridge,
        )

        assert isinstance(adapter.component_mapping["unembed"], UnembeddingBridge)

    def test_rotary_emb_is_rotary_embedding_bridge(self, adapter):
        from transformer_lens.model_bridge.generalized_components import (
            RotaryEmbeddingBridge,
        )

        assert isinstance(adapter.component_mapping["rotary_emb"], RotaryEmbeddingBridge)


class TestHrmTextWeightConversions:
    N_HEADS = 4
    D_HEAD = 8
    HIDDEN_SIZE = 32

    @pytest.fixture
    def adapter(self):
        from transformer_lens.model_bridge.supported_architectures.hrm_text import (
            HrmTextArchitectureAdapter,
        )

        cfg = _make_cfg(
            n_heads=self.N_HEADS,
            d_head=self.D_HEAD,
            d_model=self.HIDDEN_SIZE,
        )
        return HrmTextArchitectureAdapter(cfg)

    def test_weight_conversions_have_L_and_H_prefixes(self, adapter):
        keys = set(adapter.weight_processing_conversions.keys())
        assert any(k.startswith("L_blocks.") for k in keys), "Missing L_blocks prefix"
        assert any(k.startswith("H_blocks.") for k in keys), "Missing H_blocks prefix"
        assert not any(k.startswith("blocks.") for k in keys), "Should not have bare blocks prefix"

    def test_weight_conversion_count(self, adapter):
        """4 conversions per stack (q, k, v, o) × 2 stacks = 8."""
        assert len(adapter.weight_processing_conversions) == 8

    def test_preprocess_weights_empty_noop(self, adapter):
        result = adapter.preprocess_weights({})
        assert result == {}

    def test_preprocess_weights_embedding_scale_default(self, adapter):
        import torch

        state_dict = {"embed.weight": torch.ones(100, self.HIDDEN_SIZE)}
        result = adapter.preprocess_weights(state_dict)
        assert torch.equal(result["embed.weight"], state_dict["embed.weight"])

    def test_preprocess_weights_with_scale(self, adapter):
        adapter.cfg.embedding_scale = 2.0
        import torch

        state_dict = {"embed.weight": torch.ones(100, self.HIDDEN_SIZE, dtype=torch.float32)}
        result = adapter.preprocess_weights(state_dict)
        expected = torch.full((100, self.HIDDEN_SIZE), 2.0, dtype=torch.float32)
        assert torch.equal(result["embed.weight"], expected)


class TestHrmTextConfigPassthrough:
    """HRM-specific config attrs must be available on the bridge config."""

    @pytest.fixture
    def adapter(self):
        from transformer_lens.model_bridge.supported_architectures.hrm_text import (
            HrmTextArchitectureAdapter,
        )

        cfg = _make_cfg()
        for attr in (
            "H_cycles",
            "L_cycles",
            "L_bp_cycles",
            "num_layers_per_stack",
            "embedding_scale",
            "prefix_lm",
        ):
            setattr(cfg, attr, None)
        return HrmTextArchitectureAdapter(cfg)

    def test_config_has_hr_cycles_fields(self, adapter):
        for attr in ("H_cycles", "L_cycles"):
            assert hasattr(adapter.cfg, attr)

    def test_config_has_embedding_scale(self, adapter):
        assert hasattr(adapter.cfg, "embedding_scale")

    def test_config_has_prefix_lm(self, adapter):
        assert hasattr(adapter.cfg, "prefix_lm")

    def test_config_has_num_layers_per_stack(self, adapter):
        assert hasattr(adapter.cfg, "num_layers_per_stack")

    def test_config_has_L_bp_cycles(self, adapter):
        assert hasattr(adapter.cfg, "L_bp_cycles")
