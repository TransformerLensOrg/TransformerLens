"""Unit tests for the Qwen3MoeArchitectureAdapter.

All tests use programmatic TransformerBridgeConfig instances — no network access
or model downloads.
"""

import pytest

from transformer_lens.config import TransformerBridgeConfig
from transformer_lens.conversion_utils.conversion_steps.rearrange_tensor_conversion import (
    RearrangeTensorConversion,
)
from transformer_lens.conversion_utils.param_processing_conversion import (
    ParamProcessingConversion,
)
from transformer_lens.factories.architecture_adapter_factory import (
    SUPPORTED_ARCHITECTURES,
)
from transformer_lens.model_bridge.generalized_components import (
    MoEBridge,
    RMSNormalizationBridge,
)
from transformer_lens.model_bridge.supported_architectures.qwen3_moe import (
    Qwen3MoeArchitectureAdapter,
)


@pytest.fixture
def cfg() -> TransformerBridgeConfig:
    return TransformerBridgeConfig(
        d_model=64,
        d_head=16,
        n_layers=2,
        n_ctx=128,
        n_heads=4,
        n_key_value_heads=2,
        d_vocab=256,
        architecture="Qwen3MoeForCausalLM",
    )


@pytest.fixture
def adapter(cfg: TransformerBridgeConfig) -> Qwen3MoeArchitectureAdapter:
    return Qwen3MoeArchitectureAdapter(cfg)


class TestQwen3MoeAdapterConfig:
    def test_normalization_type_is_rms(self, adapter: Qwen3MoeArchitectureAdapter) -> None:
        assert adapter.cfg.normalization_type == "RMS"

    def test_positional_embedding_type_is_rotary(
        self, adapter: Qwen3MoeArchitectureAdapter
    ) -> None:
        assert adapter.cfg.positional_embedding_type == "rotary"

    def test_final_rms_is_true(self, adapter: Qwen3MoeArchitectureAdapter) -> None:
        """Qwen3MoE uses final_rms=True; OLMoE uses False."""
        assert adapter.cfg.final_rms is True

    def test_gated_mlp_is_true(self, adapter: Qwen3MoeArchitectureAdapter) -> None:
        assert adapter.cfg.gated_mlp is True

    def test_uses_rms_norm_is_true(self, adapter: Qwen3MoeArchitectureAdapter) -> None:
        assert adapter.cfg.uses_rms_norm is True

    def test_attn_implementation_is_eager(self, adapter: Qwen3MoeArchitectureAdapter) -> None:
        assert adapter.cfg.attn_implementation == "eager"

    def test_default_prepend_bos_is_false(self, adapter: Qwen3MoeArchitectureAdapter) -> None:
        assert adapter.cfg.default_prepend_bos is False

    def test_n_kv_heads_propagated(self) -> None:
        """n_key_value_heads from the loaded config is preserved."""
        cfg = TransformerBridgeConfig(
            d_model=64,
            d_head=16,
            n_layers=2,
            n_ctx=128,
            n_heads=4,
            n_key_value_heads=2,
            d_vocab=256,
            architecture="Qwen3MoeForCausalLM",
        )
        adapter = Qwen3MoeArchitectureAdapter(cfg)
        assert adapter.cfg.n_key_value_heads == 2


class TestQwen3MoeWeightConversions:
    def test_has_qkvo_keys(self, adapter: Qwen3MoeArchitectureAdapter) -> None:
        convs = adapter.weight_processing_conversions
        assert convs is not None
        assert "blocks.{i}.attn.q.weight" in convs
        assert "blocks.{i}.attn.k.weight" in convs
        assert "blocks.{i}.attn.v.weight" in convs
        assert "blocks.{i}.attn.o.weight" in convs

    def test_q_rearrange_uses_n_heads(self, adapter: Qwen3MoeArchitectureAdapter) -> None:
        """Q rearrange uses n_heads (4)."""
        convs = adapter.weight_processing_conversions
        assert convs is not None
        q_conv = convs["blocks.{i}.attn.q.weight"]
        assert isinstance(q_conv, ParamProcessingConversion)
        assert isinstance(q_conv.tensor_conversion, RearrangeTensorConversion)
        axes = q_conv.tensor_conversion.axes_lengths
        assert axes.get("n") == 4

    def test_kv_rearrange_uses_n_kv_heads(self, adapter: Qwen3MoeArchitectureAdapter) -> None:
        """K/V rearrange uses n_key_value_heads (2) for GQA."""
        convs = adapter.weight_processing_conversions
        assert convs is not None
        k_conv = convs["blocks.{i}.attn.k.weight"]
        v_conv = convs["blocks.{i}.attn.v.weight"]
        assert isinstance(k_conv, ParamProcessingConversion)
        assert isinstance(v_conv, ParamProcessingConversion)
        assert isinstance(k_conv.tensor_conversion, RearrangeTensorConversion)
        assert isinstance(v_conv.tensor_conversion, RearrangeTensorConversion)
        assert k_conv.tensor_conversion.axes_lengths.get("n") == 2
        assert v_conv.tensor_conversion.axes_lengths.get("n") == 2

    def test_o_rearrange_uses_n_heads(self, adapter: Qwen3MoeArchitectureAdapter) -> None:
        """O rearrange uses n_heads (4)."""
        convs = adapter.weight_processing_conversions
        assert convs is not None
        o_conv = convs["blocks.{i}.attn.o.weight"]
        assert isinstance(o_conv, ParamProcessingConversion)
        assert isinstance(o_conv.tensor_conversion, RearrangeTensorConversion)
        assert o_conv.tensor_conversion.axes_lengths.get("n") == 4


class TestQwen3MoeComponentMapping:
    def test_has_required_top_level_keys(self, adapter: Qwen3MoeArchitectureAdapter) -> None:
        mapping = adapter.component_mapping
        assert mapping is not None
        for key in ("embed", "rotary_emb", "blocks", "ln_final", "unembed"):
            assert key in mapping, f"Missing top-level key: {key!r}"

    def test_blocks_has_required_submodules(self, adapter: Qwen3MoeArchitectureAdapter) -> None:
        mapping = adapter.component_mapping
        assert mapping is not None
        blocks = mapping["blocks"]
        for key in ("ln1", "ln2", "attn", "mlp"):
            assert key in blocks.submodules, f"Missing blocks submodule: {key!r}"

    def test_attn_has_all_submodules(self, adapter: Qwen3MoeArchitectureAdapter) -> None:
        mapping = adapter.component_mapping
        assert mapping is not None
        attn = mapping["blocks"].submodules["attn"]
        for key in ("q", "k", "v", "o", "q_norm", "k_norm"):
            assert key in attn.submodules, f"Missing attn submodule: {key!r}"

    def test_ln1_ln2_are_rms_norm_bridges(self, adapter: Qwen3MoeArchitectureAdapter) -> None:
        mapping = adapter.component_mapping
        assert mapping is not None
        subs = mapping["blocks"].submodules
        assert isinstance(subs["ln1"], RMSNormalizationBridge)
        assert isinstance(subs["ln2"], RMSNormalizationBridge)

    def test_mlp_is_moe_bridge(self, adapter: Qwen3MoeArchitectureAdapter) -> None:
        mapping = adapter.component_mapping
        assert mapping is not None
        mlp = mapping["blocks"].submodules["mlp"]
        assert isinstance(mlp, MoEBridge)

    def test_mlp_has_gate_submodule(self, adapter: Qwen3MoeArchitectureAdapter) -> None:
        mapping = adapter.component_mapping
        assert mapping is not None
        mlp = mapping["blocks"].submodules["mlp"]
        assert "gate" in mlp.submodules

    def test_q_norm_k_norm_are_rms_norm_bridges(self, adapter: Qwen3MoeArchitectureAdapter) -> None:
        mapping = adapter.component_mapping
        assert mapping is not None
        attn_subs = mapping["blocks"].submodules["attn"].submodules
        assert isinstance(attn_subs["q_norm"], RMSNormalizationBridge)
        assert isinstance(attn_subs["k_norm"], RMSNormalizationBridge)

    def test_hf_module_paths(self, adapter: Qwen3MoeArchitectureAdapter) -> None:
        """HF module path names are mapped correctly."""
        mapping = adapter.component_mapping
        assert mapping is not None
        assert mapping["embed"].name == "model.embed_tokens"
        assert mapping["ln_final"].name == "model.norm"
        assert mapping["unembed"].name == "lm_head"
        assert mapping["blocks"].name == "model.layers"
        subs = mapping["blocks"].submodules
        assert subs["ln1"].name == "input_layernorm"
        assert subs["ln2"].name == "post_attention_layernorm"
        assert subs["attn"].name == "self_attn"
        assert subs["mlp"].name == "mlp"


class TestQwen3MoeFactoryRegistration:
    def test_factory_lookup_returns_adapter_class(self) -> None:
        assert SUPPORTED_ARCHITECTURES["Qwen3MoeForCausalLM"] is Qwen3MoeArchitectureAdapter
