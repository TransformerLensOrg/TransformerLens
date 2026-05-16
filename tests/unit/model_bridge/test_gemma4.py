"""Unit tests for Gemma4 architecture adapter registration and configuration."""

import pytest

from transformer_lens.config.TransformerBridgeConfig import TransformerBridgeConfig
from transformer_lens.factories.architecture_adapter_factory import (
    SUPPORTED_ARCHITECTURES,
    ArchitectureAdapterFactory,
)
from transformer_lens.model_bridge.supported_architectures.gemma4 import (
    Gemma4ArchitectureAdapter,
)


def _make_gemma4_cfg(**overrides):
    """Create a TransformerBridgeConfig for Gemma4 E2B."""
    defaults = dict(
        d_model=1536,
        d_head=256,
        n_heads=8,
        n_layers=35,
        n_ctx=8192,
        d_vocab=262144,
        n_key_value_heads=1,
        d_mlp=6144,
        architecture="Gemma4ForCausalLM",
    )
    defaults.update(overrides)
    extra_keys = {
        "architectures",
        "final_logit_softcapping",
        "attn_logit_softcapping",
        "hidden_size_per_layer_input",
        "num_kv_shared_layers",
        "layer_types",
    }
    extra = {k: defaults.pop(k) for k in extra_keys if k in defaults}
    cfg = TransformerBridgeConfig(**defaults)
    for k, v in extra.items():
        setattr(cfg, k, v)
    return cfg


class TestGemma4Registration:
    """Test that Gemma4ArchitectureAdapter is properly registered."""

    def test_architecture_in_supported_architectures(self):
        assert "Gemma4ForCausalLM" in SUPPORTED_ARCHITECTURES

    def test_conditional_generation_registered(self):
        assert "Gemma4ForConditionalGeneration" in SUPPORTED_ARCHITECTURES

    def test_architecture_maps_to_correct_adapter(self):
        assert SUPPORTED_ARCHITECTURES["Gemma4ForCausalLM"] is Gemma4ArchitectureAdapter

    def test_factory_selects_correct_adapter(self):
        cfg = _make_gemma4_cfg()
        adapter = ArchitectureAdapterFactory.select_architecture_adapter(cfg)
        assert isinstance(adapter, Gemma4ArchitectureAdapter)

    def test_factory_selects_conditional_generation(self):
        cfg = _make_gemma4_cfg(architecture="Gemma4ForConditionalGeneration")
        adapter = ArchitectureAdapterFactory.select_architecture_adapter(cfg)
        assert isinstance(adapter, Gemma4ArchitectureAdapter)


class TestGemma4ConfigAttributes:
    """Test Gemma4ArchitectureAdapter configuration attributes."""

    @pytest.fixture
    def adapter(self):
        cfg = _make_gemma4_cfg(architectures=["Gemma4ForCausalLM"])
        return Gemma4ArchitectureAdapter(cfg)

    def test_gated_mlp(self, adapter):
        assert adapter.cfg.gated_mlp is True

    def test_uses_rms_norm(self, adapter):
        assert adapter.cfg.uses_rms_norm is True

    def test_normalization_type(self, adapter):
        assert adapter.cfg.normalization_type == "RMS"

    def test_final_rms(self, adapter):
        assert adapter.cfg.final_rms is True

    def test_eps_attr(self, adapter):
        assert adapter.cfg.eps_attr == "rms_norm_eps"

    def test_rmsnorm_uses_offset(self, adapter):
        assert adapter.cfg.rmsnorm_uses_offset is True

    def test_positional_embedding_type(self, adapter):
        assert adapter.cfg.positional_embedding_type == "rotary"

    def test_attn_implementation(self, adapter):
        assert adapter.cfg.attn_implementation == "eager"


class TestGemma4Softcapping:
    """Test logit and attention softcapping attribute mapping."""

    def test_output_logits_soft_cap(self):
        cfg = _make_gemma4_cfg(final_logit_softcapping=30.0)
        adapter = Gemma4ArchitectureAdapter(cfg)
        assert adapter.cfg.output_logits_soft_cap == 30.0

    def test_attn_scores_soft_cap(self):
        cfg = _make_gemma4_cfg(attn_logit_softcapping=50.0)
        adapter = Gemma4ArchitectureAdapter(cfg)
        assert adapter.cfg.attn_scores_soft_cap == 50.0

    def test_no_softcapping_when_absent(self):
        cfg = _make_gemma4_cfg()
        adapter = Gemma4ArchitectureAdapter(cfg)
        # defaults are -1.0 (unchanged by adapter when softcapping not in HF config)
        assert adapter.cfg.attn_scores_soft_cap == -1.0
        assert adapter.cfg.output_logits_soft_cap == -1.0


class TestGemma4E2BConfig:
    """Test Gemma4 E-series specific config: PLE, KV sharing, layer_types."""

    def test_hidden_size_per_layer_input(self):
        cfg = _make_gemma4_cfg(hidden_size_per_layer_input=256)
        adapter = Gemma4ArchitectureAdapter(cfg)
        assert adapter.cfg.hidden_size_per_layer_input == 256

    def test_num_kv_shared_layers(self):
        cfg = _make_gemma4_cfg(num_kv_shared_layers=20)
        adapter = Gemma4ArchitectureAdapter(cfg)
        assert adapter.cfg.num_kv_shared_layers == 20

    def test_layer_types(self):
        layer_types = ["sliding_attention"] * 35
        cfg = _make_gemma4_cfg(layer_types=layer_types)
        adapter = Gemma4ArchitectureAdapter(cfg)
        assert adapter.cfg.layer_types == layer_types

    def test_ple_not_set_when_absent(self):
        cfg = _make_gemma4_cfg()
        adapter = Gemma4ArchitectureAdapter(cfg)
        assert not hasattr(adapter.cfg, "hidden_size_per_layer_input")

    def test_kv_sharing_not_set_when_absent(self):
        cfg = _make_gemma4_cfg()
        adapter = Gemma4ArchitectureAdapter(cfg)
        assert not hasattr(adapter.cfg, "num_kv_shared_layers")


class TestGemma4TextPrefix:
    """Test text prefix detection for text-only vs multimodal."""

    def test_text_prefix_causal(self):
        cfg = _make_gemma4_cfg(architectures=["Gemma4ForCausalLM"])
        adapter = Gemma4ArchitectureAdapter(cfg)
        assert adapter.text_prefix == "model"

    def test_text_prefix_conditional_generation(self):
        cfg = _make_gemma4_cfg(architectures=["Gemma4ForConditionalGeneration"])
        adapter = Gemma4ArchitectureAdapter(cfg)
        assert adapter.text_prefix == "model.language_model"


class TestGemma4ComponentMapping:
    """Test Gemma4ArchitectureAdapter component mapping."""

    @pytest.fixture
    def adapter(self):
        cfg = _make_gemma4_cfg(architectures=["Gemma4ForCausalLM"])
        return Gemma4ArchitectureAdapter(cfg)

    def test_has_embed(self, adapter):
        assert "embed" in adapter.component_mapping

    def test_has_rotary_emb(self, adapter):
        assert "rotary_emb" in adapter.component_mapping

    def test_has_blocks(self, adapter):
        assert "blocks" in adapter.component_mapping

    def test_has_ln_final(self, adapter):
        assert "ln_final" in adapter.component_mapping

    def test_has_unembed(self, adapter):
        assert "unembed" in adapter.component_mapping

    def test_embed_path_causal(self, adapter):
        assert adapter.component_mapping["embed"].name == "model.embed_tokens"

    def test_rotary_emb_path_causal(self, adapter):
        assert adapter.component_mapping["rotary_emb"].name == "model.rotary_emb"

    def test_blocks_path_causal(self, adapter):
        assert adapter.component_mapping["blocks"].name == "model.layers"

    def test_ln_final_path_causal(self, adapter):
        assert adapter.component_mapping["ln_final"].name == "model.norm"

    def test_unembed_path(self, adapter):
        assert adapter.component_mapping["unembed"].name == "lm_head"

    def test_block_submodules(self, adapter):
        block = adapter.component_mapping["blocks"]
        assert "ln1" in block.submodules
        assert "ln1_post" in block.submodules
        assert "ln2" in block.submodules
        assert "ln2_post" in block.submodules
        assert "attn" in block.submodules
        assert "mlp" in block.submodules

    def test_ln_names(self, adapter):
        block = adapter.component_mapping["blocks"]
        assert block.submodules["ln1"].name == "input_layernorm"
        assert block.submodules["ln1_post"].name == "post_attention_layernorm"
        assert block.submodules["ln2"].name == "pre_feedforward_layernorm"
        assert block.submodules["ln2_post"].name == "post_feedforward_layernorm"

    def test_attn_submodules(self, adapter):
        attn = adapter.component_mapping["blocks"].submodules["attn"]
        assert attn.name == "self_attn"
        assert "q" in attn.submodules
        assert "k" in attn.submodules
        assert "v" in attn.submodules
        assert "o" in attn.submodules
        assert "q_norm" in attn.submodules
        assert "k_norm" in attn.submodules
        assert "v_norm" in attn.submodules

    def test_attn_linear_names(self, adapter):
        attn = adapter.component_mapping["blocks"].submodules["attn"]
        assert attn.submodules["q"].name == "q_proj"
        assert attn.submodules["k"].name == "k_proj"
        assert attn.submodules["v"].name == "v_proj"
        assert attn.submodules["o"].name == "o_proj"

    def test_attn_norm_names(self, adapter):
        attn = adapter.component_mapping["blocks"].submodules["attn"]
        assert attn.submodules["q_norm"].name == "q_norm"
        assert attn.submodules["k_norm"].name == "k_norm"
        assert attn.submodules["v_norm"].name == "v_norm"

    def test_mlp_submodules(self, adapter):
        mlp = adapter.component_mapping["blocks"].submodules["mlp"]
        assert mlp.name == "mlp"
        assert "gate" in mlp.submodules
        assert "in" in mlp.submodules
        assert "out" in mlp.submodules

    def test_mlp_linear_names(self, adapter):
        mlp = adapter.component_mapping["blocks"].submodules["mlp"]
        assert mlp.submodules["gate"].name == "gate_proj"
        assert mlp.submodules["in"].name == "up_proj"
        assert mlp.submodules["out"].name == "down_proj"


class TestGemma4ComponentMappingMultimodal:
    """Test component mapping paths for multimodal (ConditionalGeneration) variant."""

    @pytest.fixture
    def adapter(self):
        cfg = _make_gemma4_cfg(
            architecture="Gemma4ForConditionalGeneration",
            architectures=["Gemma4ForConditionalGeneration"],
        )
        return Gemma4ArchitectureAdapter(cfg)

    def test_embed_path_conditional(self, adapter):
        assert adapter.component_mapping["embed"].name == "model.language_model.embed_tokens"

    def test_rotary_emb_path_conditional(self, adapter):
        assert adapter.component_mapping["rotary_emb"].name == "model.language_model.rotary_emb"

    def test_blocks_path_conditional(self, adapter):
        assert adapter.component_mapping["blocks"].name == "model.language_model.layers"

    def test_ln_final_path_conditional(self, adapter):
        assert adapter.component_mapping["ln_final"].name == "model.language_model.norm"


class TestGemma4WeightConversions:
    """Test Gemma4ArchitectureAdapter weight processing conversions exist."""

    @pytest.fixture
    def adapter(self):
        cfg = _make_gemma4_cfg(architectures=["Gemma4ForCausalLM"])
        return Gemma4ArchitectureAdapter(cfg)

    def test_qkv_weight_conversions(self, adapter):
        assert "blocks.{i}.attn.q.weight" in adapter.weight_processing_conversions
        assert "blocks.{i}.attn.k.weight" in adapter.weight_processing_conversions
        assert "blocks.{i}.attn.v.weight" in adapter.weight_processing_conversions
        assert "blocks.{i}.attn.o.weight" in adapter.weight_processing_conversions

    def test_norm_weight_conversions(self, adapter):
        assert "blocks.{i}.ln1.weight" in adapter.weight_processing_conversions
        assert "blocks.{i}.ln1_post.weight" in adapter.weight_processing_conversions
        assert "blocks.{i}.ln2.weight" in adapter.weight_processing_conversions
        assert "blocks.{i}.ln2_post.weight" in adapter.weight_processing_conversions
        assert "ln_final.weight" in adapter.weight_processing_conversions

    def test_attn_norm_weight_conversions(self, adapter):
        assert "blocks.{i}.attn.q_norm.weight" in adapter.weight_processing_conversions
        assert "blocks.{i}.attn.k_norm.weight" in adapter.weight_processing_conversions
        assert "blocks.{i}.attn.v_norm.weight" in adapter.weight_processing_conversions

    def test_mlp_weight_conversions(self, adapter):
        assert "blocks.{i}.mlp.gate.weight" in adapter.weight_processing_conversions
        assert "blocks.{i}.mlp.in.weight" in adapter.weight_processing_conversions
        assert "blocks.{i}.mlp.out.weight" in adapter.weight_processing_conversions

    def test_unembed_weight_conversion(self, adapter):
        assert "unembed.weight" in adapter.weight_processing_conversions
