"""Unit tests for Glm4MoeArchitectureAdapter — programmatic configs only."""

from types import SimpleNamespace

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
    BlockBridge,
    EmbeddingBridge,
    LinearBridge,
    MoEBridge,
    PositionEmbeddingsAttentionBridge,
    RMSNormalizationBridge,
    RotaryEmbeddingBridge,
    UnembeddingBridge,
)
from transformer_lens.model_bridge.sources.transformers import (
    determine_architecture_from_hf_config,
)
from transformer_lens.model_bridge.supported_architectures.glm4_moe import (
    Glm4MoeArchitectureAdapter,
)
from transformer_lens.tools.model_registry import (
    CANONICAL_AUTHORS_BY_ARCH,
    HF_SUPPORTED_ARCHITECTURES,
)


def _fake_hf_model(rotary_emb: object) -> SimpleNamespace:
    """Minimal HF model exposing only model.rotary_emb (no config/layers)."""
    return SimpleNamespace(model=SimpleNamespace(rotary_emb=rotary_emb))


def _fake_hf_model_with_eager_targets(rotary_emb: object) -> SimpleNamespace:
    """HF model whose top-level and layer attention impl start non-eager."""
    layers = [
        SimpleNamespace(
            self_attn=SimpleNamespace(config=SimpleNamespace(_attn_implementation="sdpa"))
        )
        for _ in range(2)
    ]
    return SimpleNamespace(
        config=SimpleNamespace(_attn_implementation="sdpa"),
        model=SimpleNamespace(rotary_emb=rotary_emb, layers=layers),
    )


class DummyAttention:
    def __init__(self) -> None:
        self.rotary_emb = None

    def set_rotary_emb(self, rotary_emb: object) -> None:
        self.rotary_emb = rotary_emb


class DummyBlock:
    def __init__(self, has_attention: bool = True) -> None:
        if has_attention:
            self.attn = DummyAttention()


class DummyBridgeModel:
    def __init__(self, blocks: list[DummyBlock]) -> None:
        self.blocks = blocks


@pytest.fixture(scope="class")
def cfg() -> TransformerBridgeConfig:
    return TransformerBridgeConfig(
        d_model=64,
        d_head=16,
        n_layers=2,
        n_ctx=128,
        n_heads=4,
        n_key_value_heads=2,
        d_vocab=256,
        architecture="Glm4MoeForCausalLM",
    )


@pytest.fixture(scope="class")
def adapter(cfg: TransformerBridgeConfig) -> Glm4MoeArchitectureAdapter:
    return Glm4MoeArchitectureAdapter(cfg)


def _mapping(adapter: Glm4MoeArchitectureAdapter) -> dict[str, object]:
    mapping = adapter.component_mapping
    assert mapping is not None
    return mapping


def _conversions(adapter: Glm4MoeArchitectureAdapter) -> dict[str, object]:
    conversions = adapter.weight_processing_conversions
    assert conversions is not None
    return conversions


def _param_conversion(adapter: Glm4MoeArchitectureAdapter, key: str) -> ParamProcessingConversion:
    conversion = _conversions(adapter)[key]
    assert isinstance(conversion, ParamProcessingConversion)
    return conversion


def _rearrange(adapter: Glm4MoeArchitectureAdapter, key: str) -> RearrangeTensorConversion:
    tensor_conversion = _param_conversion(adapter, key).tensor_conversion
    assert isinstance(tensor_conversion, RearrangeTensorConversion)
    return tensor_conversion


class TestGlm4MoeAdapterConfig:
    def test_final_rms_is_true(self, adapter: Glm4MoeArchitectureAdapter) -> None:
        assert adapter.cfg.final_rms is True

    def test_attn_implementation_is_eager(self, adapter: Glm4MoeArchitectureAdapter) -> None:
        assert adapter.cfg.attn_implementation == "eager"

    def test_default_prepend_bos_is_false(self, adapter: Glm4MoeArchitectureAdapter) -> None:
        assert adapter.cfg.default_prepend_bos is False


class TestGlm4MoeWeightConversions:
    def test_conversion_keys_are_only_qkvo(self, adapter: Glm4MoeArchitectureAdapter) -> None:
        assert set(_conversions(adapter).keys()) == {
            "blocks.{i}.attn.q.weight",
            "blocks.{i}.attn.k.weight",
            "blocks.{i}.attn.v.weight",
            "blocks.{i}.attn.o.weight",
        }

    def test_qkv_rearrange_with_no_kv_heads_falls_back_to_n_heads(self) -> None:
        minimal = Glm4MoeArchitectureAdapter(
            TransformerBridgeConfig(
                d_model=64,
                d_head=16,
                n_layers=2,
                n_ctx=128,
                n_heads=4,
                d_vocab=256,
                architecture="Glm4MoeForCausalLM",
            )
        )
        assert _rearrange(minimal, "blocks.{i}.attn.k.weight").axes_lengths["n"] == 4
        assert _rearrange(minimal, "blocks.{i}.attn.v.weight").axes_lengths["n"] == 4


class TestGlm4MoeComponentMapping:
    def test_attn_has_expected_submodules(self, adapter: Glm4MoeArchitectureAdapter) -> None:
        attn = _mapping(adapter)["blocks"].submodules["attn"]
        for key in ("q", "k", "v", "o", "q_norm", "k_norm"):
            assert key in attn.submodules, f"Missing attn submodule: {key!r}"

    def test_has_hf_module_paths(self, adapter: Glm4MoeArchitectureAdapter) -> None:
        mapping = _mapping(adapter)
        assert mapping["embed"].name == "model.embed_tokens"
        assert mapping["ln_final"].name == "model.norm"
        assert mapping["unembed"].name == "lm_head"
        assert mapping["blocks"].name == "model.layers"
        sub = mapping["blocks"].submodules
        assert sub["ln1"].name == "input_layernorm"
        assert sub["ln2"].name == "post_attention_layernorm"
        assert sub["attn"].name == "self_attn"
        assert sub["mlp"].name == "mlp"

    def test_gate_submodule_is_optional_for_dense_prefix_layers(
        self, adapter: Glm4MoeArchitectureAdapter
    ) -> None:
        mlp = _mapping(adapter)["blocks"].submodules["mlp"]
        assert isinstance(mlp, MoEBridge)
        gate = mlp.submodules["gate"]
        assert isinstance(gate, LinearBridge)
        assert getattr(gate, "optional", False) is True
        assert set(mlp.submodules.keys()) == {"gate"}


class TestGlm4MoeComponentTypes:
    def test_top_level_bridge_types(self, adapter: Glm4MoeArchitectureAdapter) -> None:
        mapping = _mapping(adapter)
        assert isinstance(mapping["embed"], EmbeddingBridge)
        assert isinstance(mapping["rotary_emb"], RotaryEmbeddingBridge)
        assert isinstance(mapping["blocks"], BlockBridge)
        assert isinstance(mapping["ln_final"], RMSNormalizationBridge)
        assert isinstance(mapping["unembed"], UnembeddingBridge)

    def test_block_submodule_types(self, adapter: Glm4MoeArchitectureAdapter) -> None:
        blocks = _mapping(adapter)["blocks"]
        assert isinstance(blocks.submodules["ln1"], RMSNormalizationBridge)
        assert isinstance(blocks.submodules["ln2"], RMSNormalizationBridge)
        assert isinstance(blocks.submodules["attn"], PositionEmbeddingsAttentionBridge)
        assert isinstance(blocks.submodules["mlp"], MoEBridge)
        attn = blocks.submodules["attn"]
        assert attn.requires_attention_mask is True
        assert attn.requires_position_embeddings is True
        assert attn.submodules["q_norm"].name == "q_norm"
        assert attn.submodules["k_norm"].name == "k_norm"


class TestGlm4MoeArchitectureRegistration:
    def test_architecture_factory_registration(self) -> None:
        assert "Glm4MoeForCausalLM" in SUPPORTED_ARCHITECTURES
        assert SUPPORTED_ARCHITECTURES["Glm4MoeForCausalLM"] is Glm4MoeArchitectureAdapter

    def test_model_registry_entries(self) -> None:
        assert "Glm4MoeForCausalLM" in HF_SUPPORTED_ARCHITECTURES
        assert CANONICAL_AUTHORS_BY_ARCH["Glm4MoeForCausalLM"] == ["zai-org"]


class TestGlm4MoeArchitectureDetection:
    def test_model_type_routes_to_glm4_moe_architecture(self) -> None:
        cfg = SimpleNamespace(model_type="glm4_moe", architectures=[])
        assert determine_architecture_from_hf_config(cfg) == "Glm4MoeForCausalLM"

    def test_architectures_list_takes_precedence(self) -> None:
        cfg = SimpleNamespace(
            model_type="glm4_moe",
            architectures=["Glm4MoeForCausalLM", "Qwen3MoeForCausalLM"],
        )
        assert determine_architecture_from_hf_config(cfg) == "Glm4MoeForCausalLM"


class TestGlm4MoeSetupComponentTesting:
    """setup_component_testing wires shared RoPE embedding and forces eager attention."""

    def test_sets_rotary_emb_on_template_attention(
        self, adapter: Glm4MoeArchitectureAdapter
    ) -> None:
        rotary_emb = object()
        attn_template = adapter.get_generalized_component("blocks.0.attn")
        assert isinstance(attn_template, PositionEmbeddingsAttentionBridge)
        assert attn_template._rotary_emb is None

        adapter.setup_component_testing(_fake_hf_model(rotary_emb))

        assert attn_template._rotary_emb is rotary_emb

    def test_sets_rotary_emb_on_each_bridge_model_attention(
        self, adapter: Glm4MoeArchitectureAdapter
    ) -> None:
        rotary_emb = object()
        bridge_model = DummyBridgeModel([DummyBlock(), DummyBlock()])

        adapter.setup_component_testing(_fake_hf_model(rotary_emb), bridge_model=bridge_model)

        for block in bridge_model.blocks:
            assert block.attn.rotary_emb is rotary_emb

    def test_skips_bridge_blocks_without_attention(
        self, adapter: Glm4MoeArchitectureAdapter
    ) -> None:
        rotary_emb = object()
        bridge_model = DummyBridgeModel([DummyBlock(), DummyBlock(has_attention=False)])

        adapter.setup_component_testing(_fake_hf_model(rotary_emb), bridge_model=bridge_model)

        assert bridge_model.blocks[0].attn.rotary_emb is rotary_emb

    def test_forces_eager_attention_implementation(
        self, adapter: Glm4MoeArchitectureAdapter
    ) -> None:
        hf_model = _fake_hf_model_with_eager_targets(object())

        adapter.setup_component_testing(hf_model)

        assert hf_model.config._attn_implementation == "eager"
        for layer in hf_model.model.layers:
            assert layer.self_attn.config._attn_implementation == "eager"

    def test_tolerates_minimal_hf_model_without_config_or_layers(
        self, adapter: Glm4MoeArchitectureAdapter
    ) -> None:
        rotary_emb = object()
        adapter.setup_component_testing(_fake_hf_model(rotary_emb))

        attn_template = adapter.get_generalized_component("blocks.0.attn")
        assert isinstance(attn_template, PositionEmbeddingsAttentionBridge)
        assert attn_template._rotary_emb is rotary_emb
