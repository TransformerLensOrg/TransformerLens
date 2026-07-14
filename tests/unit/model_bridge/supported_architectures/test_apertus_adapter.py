"""Download-free unit tests for :class:`ApertusArchitectureAdapter`.

The suite covers HF config routing, adapter selection, component paths, GQA
weight reshaping, key translation, rotary test wiring, and the temporary XiELU
meta-device compatibility patch.
"""

from types import SimpleNamespace
from typing import Any

import pytest
import torch
import torch.nn as nn
from transformers import ApertusConfig
from transformers import activations as transformers_activations

from transformer_lens.conversion_utils.conversion_steps import RearrangeTensorConversion
from transformer_lens.conversion_utils.param_processing_conversion import (
    ParamProcessingConversion,
)
from transformer_lens.factories.architecture_adapter_factory import (
    SUPPORTED_ARCHITECTURES,
    ArchitectureAdapterFactory,
)
from transformer_lens.model_bridge.generalized_components import (
    BlockBridge,
    EmbeddingBridge,
    LinearBridge,
    MLPBridge,
    PositionEmbeddingsAttentionBridge,
    RMSNormalizationBridge,
    RotaryEmbeddingBridge,
    UnembeddingBridge,
)
from transformer_lens.model_bridge.sources._bridge_builder import (
    build_bridge_config_from_hf,
)
from transformer_lens.model_bridge.sources.transformers import (
    determine_architecture_from_hf_config,
)
from transformer_lens.model_bridge.supported_architectures.apertus import (
    ApertusArchitectureAdapter,
)
from transformer_lens.tools.model_registry import (
    CANONICAL_AUTHORS_BY_ARCH,
    HF_SUPPORTED_ARCHITECTURES,
)


def _tiny_hf_config() -> ApertusConfig:
    return ApertusConfig(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=96,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=64,
        hidden_act="xielu",
        rms_norm_eps=1e-5,
    )


@pytest.fixture
def bridge_cfg():
    return build_bridge_config_from_hf(
        _tiny_hf_config(),
        "ApertusForCausalLM",
        "apertus-test",
        torch.float32,
    )


@pytest.fixture
def adapter(bridge_cfg) -> ApertusArchitectureAdapter:
    return ApertusArchitectureAdapter(bridge_cfg)


def _conversions(adapter: ApertusArchitectureAdapter) -> dict:
    conversions = adapter.weight_processing_conversions
    assert conversions is not None
    assert all(
        isinstance(conversion, ParamProcessingConversion) for conversion in conversions.values()
    )
    return conversions


def _rearrange(adapter: ApertusArchitectureAdapter, key: str) -> RearrangeTensorConversion:
    conversion = _conversions(adapter)[key].tensor_conversion
    assert isinstance(conversion, RearrangeTensorConversion)
    return conversion


class TestApertusConfigAndRegistration:
    def test_model_type_routes_to_apertus(self) -> None:
        assert determine_architecture_from_hf_config(_tiny_hf_config()) == "ApertusForCausalLM"

    def test_hf_config_fields_map_to_bridge_config(self, bridge_cfg) -> None:
        hf_cfg = _tiny_hf_config()
        assert bridge_cfg.d_vocab == hf_cfg.vocab_size
        assert bridge_cfg.d_model == hf_cfg.hidden_size
        assert bridge_cfg.d_mlp == hf_cfg.intermediate_size
        assert bridge_cfg.n_layers == hf_cfg.num_hidden_layers
        assert bridge_cfg.n_heads == hf_cfg.num_attention_heads
        assert bridge_cfg.n_key_value_heads == hf_cfg.num_key_value_heads
        assert bridge_cfg.n_ctx == hf_cfg.max_position_embeddings
        assert bridge_cfg.d_head == hf_cfg.hidden_size // hf_cfg.num_attention_heads
        assert bridge_cfg.act_fn == "xielu"
        assert bridge_cfg.eps == hf_cfg.rms_norm_eps

    def test_factory_selects_apertus_adapter(self, bridge_cfg) -> None:
        selected = ArchitectureAdapterFactory.select_architecture_adapter(bridge_cfg)
        assert isinstance(selected, ApertusArchitectureAdapter)

    def test_registration_sources_stay_in_sync(self) -> None:
        assert SUPPORTED_ARCHITECTURES["ApertusForCausalLM"] is ApertusArchitectureAdapter
        assert "ApertusForCausalLM" in HF_SUPPORTED_ARCHITECTURES
        assert CANONICAL_AUTHORS_BY_ARCH["ApertusForCausalLM"] == ["swiss-ai"]


class TestApertusAdapterConfig:
    def test_architecture_flags(self, adapter: ApertusArchitectureAdapter) -> None:
        assert adapter.cfg.normalization_type == "RMS"
        assert adapter.cfg.positional_embedding_type == "rotary"
        assert adapter.cfg.final_rms is True
        assert adapter.cfg.gated_mlp is False
        assert adapter.cfg.attn_only is False
        assert adapter.cfg.uses_rms_norm is True

    def test_eager_attention_is_required_for_hooks(
        self, adapter: ApertusArchitectureAdapter
    ) -> None:
        assert adapter.cfg.attn_implementation == "eager"


class TestApertusComponentMapping:
    def test_top_level_components(self, adapter: ApertusArchitectureAdapter) -> None:
        mapping = adapter.component_mapping
        assert set(mapping) == {"embed", "rotary_emb", "blocks", "ln_final", "unembed"}
        assert isinstance(mapping["embed"], EmbeddingBridge)
        assert isinstance(mapping["rotary_emb"], RotaryEmbeddingBridge)
        assert isinstance(mapping["blocks"], BlockBridge)
        assert isinstance(mapping["ln_final"], RMSNormalizationBridge)
        assert isinstance(mapping["unembed"], UnembeddingBridge)
        assert mapping["embed"].name == "model.embed_tokens"
        assert mapping["rotary_emb"].name == "model.rotary_emb"
        assert mapping["blocks"].name == "model.layers"
        assert mapping["ln_final"].name == "model.norm"
        assert mapping["unembed"].name == "lm_head"

    def test_block_norm_paths(self, adapter: ApertusArchitectureAdapter) -> None:
        blocks = adapter.component_mapping["blocks"]
        assert set(blocks.submodules) == {"ln1", "ln2", "attn", "mlp"}
        assert isinstance(blocks.submodules["ln1"], RMSNormalizationBridge)
        assert isinstance(blocks.submodules["ln2"], RMSNormalizationBridge)
        assert blocks.submodules["ln1"].name == "attention_layernorm"
        assert blocks.submodules["ln2"].name == "feedforward_layernorm"

    def test_attention_paths_and_types(self, adapter: ApertusArchitectureAdapter) -> None:
        attn = adapter.component_mapping["blocks"].submodules["attn"]
        assert isinstance(attn, PositionEmbeddingsAttentionBridge)
        assert attn.name == "self_attn"
        assert set(attn.submodules) == {"q", "k", "v", "o", "q_norm", "k_norm"}
        for name in ("q", "k", "v", "o"):
            assert isinstance(attn.submodules[name], LinearBridge)
        assert attn.submodules["q"].name == "q_proj"
        assert attn.submodules["k"].name == "k_proj"
        assert attn.submodules["v"].name == "v_proj"
        assert attn.submodules["o"].name == "o_proj"
        assert isinstance(attn.submodules["q_norm"], RMSNormalizationBridge)
        assert isinstance(attn.submodules["k_norm"], RMSNormalizationBridge)
        assert attn.submodules["q_norm"].name == "q_norm"
        assert attn.submodules["k_norm"].name == "k_norm"

    def test_mlp_is_non_gated(self, adapter: ApertusArchitectureAdapter) -> None:
        mlp = adapter.component_mapping["blocks"].submodules["mlp"]
        assert isinstance(mlp, MLPBridge)
        assert mlp.name == "mlp"
        assert set(mlp.submodules) == {"in", "out"}
        assert mlp.submodules["in"].name == "up_proj"
        assert mlp.submodules["out"].name == "down_proj"


class TestApertusWeightConversions:
    def test_qkvo_conversion_contract(self, adapter: ApertusArchitectureAdapter) -> None:
        assert set(_conversions(adapter)) == {
            "blocks.{i}.attn.q.weight",
            "blocks.{i}.attn.k.weight",
            "blocks.{i}.attn.v.weight",
            "blocks.{i}.attn.o.weight",
        }

    @pytest.mark.parametrize(
        ("slot", "expected_heads"),
        [("q", 4), ("k", 2), ("v", 2)],
    )
    def test_qkv_conversion_uses_correct_head_count(
        self,
        adapter: ApertusArchitectureAdapter,
        slot: str,
        expected_heads: int,
    ) -> None:
        conversion = _rearrange(adapter, f"blocks.{{i}}.attn.{slot}.weight")
        assert conversion.pattern == "(n h) m -> n m h"
        assert conversion.axes_lengths == {"n": expected_heads}

    def test_output_conversion_uses_query_heads(self, adapter: ApertusArchitectureAdapter) -> None:
        conversion = _rearrange(adapter, "blocks.{i}.attn.o.weight")
        assert conversion.pattern == "m (n h) -> n h m"
        assert conversion.axes_lengths == {"n": 4}

    @pytest.mark.parametrize(
        ("slot", "heads"),
        [("q", 4), ("k", 2), ("v", 2)],
    )
    def test_qkv_conversion_round_trip(
        self,
        adapter: ApertusArchitectureAdapter,
        slot: str,
        heads: int,
    ) -> None:
        d_head = adapter.cfg.d_head
        raw = torch.arange(heads * d_head * adapter.cfg.d_model).reshape(
            heads * d_head, adapter.cfg.d_model
        )
        conversion = _rearrange(adapter, f"blocks.{{i}}.attn.{slot}.weight")
        converted = conversion.convert(raw)
        assert converted.shape == (heads, adapter.cfg.d_model, d_head)
        assert torch.equal(conversion.revert(converted), raw)

    def test_output_conversion_round_trip(self, adapter: ApertusArchitectureAdapter) -> None:
        raw = torch.arange(adapter.cfg.d_model**2).reshape(
            adapter.cfg.d_model, adapter.cfg.d_model
        )
        conversion = _rearrange(adapter, "blocks.{i}.attn.o.weight")
        converted = conversion.convert(raw)
        assert converted.shape == (
            adapter.cfg.n_heads,
            adapter.cfg.d_head,
            adapter.cfg.d_model,
        )
        assert torch.equal(conversion.revert(converted), raw)


class TestApertusKeyTranslation:
    @pytest.mark.parametrize(
        ("hf_key", "tl_key"),
        [
            ("model.embed_tokens.weight", "embed.weight"),
            ("model.layers.1.attention_layernorm.weight", "blocks.1.ln1.weight"),
            ("model.layers.1.feedforward_layernorm.weight", "blocks.1.ln2.weight"),
            ("model.layers.1.self_attn.q_proj.weight", "blocks.1.attn.q.weight"),
            ("model.layers.1.self_attn.k_norm.weight", "blocks.1.attn.k_norm.weight"),
            ("model.layers.1.mlp.up_proj.weight", "blocks.1.mlp.in.weight"),
            ("model.layers.1.mlp.down_proj.weight", "blocks.1.mlp.out.weight"),
            ("model.norm.weight", "ln_final.weight"),
            ("lm_head.weight", "unembed.weight"),
        ],
    )
    def test_hf_keys_translate_to_canonical_paths(
        self,
        adapter: ApertusArchitectureAdapter,
        hf_key: str,
        tl_key: str,
    ) -> None:
        assert adapter.convert_hf_key_to_tl_key(hf_key) == tl_key


class _DummyAttention:
    def __init__(self) -> None:
        self.rotary_emb: object | None = None

    def set_rotary_emb(self, rotary_emb: object) -> None:
        self.rotary_emb = rotary_emb


class _DummyBlock:
    def __init__(self, with_attention: bool = True) -> None:
        if with_attention:
            self.attn = _DummyAttention()


def _fake_hf_model(rotary_emb: object, n_layers: int = 2) -> SimpleNamespace:
    layers = [
        SimpleNamespace(
            self_attn=SimpleNamespace(config=SimpleNamespace(_attn_implementation="sdpa"))
        )
        for _ in range(n_layers)
    ]
    return SimpleNamespace(
        config=SimpleNamespace(_attn_implementation="sdpa"),
        model=SimpleNamespace(rotary_emb=rotary_emb, layers=layers),
    )


class TestApertusSetupComponentTesting:
    def test_forces_eager_attention_and_wires_rotary_embeddings(
        self, adapter: ApertusArchitectureAdapter
    ) -> None:
        rotary_emb = object()
        hf_model = _fake_hf_model(rotary_emb, n_layers=3)
        bridge_model = SimpleNamespace(blocks=[_DummyBlock(), _DummyBlock(), _DummyBlock(False)])

        adapter.setup_component_testing(hf_model, bridge_model)

        assert hf_model.config._attn_implementation == "eager"
        assert all(
            layer.self_attn.config._attn_implementation == "eager"
            for layer in hf_model.model.layers
        )
        assert bridge_model.blocks[0].attn.rotary_emb is rotary_emb
        assert bridge_model.blocks[1].attn.rotary_emb is rotary_emb
        template = adapter.get_generalized_component("blocks.0.attn")
        assert template._rotary_emb is rotary_emb

    def test_bridge_model_is_optional(self, adapter: ApertusArchitectureAdapter) -> None:
        hf_model = _fake_hf_model(object())
        adapter.setup_component_testing(hf_model)
        assert hf_model.config._attn_implementation == "eager"


class TestApertusXieluCompatibilityPatch:
    def test_legacy_meta_failure_is_deferred_until_forward(
        self,
        adapter: ApertusArchitectureAdapter,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        class LegacyXIELUActivation(nn.Module):
            def __init__(self, dtype: torch.dtype = torch.float32, **kwargs: Any) -> None:
                super().__init__()
                if kwargs.get("fail_on_meta", True):
                    raise NotImplementedError("Cannot copy out of meta tensor")
                self.beta = torch.tensor(0.5, dtype=dtype)
                self.eps = torch.tensor(-1e-6, dtype=dtype)
                self._beta_scalar = float(self.beta.item())
                self._eps_scalar = float(self.eps.item())

            def forward(self, inputs: torch.Tensor) -> torch.Tensor:
                return inputs + self._beta_scalar

        monkeypatch.setattr(
            transformers_activations,
            "XIELUActivation",
            LegacyXIELUActivation,
        )

        adapter.prepare_loading("apertus-test", {})
        patched_init = LegacyXIELUActivation.__init__
        adapter.prepare_loading("apertus-test", {})

        assert LegacyXIELUActivation.__init__ is patched_init
        assert getattr(LegacyXIELUActivation, "_apertus_patched", False) is True

        activation = LegacyXIELUActivation(dtype=torch.float32)
        assert activation._beta_scalar is None
        assert activation._eps_scalar is None

        result = activation(torch.zeros(2))

        assert torch.allclose(result, torch.full((2,), 0.5))
        assert activation._beta_scalar == pytest.approx(0.5)
        assert activation._eps_scalar == pytest.approx(-1e-6)
