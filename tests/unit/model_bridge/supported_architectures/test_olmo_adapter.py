"""Unit tests for OlmoArchitectureAdapter.

Tests cover:
- config attributes and default config propagation
- component mapping structure and HF module names
- weight conversion keys, types, and GQA head counts
- PositionEmbeddingsAttentionBridge forward execution and hook shapes
- prepare_model clamp patching
- setup_component_testing eager-attention and rotary wiring
- factory registration
"""

from __future__ import annotations

import copy
from types import SimpleNamespace
from typing import Any

import pytest
import torch
import torch.nn as nn

from transformer_lens.config import TransformerBridgeConfig
from transformer_lens.conversion_utils.conversion_steps import RearrangeTensorConversion
from transformer_lens.conversion_utils.param_processing_conversion import (
    ParamProcessingConversion,
)
from transformer_lens.model_bridge.component_setup import setup_submodules
from transformer_lens.model_bridge.generalized_components import (
    BlockBridge,
    EmbeddingBridge,
    GatedMLPBridge,
    LinearBridge,
    NormalizationBridge,
    PositionEmbeddingsAttentionBridge,
    RotaryEmbeddingBridge,
    UnembeddingBridge,
)
from transformer_lens.model_bridge.supported_architectures.olmo import (
    OlmoArchitectureAdapter,
)


def _make_cfg(
    n_heads: int = 4,
    n_key_value_heads: int | None = 2,
    d_model: int = 32,
    n_layers: int = 2,
    d_mlp: int = 128,
    d_vocab: int = 100,
    n_ctx: int = 64,
) -> TransformerBridgeConfig:
    """Build a minimal bridge config for OLMo adapter unit tests."""
    return TransformerBridgeConfig(
        d_model=d_model,
        d_head=d_model // n_heads,
        n_layers=n_layers,
        n_ctx=n_ctx,
        n_heads=n_heads,
        n_key_value_heads=n_key_value_heads,
        d_vocab=d_vocab,
        d_mlp=d_mlp,
        architecture="OlmoForCausalLM",
    )


@pytest.fixture
def cfg() -> TransformerBridgeConfig:
    """Provide a small default config shared across OLMo adapter tests."""
    return _make_cfg()


@pytest.fixture
def adapter(cfg: TransformerBridgeConfig) -> OlmoArchitectureAdapter:
    """Instantiate the adapter under test from the shared config fixture."""
    return OlmoArchitectureAdapter(cfg)


class _FakeOlmoAttention(nn.Module):
    """Minimal OLMo-style attention module for bridge forward tests."""

    def __init__(self, d_model: int, n_heads: int, n_kv_heads: int) -> None:
        """Expose the attributes PositionEmbeddingsAttentionBridge expects."""
        super().__init__()
        head_dim = d_model // n_heads
        self.head_dim = head_dim
        self.num_key_value_groups = n_heads // n_kv_heads
        self.scaling = head_dim**-0.5
        self.attention_dropout = 0.0
        self.config = SimpleNamespace(_attn_implementation="sdpa")
        self.q_proj = nn.Linear(d_model, n_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(d_model, n_kv_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(d_model, n_kv_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(n_heads * head_dim, d_model, bias=False)


class _RecordingClampAttention(nn.Module):
    """Attention stub that records clip_qkv during forward."""

    def __init__(self, config: Any) -> None:
        """Keep a reference to the shared HF config and a call log."""
        super().__init__()
        self.config = config
        self.seen_clip_qkv: list[Any] = []

    def forward(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        """Record the current clip_qkv value and return a dummy tensor."""
        self.seen_clip_qkv.append(self.config.clip_qkv)
        return torch.tensor(0.0)


def _make_position_embeddings(seq_len: int, head_dim: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Return identity rotary embeddings for a deterministic attention path."""
    # Identity RoPE keeps the forward path simple while still exercising the
    # same bridge logic used for real OLMo attention.
    cos = torch.ones(1, seq_len, head_dim)
    sin = torch.zeros(1, seq_len, head_dim)
    return cos, sin


def _wire_attention_bridge(
    adapter: OlmoArchitectureAdapter,
    cfg: TransformerBridgeConfig,
) -> PositionEmbeddingsAttentionBridge:
    """Attach a fake HF attention module to a copied attention bridge."""
    # Component setup normally adds q/k/v/o as actual modules and points them at
    # the HF attention layer. For a standalone unit test we mirror that wiring
    # directly so the bridge executes the real forward path.
    attn_bridge = copy.deepcopy(adapter.get_generalized_component("blocks.0.attn"))
    assert isinstance(attn_bridge, PositionEmbeddingsAttentionBridge)
    fake_attn = _FakeOlmoAttention(
        d_model=cfg.d_model,
        n_heads=cfg.n_heads,
        n_kv_heads=cfg.n_key_value_heads or cfg.n_heads,
    )
    attn_bridge.set_original_component(fake_attn)
    setup_submodules(attn_bridge, adapter, fake_attn)
    attn_bridge.setup_hook_compatibility()
    return attn_bridge


class TestOlmoAdapterConfig:
    """Config mutations performed by OlmoArchitectureAdapter."""

    def test_default_config_propagates_gqa(self, adapter: OlmoArchitectureAdapter) -> None:
        """Any configured KV-head count should be mirrored into default_config."""
        assert adapter.default_config["n_key_value_heads"] == adapter.cfg.n_key_value_heads


class TestOlmoAdapterComponentMapping:
    """Structural tests for the adapter's component mapping."""

    @staticmethod
    def _mapping(adapter: OlmoArchitectureAdapter) -> dict[str, Any]:
        """Return the mapping and assert it was initialized."""
        mapping = adapter.component_mapping
        assert mapping is not None
        return mapping

    def test_embed_type_and_name(self, adapter: OlmoArchitectureAdapter) -> None:
        """Token embeddings should map to model.embed_tokens."""
        mapping = self._mapping(adapter)
        assert isinstance(mapping["embed"], EmbeddingBridge)
        assert mapping["embed"].name == "model.embed_tokens"

    def test_rotary_emb_type_and_name(self, adapter: OlmoArchitectureAdapter) -> None:
        """Rotary embeddings should map to the top-level OLMo rotary module."""
        mapping = self._mapping(adapter)
        assert isinstance(mapping["rotary_emb"], RotaryEmbeddingBridge)
        assert mapping["rotary_emb"].name == "model.rotary_emb"

    def test_blocks_type_and_name(self, adapter: OlmoArchitectureAdapter) -> None:
        """Transformer blocks should point at model.layers."""
        mapping = self._mapping(adapter)
        assert isinstance(mapping["blocks"], BlockBridge)
        assert mapping["blocks"].name == "model.layers"

    def test_ln_final_type_name_and_native_autograd(self, adapter: OlmoArchitectureAdapter) -> None:
        """The final LayerNorm should delegate to HF autograd for parity."""
        mapping = self._mapping(adapter)
        assert isinstance(mapping["ln_final"], NormalizationBridge)
        assert mapping["ln_final"].name == "model.norm"
        assert mapping["ln_final"].use_native_layernorm_autograd is True

    def test_unembed_type_and_name(self, adapter: OlmoArchitectureAdapter) -> None:
        """The language-model head should map to lm_head."""
        mapping = self._mapping(adapter)
        assert isinstance(mapping["unembed"], UnembeddingBridge)
        assert mapping["unembed"].name == "lm_head"

    def test_ln1_type_name_and_native_autograd(self, adapter: OlmoArchitectureAdapter) -> None:
        """ln1 should wrap the pre-attention LayerNorm with HF autograd."""
        blocks = self._mapping(adapter)["blocks"]
        ln1 = blocks.submodules["ln1"]
        assert isinstance(ln1, NormalizationBridge)
        assert ln1.name == "input_layernorm"
        assert ln1.use_native_layernorm_autograd is True

    def test_ln2_type_name_and_native_autograd(self, adapter: OlmoArchitectureAdapter) -> None:
        """ln2 should wrap the pre-MLP LayerNorm with HF autograd."""
        blocks = self._mapping(adapter)["blocks"]
        ln2 = blocks.submodules["ln2"]
        assert isinstance(ln2, NormalizationBridge)
        assert ln2.name == "post_attention_layernorm"
        assert ln2.use_native_layernorm_autograd is True

    def test_attn_mapping(self, adapter: OlmoArchitectureAdapter) -> None:
        """Attention should expose q/k/v/o projections under self_attn."""
        blocks = self._mapping(adapter)["blocks"]
        attn = blocks.submodules["attn"]
        assert isinstance(attn, PositionEmbeddingsAttentionBridge)
        assert attn.name == "self_attn"
        assert attn.requires_attention_mask is True
        assert attn.requires_position_embeddings is True

        assert isinstance(attn.submodules["q"], LinearBridge)
        assert isinstance(attn.submodules["k"], LinearBridge)
        assert isinstance(attn.submodules["v"], LinearBridge)
        assert isinstance(attn.submodules["o"], LinearBridge)

        assert attn.submodules["q"].name == "q_proj"
        assert attn.submodules["k"].name == "k_proj"
        assert attn.submodules["v"].name == "v_proj"
        assert attn.submodules["o"].name == "o_proj"

    def test_mlp_mapping(self, adapter: OlmoArchitectureAdapter) -> None:
        """The gated MLP should map gate/up/down projections correctly."""
        blocks = self._mapping(adapter)["blocks"]
        mlp = blocks.submodules["mlp"]
        assert isinstance(mlp, GatedMLPBridge)

        assert mlp.name == "mlp"
        assert mlp.submodules["gate"].name == "gate_proj"
        assert mlp.submodules["in"].name == "up_proj"
        assert mlp.submodules["out"].name == "down_proj"


class TestOlmoAdapterWeightConversions:
    """Weight conversion rules for OLMo attention projections."""

    def test_qkvo_keys_present(self, adapter: OlmoArchitectureAdapter) -> None:
        """Only Q/K/V/O projection weights need custom conversion rules."""
        convs = adapter.weight_processing_conversions
        assert convs is not None
        assert set(convs) == {
            "blocks.{i}.attn.q.weight",
            "blocks.{i}.attn.k.weight",
            "blocks.{i}.attn.v.weight",
            "blocks.{i}.attn.o.weight",
        }

    def test_q_conversion_uses_n_heads(self, adapter: OlmoArchitectureAdapter) -> None:
        """Q should split rows using the full attention head count."""
        convs = adapter.weight_processing_conversions
        assert convs is not None
        conv = convs["blocks.{i}.attn.q.weight"]
        assert isinstance(conv, ParamProcessingConversion)
        assert isinstance(conv.tensor_conversion, RearrangeTensorConversion)
        assert conv.tensor_conversion.pattern == "(n h) m -> n m h"
        assert conv.tensor_conversion.axes_lengths["n"] == adapter.cfg.n_heads

    def test_kv_conversions_use_n_key_value_heads(self, adapter: OlmoArchitectureAdapter) -> None:
        """K and V should use KV heads rather than query heads under GQA."""
        convs = adapter.weight_processing_conversions
        assert convs is not None
        for key in ("blocks.{i}.attn.k.weight", "blocks.{i}.attn.v.weight"):
            conv = convs[key]
            assert isinstance(conv, ParamProcessingConversion)
            assert isinstance(conv.tensor_conversion, RearrangeTensorConversion)
            assert conv.tensor_conversion.pattern == "(n h) m -> n m h"
            assert conv.tensor_conversion.axes_lengths["n"] == adapter.cfg.n_key_value_heads

    def test_o_conversion_uses_n_heads(self, adapter: OlmoArchitectureAdapter) -> None:
        """The output projection should split its input dimension by query heads."""
        convs = adapter.weight_processing_conversions
        assert convs is not None
        conv = convs["blocks.{i}.attn.o.weight"]
        assert isinstance(conv, ParamProcessingConversion)
        assert isinstance(conv.tensor_conversion, RearrangeTensorConversion)
        assert conv.tensor_conversion.pattern == "m (n h) -> n h m"
        assert conv.tensor_conversion.axes_lengths["n"] == adapter.cfg.n_heads


class TestOlmoAttentionBridge:
    """Forward-path tests for the adapter's attention bridge."""

    def test_forward_executes_and_matches_hook_shapes(
        self, adapter: OlmoArchitectureAdapter, cfg: TransformerBridgeConfig
    ) -> None:
        """A fake OLMo attention module should drive the real hookable forward path."""
        attn_bridge = _wire_attention_bridge(adapter, cfg)
        batch, seq_len = 2, 5
        hidden_states = torch.randn(batch, seq_len, cfg.d_model)
        position_embeddings = _make_position_embeddings(seq_len, cfg.d_head)
        # The reconstructed attention path consumes a full 4D additive/boolean
        # mask rather than the 2D convenience mask generated by get_random_inputs.
        attention_mask = torch.tril(torch.ones(batch, 1, seq_len, seq_len, dtype=torch.bool))
        attention_mask[..., -1] = False

        seen: dict[str, torch.Size] = {}

        def _record_shape(name: str):
            """Capture hook tensor shapes without modifying the live tensor."""

            def _hook(tensor: torch.Tensor, hook: Any) -> None:
                seen.setdefault(name, tensor.shape)
                return None

            return _hook

        attn_bridge.q.hook_out.add_hook(_record_shape("q"))
        attn_bridge.k.hook_out.add_hook(_record_shape("k"))
        attn_bridge.v.hook_out.add_hook(_record_shape("v"))
        attn_bridge.o.hook_in.add_hook(_record_shape("z"))

        output, pattern = attn_bridge(
            hidden_states=hidden_states,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
        )

        assert output.shape == (batch, seq_len, cfg.d_model)
        assert pattern.shape == (batch, cfg.n_heads, seq_len, seq_len)
        assert seen["q"] == torch.Size([batch, seq_len, cfg.n_heads, cfg.d_head])
        assert seen["k"] == torch.Size([batch, seq_len, cfg.n_key_value_heads, cfg.d_head])
        assert seen["v"] == torch.Size([batch, seq_len, cfg.n_key_value_heads, cfg.d_head])
        assert seen["z"] == torch.Size([batch, seq_len, cfg.n_heads, cfg.d_head])


class TestOlmoPrepareModel:
    """Behavioral tests for OLMo's in-place clamp patch."""

    def test_prepare_model_patches_clip_qkv_forward(self, adapter: OlmoArchitectureAdapter) -> None:
        """prepare_model should suppress in-place clamping during attention forward."""
        config = SimpleNamespace(clip_qkv=128.0)
        attn0 = _RecordingClampAttention(config)
        attn1 = _RecordingClampAttention(config)
        hf_model = SimpleNamespace(
            config=config,
            model=SimpleNamespace(
                layers=[
                    SimpleNamespace(self_attn=attn0),
                    SimpleNamespace(self_attn=attn1),
                ]
            ),
        )

        adapter.prepare_model(hf_model)
        attn0()
        attn1()

        assert attn0.seen_clip_qkv == [None]
        assert attn1.seen_clip_qkv == [None]
        assert config.clip_qkv == 128.0

    def test_prepare_model_is_noop_when_clip_qkv_missing(
        self, adapter: OlmoArchitectureAdapter
    ) -> None:
        """No wrapper should be installed when clip_qkv is already disabled."""
        config = SimpleNamespace(clip_qkv=None)
        attn = _RecordingClampAttention(config)
        original_forward = attn.__class__.forward
        hf_model = SimpleNamespace(
            config=config,
            model=SimpleNamespace(layers=[SimpleNamespace(self_attn=attn)]),
        )

        adapter.prepare_model(hf_model)

        assert attn.forward.__func__ is original_forward


class TestOlmoSetupComponentTesting:
    """Tests for the helper that prepares bridges for component-level checks."""

    def test_sets_rotary_emb_and_forces_eager_attention(
        self, adapter: OlmoArchitectureAdapter
    ) -> None:
        """setup_component_testing should wire RoPE references and eager attention."""
        rotary_emb = object()
        layer0_cfg = SimpleNamespace(_attn_implementation="sdpa")
        layer1_cfg = SimpleNamespace(_attn_implementation="flash_attention_2")
        hf_model = SimpleNamespace(
            config=SimpleNamespace(_attn_implementation="sdpa"),
            model=SimpleNamespace(
                rotary_emb=rotary_emb,
                layers=[
                    SimpleNamespace(self_attn=SimpleNamespace(config=layer0_cfg)),
                    SimpleNamespace(self_attn=SimpleNamespace(config=layer1_cfg)),
                ],
            ),
        )

        bridge_attn0 = copy.deepcopy(adapter.get_generalized_component("blocks.0.attn"))
        bridge_attn1 = copy.deepcopy(adapter.get_generalized_component("blocks.0.attn"))
        bridge_model = SimpleNamespace(
            blocks=[
                SimpleNamespace(attn=bridge_attn0),
                SimpleNamespace(attn=bridge_attn1),
            ]
        )

        adapter.setup_component_testing(hf_model, bridge_model=bridge_model)

        assert hf_model.config._attn_implementation == "eager"
        assert layer0_cfg._attn_implementation == "eager"
        assert layer1_cfg._attn_implementation == "eager"
        assert bridge_attn0._rotary_emb is rotary_emb
        assert bridge_attn1._rotary_emb is rotary_emb

        template_attn = adapter.get_generalized_component("blocks.0.attn")
        assert template_attn._rotary_emb is rotary_emb
