"""Unit tests for the GPTOSSArchitectureAdapter (download-free, tiny programmatic configs
plus small synthetic tensors and a fake attention module, no real checkpoints).

Covered:
- Adapter config defaults (RMSNorm with `variance_epsilon`, rotary, gated MoE MLP).
- Weight conversions: QKVO weights (no biases) with GQA-aware head counts.
- Numerical round-trips: the rearrange conversions actually reshape and revert losslessly.
- Component-mapping structure, bridge types, and HF module paths.
- Factory registration and dispatch.
- GQA forward hook shapes (Q uses n_heads, K/V use n_key_value_heads).
- setup_hook_compatibility rotary-embedding wiring on a bridge model, plus the
  setup_no_processing_hooks backward-compat alias.
"""

from types import SimpleNamespace
from typing import Any

import pytest
import torch.nn as nn
from torch import equal, ones, randn, zeros

from transformer_lens.config import TransformerBridgeConfig
from transformer_lens.conversion_utils.conversion_steps.rearrange_tensor_conversion import (
    RearrangeTensorConversion,
)
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
    GatedMLPBridge,
    LinearBridge,
    MoEBridge,
    PositionEmbeddingsAttentionBridge,
    RMSNormalizationBridge,
    RotaryEmbeddingBridge,
    UnembeddingBridge,
)
from transformer_lens.model_bridge.supported_architectures.gpt_oss import (
    GPTOSSArchitectureAdapter,
)


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
        architecture="GptOssForCausalLM",
    )


@pytest.fixture(scope="class")
def adapter(cfg: TransformerBridgeConfig) -> GPTOSSArchitectureAdapter:
    return GPTOSSArchitectureAdapter(cfg)


def _cfg(*, n_key_value_heads: int | None = 2) -> TransformerBridgeConfig:
    # Keep dimensions tiny so adapter tests do not need HF downloads or real checkpoints.
    # n_key_value_heads=None exercises the GQA fallback to n_heads in the adapter.
    return TransformerBridgeConfig(
        d_model=64,
        d_head=16,
        n_layers=2,
        n_ctx=128,
        n_heads=4,
        n_key_value_heads=n_key_value_heads,
        d_vocab=256,
        architecture="GptOssForCausalLM",
    )


def _mapping(adapter: GPTOSSArchitectureAdapter) -> dict:
    """Narrow component_mapping (Optional on the base class) to a non-None dict.

    Factored into a helper so each test stays a one-liner instead of repeating the
    `assert ... is not None` prelude in every method.
    """
    mapping = adapter.component_mapping
    assert mapping is not None
    return mapping


def _conversions(adapter: GPTOSSArchitectureAdapter) -> dict:
    """weight_processing_conversions is Optional on the base class, assert it is populated."""
    conversions = adapter.weight_processing_conversions
    assert conversions is not None
    return conversions


def _param_conversion(adapter: GPTOSSArchitectureAdapter, key: str) -> ParamProcessingConversion:
    conv = _conversions(adapter)[key]
    assert isinstance(conv, ParamProcessingConversion)
    return conv


def _rearrange(adapter: GPTOSSArchitectureAdapter, key: str) -> RearrangeTensorConversion:
    tensor_conversion = _param_conversion(adapter, key).tensor_conversion
    assert isinstance(tensor_conversion, RearrangeTensorConversion)
    return tensor_conversion


class DummyAttention:
    def __init__(self) -> None:
        self.rotary_emb = None

    def set_rotary_emb(self, rotary_emb: object) -> None:
        self.rotary_emb = rotary_emb


class DummyBlock:
    def __init__(self, has_attention: bool = True) -> None:
        if has_attention:
            self.attn = DummyAttention()


class DummyRotaryComponent:
    """Stand-in for bridge_model.rotary_emb (a component whose `original_component`
    is the HF rotary_emb instance)."""

    def __init__(self, original_component: object) -> None:
        self.original_component = original_component


class DummyBridgeModel:
    """Stand-in for a built TransformerBridge model: exposes rotary_emb and blocks."""

    def __init__(self, rotary_emb: object, blocks: list[DummyBlock]) -> None:
        self.rotary_emb = DummyRotaryComponent(rotary_emb)
        self.blocks = blocks


class FakeGPTOSSAttention(nn.Module):
    """Minimal GPT-OSS-style attention module for adapter hook-shape tests.

    GPT-OSS has no attention bias and no Q/K-norm, so this is a straightforward
    GQA module: Q is n_heads-wide, K/V are n_key_value_heads-wide, no biases.
    """

    def __init__(self, cfg: TransformerBridgeConfig) -> None:
        super().__init__()
        # PositionEmbeddingsAttentionBridge reads these HF-style attributes during forward.
        self.head_dim = cfg.d_head
        self.num_key_value_groups = cfg.n_heads // (cfg.n_key_value_heads or cfg.n_heads)
        self.scaling = cfg.d_head**-0.5
        self.attention_dropout = 0.0

        # GQA: Q has n_heads, while K/V have n_key_value_heads.
        kv_width = (cfg.n_key_value_heads or cfg.n_heads) * cfg.d_head
        self.q_proj = nn.Linear(cfg.d_model, cfg.n_heads * cfg.d_head, bias=False)
        self.k_proj = nn.Linear(cfg.d_model, kv_width, bias=False)
        self.v_proj = nn.Linear(cfg.d_model, kv_width, bias=False)
        self.o_proj = nn.Linear(cfg.n_heads * cfg.d_head, cfg.d_model, bias=False)


class TestGPTOSSAdapterConfig:
    """Adapter-owned config defaults that downstream bridge code relies on."""

    def test_normalization_type_is_rms(self, adapter: GPTOSSArchitectureAdapter) -> None:
        assert adapter.cfg.normalization_type == "RMS"

    def test_uses_rms_norm_is_true(self, adapter: GPTOSSArchitectureAdapter) -> None:
        assert adapter.cfg.uses_rms_norm is True

    def test_positional_embedding_type_is_rotary(self, adapter: GPTOSSArchitectureAdapter) -> None:
        assert adapter.cfg.positional_embedding_type == "rotary"

    def test_gated_mlp_is_true(self, adapter: GPTOSSArchitectureAdapter) -> None:
        assert adapter.cfg.gated_mlp is True

    def test_eps_attr_is_variance_epsilon(self, adapter: GPTOSSArchitectureAdapter) -> None:
        """GPT-OSS uses HF's `variance_epsilon` attribute name on RMSNorm modules,
        not the default `eps`. Downstream norm-folding reads this attribute."""
        assert adapter.cfg.eps_attr == "variance_epsilon"

    def test_n_kv_heads_propagated(self) -> None:
        adapter = GPTOSSArchitectureAdapter(_cfg(n_key_value_heads=2))
        assert adapter.cfg.n_key_value_heads == 2


class TestGPTOSSWeightConversions:
    """GPT-OSS uses the standard QKVO weight conversions (no biases), with GQA head counts."""

    def test_conversion_keys_are_exactly_qkvo_weights(
        self, adapter: GPTOSSArchitectureAdapter
    ) -> None:
        """No biases on any projection, so only the four QKVO weights are converted."""
        assert set(_conversions(adapter).keys()) == {
            "blocks.{i}.attn.q.weight",
            "blocks.{i}.attn.k.weight",
            "blocks.{i}.attn.v.weight",
            "blocks.{i}.attn.o.weight",
        }

    def test_q_weight_rearrange_uses_n_heads(self, adapter: GPTOSSArchitectureAdapter) -> None:
        rearrange = _rearrange(adapter, "blocks.{i}.attn.q.weight")
        assert rearrange.pattern == "(n h) m -> n m h"
        assert rearrange.axes_lengths.get("n") == 4

    def test_kv_weight_rearrange_uses_n_kv_heads(self, adapter: GPTOSSArchitectureAdapter) -> None:
        """GQA: K/V weights follow n_key_value_heads (2), not n_heads."""
        for slot in ("k", "v"):
            rearrange = _rearrange(adapter, f"blocks.{{i}}.attn.{slot}.weight")
            assert rearrange.pattern == "(n h) m -> n m h"
            assert rearrange.axes_lengths.get("n") == 2

    def test_o_weight_rearrange_uses_n_heads(self, adapter: GPTOSSArchitectureAdapter) -> None:
        rearrange = _rearrange(adapter, "blocks.{i}.attn.o.weight")
        assert rearrange.pattern == "m (n h) -> n h m"
        assert rearrange.axes_lengths.get("n") == 4

    def test_gqa_fallback_to_n_heads_without_kv_heads(self) -> None:
        """Without n_key_value_heads, K/V fall back to n_heads."""
        adapter = GPTOSSArchitectureAdapter(_cfg(n_key_value_heads=None))
        for slot in ("k", "v"):
            assert _rearrange(adapter, f"blocks.{{i}}.attn.{slot}.weight").axes_lengths["n"] == 4

    def test_gqa_does_not_affect_q_or_o(self, adapter: GPTOSSArchitectureAdapter) -> None:
        assert _rearrange(adapter, "blocks.{i}.attn.q.weight").axes_lengths["n"] == 4
        assert _rearrange(adapter, "blocks.{i}.attn.o.weight").axes_lengths["n"] == 4


class TestGPTOSSWeightConversionRoundTrips:
    """Run the rearrange conversions on synthetic HF-shaped tensors.

    The pattern/axis assertions above only check metadata. These confirm the
    conversions actually reshape realistic weight tensors into the split-head
    layout and revert losslessly (a rearrange operation is a pure permutation,
    so the round-trip must be exactly equal).
    """

    N_HEADS = 4
    N_KV_HEADS = 2
    D_HEAD = 16
    D_MODEL = 64

    @pytest.fixture
    def adapter(self) -> GPTOSSArchitectureAdapter:
        return GPTOSSArchitectureAdapter(_cfg(n_key_value_heads=self.N_KV_HEADS))

    def _roundtrip(self, adapter: GPTOSSArchitectureAdapter, key: str, tensor: Any) -> tuple:
        conv = _param_conversion(adapter, key)
        converted = conv.convert({key: tensor}, key)
        reverted = conv.revert(converted)
        return converted, reverted

    def test_q_weight_splits_into_n_heads(self, adapter: GPTOSSArchitectureAdapter) -> None:
        w = randn(self.N_HEADS * self.D_HEAD, self.D_MODEL)
        converted, reverted = self._roundtrip(adapter, "blocks.{i}.attn.q.weight", w)
        assert converted.shape == (self.N_HEADS, self.D_MODEL, self.D_HEAD)
        assert equal(reverted, w)

    def test_kv_weight_splits_into_n_kv_heads(self, adapter: GPTOSSArchitectureAdapter) -> None:
        for slot in ("k", "v"):
            w = randn(self.N_KV_HEADS * self.D_HEAD, self.D_MODEL)
            converted, reverted = self._roundtrip(adapter, f"blocks.{{i}}.attn.{slot}.weight", w)
            assert converted.shape == (self.N_KV_HEADS, self.D_MODEL, self.D_HEAD)
            assert equal(reverted, w)

    def test_o_weight_merges_heads(self, adapter: GPTOSSArchitectureAdapter) -> None:
        w = randn(self.D_MODEL, self.N_HEADS * self.D_HEAD)
        converted, reverted = self._roundtrip(adapter, "blocks.{i}.attn.o.weight", w)
        assert converted.shape == (self.N_HEADS, self.D_HEAD, self.D_MODEL)
        assert equal(reverted, w)


class TestGPTOSSComponentMapping:
    """Structure of the component mapping: required keys and submodules."""

    def test_has_required_top_level_keys(self, adapter: GPTOSSArchitectureAdapter) -> None:
        mapping = _mapping(adapter)
        for key in ("embed", "rotary_emb", "blocks", "ln_final", "unembed"):
            assert key in mapping, f"Missing top-level key: {key!r}"

    def test_blocks_has_required_submodules(self, adapter: GPTOSSArchitectureAdapter) -> None:
        blocks = _mapping(adapter)["blocks"]
        for key in ("ln1", "ln2", "attn", "mlp"):
            assert key in blocks.submodules, f"Missing blocks submodule: {key!r}"

    def test_attn_has_qkvo_submodules_only(self, adapter: GPTOSSArchitectureAdapter) -> None:
        """GPT-OSS attention has no Q/K-norm submodules, only Q, K, V, O."""
        attn = _mapping(adapter)["blocks"].submodules["attn"]
        assert set(attn.submodules.keys()) == {"q", "k", "v", "o"}

    def test_ln1_ln2_are_rms_norm_bridges(self, adapter: GPTOSSArchitectureAdapter) -> None:
        subs = _mapping(adapter)["blocks"].submodules
        assert isinstance(subs["ln1"], RMSNormalizationBridge)
        assert isinstance(subs["ln2"], RMSNormalizationBridge)

    def test_mlp_has_no_submodules(self, adapter: GPTOSSArchitectureAdapter) -> None:
        """GPT-OSS exposes no router submodule on the MoE block, the entire MoE module
        is wrapped opaquely by MoEBridge."""
        mlp = _mapping(adapter)["blocks"].submodules["mlp"]
        assert mlp.submodules == {}

    def test_hf_module_paths(self, adapter: GPTOSSArchitectureAdapter) -> None:
        mapping = _mapping(adapter)
        assert mapping["embed"].name == "model.embed_tokens"
        assert mapping["rotary_emb"].name == "model.rotary_emb"
        assert mapping["ln_final"].name == "model.norm"
        assert mapping["unembed"].name == "lm_head"
        assert mapping["blocks"].name == "model.layers"
        subs = mapping["blocks"].submodules
        assert subs["ln1"].name == "input_layernorm"
        assert subs["ln2"].name == "post_attention_layernorm"
        assert subs["attn"].name == "self_attn"
        assert subs["mlp"].name == "mlp"


class TestGPTOSSComponentTypes:
    """Top-level bridge classes, guarding against silent type substitution."""

    def test_embed_is_embedding_bridge(self, adapter: GPTOSSArchitectureAdapter) -> None:
        assert isinstance(_mapping(adapter)["embed"], EmbeddingBridge)

    def test_rotary_emb_is_rotary_bridge(self, adapter: GPTOSSArchitectureAdapter) -> None:
        assert isinstance(_mapping(adapter)["rotary_emb"], RotaryEmbeddingBridge)

    def test_blocks_is_block_bridge(self, adapter: GPTOSSArchitectureAdapter) -> None:
        assert isinstance(_mapping(adapter)["blocks"], BlockBridge)

    def test_ln_final_is_rms_norm_bridge(self, adapter: GPTOSSArchitectureAdapter) -> None:
        assert isinstance(_mapping(adapter)["ln_final"], RMSNormalizationBridge)

    def test_unembed_is_unembedding_bridge(self, adapter: GPTOSSArchitectureAdapter) -> None:
        assert isinstance(_mapping(adapter)["unembed"], UnembeddingBridge)


class TestGPTOSSBlockSubmodules:
    """BlockBridge submodule types and HF paths."""

    def test_attn_is_position_embeddings_attention(
        self, adapter: GPTOSSArchitectureAdapter
    ) -> None:
        attn = _mapping(adapter)["blocks"].submodules["attn"]
        assert isinstance(attn, PositionEmbeddingsAttentionBridge)

    def test_attn_requires_attention_mask_and_position_embeddings(
        self, adapter: GPTOSSArchitectureAdapter
    ) -> None:
        """GPT-OSS attention forward needs both an attention mask and position embeddings."""
        attn = _mapping(adapter)["blocks"].submodules["attn"]
        assert attn.requires_attention_mask is True
        assert attn.requires_position_embeddings is True

    def test_attn_qkvo_submodule_types_and_paths(self, adapter: GPTOSSArchitectureAdapter) -> None:
        attn = _mapping(adapter)["blocks"].submodules["attn"]
        for sub_name, expected_path in (
            ("q", "q_proj"),
            ("k", "k_proj"),
            ("v", "v_proj"),
            ("o", "o_proj"),
        ):
            sub = attn.submodules[sub_name]
            assert isinstance(sub, LinearBridge)
            assert sub.name == expected_path


class TestGPTOSSMoEStructure:
    """MoE structural invariants distinguishing GPT-OSS from a dense decoder."""

    def test_mlp_is_moe_not_gated_mlp(self, adapter: GPTOSSArchitectureAdapter) -> None:
        mlp = _mapping(adapter)["blocks"].submodules["mlp"]
        assert isinstance(mlp, MoEBridge)
        assert not isinstance(mlp, GatedMLPBridge)

    def test_mlp_submodules_is_empty(self, adapter: GPTOSSArchitectureAdapter) -> None:
        """The entire MoE block (router + experts) is wrapped opaquely by MoEBridge,
        no submodules are exposed for hook access."""
        mlp = _mapping(adapter)["blocks"].submodules["mlp"]
        assert mlp.submodules == {}


class TestGPTOSSGQAHookShapes:
    """Wire a fake attention module into the bridge and verify GQA hook shapes.

    Spec assertions cannot prove the bridge reshapes activations correctly.
    Here Q must surface n_heads while K/V surface n_key_value_heads,
    which is the whole point of grouped-query attention.
    """

    N_HEADS = 4
    N_KV_HEADS = 2
    D_MODEL = 64
    D_HEAD = D_MODEL // N_HEADS
    BATCH = 2
    SEQ = 8

    @pytest.fixture
    def wired_attn_bridge(self) -> PositionEmbeddingsAttentionBridge:
        adapter = GPTOSSArchitectureAdapter(_cfg(n_key_value_heads=self.N_KV_HEADS))
        fake_attn = FakeGPTOSSAttention(adapter.cfg)
        attn_bridge = _mapping(adapter)["blocks"].submodules["attn"]
        assert isinstance(attn_bridge, PositionEmbeddingsAttentionBridge)
        attn_bridge.set_original_component(fake_attn)
        # A full TransformerBridge build materializes these child bridge modules for us.
        # This unit test wires them by hand so it can stay download-free.
        for name, original in {
            "q": fake_attn.q_proj,
            "k": fake_attn.k_proj,
            "v": fake_attn.v_proj,
            "o": fake_attn.o_proj,
        }.items():
            submodule = attn_bridge.submodules[name]
            submodule.set_original_component(original)
            attn_bridge.add_module(name, submodule)
        attn_bridge.setup_hook_compatibility()
        return attn_bridge

    def _run_and_capture(self, attn_bridge: PositionEmbeddingsAttentionBridge) -> tuple:
        captured: dict = {}

        def _capture(name: str) -> Any:
            def _hook(x: Any, hook: Any) -> Any:
                captured[name] = x.detach()
                return x

            return _hook

        attn_bridge.q.hook_out.add_hook(_capture("q"))
        attn_bridge.k.hook_out.add_hook(_capture("k"))
        attn_bridge.v.hook_out.add_hook(_capture("v"))

        hidden = randn(self.BATCH, self.SEQ, self.D_MODEL)
        # Identity RoPE inputs keep this test focused on hook reshaping, not rotation math.
        cos = ones(1, self.SEQ, self.D_HEAD)
        sin = zeros(1, self.SEQ, self.D_HEAD)
        out = attn_bridge(hidden, position_embeddings=(cos, sin))
        # The attention bridge may return either a bare tensor or an (output, ...) tuple.
        out_tensor = out[0] if isinstance(out, tuple) else out

        return captured["q"], captured["k"], captured["v"], out_tensor

    def test_hook_q_uses_n_heads(
        self, wired_attn_bridge: PositionEmbeddingsAttentionBridge
    ) -> None:
        q, _, _, _ = self._run_and_capture(wired_attn_bridge)
        assert q.shape == (self.BATCH, self.SEQ, self.N_HEADS, self.D_HEAD)

    def test_hook_kv_use_n_kv_heads(
        self, wired_attn_bridge: PositionEmbeddingsAttentionBridge
    ) -> None:
        _, k, v, _ = self._run_and_capture(wired_attn_bridge)
        assert k.shape == (self.BATCH, self.SEQ, self.N_KV_HEADS, self.D_HEAD)
        assert v.shape == (self.BATCH, self.SEQ, self.N_KV_HEADS, self.D_HEAD)

    def test_attn_output_shape(self, wired_attn_bridge: PositionEmbeddingsAttentionBridge) -> None:
        _, _, _, out = self._run_and_capture(wired_attn_bridge)
        assert out.shape == (self.BATCH, self.SEQ, self.D_MODEL)


class TestGPTOSSFactoryRegistration:
    """GPT-OSS is registered in the factory and dispatched from a matching config."""

    def test_factory_lookup_returns_adapter_class(self) -> None:
        assert SUPPORTED_ARCHITECTURES["GptOssForCausalLM"] is GPTOSSArchitectureAdapter

    def test_factory_selects_correct_adapter(self) -> None:
        adapter = ArchitectureAdapterFactory.select_architecture_adapter(_cfg())
        assert isinstance(adapter, GPTOSSArchitectureAdapter)


class TestGPTOSSSetupHookCompatibility:
    """setup_hook_compatibility wires the bridge model's rotary_emb onto every
    attention bridge instance.

    GPT-OSS exposes setup_hook_compatibility(bridge_model): it reads the rotary
    module from bridge_model.rotary_emb.original_component and acts only on
    built bridge instances (no template wiring, no eager-attention forcing).
    """

    def test_no_op_on_none_bridge_model(self, adapter: GPTOSSArchitectureAdapter) -> None:
        """Must not raise when called with bridge_model=None (the guard branch)."""
        adapter.setup_hook_compatibility(None)

    def test_no_op_when_bridge_model_has_no_rotary_emb(
        self, adapter: GPTOSSArchitectureAdapter
    ) -> None:
        """Must not raise when bridge_model exposes no rotary_emb attribute."""
        bridge_model = SimpleNamespace(blocks=[DummyBlock()])
        adapter.setup_hook_compatibility(bridge_model)
        # The attn was not touched (no rotary_emb to propagate).
        assert bridge_model.blocks[0].attn.rotary_emb is None

    def test_sets_rotary_emb_on_each_bridge_block_attention(
        self, adapter: GPTOSSArchitectureAdapter
    ) -> None:
        rotary_emb = object()
        bridge_model = DummyBridgeModel(rotary_emb, [DummyBlock(), DummyBlock(), DummyBlock()])

        adapter.setup_hook_compatibility(bridge_model)

        for block in bridge_model.blocks:
            assert block.attn.rotary_emb is rotary_emb

    def test_skips_blocks_without_attention(self, adapter: GPTOSSArchitectureAdapter) -> None:
        rotary_emb = object()
        bridge_model = DummyBridgeModel(rotary_emb, [DummyBlock(), DummyBlock(has_attention=False)])

        adapter.setup_hook_compatibility(bridge_model)

        assert bridge_model.blocks[0].attn.rotary_emb is rotary_emb
        # Second block had no attn. No error and no spurious attribute added.
        assert not hasattr(bridge_model.blocks[1], "attn")

    def test_setup_no_processing_hooks_is_backward_compat_alias(
        self, adapter: GPTOSSArchitectureAdapter
    ) -> None:
        """`setup_no_processing_hooks` is documented as a backward-compatibility alias
        that must produce the same wiring as setup_hook_compatibility."""
        rotary_emb = object()
        bridge_model = DummyBridgeModel(rotary_emb, [DummyBlock(), DummyBlock()])

        adapter.setup_no_processing_hooks(bridge_model)

        for block in bridge_model.blocks:
            assert block.attn.rotary_emb is rotary_emb


class TestGPTOSSArchitectureGuards:
    """Guards against drift from GPT-OSS conventions."""

    def test_no_norm_offset_conversions(self, adapter: GPTOSSArchitectureAdapter) -> None:
        """LLaMA-style RMSNorm, with no normalization weights in the conversion map."""
        for key in _conversions(adapter):
            assert "ln1" not in key
            assert "ln2" not in key
            assert "ln_final" not in key

    def test_no_bias_conversions(self, adapter: GPTOSSArchitectureAdapter) -> None:
        """GPT-OSS has no biases on any projection."""
        for key in _conversions(adapter):
            assert not key.endswith(".bias")

    def test_attn_is_not_optional(self, adapter: GPTOSSArchitectureAdapter) -> None:
        """Every layer has self_attn (no hybrid/optional attention)."""
        attn = _mapping(adapter)["blocks"].submodules["attn"]
        assert getattr(attn, "optional", False) is False
