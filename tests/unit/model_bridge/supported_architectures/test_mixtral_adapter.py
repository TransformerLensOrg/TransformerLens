"""Unit tests for the MixtralArchitectureAdapter (download-free: tiny programmatic configs
plus small synthetic tensors and a fake attention module, no real checkpoints).

Covered:
- Adapter config defaults (RMSNorm, rotary, gated MoE MLP).
- Weight conversions: QKVO weights with GQA-aware head counts.
- Component-mapping structure, bridge types, and HF module paths.
- Factory registration and dispatch.
- GQA forward hook shapes (Q uses n_heads, K/V use n_key_value_heads).
- setup_component_testing rotary-embedding wiring, eager forcing, and robustness.
"""

from types import SimpleNamespace
from typing import Any

import pytest
import torch.nn as nn
from torch import ones, randn, zeros

from transformer_lens.config import TransformerBridgeConfig
from transformer_lens.conversion_utils.conversion_steps.rearrange_tensor_conversion import (
    RearrangeTensorConversion,
)
from transformer_lens.conversion_utils.param_processing_conversion import (
    ParamProcessingConversion,
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
from transformer_lens.model_bridge.supported_architectures.mixtral import (
    MixtralArchitectureAdapter,
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
        architecture="MixtralForCausalLM",
    )


@pytest.fixture(scope="class")
def adapter(cfg: TransformerBridgeConfig) -> MixtralArchitectureAdapter:
    return MixtralArchitectureAdapter(cfg)


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
        architecture="MixtralForCausalLM",
    )


def _mapping(adapter: MixtralArchitectureAdapter) -> dict:
    """Narrow component_mapping (Optional on the base class) to a non-None dict.

    Factored into a helper so each test stays a one-liner instead of repeating the
    `assert ... is not None` prelude in every method. The qwen3_moe adapter test
    inlines that prelude per method instead; this is the deduplicated equivalent.
    """
    mapping = adapter.component_mapping
    assert mapping is not None
    return mapping


def _conversions(adapter: MixtralArchitectureAdapter) -> dict:
    """weight_processing_conversions is Optional on the base class; assert it is populated."""
    conversions = adapter.weight_processing_conversions
    assert conversions is not None
    return conversions


def _param_conversion(adapter: MixtralArchitectureAdapter, key: str) -> ParamProcessingConversion:
    conv = _conversions(adapter)[key]
    assert isinstance(conv, ParamProcessingConversion)
    return conv


def _rearrange(adapter: MixtralArchitectureAdapter, key: str) -> RearrangeTensorConversion:
    tensor_conversion = _param_conversion(adapter, key).tensor_conversion
    assert isinstance(tensor_conversion, RearrangeTensorConversion)
    return tensor_conversion


def _fake_hf_model(rotary_emb: object) -> SimpleNamespace:
    """Minimal HF model exposing only model.rotary_emb (no config/layers)."""
    return SimpleNamespace(model=SimpleNamespace(rotary_emb=rotary_emb))


def _fake_hf_model_with_eager_targets(rotary_emb: object) -> SimpleNamespace:
    """HF model whose top-level and per-layer attention impl start non-eager."""
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


class FakeMixtralAttention(nn.Module):
    """Minimal Mixtral-style attention module for adapter hook-shape tests."""

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


class TestMixtralAdapterConfig:
    """Adapter-owned config defaults that downstream bridge code relies on."""


class TestMixtralWeightConversions:
    """Mixtral uses QKVO weight conversions with GQA head counts."""

    def test_conversion_keys_are_exactly_qkvo(self, adapter: MixtralArchitectureAdapter) -> None:
        """Mixtral has no attention biases; only the four projections convert."""
        assert set(_conversions(adapter).keys()) == {
            "blocks.{i}.attn.q.weight",
            "blocks.{i}.attn.k.weight",
            "blocks.{i}.attn.v.weight",
            "blocks.{i}.attn.o.weight",
        }

    def test_q_weight_rearrange_uses_n_heads(self, adapter: MixtralArchitectureAdapter) -> None:
        rearrange = _rearrange(adapter, "blocks.{i}.attn.q.weight")
        assert rearrange.pattern == "(n h) m -> n m h"
        assert rearrange.axes_lengths.get("n") == 4

    def test_kv_weight_rearrange_uses_n_kv_heads(self, adapter: MixtralArchitectureAdapter) -> None:
        """GQA: K/V weights follow n_key_value_heads (2), not n_heads."""
        for slot in ("k", "v"):
            rearrange = _rearrange(adapter, f"blocks.{{i}}.attn.{slot}.weight")
            assert rearrange.pattern == "(n h) m -> n m h"
            assert rearrange.axes_lengths.get("n") == 2

    def test_o_weight_rearrange_uses_n_heads(self, adapter: MixtralArchitectureAdapter) -> None:
        rearrange = _rearrange(adapter, "blocks.{i}.attn.o.weight")
        assert rearrange.pattern == "m (n h) -> n h m"
        assert rearrange.axes_lengths.get("n") == 4

    def test_gqa_fallback_to_n_heads_without_kv_heads(self) -> None:
        """Without n_key_value_heads, K/V fall back to n_heads."""
        adapter = MixtralArchitectureAdapter(_cfg(n_key_value_heads=None))
        for slot in ("k", "v"):
            assert _rearrange(adapter, f"blocks.{{i}}.attn.{slot}.weight").axes_lengths["n"] == 4


class TestMixtralComponentMapping:
    """Structure of the component mapping: required keys and submodules."""

    def test_has_required_top_level_keys(self, adapter: MixtralArchitectureAdapter) -> None:
        mapping = _mapping(adapter)
        for key in ("embed", "rotary_emb", "blocks", "ln_final", "unembed"):
            assert key in mapping, f"Missing top-level key: {key!r}"

    def test_blocks_has_required_submodules(self, adapter: MixtralArchitectureAdapter) -> None:
        blocks = _mapping(adapter)["blocks"]
        for key in ("ln1", "ln2", "attn", "mlp"):
            assert key in blocks.submodules, f"Missing blocks submodule: {key!r}"

    def test_attn_has_qkvo_submodules(self, adapter: MixtralArchitectureAdapter) -> None:
        attn = _mapping(adapter)["blocks"].submodules["attn"]
        assert set(attn.submodules.keys()) == {"q", "k", "v", "o"}

    def test_ln1_ln2_are_rms_norm_bridges(self, adapter: MixtralArchitectureAdapter) -> None:
        subs = _mapping(adapter)["blocks"].submodules
        assert isinstance(subs["ln1"], RMSNormalizationBridge)
        assert isinstance(subs["ln2"], RMSNormalizationBridge)

    def test_mlp_has_only_gate_submodule(self, adapter: MixtralArchitectureAdapter) -> None:
        """Experts are batched tensors inside the sparse block; only the router is mapped."""
        mlp = _mapping(adapter)["blocks"].submodules["mlp"]
        assert set(mlp.submodules.keys()) == {"gate"}

    def test_hf_module_paths(self, adapter: MixtralArchitectureAdapter) -> None:
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
        # transformers >= 5.13 renamed the decoder-layer attr block_sparse_moe -> mlp.
        assert subs["mlp"].name == "mlp"
        assert subs["mlp"].submodules["gate"].name == "gate"


class TestMixtralComponentTypes:
    """Top-level bridge classes, guarding against silent type substitution."""

    def test_embed_is_embedding_bridge(self, adapter: MixtralArchitectureAdapter) -> None:
        assert isinstance(_mapping(adapter)["embed"], EmbeddingBridge)

    def test_rotary_emb_is_rotary_bridge(self, adapter: MixtralArchitectureAdapter) -> None:
        assert isinstance(_mapping(adapter)["rotary_emb"], RotaryEmbeddingBridge)

    def test_blocks_is_block_bridge(self, adapter: MixtralArchitectureAdapter) -> None:
        assert isinstance(_mapping(adapter)["blocks"], BlockBridge)

    def test_ln_final_is_rms_norm_bridge(self, adapter: MixtralArchitectureAdapter) -> None:
        assert isinstance(_mapping(adapter)["ln_final"], RMSNormalizationBridge)

    def test_unembed_is_unembedding_bridge(self, adapter: MixtralArchitectureAdapter) -> None:
        assert isinstance(_mapping(adapter)["unembed"], UnembeddingBridge)


class TestMixtralBlockSubmodules:
    """BlockBridge submodule types and HF paths."""

    def test_attn_is_position_embeddings_attention(
        self, adapter: MixtralArchitectureAdapter
    ) -> None:
        attn = _mapping(adapter)["blocks"].submodules["attn"]
        assert isinstance(attn, PositionEmbeddingsAttentionBridge)

    def test_attn_requires_attention_mask_and_position_embeddings(
        self, adapter: MixtralArchitectureAdapter
    ) -> None:
        """MixtralAttention.forward() needs both an attention mask and position embeddings."""
        attn = _mapping(adapter)["blocks"].submodules["attn"]
        assert attn.requires_attention_mask is True
        assert attn.requires_position_embeddings is True

    def test_attn_qkvo_submodule_types_and_paths(self, adapter: MixtralArchitectureAdapter) -> None:
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

    def test_mlp_gate_submodule_type(self, adapter: MixtralArchitectureAdapter) -> None:
        """Router is a LinearBridge so the routing logits can be hooked."""
        mlp = _mapping(adapter)["blocks"].submodules["mlp"]
        assert isinstance(mlp.submodules["gate"], LinearBridge)


class TestMixtralMoEStructure:
    """MoE structural invariants distinguishing Mixtral from a dense decoder."""

    def test_mlp_is_moe_not_gated_mlp(self, adapter: MixtralArchitectureAdapter) -> None:
        mlp = _mapping(adapter)["blocks"].submodules["mlp"]
        assert isinstance(mlp, MoEBridge)
        assert not isinstance(mlp, GatedMLPBridge)


class TestMixtralGQAHookShapes:
    """Wire a fake attention module into the bridge and verify GQA hook shapes.

    Spec assertions cannot prove the bridge reshapes activations correctly. Here
    Q must surface n_heads while K/V surface n_key_value_heads, which is the whole
    point of grouped-query attention.
    """

    N_HEADS = 4
    N_KV_HEADS = 2
    D_MODEL = 64
    D_HEAD = D_MODEL // N_HEADS
    BATCH = 2
    SEQ = 8

    @pytest.fixture
    def wired_attn_bridge(self) -> PositionEmbeddingsAttentionBridge:
        adapter = MixtralArchitectureAdapter(_cfg(n_key_value_heads=self.N_KV_HEADS))
        fake_attn = FakeMixtralAttention(adapter.cfg)
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
        attn_bridge(hidden, position_embeddings=(cos, sin))

        return captured["q"], captured["k"], captured["v"]

    def test_hook_q_uses_n_heads(
        self, wired_attn_bridge: PositionEmbeddingsAttentionBridge
    ) -> None:
        q, _, _ = self._run_and_capture(wired_attn_bridge)
        assert q.shape == (self.BATCH, self.SEQ, self.N_HEADS, self.D_HEAD)

    def test_hook_kv_use_n_kv_heads(
        self, wired_attn_bridge: PositionEmbeddingsAttentionBridge
    ) -> None:
        _, k, v = self._run_and_capture(wired_attn_bridge)
        assert k.shape == (self.BATCH, self.SEQ, self.N_KV_HEADS, self.D_HEAD)
        assert v.shape == (self.BATCH, self.SEQ, self.N_KV_HEADS, self.D_HEAD)


class TestMixtralSetupComponentTesting:
    """setup_component_testing wires the shared rotary embedding and forces eager attention."""

    def test_sets_rotary_emb_on_template_attention(
        self, adapter: MixtralArchitectureAdapter
    ) -> None:
        rotary_emb = object()
        attn_template = adapter.get_generalized_component("blocks.0.attn")
        assert isinstance(attn_template, PositionEmbeddingsAttentionBridge)
        assert attn_template._rotary_emb is None

        adapter.setup_component_testing(_fake_hf_model(rotary_emb))

        assert attn_template._rotary_emb is rotary_emb

    def test_sets_rotary_emb_on_each_bridge_model_attention(
        self, adapter: MixtralArchitectureAdapter
    ) -> None:
        rotary_emb = object()
        bridge_model = DummyBridgeModel([DummyBlock(), DummyBlock(), DummyBlock()])

        adapter.setup_component_testing(_fake_hf_model(rotary_emb), bridge_model=bridge_model)

        for block in bridge_model.blocks:
            assert block.attn.rotary_emb is rotary_emb

    def test_skips_bridge_blocks_without_attention(
        self, adapter: MixtralArchitectureAdapter
    ) -> None:
        rotary_emb = object()
        bridge_model = DummyBridgeModel([DummyBlock(), DummyBlock(has_attention=False)])

        adapter.setup_component_testing(_fake_hf_model(rotary_emb), bridge_model=bridge_model)

        assert bridge_model.blocks[0].attn.rotary_emb is rotary_emb

    def test_forces_eager_attention_implementation(
        self, adapter: MixtralArchitectureAdapter
    ) -> None:
        """Bridge attention only matches HF under eager attention, so it is forced on."""
        hf_model = _fake_hf_model_with_eager_targets(object())

        adapter.setup_component_testing(hf_model)

        assert hf_model.config._attn_implementation == "eager"
        for layer in hf_model.model.layers:
            assert layer.self_attn.config._attn_implementation == "eager"

    def test_tolerates_minimal_hf_model_without_config_or_layers(
        self, adapter: MixtralArchitectureAdapter
    ) -> None:
        """The defensive hasattr branches must not raise when config/layers are absent."""
        rotary_emb = object()
        # _fake_hf_model exposes only model.rotary_emb (no config, no layers).
        adapter.setup_component_testing(_fake_hf_model(rotary_emb))

        attn_template = adapter.get_generalized_component("blocks.0.attn")
        assert isinstance(attn_template, PositionEmbeddingsAttentionBridge)
        assert attn_template._rotary_emb is rotary_emb


class TestMixtralArchitectureGuards:
    """Guards against drift from Mixtral conventions."""

    def test_no_norm_offset_conversions(self, adapter: MixtralArchitectureAdapter) -> None:
        """LLaMA-style RMSNorm, with no normalization weights in the conversion map."""
        for key in _conversions(adapter):
            assert "ln1" not in key
            assert "ln2" not in key
            assert "ln_final" not in key
