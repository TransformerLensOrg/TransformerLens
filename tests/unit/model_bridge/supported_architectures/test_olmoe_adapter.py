"""Unit tests for the OlmoeArchitectureAdapter (download-free, tiny programmatic configs
plus small synthetic tensors and a fake attention module, no real checkpoints).

Covered:
- Adapter config defaults (RMSNorm, rotary, gated MoE MLP, eager attention).
- Weight conversions: QKVO weights (no biases) with GQA-aware head counts.
- Numerical round-trips: the rearrange conversions actually reshape and revert losslessly.
- Component-mapping structure, bridge types, and HF module paths (incl. Q/K-norm).
- Factory registration and dispatch.
- GQA forward hook shapes (Q uses n_heads, K/V use n_key_value_heads) with Q/K-norm wired.
- setup_component_testing rotary-embedding wiring, eager forcing, and robustness.
- clip_qkv clamp parity with HF attention (clamp applies after q/k norm, matching HF order).
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
    MoERouterBridge,
    PositionEmbeddingsAttentionBridge,
    RMSNormalizationBridge,
    RotaryEmbeddingBridge,
    UnembeddingBridge,
)
from transformer_lens.model_bridge.supported_architectures.olmoe import (
    OlmoeArchitectureAdapter,
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
        architecture="OlmoeForCausalLM",
    )


@pytest.fixture(scope="class")
def adapter(cfg: TransformerBridgeConfig) -> OlmoeArchitectureAdapter:
    return OlmoeArchitectureAdapter(cfg)


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
        architecture="OlmoeForCausalLM",
    )


def _mapping(adapter: OlmoeArchitectureAdapter) -> dict:
    """Narrow component_mapping (Optional on the base class) to a non-None dict."""
    mapping = adapter.component_mapping
    assert mapping is not None
    return mapping


def _conversions(adapter: OlmoeArchitectureAdapter) -> dict:
    """weight_processing_conversions is Optional on the base class; assert it is populated."""
    conversions = adapter.weight_processing_conversions
    assert conversions is not None
    return conversions


def _param_conversion(adapter: OlmoeArchitectureAdapter, key: str) -> ParamProcessingConversion:
    conv = _conversions(adapter)[key]
    assert isinstance(conv, ParamProcessingConversion)
    return conv


def _rearrange(adapter: OlmoeArchitectureAdapter, key: str) -> RearrangeTensorConversion:
    tensor_conversion = _param_conversion(adapter, key).tensor_conversion
    assert isinstance(tensor_conversion, RearrangeTensorConversion)
    return tensor_conversion


def _fake_hf_model(rotary_emb: object) -> SimpleNamespace:
    """Minimal HF model exposing only model.rotary_emb (no config/layers)."""
    return SimpleNamespace(model=SimpleNamespace(rotary_emb=rotary_emb))


def _fake_hf_model_with_eager_targets(rotary_emb: object) -> SimpleNamespace:
    """HF model whose top-level and per-layer attention implementation start non-eager."""
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


class FakeOlmoeAttention(nn.Module):
    """Minimal OLMoE-style attention module for adapter hook-shape tests.

    OLMoE has no attention bias, and applies RMSNorm to the flattened Q and K
    projections (OLMo-2 "pre_reshape" phase): q_norm over n_heads * head_dim and
    k_norm over n_key_value_heads * head_dim. The bridge's set_original_component
    inspects q_norm.weight to pick that phase, so the norms must be present and
    correctly shaped here.
    """

    def __init__(self, cfg: TransformerBridgeConfig) -> None:
        super().__init__()
        # PositionEmbeddingsAttentionBridge reads these HF-style attributes during forward.
        self.head_dim = cfg.d_head
        self.num_key_value_groups = cfg.n_heads // (cfg.n_key_value_heads or cfg.n_heads)
        self.scaling = cfg.d_head**-0.5
        self.attention_dropout = 0.0

        # GQA: Q has n_heads, while K/V have n_key_value_heads.
        n_kv = cfg.n_key_value_heads or cfg.n_heads
        kv_width = n_kv * cfg.d_head
        self.q_proj = nn.Linear(cfg.d_model, cfg.n_heads * cfg.d_head, bias=False)
        self.k_proj = nn.Linear(cfg.d_model, kv_width, bias=False)
        self.v_proj = nn.Linear(cfg.d_model, kv_width, bias=False)
        self.o_proj = nn.Linear(cfg.n_heads * cfg.d_head, cfg.d_model, bias=False)
        # Pre-reshape RMSNorm over the flattened head dimension.
        self.q_norm = nn.RMSNorm(cfg.n_heads * cfg.d_head)
        self.k_norm = nn.RMSNorm(kv_width)


class TestOlmoeAdapterConfig:
    """Adapter-owned config defaults that downstream bridge code relies on."""

    def test_final_rms_is_false(self, adapter: OlmoeArchitectureAdapter) -> None:
        """OLMoE does not apply a final RMS fold."""
        assert adapter.cfg.final_rms is False


class TestOlmoeWeightConversions:
    """OLMoE uses the standard QKVO weight conversions (no biases), with GQA head counts."""

    def test_conversion_keys_are_exactly_qkvo_weights(
        self, adapter: OlmoeArchitectureAdapter
    ) -> None:
        """No biases on any projection, so only the four QKVO weights are converted."""
        assert set(_conversions(adapter).keys()) == {
            "blocks.{i}.attn.q.weight",
            "blocks.{i}.attn.k.weight",
            "blocks.{i}.attn.v.weight",
            "blocks.{i}.attn.o.weight",
        }

    def test_q_weight_rearrange_uses_n_heads(self, adapter: OlmoeArchitectureAdapter) -> None:
        rearrange = _rearrange(adapter, "blocks.{i}.attn.q.weight")
        assert rearrange.pattern == "(n h) m -> n m h"
        assert rearrange.axes_lengths.get("n") == 4

    def test_kv_weight_rearrange_uses_n_kv_heads(self, adapter: OlmoeArchitectureAdapter) -> None:
        """GQA: K/V weights follow n_key_value_heads (2), not n_heads."""
        for slot in ("k", "v"):
            rearrange = _rearrange(adapter, f"blocks.{{i}}.attn.{slot}.weight")
            assert rearrange.pattern == "(n h) m -> n m h"
            assert rearrange.axes_lengths.get("n") == 2

    def test_o_weight_rearrange_uses_n_heads(self, adapter: OlmoeArchitectureAdapter) -> None:
        rearrange = _rearrange(adapter, "blocks.{i}.attn.o.weight")
        assert rearrange.pattern == "m (n h) -> n h m"
        assert rearrange.axes_lengths.get("n") == 4

    def test_gqa_fallback_to_n_heads_without_kv_heads(self) -> None:
        """Without n_key_value_heads, K/V fall back to n_heads."""
        adapter = OlmoeArchitectureAdapter(_cfg(n_key_value_heads=None))
        for slot in ("k", "v"):
            assert _rearrange(adapter, f"blocks.{{i}}.attn.{slot}.weight").axes_lengths["n"] == 4

    def test_gqa_does_not_affect_q_or_o(self, adapter: OlmoeArchitectureAdapter) -> None:
        assert _rearrange(adapter, "blocks.{i}.attn.q.weight").axes_lengths["n"] == 4
        assert _rearrange(adapter, "blocks.{i}.attn.o.weight").axes_lengths["n"] == 4


class TestOlmoeComponentMapping:
    """Structure of the component mapping: required keys and submodules."""

    def test_has_required_top_level_keys(self, adapter: OlmoeArchitectureAdapter) -> None:
        mapping = _mapping(adapter)
        for key in ("embed", "rotary_emb", "blocks", "ln_final", "unembed"):
            assert key in mapping, f"Missing top-level key: {key!r}"

    def test_blocks_has_required_submodules(self, adapter: OlmoeArchitectureAdapter) -> None:
        blocks = _mapping(adapter)["blocks"]
        for key in ("ln1", "ln2", "attn", "mlp"):
            assert key in blocks.submodules, f"Missing blocks submodule: {key!r}"

    def test_attn_has_qkvo_and_qk_norm_submodules(self, adapter: OlmoeArchitectureAdapter) -> None:
        """OLMoE adds Q/K normalization to QKVO, applied over the flattened n_heads * head_dim."""
        attn = _mapping(adapter)["blocks"].submodules["attn"]
        assert set(attn.submodules.keys()) == {"q", "k", "v", "o", "q_norm", "k_norm"}

    def test_ln1_ln2_are_rms_norm_bridges(self, adapter: OlmoeArchitectureAdapter) -> None:
        subs = _mapping(adapter)["blocks"].submodules
        assert isinstance(subs["ln1"], RMSNormalizationBridge)
        assert isinstance(subs["ln2"], RMSNormalizationBridge)

    def test_qk_norm_are_rms_norm_bridges(self, adapter: OlmoeArchitectureAdapter) -> None:
        attn_subs = _mapping(adapter)["blocks"].submodules["attn"].submodules
        assert isinstance(attn_subs["q_norm"], RMSNormalizationBridge)
        assert isinstance(attn_subs["k_norm"], RMSNormalizationBridge)

    def test_mlp_has_only_gate_submodule(self, adapter: OlmoeArchitectureAdapter) -> None:
        """Experts are batched tensors inside the MoE block; only the router is mapped."""
        mlp = _mapping(adapter)["blocks"].submodules["mlp"]
        assert set(mlp.submodules.keys()) == {"gate"}

    def test_hf_module_paths(self, adapter: OlmoeArchitectureAdapter) -> None:
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
        assert subs["mlp"].submodules["gate"].name == "gate"
        attn_subs = subs["attn"].submodules
        assert attn_subs["q_norm"].name == "q_norm"
        assert attn_subs["k_norm"].name == "k_norm"


class TestOlmoeComponentTypes:
    """Top-level bridge classes, guarding against silent type substitution."""

    def test_embed_is_embedding_bridge(self, adapter: OlmoeArchitectureAdapter) -> None:
        assert isinstance(_mapping(adapter)["embed"], EmbeddingBridge)

    def test_rotary_emb_is_rotary_bridge(self, adapter: OlmoeArchitectureAdapter) -> None:
        assert isinstance(_mapping(adapter)["rotary_emb"], RotaryEmbeddingBridge)

    def test_blocks_is_block_bridge(self, adapter: OlmoeArchitectureAdapter) -> None:
        assert isinstance(_mapping(adapter)["blocks"], BlockBridge)

    def test_ln_final_is_rms_norm_bridge(self, adapter: OlmoeArchitectureAdapter) -> None:
        assert isinstance(_mapping(adapter)["ln_final"], RMSNormalizationBridge)

    def test_unembed_is_unembedding_bridge(self, adapter: OlmoeArchitectureAdapter) -> None:
        assert isinstance(_mapping(adapter)["unembed"], UnembeddingBridge)


class TestOlmoeBlockSubmodules:
    """BlockBridge submodule types and HF paths."""

    def test_attn_is_position_embeddings_attention(self, adapter: OlmoeArchitectureAdapter) -> None:
        attn = _mapping(adapter)["blocks"].submodules["attn"]
        assert isinstance(attn, PositionEmbeddingsAttentionBridge)

    def test_attn_requires_attention_mask_and_position_embeddings(
        self, adapter: OlmoeArchitectureAdapter
    ) -> None:
        """OLMoE attention forward needs both an attention mask and position embeddings."""
        attn = _mapping(adapter)["blocks"].submodules["attn"]
        assert attn.requires_attention_mask is True
        assert attn.requires_position_embeddings is True

    def test_attn_qkvo_submodule_types_and_paths(self, adapter: OlmoeArchitectureAdapter) -> None:
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

    def test_mlp_gate_submodule_type(self, adapter: OlmoeArchitectureAdapter) -> None:
        """Router is a LinearBridge so the routing logits can be hooked."""
        mlp = _mapping(adapter)["blocks"].submodules["mlp"]
        assert isinstance(mlp.submodules["gate"], MoERouterBridge)


class TestOlmoeMoEStructure:
    """MoE structural invariants distinguishing OLMoE from a dense decoder."""

    def test_mlp_is_moe_not_gated_mlp(self, adapter: OlmoeArchitectureAdapter) -> None:
        mlp = _mapping(adapter)["blocks"].submodules["mlp"]
        assert isinstance(mlp, MoEBridge)
        assert not isinstance(mlp, GatedMLPBridge)


class TestOlmoeGQAHookShapes:
    """Wire a fake attention module into the bridge and verify GQA hook shapes.

    Spec assertions cannot prove the bridge reshapes activations correctly.
    Here Q must surface n_heads while K/V surface n_key_value_heads, which is the
    whole point of grouped-query attention. The fake carries OLMoE's pre-reshape
    Q/K norms so the bridge takes its Q/K-norm code path.
    """

    N_HEADS = 4
    N_KV_HEADS = 2
    D_MODEL = 64
    D_HEAD = D_MODEL // N_HEADS
    BATCH = 2
    SEQ = 8

    @pytest.fixture
    def wired_attn_bridge(self) -> PositionEmbeddingsAttentionBridge:
        adapter = OlmoeArchitectureAdapter(_cfg(n_key_value_heads=self.N_KV_HEADS))
        fake_attn = FakeOlmoeAttention(adapter.cfg)
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
            "q_norm": fake_attn.q_norm,
            "k_norm": fake_attn.k_norm,
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


class TestOlmoeSetupComponentTesting:
    """setup_component_testing wires the shared rotary embedding and forces eager attention."""

    def test_sets_rotary_emb_on_template_attention(self, adapter: OlmoeArchitectureAdapter) -> None:
        rotary_emb = object()
        attn_template = adapter.get_generalized_component("blocks.0.attn")
        assert isinstance(attn_template, PositionEmbeddingsAttentionBridge)
        assert attn_template._rotary_emb is None

        adapter.setup_component_testing(_fake_hf_model(rotary_emb))

        assert attn_template._rotary_emb is rotary_emb

    def test_sets_rotary_emb_on_each_bridge_model_attention(
        self, adapter: OlmoeArchitectureAdapter
    ) -> None:
        rotary_emb = object()
        bridge_model = DummyBridgeModel([DummyBlock(), DummyBlock(), DummyBlock()])

        adapter.setup_component_testing(_fake_hf_model(rotary_emb), bridge_model=bridge_model)

        for block in bridge_model.blocks:
            assert block.attn.rotary_emb is rotary_emb

    def test_skips_bridge_blocks_without_attention(self, adapter: OlmoeArchitectureAdapter) -> None:
        rotary_emb = object()
        bridge_model = DummyBridgeModel([DummyBlock(), DummyBlock(has_attention=False)])

        adapter.setup_component_testing(_fake_hf_model(rotary_emb), bridge_model=bridge_model)

        assert bridge_model.blocks[0].attn.rotary_emb is rotary_emb

    def test_forces_eager_attention_implementation(self, adapter: OlmoeArchitectureAdapter) -> None:
        """Bridge attention only matches HF under eager attention, so it is forced on."""
        hf_model = _fake_hf_model_with_eager_targets(object())

        adapter.setup_component_testing(hf_model)

        assert hf_model.config._attn_implementation == "eager"
        for layer in hf_model.model.layers:
            assert layer.self_attn.config._attn_implementation == "eager"

    def test_tolerates_minimal_hf_model_without_config_or_layers(
        self, adapter: OlmoeArchitectureAdapter
    ) -> None:
        """The defensive hasattr branches must not raise when config/layers are absent."""
        rotary_emb = object()
        # _fake_hf_model exposes only model.rotary_emb (no config, no layers).
        adapter.setup_component_testing(_fake_hf_model(rotary_emb))

        attn_template = adapter.get_generalized_component("blocks.0.attn")
        assert isinstance(attn_template, PositionEmbeddingsAttentionBridge)
        assert attn_template._rotary_emb is rotary_emb


class TestOlmoeClipQkv:
    """clip_qkv clamps after the q/k norms (HF order: norm first, then clamp)."""

    def test_bridge_matches_hf_with_active_clamp(
        self, adapter: OlmoeArchitectureAdapter, cfg: TransformerBridgeConfig
    ) -> None:
        import copy

        import torch
        from transformers import OlmoeConfig
        from transformers.models.olmoe.modeling_olmoe import OlmoeAttention

        from transformer_lens.model_bridge.component_setup import setup_submodules

        # fork_rng: deterministic setup without mutating the global RNG stream.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(0)
            # RMSNorm output is unit-RMS scale, so clip_qkv=0.5 clamps aggressively —
            # a clamp placed before the norms would be rescaled and fail parity.
            hf_config = OlmoeConfig(
                hidden_size=cfg.d_model,
                num_attention_heads=cfg.n_heads,
                num_key_value_heads=cfg.n_key_value_heads,
                clip_qkv=0.5,
            )
            hf_config._attn_implementation = "eager"
            hf_attn = OlmoeAttention(hf_config, layer_idx=0)
            batch, seq = 2, 5
            hidden_states = randn(batch, seq, cfg.d_model)

        attn_bridge = copy.deepcopy(adapter.get_generalized_component("blocks.0.attn"))
        assert isinstance(attn_bridge, PositionEmbeddingsAttentionBridge)
        attn_bridge.set_original_component(hf_attn)
        setup_submodules(attn_bridge, adapter, hf_attn)
        attn_bridge.setup_hook_compatibility()

        head_dim = cfg.d_model // cfg.n_heads
        position_embeddings = (ones(1, seq, head_dim), zeros(1, seq, head_dim))
        # 4D additive causal mask — authoritative for both HF eager and the bridge.
        causal = torch.tril(torch.ones(seq, seq, dtype=torch.bool))
        attention_mask = zeros(batch, 1, seq, seq).masked_fill(
            ~causal, torch.finfo(torch.float32).min
        )

        with torch.no_grad():
            hf_out = hf_attn(
                hidden_states,
                position_embeddings=position_embeddings,
                attention_mask=attention_mask,
            )[0]
            bridge_out = attn_bridge(
                hidden_states=hidden_states,
                position_embeddings=position_embeddings,
                attention_mask=attention_mask,
            )[0]

        torch.testing.assert_close(bridge_out, hf_out, atol=1e-5, rtol=1e-4)

        # Disabling the clamp must change the bridge output, or parity is vacuous.
        hf_config.clip_qkv = None
        with torch.no_grad():
            unclamped_out = attn_bridge(
                hidden_states=hidden_states,
                position_embeddings=position_embeddings,
                attention_mask=attention_mask,
            )[0]
        assert not torch.allclose(unclamped_out, hf_out, atol=1e-5)


class TestOlmoeArchitectureGuards:
    """Guards against drift from OLMoE conventions."""

    def test_no_norm_offset_conversions(self, adapter: OlmoeArchitectureAdapter) -> None:
        """LLaMA-style RMSNorm, with no normalization weights in the conversion map."""
        for key in _conversions(adapter):
            assert "ln1" not in key
            assert "ln2" not in key
            assert "ln_final" not in key

    def test_no_bias_conversions(self, adapter: OlmoeArchitectureAdapter) -> None:
        """OLMoE has no biases on any projection."""
        for key in _conversions(adapter):
            assert not key.endswith(".bias")

    def test_attn_is_not_optional(self, adapter: OlmoeArchitectureAdapter) -> None:
        """Every layer has self_attn (no hybrid/optional attention)."""
        attn = _mapping(adapter)["blocks"].submodules["attn"]
        assert getattr(attn, "optional", False) is False
