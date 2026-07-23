"""Unit tests for the Olmo2ArchitectureAdapter (download-free, tiny programmatic configs
plus small synthetic tensors and a fake attention module, no real checkpoints).

Covered:
- Anti-drift config: supports_fold_ln=False (post-norm cannot fold).
- Weight conversions: the adapter uses the base QKVO helper, so only the exact
  key set is asserted at the adapter layer.
- Component-mapping structure, bridge types, and HF module paths.
- Post-norm block wiring: ln1 maps to post_attention_layernorm and ln2 maps to
  post_feedforward_layernorm. This is the central arch-specific decision.
- Q/K-norm submodules under attention with the correct HF names.
- hook_alias_overrides: hook_resid_mid points at mlp.hook_in (the true post-attn
  pre-mlp residual under post-norm), overriding BlockBridge's default ln2.hook_in.
- GQA forward hook shapes for Q and K/V with Q/K-norm wired into the fake attention.
- setup_component_testing: rotary wiring on template and bridge-model attentions,
  plus forcing eager attention on the HF model and per-layer self_attn configs.
- Architecture guards.
"""

from types import SimpleNamespace
from typing import Any

import pytest
import torch.nn as nn
from torch import ones, randn, zeros

from transformer_lens.config import TransformerBridgeConfig
from transformer_lens.model_bridge.generalized_components import (
    BlockBridge,
    EmbeddingBridge,
    GatedMLPBridge,
    LinearBridge,
    PositionEmbeddingsAttentionBridge,
    RMSNormalizationBridge,
    RotaryEmbeddingBridge,
    UnembeddingBridge,
)
from transformer_lens.model_bridge.supported_architectures.olmo2 import (
    Olmo2ArchitectureAdapter,
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
        architecture="Olmo2ForCausalLM",
    )


@pytest.fixture(scope="class")
def adapter(cfg: TransformerBridgeConfig) -> Olmo2ArchitectureAdapter:
    return Olmo2ArchitectureAdapter(cfg)


def _cfg(*, n_key_value_heads: int | None = 2) -> TransformerBridgeConfig:
    # Keep dimensions tiny so adapter tests do not need HF downloads or real checkpoints.
    return TransformerBridgeConfig(
        d_model=64,
        d_head=16,
        n_layers=2,
        n_ctx=128,
        n_heads=4,
        n_key_value_heads=n_key_value_heads,
        d_vocab=256,
        architecture="Olmo2ForCausalLM",
    )


def _mapping(adapter: Olmo2ArchitectureAdapter) -> dict:
    """Narrow component_mapping (Optional on the base class) to a non-None dict."""
    mapping = adapter.component_mapping
    assert mapping is not None
    return mapping


def _conversions(adapter: Olmo2ArchitectureAdapter) -> dict:
    """weight_processing_conversions is Optional on the base class, assert it is populated."""
    conversions = adapter.weight_processing_conversions
    assert conversions is not None
    return conversions


def _fake_hf_model(rotary_emb: object) -> SimpleNamespace:
    """Minimal HF model exposing only model.rotary_emb (no config, no layers)."""
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


class FakeOlmo2Attention(nn.Module):
    """Minimal OLMo-2-style attention module for adapter hook-shape tests.

    OLMo 2 has no attention bias and applies RMSNorm to the flattened Q and K
    projections (pre-reshape phase): q_norm over n_heads * head_dim and k_norm
    over n_key_value_heads * head_dim. Matches HF's Olmo2Attention shape.
    """

    def __init__(self, cfg: TransformerBridgeConfig) -> None:
        super().__init__()
        # PositionEmbeddingsAttentionBridge reads these HF-style attributes during forward.
        self.head_dim = cfg.d_head
        self.num_key_value_groups = cfg.n_heads // (cfg.n_key_value_heads or cfg.n_heads)
        self.scaling = cfg.d_head**-0.5
        self.attention_dropout = 0.0

        n_kv = cfg.n_key_value_heads or cfg.n_heads
        kv_width = n_kv * cfg.d_head
        self.q_proj = nn.Linear(cfg.d_model, cfg.n_heads * cfg.d_head, bias=False)
        self.k_proj = nn.Linear(cfg.d_model, kv_width, bias=False)
        self.v_proj = nn.Linear(cfg.d_model, kv_width, bias=False)
        self.o_proj = nn.Linear(cfg.n_heads * cfg.d_head, cfg.d_model, bias=False)
        # Pre-reshape RMSNorm over the flattened head dimension.
        self.q_norm = nn.RMSNorm(cfg.n_heads * cfg.d_head)
        self.k_norm = nn.RMSNorm(kv_width)


class TestOlmo2AdapterConfig:
    """Anti-drift config: post-norm forces supports_fold_ln=False because folding
    a norm that runs AFTER attention/MLP would corrupt the weights."""

    def test_supports_fold_ln_is_false(self, adapter: Olmo2ArchitectureAdapter) -> None:
        """OLMo 2 is post-norm: RMSNorm applies after attention/MLP, not before.
        Folding LN into QKV/MLP weights would be incorrect."""
        assert adapter.supports_fold_ln is False


class TestOlmo2WeightConversions:
    """The adapter uses `self._qkvo_weight_conversions()` from the base helper with no
    overrides. Per the unit-test guide, the rearrange patterns and the GQA n_kv_heads
    axis are the base helper's responsibility and are covered by base-class tests.
    The adapter-owned decision here is the exact set of conversion keys: four QKVO
    weights, no biases, no extras."""

    def test_conversion_keys_are_exactly_qkvo_weights(
        self, adapter: Olmo2ArchitectureAdapter
    ) -> None:
        assert set(_conversions(adapter).keys()) == {
            "blocks.{i}.attn.q.weight",
            "blocks.{i}.attn.k.weight",
            "blocks.{i}.attn.v.weight",
            "blocks.{i}.attn.o.weight",
        }


class TestOlmo2ComponentMapping:
    """Structure of the component mapping: required keys and HF module paths."""

    def test_has_required_top_level_keys(self, adapter: Olmo2ArchitectureAdapter) -> None:
        mapping = _mapping(adapter)
        for key in ("embed", "rotary_emb", "blocks", "ln_final", "unembed"):
            assert key in mapping, f"Missing top-level key: {key!r}"

    def test_top_level_hf_paths(self, adapter: Olmo2ArchitectureAdapter) -> None:
        mapping = _mapping(adapter)
        assert mapping["embed"].name == "model.embed_tokens"
        assert mapping["rotary_emb"].name == "model.rotary_emb"
        assert mapping["blocks"].name == "model.layers"
        assert mapping["ln_final"].name == "model.norm"
        assert mapping["unembed"].name == "lm_head"


class TestOlmo2ComponentTypes:
    """Bridge classes selected for each component slot."""

    def test_rotary_emb_is_rotary_bridge(self, adapter: Olmo2ArchitectureAdapter) -> None:
        assert isinstance(_mapping(adapter)["rotary_emb"], RotaryEmbeddingBridge)

    def test_blocks_is_block_bridge(self, adapter: Olmo2ArchitectureAdapter) -> None:
        assert isinstance(_mapping(adapter)["blocks"], BlockBridge)

    def test_ln_final_is_rms_normalization_bridge(self, adapter: Olmo2ArchitectureAdapter) -> None:
        assert isinstance(_mapping(adapter)["ln_final"], RMSNormalizationBridge)

    def test_block_attn_is_position_embeddings_bridge(
        self, adapter: Olmo2ArchitectureAdapter
    ) -> None:
        block = _mapping(adapter)["blocks"]
        assert isinstance(block.submodules["attn"], PositionEmbeddingsAttentionBridge)

    def test_block_mlp_is_gated_mlp_bridge(self, adapter: Olmo2ArchitectureAdapter) -> None:
        block = _mapping(adapter)["blocks"]
        assert isinstance(block.submodules["mlp"], GatedMLPBridge)

    def test_block_norms_are_rms(self, adapter: Olmo2ArchitectureAdapter) -> None:
        block = _mapping(adapter)["blocks"]
        assert isinstance(block.submodules["ln1"], RMSNormalizationBridge)
        assert isinstance(block.submodules["ln2"], RMSNormalizationBridge)

    def test_embed_and_unembed_are_correct_bridge_types(
        self, adapter: Olmo2ArchitectureAdapter
    ) -> None:
        mapping = _mapping(adapter)
        assert isinstance(mapping["embed"], EmbeddingBridge)
        assert isinstance(mapping["unembed"], UnembeddingBridge)

    def test_attn_q_k_v_o_are_linear_bridges(self, adapter: Olmo2ArchitectureAdapter) -> None:
        attn = _mapping(adapter)["blocks"].submodules["attn"]
        for slot in ("q", "k", "v", "o"):
            assert isinstance(attn.submodules[slot], LinearBridge)

    def test_mlp_gate_in_out_are_linear_bridges(self, adapter: Olmo2ArchitectureAdapter) -> None:
        mlp = _mapping(adapter)["blocks"].submodules["mlp"]
        for slot in ("gate", "in", "out"):
            assert isinstance(mlp.submodules[slot], LinearBridge)


class TestOlmo2PostNormBlockWiring:
    """OLMo 2 is post-norm: RMSNorm applies AFTER attention (ln1) and AFTER MLP (ln2).
    The HF module names diverge from the pre-norm Llama family default, where ln1
    would be `input_layernorm` and ln2 would be `post_attention_layernorm`. This is
    the central arch-specific decision and the single test most likely to catch a
    porting regression."""

    def test_ln1_maps_to_post_attention_layernorm(self, adapter: Olmo2ArchitectureAdapter) -> None:
        block = _mapping(adapter)["blocks"]
        assert block.submodules["ln1"].name == "post_attention_layernorm"

    def test_ln2_maps_to_post_feedforward_layernorm(
        self, adapter: Olmo2ArchitectureAdapter
    ) -> None:
        block = _mapping(adapter)["blocks"]
        assert block.submodules["ln2"].name == "post_feedforward_layernorm"

    def test_block_submodule_set(self, adapter: Olmo2ArchitectureAdapter) -> None:
        """Exact block submodule set: sequential transformer with both ln1 and ln2."""
        block = _mapping(adapter)["blocks"]
        assert set(block.submodules.keys()) == {"ln1", "ln2", "attn", "mlp"}


class TestOlmo2QKNormStructure:
    """OLMo 2 applies a pre-reshape RMSNorm to the flattened Q and K projection outputs
    (`q_norm(q_proj(x))`, then reshape into heads). The adapter exposes those norms as
    `q_norm` and `k_norm` submodules under the attention bridge, with HF names matching."""

    def test_attn_submodule_set(self, adapter: Olmo2ArchitectureAdapter) -> None:
        attn = _mapping(adapter)["blocks"].submodules["attn"]
        assert set(attn.submodules.keys()) == {"q", "k", "v", "o", "q_norm", "k_norm"}

    def test_q_norm_is_rms_normalization_bridge(self, adapter: Olmo2ArchitectureAdapter) -> None:
        attn = _mapping(adapter)["blocks"].submodules["attn"]
        assert isinstance(attn.submodules["q_norm"], RMSNormalizationBridge)

    def test_k_norm_is_rms_normalization_bridge(self, adapter: Olmo2ArchitectureAdapter) -> None:
        attn = _mapping(adapter)["blocks"].submodules["attn"]
        assert isinstance(attn.submodules["k_norm"], RMSNormalizationBridge)

    def test_q_norm_uses_q_norm_hf_name(self, adapter: Olmo2ArchitectureAdapter) -> None:
        attn = _mapping(adapter)["blocks"].submodules["attn"]
        assert attn.submodules["q_norm"].name == "q_norm"

    def test_k_norm_uses_k_norm_hf_name(self, adapter: Olmo2ArchitectureAdapter) -> None:
        attn = _mapping(adapter)["blocks"].submodules["attn"]
        assert attn.submodules["k_norm"].name == "k_norm"


class TestOlmo2HookAliasOverrides:
    """Under post-norm, `ln2.hook_in` no longer captures the residual between
    attention and MLP (ln2 applies AFTER MLP, so ln2.hook_in is the MLP output).
    The true post-attn pre-mlp residual is at `mlp.hook_in`. The adapter overrides
    `hook_resid_mid` accordingly. BlockBridge's default would point at the wrong
    tensor without this override."""

    def test_hook_resid_mid_points_at_mlp_hook_in(self, adapter: Olmo2ArchitectureAdapter) -> None:
        block = _mapping(adapter)["blocks"]
        assert block.hook_aliases["hook_resid_mid"] == "mlp.hook_in"


class TestOlmo2GQAHookShapes:
    """Wire a fake attention module into the bridge and verify GQA hook shapes.

    Q must surface n_heads while K/V surface n_key_value_heads, which is the whole
    point of grouped-query attention. The fake carries OLMo 2's pre-reshape Q/K norms
    so the bridge takes its Q/K-norm code path.
    """

    N_HEADS = 4
    N_KV_HEADS = 2
    D_MODEL = 64
    D_HEAD = D_MODEL // N_HEADS
    BATCH = 2
    SEQ = 8

    @pytest.fixture
    def wired_attn_bridge(self) -> PositionEmbeddingsAttentionBridge:
        adapter = Olmo2ArchitectureAdapter(_cfg(n_key_value_heads=self.N_KV_HEADS))
        fake_attn = FakeOlmo2Attention(adapter.cfg)
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


class TestOlmo2SetupComponentTesting:
    """setup_component_testing wires the shared rotary embedding onto the template
    attention bridge and onto each bridge-model block's attention. It also forces
    eager attention on the HF model (top-level config and per-layer self_attn.config)
    for numerical parity with the bridge's eager-mode reimplementation."""

    def test_sets_rotary_emb_on_template_attention(self, adapter: Olmo2ArchitectureAdapter) -> None:
        rotary_emb = object()
        attn_template = adapter.get_generalized_component("blocks.0.attn")
        assert isinstance(attn_template, PositionEmbeddingsAttentionBridge)

        adapter.setup_component_testing(_fake_hf_model(rotary_emb))

        assert attn_template._rotary_emb is rotary_emb

    def test_sets_rotary_emb_on_each_bridge_model_attention(
        self, adapter: Olmo2ArchitectureAdapter
    ) -> None:
        rotary_emb = object()
        bridge_model = DummyBridgeModel([DummyBlock(), DummyBlock(), DummyBlock()])

        adapter.setup_component_testing(_fake_hf_model(rotary_emb), bridge_model=bridge_model)

        for block in bridge_model.blocks:
            assert block.attn.rotary_emb is rotary_emb

    def test_skips_bridge_blocks_without_attention(self, adapter: Olmo2ArchitectureAdapter) -> None:
        rotary_emb = object()
        bridge_model = DummyBridgeModel([DummyBlock(), DummyBlock(has_attention=False)])

        adapter.setup_component_testing(_fake_hf_model(rotary_emb), bridge_model=bridge_model)

        assert bridge_model.blocks[0].attn.rotary_emb is rotary_emb

    def test_forces_eager_attention_implementation(self, adapter: Olmo2ArchitectureAdapter) -> None:
        """Bridge attention only matches HF under eager attention, so it is forced on
        at both the top-level config and on each per-layer self_attn.config."""
        hf_model = _fake_hf_model_with_eager_targets(object())

        adapter.setup_component_testing(hf_model)

        assert hf_model.config._attn_implementation == "eager"
        for layer in hf_model.model.layers:
            assert layer.self_attn.config._attn_implementation == "eager"

    def test_tolerates_minimal_hf_model_without_config_or_layers(
        self, adapter: Olmo2ArchitectureAdapter
    ) -> None:
        """The defensive hasattr branches must not raise when config/layers are absent."""
        rotary_emb = object()
        # _fake_hf_model exposes only model.rotary_emb (no config, no layers).
        adapter.setup_component_testing(_fake_hf_model(rotary_emb))

        attn_template = adapter.get_generalized_component("blocks.0.attn")
        assert isinstance(attn_template, PositionEmbeddingsAttentionBridge)
        assert attn_template._rotary_emb is rotary_emb
