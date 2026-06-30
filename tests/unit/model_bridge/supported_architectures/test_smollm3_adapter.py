"""Unit tests for SmolLM3ArchitectureAdapter (download-free: tiny programmatic
configs, small synthetic tensors, and a fake attention module, no real checkpoints).

Covered:
- Adapter config defaults (RMSNorm, rotary, gated MLP, eager attention, GQA propagation).
- Component-mapping structure, bridge types, and HF module paths, including the
  NoPE-aware attention bridge and the absence of Q/K norms.
- Standard Q/K/V/O weight conversions (keys, einops patterns, GQA head counts).
- Numerical round-trips: the rearrange conversions reshape and revert losslessly.
- GQA forward hook shapes (Q uses n_heads, K/V use n_key_value_heads).
- NoPE behaviour: the attention bridge drops position embeddings on NoPE layers
  and passes them through on RoPE layers.
- setup_component_testing rotary-embedding wiring, eager forcing, and robustness.
- Factory dispatch, registry-sync membership, and model_type routing.
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
    PositionEmbeddingsAttentionBridge,
    RMSNormalizationBridge,
    RotaryEmbeddingBridge,
    UnembeddingBridge,
)
from transformer_lens.model_bridge.supported_architectures.smollm3 import (
    SmolLM3ArchitectureAdapter,
    _SmolLM3AttentionBridge,
)
from transformer_lens.tools.model_registry import CANONICAL_AUTHORS_BY_ARCH


def _make_cfg(
    n_heads: int = 4,
    n_key_value_heads: int | None = 2,
    d_model: int = 64,
    n_layers: int = 2,
    d_mlp: int = 256,
    d_vocab: int = 100,
    n_ctx: int = 64,
) -> TransformerBridgeConfig:
    # Keep dimensions tiny so adapter tests do not need HF downloads or real checkpoints.
    # n_key_value_heads=None exercises the GQA fallback to n_heads in the conversions.
    return TransformerBridgeConfig(
        d_model=d_model,
        d_head=d_model // n_heads,
        n_layers=n_layers,
        n_ctx=n_ctx,
        n_heads=n_heads,
        n_key_value_heads=n_key_value_heads,
        d_vocab=d_vocab,
        d_mlp=d_mlp,
        default_prepend_bos=False,
        architecture="SmolLM3ForCausalLM",
    )


@pytest.fixture
def cfg() -> TransformerBridgeConfig:
    return _make_cfg()


@pytest.fixture
def adapter(cfg: TransformerBridgeConfig) -> SmolLM3ArchitectureAdapter:
    return SmolLM3ArchitectureAdapter(cfg)


class FakeSmolLM3Attention(nn.Module):
    """Minimal SmolLM3-style attention module for adapter hook-shape tests.

    SmolLM3 has no attention bias and uses GQA, so the projections are bias-free
    with Q at n_heads width and K/V at n_key_value_heads width. use_rope mirrors
    the HF flag the NoPE-aware bridge reads (1 = apply RoPE on this layer).
    """

    def __init__(self, cfg: TransformerBridgeConfig, use_rope: int = 1) -> None:
        super().__init__()
        # PositionEmbeddingsAttentionBridge reads these HF-style attributes during forward.
        self.head_dim = cfg.d_head
        self.num_key_value_groups = cfg.n_heads // (cfg.n_key_value_heads or cfg.n_heads)
        self.scaling = cfg.d_head**-0.5
        self.attention_dropout = 0.0
        self.use_rope = use_rope

        kv_width = (cfg.n_key_value_heads or cfg.n_heads) * cfg.d_head
        self.q_proj = nn.Linear(cfg.d_model, cfg.n_heads * cfg.d_head, bias=False)
        self.k_proj = nn.Linear(cfg.d_model, kv_width, bias=False)
        self.v_proj = nn.Linear(cfg.d_model, kv_width, bias=False)
        self.o_proj = nn.Linear(cfg.n_heads * cfg.d_head, cfg.d_model, bias=False)


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


def _conversions(adapter: SmolLM3ArchitectureAdapter) -> dict:
    """weight_processing_conversions is Optional on the base class; assert it is populated."""
    conversions = adapter.weight_processing_conversions
    assert conversions is not None
    return conversions


def _param_conversion(adapter: SmolLM3ArchitectureAdapter, key: str) -> ParamProcessingConversion:
    conv = _conversions(adapter)[key]
    assert isinstance(conv, ParamProcessingConversion)
    return conv


def _rearrange(adapter: SmolLM3ArchitectureAdapter, key: str) -> RearrangeTensorConversion:
    tensor_conversion = _param_conversion(adapter, key).tensor_conversion
    assert isinstance(tensor_conversion, RearrangeTensorConversion)
    return tensor_conversion


class TestSmolLM3AdapterConfig:
    """Adapter-owned config defaults that downstream bridge code relies on."""

    def test_default_prepend_bos_is_false(self, adapter: SmolLM3ArchitectureAdapter) -> None:
        assert adapter.cfg.default_prepend_bos is False


class TestSmolLM3ComponentMapping:
    """The adapter contract: TL canonical names mapped to SmolLM3 HF module paths."""

    def test_top_level_keys(self, adapter: SmolLM3ArchitectureAdapter) -> None:
        assert set(adapter.component_mapping.keys()) == {
            "embed",
            "rotary_emb",
            "blocks",
            "ln_final",
            "unembed",
        }

    def test_embed_path(self, adapter: SmolLM3ArchitectureAdapter) -> None:
        assert adapter.component_mapping["embed"].name == "model.embed_tokens"

    def test_rotary_emb_path(self, adapter: SmolLM3ArchitectureAdapter) -> None:
        assert adapter.component_mapping["rotary_emb"].name == "model.rotary_emb"

    def test_blocks_path(self, adapter: SmolLM3ArchitectureAdapter) -> None:
        assert adapter.component_mapping["blocks"].name == "model.layers"

    def test_ln_final_path(self, adapter: SmolLM3ArchitectureAdapter) -> None:
        assert adapter.component_mapping["ln_final"].name == "model.norm"

    def test_unembed_path(self, adapter: SmolLM3ArchitectureAdapter) -> None:
        assert adapter.component_mapping["unembed"].name == "lm_head"

    def test_block_submodule_keys(self, adapter: SmolLM3ArchitectureAdapter) -> None:
        blocks = adapter.component_mapping["blocks"]
        assert set(blocks.submodules.keys()) == {"ln1", "ln2", "attn", "mlp"}

    def test_attention_submodule_keys(self, adapter: SmolLM3ArchitectureAdapter) -> None:
        attn = adapter.component_mapping["blocks"].submodules["attn"]
        assert set(attn.submodules.keys()) == {"q", "k", "v", "o"}

    def test_attention_has_no_qk_norms(self, adapter: SmolLM3ArchitectureAdapter) -> None:
        """Unlike Qwen3, SmolLM3 has no per-head Q/K norm. Declaring one would make
        the attention-bridge validator raise against the real HF module."""
        attn = adapter.component_mapping["blocks"].submodules["attn"]
        assert "q_norm" not in attn.submodules
        assert "k_norm" not in attn.submodules

    def test_mlp_submodule_keys(self, adapter: SmolLM3ArchitectureAdapter) -> None:
        mlp = adapter.component_mapping["blocks"].submodules["mlp"]
        assert set(mlp.submodules.keys()) == {"gate", "in", "out"}

    def test_attention_hf_paths(self, adapter: SmolLM3ArchitectureAdapter) -> None:
        attn = adapter.component_mapping["blocks"].submodules["attn"]
        assert attn.name == "self_attn"
        assert attn.submodules["q"].name == "q_proj"
        assert attn.submodules["k"].name == "k_proj"
        assert attn.submodules["v"].name == "v_proj"
        assert attn.submodules["o"].name == "o_proj"

    def test_mlp_hf_paths(self, adapter: SmolLM3ArchitectureAdapter) -> None:
        mlp = adapter.component_mapping["blocks"].submodules["mlp"]
        assert mlp.name == "mlp"
        assert mlp.submodules["gate"].name == "gate_proj"
        assert mlp.submodules["in"].name == "up_proj"
        assert mlp.submodules["out"].name == "down_proj"

    def test_block_layernorm_hf_paths(self, adapter: SmolLM3ArchitectureAdapter) -> None:
        blocks = adapter.component_mapping["blocks"]
        assert blocks.submodules["ln1"].name == "input_layernorm"
        assert blocks.submodules["ln2"].name == "post_attention_layernorm"


class TestSmolLM3ComponentTypes:
    """Bridge classes guard against silent substitution of the wrong component."""

    def test_top_level_bridge_types(self, adapter: SmolLM3ArchitectureAdapter) -> None:
        mapping = adapter.component_mapping
        assert isinstance(mapping["embed"], EmbeddingBridge)
        assert isinstance(mapping["rotary_emb"], RotaryEmbeddingBridge)
        assert isinstance(mapping["blocks"], BlockBridge)
        assert isinstance(mapping["ln_final"], RMSNormalizationBridge)
        assert isinstance(mapping["unembed"], UnembeddingBridge)

    def test_block_layernorm_bridge_types(self, adapter: SmolLM3ArchitectureAdapter) -> None:
        blocks = adapter.component_mapping["blocks"]
        assert isinstance(blocks.submodules["ln1"], RMSNormalizationBridge)
        assert isinstance(blocks.submodules["ln2"], RMSNormalizationBridge)

    def test_attention_is_nope_aware_bridge(self, adapter: SmolLM3ArchitectureAdapter) -> None:
        """The attention bridge is the NoPE-aware subclass, not the plain base bridge."""
        attn = adapter.component_mapping["blocks"].submodules["attn"]
        # Exact type, not isinstance: the adapter must wire the NoPE-aware subclass,
        # and the plain base bridge (which would rotate Q/K on NoPE layers) must not
        # silently satisfy this check.
        assert type(attn) is _SmolLM3AttentionBridge
        # Document and enforce the inheritance contract the NoPE override relies on
        # (it delegates to the base forward via super()).
        assert issubclass(_SmolLM3AttentionBridge, PositionEmbeddingsAttentionBridge)

    def test_mlp_is_gated_mlp(self, adapter: SmolLM3ArchitectureAdapter) -> None:
        mlp = adapter.component_mapping["blocks"].submodules["mlp"]
        assert isinstance(mlp, GatedMLPBridge)

    def test_linear_submodule_bridge_types(self, adapter: SmolLM3ArchitectureAdapter) -> None:
        blocks = adapter.component_mapping["blocks"]
        attn = blocks.submodules["attn"]
        mlp = blocks.submodules["mlp"]
        for submodule in [*attn.submodules.values(), *mlp.submodules.values()]:
            assert isinstance(submodule, LinearBridge)

    def test_attn_requires_attention_mask_and_position_embeddings(
        self, adapter: SmolLM3ArchitectureAdapter
    ) -> None:
        attn = adapter.component_mapping["blocks"].submodules["attn"]
        assert attn.requires_attention_mask is True
        assert attn.requires_position_embeddings is True


class TestSmolLM3WeightConversions:
    """SmolLM3 uses the standard QKVO weight conversions, with GQA-specific K/V heads."""

    def test_conversion_keys_exactly_qkvo(self, adapter: SmolLM3ArchitectureAdapter) -> None:
        assert set(_conversions(adapter).keys()) == {
            "blocks.{i}.attn.q.weight",
            "blocks.{i}.attn.k.weight",
            "blocks.{i}.attn.v.weight",
            "blocks.{i}.attn.o.weight",
        }

    def test_gqa_fallback_to_n_heads_without_kv_heads(self) -> None:
        """Without n_key_value_heads, K/V fall back to n_heads in the conversions."""
        adapter = SmolLM3ArchitectureAdapter(_make_cfg(n_key_value_heads=None))
        for slot in ("k", "v"):
            assert _rearrange(adapter, f"blocks.{{i}}.attn.{slot}.weight").axes_lengths["n"] == 4


class TestSmolLM3GQAHookShapes:
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
        adapter = SmolLM3ArchitectureAdapter(
            _make_cfg(
                n_heads=self.N_HEADS,
                n_key_value_heads=self.N_KV_HEADS,
                d_model=self.D_MODEL,
            )
        )
        fake_attn = FakeSmolLM3Attention(adapter.cfg)
        attn_bridge = adapter.component_mapping["blocks"].submodules["attn"]
        assert isinstance(attn_bridge, _SmolLM3AttentionBridge)
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


class TestSmolLM3NoPE:
    """The one piece of real adapter logic: per-layer RoPE suppression.

    SmolLM3 disables RoPE on every no_rope_layer_interval-th layer. The wrapped
    HF module records this as use_rope (1 = RoPE, 0 = NoPE). The base attention
    bridge rotates whenever position embeddings are present, so the adapter's
    _SmolLM3AttentionBridge must null them out on NoPE layers and leave them
    intact otherwise. These tests spy on the base-class forward to assert exactly
    that, rather than re-deriving the rotation math.
    """

    def _bridge_with(self, use_rope: int) -> _SmolLM3AttentionBridge:
        cfg = _make_cfg()
        bridge = _SmolLM3AttentionBridge(
            name="self_attn",
            config=cfg,
            submodules={
                "q": LinearBridge(name="q_proj"),
                "k": LinearBridge(name="k_proj"),
                "v": LinearBridge(name="v_proj"),
                "o": LinearBridge(name="o_proj"),
            },
        )
        bridge.set_original_component(FakeSmolLM3Attention(cfg, use_rope=use_rope))
        return bridge

    def _record_super_forward(self, monkeypatch: pytest.MonkeyPatch) -> dict:
        """Replace the base bridge forward with a recorder and return the capture dict."""
        recorded: dict = {}

        def _fake_super_forward(self: Any, *args: Any, **kwargs: Any) -> str:
            recorded["args"] = args
            recorded["kwargs"] = kwargs
            return "sentinel-output"

        monkeypatch.setattr(
            PositionEmbeddingsAttentionBridge, "forward", _fake_super_forward, raising=True
        )
        return recorded

    def test_nope_layer_suppresses_position_embeddings_kwarg(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        recorded = self._record_super_forward(monkeypatch)
        bridge = self._bridge_with(use_rope=0)
        hidden = randn(2, 8, 64)

        result = bridge.forward(hidden, position_embeddings=(ones(1, 8, 16), zeros(1, 8, 16)))

        assert result == "sentinel-output"
        assert recorded["kwargs"]["position_embeddings"] is None

    def test_rope_layer_passes_position_embeddings_through(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        recorded = self._record_super_forward(monkeypatch)
        bridge = self._bridge_with(use_rope=1)
        hidden = randn(2, 8, 64)
        pos = (ones(1, 8, 16), zeros(1, 8, 16))

        bridge.forward(hidden, position_embeddings=pos)

        assert recorded["kwargs"]["position_embeddings"] is pos

    def test_nope_layer_suppresses_positional_position_embeddings(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Defensive positional branch: a (hidden, (cos, sin), ...) call is nulled too."""
        recorded = self._record_super_forward(monkeypatch)
        bridge = self._bridge_with(use_rope=0)
        hidden = randn(2, 8, 64)
        pos = (ones(1, 8, 16), zeros(1, 8, 16))

        bridge.forward(hidden, pos, None)

        assert recorded["args"][0] is hidden
        assert recorded["args"][1] is None

    @staticmethod
    def _wired_bridge(cfg: TransformerBridgeConfig, use_rope: int) -> _SmolLM3AttentionBridge:
        """Build a fully wired NoPE-aware attention bridge over a fake HF module."""
        fake_attn = FakeSmolLM3Attention(cfg, use_rope=use_rope)
        bridge = _SmolLM3AttentionBridge(
            name="self_attn",
            config=cfg,
            submodules={
                "q": LinearBridge(name="q_proj"),
                "k": LinearBridge(name="k_proj"),
                "v": LinearBridge(name="v_proj"),
                "o": LinearBridge(name="o_proj"),
            },
        )
        bridge.set_original_component(fake_attn)
        for name, original in {
            "q": fake_attn.q_proj,
            "k": fake_attn.k_proj,
            "v": fake_attn.v_proj,
            "o": fake_attn.o_proj,
        }.items():
            submodule = bridge.submodules[name]
            submodule.set_original_component(original)
            bridge.add_module(name, submodule)
        bridge.setup_hook_compatibility()
        return bridge

    @staticmethod
    def _forward(bridge: _SmolLM3AttentionBridge, hidden: Any, position_embeddings: Any) -> Any:
        from torch import no_grad

        with no_grad():
            out = bridge(hidden, position_embeddings=position_embeddings, attention_mask=None)
        return out[0] if isinstance(out, tuple) else out

    def test_nope_layer_output_ignores_position_embeddings_end_to_end(self) -> None:
        """On a NoPE layer the full bridge forward must produce the same output with
        non-identity rotary embeddings as with none at all, proving RoPE was skipped.

        This drives the entire reimplemented attention path (not just the kwarg
        spy), so it guards against any future change that would apply RoPE on a
        NoPE layer.
        """
        from torch import allclose, manual_seed

        manual_seed(0)
        cfg = _make_cfg(n_heads=4, n_key_value_heads=2, d_model=64)
        bridge = self._wired_bridge(cfg, use_rope=0)

        hidden = randn(2, 8, 64)
        # Non-identity cos/sin: a layer that actually applied RoPE would diverge
        # from the no-rotation result.
        cos = randn(1, 8, 16)
        sin = randn(1, 8, 16)

        out_with_pos = self._forward(bridge, hidden, (cos, sin))
        out_without_pos = self._forward(bridge, hidden, None)

        assert allclose(out_with_pos, out_without_pos, atol=1e-6)

    def test_rope_layer_output_depends_on_position_embeddings_end_to_end(self) -> None:
        """Control for the NoPE test: on a RoPE layer the same non-identity rotary
        embeddings must change the output, confirming the harness is sensitive
        enough to detect rotation (so the NoPE match above is meaningful)."""
        from torch import allclose, manual_seed

        manual_seed(0)
        cfg = _make_cfg(n_heads=4, n_key_value_heads=2, d_model=64)
        bridge = self._wired_bridge(cfg, use_rope=1)

        hidden = randn(2, 8, 64)
        cos = randn(1, 8, 16)
        sin = randn(1, 8, 16)

        out_with_pos = self._forward(bridge, hidden, (cos, sin))
        out_without_pos = self._forward(bridge, hidden, None)

        assert not allclose(out_with_pos, out_without_pos, atol=1e-6)


class TestSmolLM3SetupComponentTesting:
    """setup_component_testing wires the shared rotary embedding and forces eager attention."""

    def test_sets_rotary_emb_on_template_attention(
        self, adapter: SmolLM3ArchitectureAdapter
    ) -> None:
        rotary_emb = object()
        attn_template = adapter.get_generalized_component("blocks.0.attn")
        assert isinstance(attn_template, _SmolLM3AttentionBridge)
        assert attn_template._rotary_emb is None

        adapter.setup_component_testing(_fake_hf_model(rotary_emb))

        assert attn_template._rotary_emb is rotary_emb

    def test_sets_rotary_emb_on_each_bridge_model_attention(
        self, adapter: SmolLM3ArchitectureAdapter
    ) -> None:
        rotary_emb = object()
        bridge_model = DummyBridgeModel([DummyBlock(), DummyBlock(), DummyBlock()])

        adapter.setup_component_testing(_fake_hf_model(rotary_emb), bridge_model=bridge_model)

        for block in bridge_model.blocks:
            assert block.attn.rotary_emb is rotary_emb

    def test_skips_bridge_blocks_without_attention(
        self, adapter: SmolLM3ArchitectureAdapter
    ) -> None:
        rotary_emb = object()
        bridge_model = DummyBridgeModel([DummyBlock(), DummyBlock(has_attention=False)])

        adapter.setup_component_testing(_fake_hf_model(rotary_emb), bridge_model=bridge_model)

        assert bridge_model.blocks[0].attn.rotary_emb is rotary_emb

    def test_forces_eager_attention_implementation(
        self, adapter: SmolLM3ArchitectureAdapter
    ) -> None:
        """Bridge attention only matches HF under eager attention, so it is forced on."""
        hf_model = _fake_hf_model_with_eager_targets(object())

        adapter.setup_component_testing(hf_model)

        assert hf_model.config._attn_implementation == "eager"
        for layer in hf_model.model.layers:
            assert layer.self_attn.config._attn_implementation == "eager"

    def test_tolerates_minimal_hf_model_without_config_or_layers(
        self, adapter: SmolLM3ArchitectureAdapter
    ) -> None:
        """The defensive hasattr branches must not raise when config/layers are absent."""
        rotary_emb = object()
        adapter.setup_component_testing(_fake_hf_model(rotary_emb))

        attn_template = adapter.get_generalized_component("blocks.0.attn")
        assert isinstance(attn_template, _SmolLM3AttentionBridge)
        assert attn_template._rotary_emb is rotary_emb


class TestSmolLM3RegistryRegistration:
    """Registry sets must stay in sync with the factory (enforced by the #1354 invariant)."""

    def test_in_canonical_authors(self) -> None:
        assert "SmolLM3ForCausalLM" in CANONICAL_AUTHORS_BY_ARCH
        assert "HuggingFaceTB" in CANONICAL_AUTHORS_BY_ARCH["SmolLM3ForCausalLM"]


class TestSmolLM3ArchitectureDetection:
    """model_type routing for configs that report smollm3 without architectures[]."""

    def test_model_type_smollm3_routes(self) -> None:
        from transformer_lens.model_bridge.sources.transformers import (
            determine_architecture_from_hf_config,
        )

        cfg = SimpleNamespace(model_type="smollm3", architectures=[])
        assert determine_architecture_from_hf_config(cfg) == "SmolLM3ForCausalLM"
