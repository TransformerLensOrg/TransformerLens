"""Unit tests for the StableLmArchitectureAdapter (download-free, tiny programmatic configs
plus small synthetic tensors and a fake attention module, no real checkpoints).

Covered:
- Anti-drift config defaults: `normalization_type="LN"`, `uses_rms_norm=False`, and
  `attn_implementation="eager"`. Each is read by a distinct consumer (bridge selection,
  NormalizationBridge runtime behavior, model-boot attention class).
- Weight conversions: QKVO weights via the base helper, Q/K/V biases inline with
  GQA-aware head counts, plus the no-`n_key_value_heads` fallback.
- Component-mapping structure, bridge types (NormalizationBridge for LN, not RMS), and HF paths.
- Block submodules across both `parallel_attn_mlp` branches (with and without `ln2`).
- GQA forward hook shapes: a fake attention module confirms Q uses `n_heads` while
  K/V use `n_key_value_heads`.
- `setup_hook_compatibility`: QK-LayerNorm hook injection on StableLM v2 models, including
  the no-op-when-absent path and defensive guards.
- `setup_component_testing`: rotary embedding wiring on the template attention bridge and
  on each bridge-model block, plus forcing eager attention on the HF model and per-layer
  self-attention configs.
- Architecture guards against drift.
"""

from types import SimpleNamespace
from typing import Any

import pytest
import torch
import torch.nn as nn
from torch import ones, randn, zeros

from transformer_lens.config import TransformerBridgeConfig
from transformer_lens.conversion_utils.conversion_steps.rearrange_tensor_conversion import (
    RearrangeTensorConversion,
)
from transformer_lens.conversion_utils.param_processing_conversion import (
    ParamProcessingConversion,
)
from transformer_lens.hook_points import HookPoint
from transformer_lens.model_bridge.generalized_components import (
    BlockBridge,
    EmbeddingBridge,
    GatedMLPBridge,
    LinearBridge,
    NormalizationBridge,
    ParallelBlockBridge,
    PositionEmbeddingsAttentionBridge,
    RotaryEmbeddingBridge,
    UnembeddingBridge,
)
from transformer_lens.model_bridge.supported_architectures.stablelm import (
    StableLmArchitectureAdapter,
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
        architecture="StableLmForCausalLM",
    )


@pytest.fixture(scope="class")
def adapter(cfg: TransformerBridgeConfig) -> StableLmArchitectureAdapter:
    return StableLmArchitectureAdapter(cfg)


def _cfg(
    *,
    n_key_value_heads: int | None = 2,
    parallel_attn_mlp: bool = False,
) -> TransformerBridgeConfig:
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
        architecture="StableLmForCausalLM",
        parallel_attn_mlp=parallel_attn_mlp,
    )


def _mapping(adapter: StableLmArchitectureAdapter) -> dict:
    """Narrow component_mapping (Optional on the base class) to a non-None dict.

    Factored into a helper so each test stays a one-liner instead of repeating the
    `assert ... is not None` prelude in every method.
    """
    mapping = adapter.component_mapping
    assert mapping is not None
    return mapping


def _conversions(adapter: StableLmArchitectureAdapter) -> dict:
    """weight_processing_conversions is Optional on the base class, assert it is populated."""
    conversions = adapter.weight_processing_conversions
    assert conversions is not None
    return conversions


def _param_conversion(adapter: StableLmArchitectureAdapter, key: str) -> ParamProcessingConversion:
    conv = _conversions(adapter)[key]
    assert isinstance(conv, ParamProcessingConversion)
    return conv


def _rearrange(adapter: StableLmArchitectureAdapter, key: str) -> RearrangeTensorConversion:
    tensor_conversion = _param_conversion(adapter, key).tensor_conversion
    assert isinstance(tensor_conversion, RearrangeTensorConversion)
    return tensor_conversion


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


class FakeStableLMAttention(nn.Module):
    """Minimal StableLM-style attention module for adapter hook-shape tests.

    Q is n_heads-wide, K/V are n_key_value_heads-wide, with no biases by default
    (use_qkv_bias is False on stock StableLM).
    """

    def __init__(self, cfg: TransformerBridgeConfig) -> None:
        super().__init__()
        # PositionEmbeddingsAttentionBridge reads these HF-style attributes during forward.
        self.head_dim = cfg.d_head
        self.num_key_value_groups = cfg.n_heads // (cfg.n_key_value_heads or cfg.n_heads)
        self.scaling = cfg.d_head**-0.5
        self.attention_dropout = 0.0

        kv_width = (cfg.n_key_value_heads or cfg.n_heads) * cfg.d_head
        self.q_proj = nn.Linear(cfg.d_model, cfg.n_heads * cfg.d_head, bias=False)
        self.k_proj = nn.Linear(cfg.d_model, kv_width, bias=False)
        self.v_proj = nn.Linear(cfg.d_model, kv_width, bias=False)
        self.o_proj = nn.Linear(cfg.n_heads * cfg.d_head, cfg.d_model, bias=False)


class _DummyHFAttn(nn.Module):
    """HF attention stand-in for the QK-LayerNorm setup_hook_compatibility tests.

    `qk_layernorm` is read as a bool flag. `q_layernorm` and `k_layernorm` are
    nn.Modules whose `forward` the adapter wraps to fire the new hook points.
    """

    def __init__(self, qk_layernorm: bool = True) -> None:
        super().__init__()
        self.qk_layernorm = qk_layernorm
        self.q_layernorm = nn.Identity()
        self.k_layernorm = nn.Identity()


class _DummyAttnBridge(nn.Module):
    """nn.Module stand-in for the attention bridge.

    The adapter calls `add_module` on the bridge to register HookPoints, which
    only works on nn.Modules.
    """

    def __init__(self, original_component: Any | None = None) -> None:
        super().__init__()
        self.original_component = original_component


class _DummyBlockWithAttn:
    def __init__(self, attn: Any) -> None:
        self.attn = attn


class _DummyBlockNoAttn:
    pass


class _DummyHookBridge:
    """Stand-in for a built TransformerBridge: exposes `blocks` and a `_hook_registry` dict."""

    def __init__(self, blocks: list[Any]) -> None:
        self.blocks = blocks
        self._hook_registry: dict[str, HookPoint] = {}


class TestStableLMAdapterConfig:
    """Anti-drift config flags: StableLM deviates from the Llama-family default of RMSNorm
    and deliberately forces eager attention for parity with the bridge's eager-mode
    reimplementation of attention."""

    def test_normalization_type_is_ln(self, adapter: StableLmArchitectureAdapter) -> None:
        """StableLM uses standard LayerNorm, not RMSNorm like the rest of the Llama family."""
        assert adapter.cfg.normalization_type == "LN"

    def test_uses_rms_norm_is_false(self, adapter: StableLmArchitectureAdapter) -> None:
        """Paired with normalization_type=LN, but consumed by a different code path:
        NormalizationBridge.uses_rms_norm reads this flag at forward time to decide
        whether the bridge behaves as RMSNorm or LayerNorm. The ComponentTypes tests
        only verify class identity (NormalizationBridge), not runtime behavior, so a
        silent flip here would slip past them. Sibling LN-anti-drift adapters
        (cohere, gpt_bigcode) keep the same assertion for the same reason."""
        assert adapter.cfg.uses_rms_norm is False

    def test_attn_implementation_is_eager(self, adapter: StableLmArchitectureAdapter) -> None:
        """cfg.attn_implementation is read at model boot in sources/transformers.py and
        passed to HF's loader, which instantiates different attention modules per value.
        TestStableLMSetupComponentTesting only covers the post-load fixup path. The
        load-time selection path is only guarded by this assertion."""
        assert adapter.cfg.attn_implementation == "eager"


class TestStableLMWeightConversions:
    """StableLM declares QKVO weight conversions via the base helper plus Q/K/V bias
    conversions inline (for the optional `use_qkv_bias=True` variants like stable-code-3b).
    The bias rearranges are the adapter's own choice, so their patterns and head-count
    axes are worth asserting. The QKVO weight patterns come from the base helper and are
    covered by the base class's own tests."""

    def test_conversion_keys_are_exactly_qkvo_weights_plus_qkv_biases(
        self, adapter: StableLmArchitectureAdapter
    ) -> None:
        """O has no bias (no `b_O` in the conversion map), and the MLP has no biases."""
        assert set(_conversions(adapter).keys()) == {
            "blocks.{i}.attn.q.weight",
            "blocks.{i}.attn.k.weight",
            "blocks.{i}.attn.v.weight",
            "blocks.{i}.attn.o.weight",
            "blocks.{i}.attn.q.bias",
            "blocks.{i}.attn.k.bias",
            "blocks.{i}.attn.v.bias",
        }

    def test_q_bias_rearrange_uses_n_heads(self, adapter: StableLmArchitectureAdapter) -> None:
        rearrange = _rearrange(adapter, "blocks.{i}.attn.q.bias")
        assert rearrange.pattern == "(n h) -> n h"
        assert rearrange.axes_lengths.get("n") == 4

    def test_kv_bias_rearrange_uses_n_kv_heads(self, adapter: StableLmArchitectureAdapter) -> None:
        """GQA: K/V biases follow n_key_value_heads (2), not n_heads."""
        for slot in ("k", "v"):
            rearrange = _rearrange(adapter, f"blocks.{{i}}.attn.{slot}.bias")
            assert rearrange.pattern == "(n h) -> n h"
            assert rearrange.axes_lengths.get("n") == 2

    def test_gqa_fallback_to_n_heads_without_kv_heads(self) -> None:
        """Without n_key_value_heads, K/V biases fall back to n_heads (the `or self.cfg.n_heads`
        clause in the adapter)."""
        adapter = StableLmArchitectureAdapter(_cfg(n_key_value_heads=None))
        for slot in ("k", "v"):
            assert _rearrange(adapter, f"blocks.{{i}}.attn.{slot}.bias").axes_lengths["n"] == 4


class TestStableLMComponentMapping:
    """Structure of the component mapping: required keys and HF module paths."""

    def test_has_required_top_level_keys(self, adapter: StableLmArchitectureAdapter) -> None:
        mapping = _mapping(adapter)
        for key in ("embed", "rotary_emb", "blocks", "ln_final", "unembed"):
            assert key in mapping, f"Missing top-level key: {key!r}"

    def test_top_level_hf_paths(self, adapter: StableLmArchitectureAdapter) -> None:
        mapping = _mapping(adapter)
        assert mapping["embed"].name == "model.embed_tokens"
        assert mapping["rotary_emb"].name == "model.rotary_emb"
        assert mapping["blocks"].name == "model.layers"
        assert mapping["ln_final"].name == "model.norm"
        assert mapping["unembed"].name == "lm_head"


class TestStableLMComponentTypes:
    """Bridge classes selected for each component slot. The norms are deliberately
    NormalizationBridge (LayerNorm), not RMSNormalizationBridge: this is the structural
    consequence of `normalization_type='LN'` in the AdapterConfig tests above."""

    def test_rotary_emb_is_rotary_bridge(self, adapter: StableLmArchitectureAdapter) -> None:
        assert isinstance(_mapping(adapter)["rotary_emb"], RotaryEmbeddingBridge)

    def test_ln_final_is_layernorm_not_rms(self, adapter: StableLmArchitectureAdapter) -> None:
        """LN, not RMSNorm. The anti-drift normalization_type flag has to wire here."""
        ln_final = _mapping(adapter)["ln_final"]
        assert isinstance(ln_final, NormalizationBridge)

    def test_block_attn_is_position_embeddings_bridge(
        self, adapter: StableLmArchitectureAdapter
    ) -> None:
        block = _mapping(adapter)["blocks"]
        assert isinstance(block.submodules["attn"], PositionEmbeddingsAttentionBridge)

    def test_block_mlp_is_gated_mlp_bridge(self, adapter: StableLmArchitectureAdapter) -> None:
        block = _mapping(adapter)["blocks"]
        assert isinstance(block.submodules["mlp"], GatedMLPBridge)

    def test_block_norms_are_layernorm_not_rms(self, adapter: StableLmArchitectureAdapter) -> None:
        block = _mapping(adapter)["blocks"]
        assert isinstance(block.submodules["ln1"], NormalizationBridge)
        assert isinstance(block.submodules["ln2"], NormalizationBridge)

    def test_embed_and_unembed_are_correct_bridge_types(
        self, adapter: StableLmArchitectureAdapter
    ) -> None:
        mapping = _mapping(adapter)
        assert isinstance(mapping["embed"], EmbeddingBridge)
        assert isinstance(mapping["unembed"], UnembeddingBridge)

    def test_attn_q_k_v_o_are_linear_bridges(self, adapter: StableLmArchitectureAdapter) -> None:
        attn = _mapping(adapter)["blocks"].submodules["attn"]
        for slot in ("q", "k", "v", "o"):
            assert isinstance(attn.submodules[slot], LinearBridge)

    def test_mlp_gate_in_out_are_linear_bridges(self, adapter: StableLmArchitectureAdapter) -> None:
        mlp = _mapping(adapter)["blocks"].submodules["mlp"]
        for slot in ("gate", "in", "out"):
            assert isinstance(mlp.submodules[slot], LinearBridge)


class TestStableLMBlockSubmodulesDefault:
    """Default branch (`parallel_attn_mlp=False`): sequential residual with separate ln1
    and ln2 norms. The block carries four submodules: ln1, ln2, attn, mlp."""

    def test_default_block_submodule_keys(self, adapter: StableLmArchitectureAdapter) -> None:
        block = _mapping(adapter)["blocks"]
        # Strict type identity: the sequential branch must NOT pick ParallelBlockBridge
        # (which is a BlockBridge subclass), so a plain `isinstance` would not discriminate.
        assert type(block) is BlockBridge
        assert set(block.submodules.keys()) == {"ln1", "ln2", "attn", "mlp"}

    def test_ln2_uses_post_attention_layernorm_hf_name(
        self, adapter: StableLmArchitectureAdapter
    ) -> None:
        block = _mapping(adapter)["blocks"]
        assert block.submodules["ln2"].name == "post_attention_layernorm"

    def test_ln1_uses_input_layernorm_hf_name(self, adapter: StableLmArchitectureAdapter) -> None:
        block = _mapping(adapter)["blocks"]
        assert block.submodules["ln1"].name == "input_layernorm"


class TestStableLMBlockSubmodulesParallelResidual:
    """Parallel-residual branch (`parallel_attn_mlp=True`): both attn and MLP read from
    ln1's output, so HF sets post_attention_layernorm=None. The block carries three
    submodules (ln1, attn, mlp) inside a `ParallelBlockBridge` container."""

    @pytest.fixture
    def parallel_adapter(self) -> StableLmArchitectureAdapter:
        return StableLmArchitectureAdapter(_cfg(parallel_attn_mlp=True))

    def test_parallel_block_submodule_keys(
        self, parallel_adapter: StableLmArchitectureAdapter
    ) -> None:
        """Container is ParallelBlockBridge so the no-ln2 layout (attn + mlp without ln2) is the supported shape that BlockBridge would otherwise reject."""
        block = _mapping(parallel_adapter)["blocks"]
        assert isinstance(block, ParallelBlockBridge)
        assert set(block.submodules.keys()) == {"ln1", "attn", "mlp"}


class TestStableLMGQAHookShapes:
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
        adapter = StableLmArchitectureAdapter(_cfg(n_key_value_heads=self.N_KV_HEADS))
        fake_attn = FakeStableLMAttention(adapter.cfg)
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


class TestStableLMSetupHookCompatibility:
    """`setup_hook_compatibility` injects `hook_q_layernorm` and `hook_k_layernorm` HookPoints
    onto every attention bridge whose underlying HF attention has `qk_layernorm=True`
    (StableLM v2 models like stablelm-2-12b). The hooks are added as bridge submodules,
    registered in `bridge._hook_registry` with canonical TL-style names, and the HF
    q_layernorm/k_layernorm forward methods are wrapped to fire them.

    Coverage: every branch of the override, including the no-op-when-disabled path and
    each defensive guard.
    """

    @pytest.fixture
    def adapter_only(self) -> StableLmArchitectureAdapter:
        return StableLmArchitectureAdapter(_cfg())

    def _hook_bridge(self, blocks: list[Any]) -> _DummyHookBridge:
        return _DummyHookBridge(blocks)

    def test_no_op_when_bridge_has_no_blocks(
        self, adapter_only: StableLmArchitectureAdapter
    ) -> None:
        """Guard branch: a bridge without `.blocks` must not raise."""
        bridge = SimpleNamespace()  # no blocks attribute at all
        adapter_only.setup_hook_compatibility(bridge)

    def test_no_op_when_qk_layernorm_disabled(
        self, adapter_only: StableLmArchitectureAdapter
    ) -> None:
        """Stock StableLM (v1) has qk_layernorm=False, so no hooks are injected."""
        hf_attn = _DummyHFAttn(qk_layernorm=False)
        attn_bridge = _DummyAttnBridge(original_component=hf_attn)
        bridge = self._hook_bridge([_DummyBlockWithAttn(attn_bridge)])

        adapter_only.setup_hook_compatibility(bridge)

        assert not hasattr(attn_bridge, "hook_q_layernorm")
        assert not hasattr(attn_bridge, "hook_k_layernorm")
        assert bridge._hook_registry == {}

    def test_skips_block_without_attn(self, adapter_only: StableLmArchitectureAdapter) -> None:
        """A block with no `attn` attribute must be silently skipped."""
        hf_attn = _DummyHFAttn(qk_layernorm=True)
        attn_bridge_with = _DummyAttnBridge(original_component=hf_attn)
        bridge = self._hook_bridge([_DummyBlockNoAttn(), _DummyBlockWithAttn(attn_bridge_with)])

        adapter_only.setup_hook_compatibility(bridge)

        # Block 1 still received hooks, block 0 had no attn and was skipped without error.
        assert hasattr(attn_bridge_with, "hook_q_layernorm")
        assert "blocks.1.attn.hook_q_layernorm" in bridge._hook_registry

    def test_skips_block_with_none_original_component(
        self, adapter_only: StableLmArchitectureAdapter
    ) -> None:
        """When the attention bridge has no original_component (not yet built), skip
        rather than raise."""
        attn_bridge = _DummyAttnBridge(original_component=None)
        bridge = self._hook_bridge([_DummyBlockWithAttn(attn_bridge)])

        adapter_only.setup_hook_compatibility(bridge)

        assert not hasattr(attn_bridge, "hook_q_layernorm")
        assert bridge._hook_registry == {}

    def test_adds_hook_modules_to_attn_bridge(
        self, adapter_only: StableLmArchitectureAdapter
    ) -> None:
        """Happy path: HookPoints are added as submodules of the attention bridge."""
        hf_attn = _DummyHFAttn(qk_layernorm=True)
        attn_bridge = _DummyAttnBridge(original_component=hf_attn)
        bridge = self._hook_bridge([_DummyBlockWithAttn(attn_bridge)])

        adapter_only.setup_hook_compatibility(bridge)

        assert isinstance(attn_bridge.hook_q_layernorm, HookPoint)
        assert isinstance(attn_bridge.hook_k_layernorm, HookPoint)

    def test_registers_hooks_in_bridge_hook_registry_with_canonical_names(
        self, adapter_only: StableLmArchitectureAdapter
    ) -> None:
        """Hooks are registered under TL-canonical names so the registry scanner can find them
        (the scanner skips _original_component subtrees, so the adapter wires registry entries
        directly)."""
        blocks = []
        for _ in range(3):
            hf_attn = _DummyHFAttn(qk_layernorm=True)
            attn_bridge = _DummyAttnBridge(original_component=hf_attn)
            blocks.append(_DummyBlockWithAttn(attn_bridge))
        bridge = self._hook_bridge(blocks)

        adapter_only.setup_hook_compatibility(bridge)

        expected = set()
        for i in range(3):
            expected.add(f"blocks.{i}.attn.hook_q_layernorm")
            expected.add(f"blocks.{i}.attn.hook_k_layernorm")
        assert set(bridge._hook_registry.keys()) == expected
        # And every registry entry points at the same HookPoint object on the bridge.
        for i, block in enumerate(blocks):
            assert (
                bridge._hook_registry[f"blocks.{i}.attn.hook_q_layernorm"]
                is block.attn.hook_q_layernorm
            )
            assert (
                bridge._hook_registry[f"blocks.{i}.attn.hook_k_layernorm"]
                is block.attn.hook_k_layernorm
            )

    def test_q_layernorm_forward_wrap_fires_hook(
        self, adapter_only: StableLmArchitectureAdapter
    ) -> None:
        """Behavioral assertion: calling the HF q_layernorm.forward fires the new hook
        with the layernorm output. Without the wrap, the hook would never run."""
        hf_attn = _DummyHFAttn(qk_layernorm=True)
        attn_bridge = _DummyAttnBridge(original_component=hf_attn)
        bridge = self._hook_bridge([_DummyBlockWithAttn(attn_bridge)])
        adapter_only.setup_hook_compatibility(bridge)

        captured: dict[str, torch.Tensor] = {}

        def _capture(x: torch.Tensor, hook: HookPoint) -> torch.Tensor:
            captured["q"] = x.detach()
            return x

        attn_bridge.hook_q_layernorm.add_hook(_capture)
        x = randn(2, 4, 16)
        out = hf_attn.q_layernorm.forward(x)
        # nn.Identity passes through, so the hook saw exactly x.
        assert "q" in captured
        assert torch.equal(captured["q"], x)
        assert torch.equal(out, x)

    def test_k_layernorm_forward_wrap_fires_hook(
        self, adapter_only: StableLmArchitectureAdapter
    ) -> None:
        """Same behavioral assertion for the K-side wrap."""
        hf_attn = _DummyHFAttn(qk_layernorm=True)
        attn_bridge = _DummyAttnBridge(original_component=hf_attn)
        bridge = self._hook_bridge([_DummyBlockWithAttn(attn_bridge)])
        adapter_only.setup_hook_compatibility(bridge)

        captured: dict[str, torch.Tensor] = {}

        def _capture(x: torch.Tensor, hook: HookPoint) -> torch.Tensor:
            captured["k"] = x.detach()
            return x

        attn_bridge.hook_k_layernorm.add_hook(_capture)
        x = randn(2, 4, 16)
        out = hf_attn.k_layernorm.forward(x)
        assert "k" in captured
        assert torch.equal(captured["k"], x)
        assert torch.equal(out, x)


class TestStableLMSetupComponentTesting:
    """`setup_component_testing` wires the shared rotary embedding onto the template
    attention bridge and onto each bridge-model block's attention. It also forces eager
    attention on the HF model (top-level config and per-layer self_attn.config) for
    numerical parity with the bridge's eager-mode reimplementation."""

    def test_sets_rotary_emb_on_template_attention(
        self, adapter: StableLmArchitectureAdapter
    ) -> None:
        rotary_emb = object()
        attn_template = adapter.get_generalized_component("blocks.0.attn")
        assert isinstance(attn_template, PositionEmbeddingsAttentionBridge)

        adapter.setup_component_testing(_fake_hf_model(rotary_emb))

        assert attn_template._rotary_emb is rotary_emb

    def test_sets_rotary_emb_on_each_bridge_model_attention(
        self, adapter: StableLmArchitectureAdapter
    ) -> None:
        rotary_emb = object()
        bridge_model = DummyBridgeModel([DummyBlock(), DummyBlock(), DummyBlock()])

        adapter.setup_component_testing(_fake_hf_model(rotary_emb), bridge_model=bridge_model)

        for block in bridge_model.blocks:
            assert block.attn.rotary_emb is rotary_emb

    def test_skips_bridge_blocks_without_attention(
        self, adapter: StableLmArchitectureAdapter
    ) -> None:
        rotary_emb = object()
        bridge_model = DummyBridgeModel([DummyBlock(), DummyBlock(has_attention=False)])

        adapter.setup_component_testing(_fake_hf_model(rotary_emb), bridge_model=bridge_model)

        assert bridge_model.blocks[0].attn.rotary_emb is rotary_emb

    def test_forces_eager_attention_implementation(
        self, adapter: StableLmArchitectureAdapter
    ) -> None:
        """Bridge attention only matches HF under eager attention, so it is forced on
        at both the top-level config and on each per-layer self_attn.config."""
        hf_model = _fake_hf_model_with_eager_targets(object())

        adapter.setup_component_testing(hf_model)

        assert hf_model.config._attn_implementation == "eager"
        for layer in hf_model.model.layers:
            assert layer.self_attn.config._attn_implementation == "eager"

    def test_tolerates_minimal_hf_model_without_config_or_layers(
        self, adapter: StableLmArchitectureAdapter
    ) -> None:
        """The defensive hasattr branches must not raise when config/layers are absent."""
        rotary_emb = object()
        # _fake_hf_model exposes only model.rotary_emb (no config, no layers).
        adapter.setup_component_testing(_fake_hf_model(rotary_emb))

        attn_template = adapter.get_generalized_component("blocks.0.attn")
        assert isinstance(attn_template, PositionEmbeddingsAttentionBridge)
        assert attn_template._rotary_emb is rotary_emb
