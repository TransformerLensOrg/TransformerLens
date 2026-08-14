"""Unit tests for OpenElmArchitectureAdapter.

Tests cover:
- Config attribute validation (all required attributes are set correctly)
- Component mapping structure (correct bridge types and HF module names)
- Weight conversion keys (empty for OpenELM — native attention handles all variants)
"""

import pytest
import torch

from transformer_lens.config import TransformerBridgeConfig
from transformer_lens.model_bridge.generalized_components import (
    AttentionBridge,
    BlockBridge,
    EmbeddingBridge,
    LinearBridge,
    MLPBridge,
    RMSNormalizationBridge,
    UnembeddingBridge,
)
from transformer_lens.model_bridge.supported_architectures.openelm import (
    OpenElmArchitectureAdapter,
)

# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------


def _make_cfg(
    n_heads: int = 4,
    d_model: int = 64,
    n_layers: int = 2,
    d_mlp: int = 256,
    d_vocab: int = 1000,
    n_ctx: int = 512,
) -> TransformerBridgeConfig:
    """Return a minimal TransformerBridgeConfig for OpenELM adapter tests."""
    return TransformerBridgeConfig(
        d_model=d_model,
        d_head=d_model // n_heads,
        n_layers=n_layers,
        n_ctx=n_ctx,
        n_heads=n_heads,
        d_vocab=d_vocab,
        d_mlp=d_mlp,
        default_prepend_bos=True,
        architecture="OpenELMForCausalLM",
    )


@pytest.fixture
def cfg() -> TransformerBridgeConfig:
    return _make_cfg()


@pytest.fixture
def adapter(cfg: TransformerBridgeConfig) -> OpenElmArchitectureAdapter:
    return OpenElmArchitectureAdapter(cfg)


# ---------------------------------------------------------------------------
# Config attribute tests
# ---------------------------------------------------------------------------


class TestOpenElmAdapterConfig:
    """Anti-drift flags and non-obvious config choices for OpenElmArchitectureAdapter."""

    def test_final_rms_is_true(self, adapter: OpenElmArchitectureAdapter) -> None:
        assert adapter.cfg.final_rms is True

    def test_uses_rms_norm_is_true(self, adapter: OpenElmArchitectureAdapter) -> None:
        assert adapter.cfg.uses_rms_norm is True

    def test_tokenizer_name(self, adapter: OpenElmArchitectureAdapter) -> None:
        """OpenELM has no bundled tokenizer — uses LLaMA-2 tokenizer as proxy."""
        assert adapter.cfg.tokenizer_name == "NousResearch/Llama-2-7b-hf"


# ---------------------------------------------------------------------------
# Component mapping structure tests
# ---------------------------------------------------------------------------


class TestOpenElmAdapterComponentMapping:
    """Tests that component_mapping has the correct bridge types and HF module names."""

    # -- Top-level keys --

    def test_embed_is_embedding_bridge(self, adapter: OpenElmArchitectureAdapter) -> None:
        assert isinstance(adapter.component_mapping["embed"], EmbeddingBridge)

    def test_embed_name(self, adapter: OpenElmArchitectureAdapter) -> None:
        assert adapter.component_mapping["embed"].name == "transformer.token_embeddings"

    def test_no_pos_embed_key(self, adapter: OpenElmArchitectureAdapter) -> None:
        """OpenELM uses per-layer rotary embeddings — no shared positional embedding."""
        assert "pos_embed" not in adapter.component_mapping

    def test_no_rotary_emb_key(self, adapter: OpenElmArchitectureAdapter) -> None:
        """OpenELM RoPE is embedded per-layer in attention, not a top-level component."""
        assert "rotary_emb" not in adapter.component_mapping

    def test_blocks_is_block_bridge(self, adapter: OpenElmArchitectureAdapter) -> None:
        assert isinstance(adapter.component_mapping["blocks"], BlockBridge)

    def test_blocks_name(self, adapter: OpenElmArchitectureAdapter) -> None:
        assert adapter.component_mapping["blocks"].name == "transformer.layers"

    def test_ln_final_is_rms_normalization_bridge(
        self, adapter: OpenElmArchitectureAdapter
    ) -> None:
        assert isinstance(adapter.component_mapping["ln_final"], RMSNormalizationBridge)

    def test_ln_final_name(self, adapter: OpenElmArchitectureAdapter) -> None:
        assert adapter.component_mapping["ln_final"].name == "transformer.norm"

    def test_unembed_is_unembedding_bridge(self, adapter: OpenElmArchitectureAdapter) -> None:
        assert isinstance(adapter.component_mapping["unembed"], UnembeddingBridge)

    def test_unembed_name(self, adapter: OpenElmArchitectureAdapter) -> None:
        assert adapter.component_mapping["unembed"].name == "lm_head"

    # -- Block submodules --

    def test_blocks_ln1_is_rms_normalization_bridge(
        self, adapter: OpenElmArchitectureAdapter
    ) -> None:
        assert isinstance(
            adapter.component_mapping["blocks"].submodules["ln1"], RMSNormalizationBridge
        )

    def test_blocks_ln1_name(self, adapter: OpenElmArchitectureAdapter) -> None:
        assert adapter.component_mapping["blocks"].submodules["ln1"].name == "attn_norm"

    def test_blocks_ln2_is_rms_normalization_bridge(
        self, adapter: OpenElmArchitectureAdapter
    ) -> None:
        assert isinstance(
            adapter.component_mapping["blocks"].submodules["ln2"], RMSNormalizationBridge
        )

    def test_blocks_ln2_name(self, adapter: OpenElmArchitectureAdapter) -> None:
        assert adapter.component_mapping["blocks"].submodules["ln2"].name == "ffn_norm"

    def test_attn_is_attention_bridge(self, adapter: OpenElmArchitectureAdapter) -> None:
        blocks = adapter.component_mapping["blocks"]
        assert isinstance(blocks.submodules["attn"], AttentionBridge)

    def test_attn_name(self, adapter: OpenElmArchitectureAdapter) -> None:
        blocks = adapter.component_mapping["blocks"]
        assert blocks.submodules["attn"].name == "attn"

    def test_attn_requires_attention_mask(self, adapter: OpenElmArchitectureAdapter) -> None:
        attn = adapter.component_mapping["blocks"].submodules["attn"]
        assert attn.requires_attention_mask is True

    def test_attn_qkv_is_linear_bridge(self, adapter: OpenElmArchitectureAdapter) -> None:
        """OpenELM uses a combined QKV projection (not separate q/k/v)."""
        attn = adapter.component_mapping["blocks"].submodules["attn"]
        assert isinstance(attn.submodules["qkv"], LinearBridge)

    def test_attn_qkv_name(self, adapter: OpenElmArchitectureAdapter) -> None:
        attn = adapter.component_mapping["blocks"].submodules["attn"]
        assert attn.submodules["qkv"].name == "qkv_proj"

    def test_attn_o_is_linear_bridge(self, adapter: OpenElmArchitectureAdapter) -> None:
        attn = adapter.component_mapping["blocks"].submodules["attn"]
        assert isinstance(attn.submodules["o"], LinearBridge)

    def test_attn_o_name(self, adapter: OpenElmArchitectureAdapter) -> None:
        attn = adapter.component_mapping["blocks"].submodules["attn"]
        assert attn.submodules["o"].name == "out_proj"

    def test_mlp_is_mlp_bridge(self, adapter: OpenElmArchitectureAdapter) -> None:
        blocks = adapter.component_mapping["blocks"]
        assert isinstance(blocks.submodules["mlp"], MLPBridge)

    def test_mlp_name(self, adapter: OpenElmArchitectureAdapter) -> None:
        """OpenELM names its MLP submodule 'ffn' (feedforward network)."""
        assert adapter.component_mapping["blocks"].submodules["mlp"].name == "ffn"

    def test_mlp_gate_and_in_are_synthesized_from_the_fused_proj_1(
        self, adapter: OpenElmArchitectureAdapter
    ) -> None:
        """proj_1 is a fused gate+up projection, so there is no submodule that
        binds it directly: the bridge splits it and registers "gate"/"in" as
        synthesized halves. Binding "in" to proj_1 (as before) made hook_pre the
        concatenated pre-GLU tensor."""
        mlp = adapter.component_mapping["blocks"].submodules["mlp"]
        assert {"gate", "in"} <= set(mlp.submodules)
        assert mlp.submodules["in"].name == "in"
        assert mlp.submodules["gate"].name == "gate"

    def test_mlp_out_name(self, adapter: OpenElmArchitectureAdapter) -> None:
        mlp = adapter.component_mapping["blocks"].submodules["mlp"]
        assert mlp.submodules["out"].name == "proj_2"


# ---------------------------------------------------------------------------
# Weight processing conversion tests
# ---------------------------------------------------------------------------


class TestOpenElmAdapterWeightConversions:
    """Tests that weight_processing_conversions is empty for OpenELM.

    OpenELM uses per-layer varying head counts and FFN dimensions handled
    entirely by native HuggingFace attention — no static weight rearrangements
    are needed at the bridge level.
    """

    def test_no_weight_processing_conversions(self, adapter: OpenElmArchitectureAdapter) -> None:
        assert len(adapter.weight_processing_conversions) == 0


class TestOpenElmFusedProjections:
    """OpenELM fuses both QKV and gate+up, and the mapping ignored both.

    `qkv_proj` is a single module, so the inherited hook_q/hook_k/hook_v aliases
    could never resolve — they showed up as dead aliases in the resolution audit
    (and per-layer head counts mean a uniform split would be wrong anyway).
    `proj_1` fuses gate and up, so a plain MLPBridge served `hook_pre` as the
    concatenated pre-GLU tensor with no `hook_pre_linear` at all.
    """

    @staticmethod
    def _adapter():
        from tests.unit.model_bridge.supported_architectures.helpers import (
            make_bridge_cfg,
        )
        from transformer_lens.factories.architecture_adapter_factory import (
            ArchitectureAdapterFactory,
        )

        return ArchitectureAdapterFactory.select_architecture_adapter(
            make_bridge_cfg("OpenELMForCausalLM", d_head=8)
        )

    def test_dead_qkv_aliases_are_stripped(self) -> None:
        attn = self._adapter().component_mapping["blocks"].submodules["attn"]
        for dead in ("hook_q", "hook_k", "hook_v"):
            assert dead not in attn.hook_aliases, f"{dead} cannot resolve on a fused qkv_proj"
        # hook_z stays: it resolves through out_proj.
        assert attn.hook_aliases["hook_z"] == "o.hook_in"

    def test_fused_gate_up_exposes_the_neuron_basis(self) -> None:
        from transformer_lens.model_bridge.generalized_components import (
            JointGateUpMLPBridge,
        )

        mlp = self._adapter().component_mapping["blocks"].submodules["mlp"]
        # Alias constants live on the class; resolution is covered by
        # test_hook_alias_resolution.py and behavior by the parity tests below.
        assert isinstance(mlp, JointGateUpMLPBridge)
        assert mlp.submodules["out"].name == "proj_2"

    def test_proj_1_splits_gate_first(self) -> None:
        """HF's OpenELMFeedForwardNetwork applies SiLU to the FIRST chunk, so
        the first half is the gate. Shapes cannot catch a swap — both halves are
        [d_mlp, d_model] — so this pins the halves by value."""
        import torch

        from transformer_lens.model_bridge.supported_architectures.openelm import (
            OpenElmArchitectureAdapter,
        )

        class _FusedFFN(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.proj_1 = torch.nn.Linear(8, 32, bias=False)
                self.proj_2 = torch.nn.Linear(16, 8, bias=False)

        module = _FusedFFN()
        mlp = self._adapter().component_mapping["blocks"].submodules["mlp"]
        gate, up = mlp.split_gate_up_matrix(module)
        fused = module.proj_1.weight
        torch.testing.assert_close(gate.weight, fused[:16])
        torch.testing.assert_close(up.weight, fused[16:])
        assert not torch.allclose(gate.weight, up.weight)


class TestOpenElmFFNNumericalParity:
    """The reconstructed FFN must match HF numerically, not just structurally.

    JointGateUpMLPBridge has no delegation branch, so it always recomputes
    act(gate)*up — and both activation sources missed OpenELM: the module
    stores `self.act` (probe expected activation_fn/act_fn) and the config
    spells it `activation_fn_name` (conversion probed four other names). The
    silent fallback was cfg.act_fn's "relu" default: ~30% FFN divergence on
    every load, invisible to structural tests.
    """

    class _FFN(torch.nn.Module):
        """OpenELM-shaped: fused proj_1, `self.act`, proj_2."""

        def __init__(self) -> None:
            super().__init__()
            self.proj_1 = torch.nn.Linear(8, 32, bias=False)
            self.proj_2 = torch.nn.Linear(16, 8, bias=False)
            self.act = torch.nn.SiLU()

        def forward(self, x):
            gate, up = self.proj_1(x).chunk(2, dim=-1)
            return self.proj_2(self.act(gate) * up)

    def _bridge(self, module):
        import copy

        from tests.unit.model_bridge.supported_architectures.helpers import (
            make_bridge_cfg,
        )
        from transformer_lens.factories.architecture_adapter_factory import (
            ArchitectureAdapterFactory,
        )
        from transformer_lens.model_bridge.component_setup import setup_submodules

        adapter = ArchitectureAdapterFactory.select_architecture_adapter(
            make_bridge_cfg("OpenELMForCausalLM", d_model=8, n_heads=2, d_head=4)
        )
        bridge = copy.deepcopy(adapter.component_mapping["blocks"].submodules["mlp"])
        bridge.set_original_component(module)
        setup_submodules(bridge, adapter, module)
        return adapter, bridge

    def test_reconstructed_ffn_matches_hf(self) -> None:
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(0)
            module = self._FFN()
            x = torch.randn(2, 5, 8)
        _, bridge = self._bridge(module)
        with torch.no_grad():
            torch.testing.assert_close(bridge(x), module(x))
        # The fixture must be able to tell silu from relu, or the parity
        # assertion above cannot catch the activation falling back.
        with torch.no_grad():
            gate, up = module.proj_1(x).chunk(2, dim=-1)
            silu_out = module.proj_2(torch.nn.functional.silu(gate) * up)
            relu_out = module.proj_2(torch.nn.functional.relu(gate) * up)
        assert not torch.allclose(silu_out, relu_out, atol=1e-4)

    def test_module_activation_is_captured(self) -> None:
        module = self._FFN()
        _, bridge = self._bridge(module)
        assert bridge._activation_fn is module.act

    def test_config_maps_activation_fn_name(self) -> None:
        """The config half: cfg.act_fn must not silently keep its 'relu'
        default when HF spells the field activation_fn_name (OpenELM)."""
        from transformers import PretrainedConfig

        from transformer_lens.model_bridge.sources.transformers import (
            map_default_transformer_lens_config,
        )

        hf_config = PretrainedConfig(hidden_size=8, activation_fn_name="swish")
        tl_config = map_default_transformer_lens_config(hf_config)
        assert tl_config.act_fn == "swish"
        # resolve_activation_fn treats "swish" as silu; pin that too, or the
        # mapping could land on a name the resolver does not know.
        from transformer_lens.model_bridge.generalized_components.gated_mlp import (
            resolve_activation_fn,
        )

        assert resolve_activation_fn(tl_config) is torch.nn.functional.silu


def test_ungated_ffn_is_refused_at_boot() -> None:
    """ffn_with_glu=False would make the fused split halve a non-fused
    projection — boot succeeded, then the first forward crashed in a mat-mul
    far from the cause. Unreachable on published checkpoints (default True),
    so the guard is the only thing standing between the config and the crash."""
    import types

    adapter = OpenElmArchitectureAdapter(_make_cfg())

    class _UngatedFFN(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.ffn_with_glu = False
            self.proj_1 = torch.nn.Linear(8, 16, bias=False)
            self.proj_2 = torch.nn.Linear(16, 8, bias=False)

    model = torch.nn.Module()
    model.add_module("ffn", _UngatedFFN())
    model.config = types.SimpleNamespace(attn_implementation="eager")

    with pytest.raises(NotImplementedError, match="ffn_with_glu"):
        adapter.prepare_model(model)


def test_prepare_loading_installs_dynamic_cache_compat(monkeypatch) -> None:
    """apple's frozen remote code calls DynamicCache.from_legacy_cache /
    to_legacy_cache (removed in transformers>=5); generation died on the first
    step because OpenELM's prepare_loading never installed the shared compat
    that phi3/internlm2/baichuan already use."""
    import transformer_lens.model_bridge.supported_architectures.openelm as openelm_mod

    calls: list = []
    monkeypatch.setattr(openelm_mod, "patch_dynamic_cache_v5", lambda: calls.append(True))
    OpenElmArchitectureAdapter(_make_cfg()).prepare_loading("apple/OpenELM-270M", {})
    assert calls, "prepare_loading must install the DynamicCache v5 compat"
