"""Unit tests for OptArchitectureAdapter.

Tests cover:
- Config attribute validation
- Post-norm support flags
- Weight conversion keys
- Component mapping structure
- OPT-350m projection mapping
"""

import pytest
import torch

from transformer_lens.config import TransformerBridgeConfig
from transformer_lens.model_bridge.generalized_components import (
    AttentionBridge,
    BlockBridge,
    EmbeddingBridge,
    LinearBridge,
    NormalizationBridge,
    PosEmbedBridge,
    UnembeddingBridge,
)
from transformer_lens.model_bridge.supported_architectures.opt import (
    OptArchitectureAdapter,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_cfg(
    n_heads: int = 4,
    d_model: int = 64,
    n_layers: int = 2,
    d_mlp: int = 256,
    d_vocab: int = 1000,
    n_ctx: int = 512,
    do_layer_norm_before: bool = True,
    word_embed_proj_dim: int | None = None,
) -> TransformerBridgeConfig:
    """Return a minimal TransformerBridgeConfig for OPT adapter tests."""
    cfg = TransformerBridgeConfig(
        d_model=d_model,
        d_head=d_model // n_heads,
        n_layers=n_layers,
        n_ctx=n_ctx,
        n_heads=n_heads,
        d_vocab=d_vocab,
        d_mlp=d_mlp,
        default_prepend_bos=True,
        architecture="OPTForCausalLM",
    )
    cfg.do_layer_norm_before = do_layer_norm_before
    if word_embed_proj_dim is not None:
        cfg.word_embed_proj_dim = word_embed_proj_dim
    return cfg


@pytest.fixture
def cfg() -> TransformerBridgeConfig:
    return _make_cfg()


@pytest.fixture
def adapter(cfg: TransformerBridgeConfig) -> OptArchitectureAdapter:
    return OptArchitectureAdapter(cfg)


# ---------------------------------------------------------------------------
# Config attribute tests
# ---------------------------------------------------------------------------


class TestOptAdapterConfig:
    """Adapter must set all required config attributes to the correct values."""

    def test_normalization_type_is_ln(self, adapter: OptArchitectureAdapter) -> None:
        assert adapter.cfg.normalization_type == "LN"

    def test_standard_pos_embedding_omits_rotary_attn_hooks(
        self, adapter: OptArchitectureAdapter
    ) -> None:
        """Standard (learned) positional embeddings build attention WITHOUT rotary hooks.

        AttentionBridge only creates hook_rot_q/hook_rot_k when
        positional_embedding_type == "rotary"; OPT's "standard" setting must leave
        them off. Flipping OPT to rotary would add these hooks and fail this test.
        """
        attn = adapter.component_mapping["blocks"].submodules["attn"]
        assert not hasattr(attn, "hook_rot_q")
        assert not hasattr(attn, "hook_rot_k")


class TestOptAdapterPostNorm:
    """Post-norm OPT disables transforms that require pre-norm semantics."""

    def test_post_norm_disables_fold_ln(self) -> None:
        adapter = OptArchitectureAdapter(_make_cfg(do_layer_norm_before=False))
        assert adapter.supports_fold_ln is False

    def test_post_norm_disables_center_writing_weights(self) -> None:
        adapter = OptArchitectureAdapter(_make_cfg(do_layer_norm_before=False))
        assert adapter.supports_center_writing_weights is False


# ---------------------------------------------------------------------------
# Weight processing conversion tests
# ---------------------------------------------------------------------------


class TestOptAdapterWeightConversions:
    """Adapter must define exactly the four standard QKVO weight conversions."""

    def test_q_weight_key_present(self, adapter: OptArchitectureAdapter) -> None:
        assert "blocks.{i}.attn.q.weight" in adapter.weight_processing_conversions

    def test_k_weight_key_present(self, adapter: OptArchitectureAdapter) -> None:
        assert "blocks.{i}.attn.k.weight" in adapter.weight_processing_conversions

    def test_v_weight_key_present(self, adapter: OptArchitectureAdapter) -> None:
        assert "blocks.{i}.attn.v.weight" in adapter.weight_processing_conversions

    def test_o_weight_key_present(self, adapter: OptArchitectureAdapter) -> None:
        assert "blocks.{i}.attn.o.weight" in adapter.weight_processing_conversions

    def test_exactly_four_conversion_keys(self, adapter: OptArchitectureAdapter) -> None:
        assert len(adapter.weight_processing_conversions) == 4


# ---------------------------------------------------------------------------
# Component mapping structure tests
# ---------------------------------------------------------------------------


class TestOptAdapterComponentMapping:
    """Component mapping must have the correct bridge types and HF module paths."""

    def test_embed_is_embedding_bridge(self, adapter: OptArchitectureAdapter) -> None:
        assert isinstance(adapter.component_mapping["embed"], EmbeddingBridge)

    def test_embed_name(self, adapter: OptArchitectureAdapter) -> None:
        assert adapter.component_mapping["embed"].name == "model.decoder.embed_tokens"

    def test_pos_embed_is_pos_embed_bridge(self, adapter: OptArchitectureAdapter) -> None:
        assert isinstance(adapter.component_mapping["pos_embed"], PosEmbedBridge)

    def test_pos_embed_name(self, adapter: OptArchitectureAdapter) -> None:
        assert adapter.component_mapping["pos_embed"].name == "model.decoder.embed_positions"

    def test_blocks_is_block_bridge(self, adapter: OptArchitectureAdapter) -> None:
        assert isinstance(adapter.component_mapping["blocks"], BlockBridge)

    def test_blocks_name(self, adapter: OptArchitectureAdapter) -> None:
        assert adapter.component_mapping["blocks"].name == "model.decoder.layers"

    def test_ln_final_is_normalization_bridge(self, adapter: OptArchitectureAdapter) -> None:
        assert isinstance(adapter.component_mapping["ln_final"], NormalizationBridge)

    def test_ln_final_name(self, adapter: OptArchitectureAdapter) -> None:
        assert adapter.component_mapping["ln_final"].name == "model.decoder.final_layer_norm"

    def test_unembed_is_unembedding_bridge(self, adapter: OptArchitectureAdapter) -> None:
        assert isinstance(adapter.component_mapping["unembed"], UnembeddingBridge)

    def test_unembed_name(self, adapter: OptArchitectureAdapter) -> None:
        assert adapter.component_mapping["unembed"].name == "lm_head"

    def test_ln1_is_normalization_bridge(self, adapter: OptArchitectureAdapter) -> None:
        blocks = adapter.component_mapping["blocks"]
        assert isinstance(blocks.submodules["ln1"], NormalizationBridge)

    def test_ln1_name(self, adapter: OptArchitectureAdapter) -> None:
        blocks = adapter.component_mapping["blocks"]
        assert blocks.submodules["ln1"].name == "self_attn_layer_norm"

    def test_attn_is_attention_bridge(self, adapter: OptArchitectureAdapter) -> None:
        blocks = adapter.component_mapping["blocks"]
        assert isinstance(blocks.submodules["attn"], AttentionBridge)

    def test_attn_name(self, adapter: OptArchitectureAdapter) -> None:
        blocks = adapter.component_mapping["blocks"]
        assert blocks.submodules["attn"].name == "self_attn"

    def test_attn_requires_attention_mask(self, adapter: OptArchitectureAdapter) -> None:
        blocks = adapter.component_mapping["blocks"]
        assert blocks.submodules["attn"].requires_attention_mask is True

    def test_attn_attention_mask_4d(self, adapter: OptArchitectureAdapter) -> None:
        blocks = adapter.component_mapping["blocks"]
        assert blocks.submodules["attn"].attention_mask_4d is True

    def test_attn_q_name(self, adapter: OptArchitectureAdapter) -> None:
        attn = adapter.component_mapping["blocks"].submodules["attn"]
        assert attn.submodules["q"].name == "q_proj"

    def test_attn_k_name(self, adapter: OptArchitectureAdapter) -> None:
        attn = adapter.component_mapping["blocks"].submodules["attn"]
        assert attn.submodules["k"].name == "k_proj"

    def test_attn_v_name(self, adapter: OptArchitectureAdapter) -> None:
        attn = adapter.component_mapping["blocks"].submodules["attn"]
        assert attn.submodules["v"].name == "v_proj"

    def test_attn_o_name(self, adapter: OptArchitectureAdapter) -> None:
        attn = adapter.component_mapping["blocks"].submodules["attn"]
        assert attn.submodules["o"].name == "out_proj"

    def test_ln2_is_normalization_bridge(self, adapter: OptArchitectureAdapter) -> None:
        blocks = adapter.component_mapping["blocks"]
        assert isinstance(blocks.submodules["ln2"], NormalizationBridge)

    def test_ln2_name(self, adapter: OptArchitectureAdapter) -> None:
        blocks = adapter.component_mapping["blocks"]
        assert blocks.submodules["ln2"].name == "final_layer_norm"

    def test_mlp_exposes_neuron_hooks(self, adapter: OptArchitectureAdapter) -> None:
        """MLPBridge(name=None), not SymbolicBridge: fc1/fc2 sit directly on the
        block with no MLP container, and SymbolicBridge exposed no
        hook_pre/hook_post. Asserting the aliases rather than the class is
        the point — the class is only the means to those hooks."""
        blocks = adapter.component_mapping["blocks"]
        mlp = blocks.submodules["mlp"]
        assert mlp.name is None, "containerless: fc1/fc2 are block attributes"
        assert mlp.hook_aliases == {"hook_pre": "in.hook_out", "hook_post": "out.hook_in"}
        # fc2 IS the mlp output, so the block alias must reach it; otherwise
        # hook_mlp_out targets a HookPoint that never fires.
        assert blocks.hook_aliases["hook_mlp_out"] == "mlp.out.hook_out"

    def test_mlp_fc1_name(self, adapter: OptArchitectureAdapter) -> None:
        mlp = adapter.component_mapping["blocks"].submodules["mlp"]
        assert mlp.submodules["in"].name == "fc1"

    def test_mlp_fc2_name(self, adapter: OptArchitectureAdapter) -> None:
        mlp = adapter.component_mapping["blocks"].submodules["mlp"]
        assert mlp.submodules["out"].name == "fc2"


# ---------------------------------------------------------------------------
# OPT-350m special path tests
# ---------------------------------------------------------------------------


class TestOpt350mProjectionMapping:
    """OPT-350m uses project_in/project_out instead of final_layer_norm."""

    def test_project_bridges_absent_in_standard_path(self, adapter: OptArchitectureAdapter) -> None:
        assert "project_in" not in adapter.component_mapping
        assert "project_out" not in adapter.component_mapping

    def test_ln_final_absent_when_word_embed_proj_dim_differs(self) -> None:
        adapter = OptArchitectureAdapter(_make_cfg(d_model=64, word_embed_proj_dim=32))
        assert "ln_final" not in adapter.component_mapping

    def test_project_in_present_when_word_embed_proj_dim_differs(self) -> None:
        adapter = OptArchitectureAdapter(_make_cfg(d_model=64, word_embed_proj_dim=32))
        assert isinstance(adapter.component_mapping["project_in"], LinearBridge)
        assert adapter.component_mapping["project_in"].name == "model.decoder.project_in"

    def test_project_out_present_when_word_embed_proj_dim_differs(self) -> None:
        adapter = OptArchitectureAdapter(_make_cfg(d_model=64, word_embed_proj_dim=32))
        assert isinstance(adapter.component_mapping["project_out"], LinearBridge)
        assert adapter.component_mapping["project_out"].name == "model.decoder.project_out"


class TestOptMLPHookShapes:
    """OPT reshapes to [batch*seq, d] around ln2/fc1/fc2, so the newly-live MLP
    hooks fired flattened — silently wrong for position-indexed patching,
    crashing for `b s d` einops.
    """

    D_MODEL, BATCH, SEQ = 16, 2, 5

    class _FlatteningLayer(torch.nn.Module):
        """OPTDecoderLayer's FFN half, with the real 2D reshape."""

        def __init__(self, d_model):
            super().__init__()
            # Attention half present only so block setup can bind submodules;
            # this fake's forward exercises the FFN half alone.
            self.self_attn_layer_norm = torch.nn.LayerNorm(d_model)
            self.self_attn = torch.nn.Module()
            for name in ("q_proj", "k_proj", "v_proj", "out_proj"):
                self.self_attn.add_module(name, torch.nn.Linear(d_model, d_model))
            self.final_layer_norm = torch.nn.LayerNorm(d_model)
            self.fc1 = torch.nn.Linear(d_model, 4 * d_model)
            self.fc2 = torch.nn.Linear(4 * d_model, d_model)

        def forward(self, hidden_states, **kwargs):
            # Mirrors modeling_opt exactly: residual captured 2D, added 2D,
            # reshaped back only at the end. The 2D residual is load-bearing
            # for the tests — a hook that returns 3D without the revert step
            # would broadcast-crash here, as it does in real OPT.
            shape = hidden_states.shape
            hidden_states = hidden_states.reshape(-1, hidden_states.size(-1))
            residual = hidden_states
            hidden_states = self.final_layer_norm(hidden_states)
            hidden_states = self.fc2(torch.nn.functional.relu(self.fc1(hidden_states)))
            return ((residual + hidden_states).view(shape),)

    def _wired_block(self):
        import copy

        from transformer_lens.model_bridge.component_setup import setup_submodules

        adapter = OptArchitectureAdapter(_make_cfg())
        block = copy.deepcopy(adapter.component_mapping["blocks"])
        layer = self._FlatteningLayer(self.D_MODEL)
        block.set_original_component(layer)
        setup_submodules(block, adapter, layer)
        return block

    def test_mlp_hooks_fire_3d(self) -> None:
        block = self._wired_block()
        mlp = block.submodules["mlp"]
        seen: dict = {}
        mlp.submodules["in"].hook_out.add_hook(
            lambda t, hook: seen.__setitem__("pre", tuple(t.shape))
        )
        mlp.submodules["out"].hook_out.add_hook(
            lambda t, hook: seen.__setitem__("mlp_out", tuple(t.shape))
        )
        block.submodules["ln2"].hook_out.add_hook(
            lambda t, hook: seen.__setitem__("ln2", tuple(t.shape))
        )
        with torch.no_grad():
            block(torch.randn(self.BATCH, self.SEQ, self.D_MODEL))

        assert seen["pre"] == (self.BATCH, self.SEQ, 4 * self.D_MODEL), seen
        assert seen["mlp_out"] == (self.BATCH, self.SEQ, self.D_MODEL), seen
        assert seen["ln2"] == (self.BATCH, self.SEQ, self.D_MODEL), seen

    def test_editing_hook_write_lands_positionally(self) -> None:
        """The revert half: a 3D write from a hook must reach HF's 2D math at
        the right positions — this is what flat hooks silently corrupt."""
        block = self._wired_block()
        mlp = block.submodules["mlp"]

        def zero_last_position(t, hook):
            t = t.clone()
            t[:, -1, :] = 0.0
            return t

        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(0)
            x = torch.randn(self.BATCH, self.SEQ, self.D_MODEL)
        # Reference BEFORE attaching the hook — the wrapped modules are shared,
        # so a post-hook "reference" would be edited too.
        with torch.no_grad():
            reference = block(x)[0]
        mlp.submodules["out"].hook_out.add_hook(zero_last_position)
        with torch.no_grad():
            edited = block(x)[0]
        # Positions 0..S-2 must be untouched; the write hits only position -1.
        torch.testing.assert_close(edited[:, :-1], reference[:, :-1])
        assert not torch.allclose(edited[:, -1], reference[:, -1])

    def test_hookless_forward_is_untouched(self) -> None:
        """Conversions must be inert with no hooks attached."""
        block = self._wired_block()
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(1)
            x = torch.randn(self.BATCH, self.SEQ, self.D_MODEL)
        with torch.no_grad():
            torch.testing.assert_close(block(x)[0], block.original_component(x)[0])


def test_containerless_mlp_direct_call_raises() -> None:
    """block.mlp(x) used to silently execute the WHOLE decoder layer —
    component setup binds the parent layer as original_component when
    name=None. The old SymbolicBridge raised here; MLPBridge must too."""
    import copy

    from transformer_lens.model_bridge.component_setup import setup_submodules

    adapter = OptArchitectureAdapter(_make_cfg())
    block = copy.deepcopy(adapter.component_mapping["blocks"])
    layer = TestOptMLPHookShapes._FlatteningLayer(16)
    block.set_original_component(layer)
    setup_submodules(block, adapter, layer)

    with pytest.raises(RuntimeError, match="containerless"):
        block.submodules["mlp"](torch.randn(2, 5, 16))


class TestOptHookMlpInShape:
    """hook_mlp_in fires from a ln2 pre-hook inside the flattened region and is
    the block's own HookPoint, so the submodule stamping missed it: 2D reads,
    broadcast-failing 3D writes.
    """

    def _wired_block(self):
        import copy

        from transformer_lens.model_bridge.component_setup import setup_submodules

        adapter = OptArchitectureAdapter(_make_cfg())
        block = copy.deepcopy(adapter.component_mapping["blocks"])
        layer = TestOptMLPHookShapes._FlatteningLayer(16)
        block.set_original_component(layer)
        setup_submodules(block, adapter, layer)
        # The capture pre-hook is gated on this; the bridge normally toggles it
        # via set_use_hook_mlp_in, which this harness bypasses.
        block._use_hook_mlp_in = True
        return block

    def test_reads_are_3d(self) -> None:
        block = self._wired_block()
        seen: dict = {}
        block.hook_mlp_in.add_hook(lambda t, hook: seen.__setitem__("mlp_in", tuple(t.shape)))
        with torch.no_grad():
            block(torch.randn(2, 5, 16))
        assert seen.get("mlp_in") == (2, 5, 16), seen

    def test_a_3d_write_lands_positionally(self) -> None:
        block = self._wired_block()
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(0)
            x = torch.randn(2, 5, 16)
        with torch.no_grad():
            reference = block(x)[0]

        def zero_last_position(t, hook):
            t = t.clone()
            t[:, -1, :] = 0.0
            return t

        block.hook_mlp_in.add_hook(zero_last_position)
        with torch.no_grad():
            edited = block(x)[0]
        torch.testing.assert_close(edited[:, :-1], reference[:, :-1])
        assert not torch.allclose(edited[:, -1], reference[:, -1])
