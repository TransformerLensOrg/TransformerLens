"""Unit tests for PretrainArchitectureAdapter and DenseOrMoEFeedForwardBridge
-- component mapping, structural dispatch, and bridge-construction-level
behavior. Container, builder, and wrapped-model lifecycle behavior (kwarg
filtering, output-contract normalization, hook registration for the hidden
MLP delegate, train/eval, dtype/device) lives in
test_pretrain_model_container.py.

Fully self-contained (see _pretrain_mocks.py): tiny mock `nn.Module`s, no
external package dependency, no network access.
"""
from __future__ import annotations

import pytest
import torch

from transformer_lens.model_bridge.generalized_components import (
    DelegatedAttentionBlockBridge,
)
from transformer_lens.model_bridge.sources._bridge_builder import (
    build_bridge_from_module,
)
from transformer_lens.model_bridge.supported_architectures.pretrain import (
    ARCHITECTURE_NAME,
    DenseOrMoEFeedForwardBridge,
    NativeForwardAttentionBridge,
    PretrainArchitectureAdapter,
    PretrainModelContainer,
    build_pretrain_bridge,
)

from ._pretrain_mocks import (
    MalformedMLP,
    TinyDenseMLP,
    TinyMoE,
    TinyPretrainModel,
    make_cfg,
)

_make_cfg = make_cfg  # local alias, matches call sites below


class TestPretrainAdapterConstruction:
    def test_adapter_sets_required_config_flags(self) -> None:
        adapter = PretrainArchitectureAdapter(_make_cfg())
        assert adapter.cfg.normalization_type == "RMS"
        assert adapter.cfg.positional_embedding_type == "rotary"
        assert adapter.cfg.final_rms is True
        assert adapter.cfg.gated_mlp is True
        assert adapter.cfg.attn_only is False

    def test_adapter_mutates_the_passed_in_cfg_object_in_place(self) -> None:
        """Documented, intentional behavior (matches nanogpt.py's adapter
        convention) -- not an accidental side effect. This test guards the
        specific claim: the object passed in is the object mutated, not a
        copy."""
        cfg = _make_cfg()
        adapter = PretrainArchitectureAdapter(cfg)
        assert adapter.cfg is cfg
        assert cfg.normalization_type == "RMS"

    def test_component_mapping_has_expected_top_level_keys(self) -> None:
        adapter = PretrainArchitectureAdapter(_make_cfg())
        mapping = adapter.get_component_mapping()
        assert set(mapping.keys()) == {"embed", "blocks", "ln_final", "unembed"}

    def test_mlp_mapping_always_uses_the_dispatcher(self) -> None:
        """Unconditional now -- no cfg.num_experts branch to depend on."""
        adapter = PretrainArchitectureAdapter(_make_cfg())
        mlp_component = adapter.component_mapping["blocks"].submodules["mlp"]
        assert isinstance(mlp_component, DenseOrMoEFeedForwardBridge)


class TestNativeForwardAttentionBridge:
    def test_opaque_attention_bridge_does_not_advertise_per_head_aliases(self) -> None:
        """The opaque attention wrap has no Q/K/V/O submodules, so it must
        not advertise AttentionBridge's class-level hook_aliases/
        property_aliases (which assume those submodules exist) or the
        split-QKV-fork machinery -- see NativeForwardAttentionBridge's
        docstring."""
        adapter = PretrainArchitectureAdapter(_make_cfg())
        attn = adapter.component_mapping["blocks"].submodules["attn"]

        assert isinstance(attn, NativeForwardAttentionBridge)
        assert attn.hook_aliases == {}
        assert attn.property_aliases == {}
        assert attn.supports_split_qkv_fork is False

        # Also check the actual constructed runtime bridge, not just the
        # adapter's own component_mapping -- proves bridge construction
        # didn't reintroduce the aliases along the way.
        cfg = _make_cfg(n_layers=1)
        model = TinyPretrainModel(
            d_model=32, n_heads=4, n_layers=1, d_ff=64, vocab_size=256
        )
        bridge = build_pretrain_bridge(model, cfg)
        assert bridge.blocks[0].attn.hook_aliases == {}

    def test_block_level_attention_output_still_reaches_hook_out(self) -> None:
        """Clearing the per-head aliases must not also silence the block's
        own attn.hook_out -- that's the actual hook point this adapter's
        block-level hook coverage relies on."""
        cfg = _make_cfg(n_layers=1)
        model = TinyPretrainModel(
            d_model=32, n_heads=4, n_layers=1, d_ff=64, vocab_size=256
        )
        bridge = build_pretrain_bridge(model, cfg)
        tokens = torch.randint(0, 256, (1, 5))
        _, cache = bridge.run_with_cache(tokens)
        assert "blocks.0.attn.hook_out" in cache

    def test_blocks_use_delegated_attention_block_bridge(self) -> None:
        """The adapter reuses DelegatedAttentionBlockBridge rather than
        plain BlockBridge, since NativeForwardAttentionBridge's
        supports_split_qkv_fork = False means the block-level
        hook_attn_in/hook_q_input/hook_k_input/hook_v_input aliases would
        otherwise dangle (point at HookPoints that are never created)."""
        adapter = PretrainArchitectureAdapter(_make_cfg())
        assert isinstance(adapter.component_mapping["blocks"], DelegatedAttentionBlockBridge)

    def test_split_qkv_hook_names_are_absent_but_attention_output_remains_available(
        self,
    ) -> None:
        """hook_attn_in/hook_q_input/hook_k_input/hook_v_input must not
        appear as resolvable hook names anywhere in the built bridge --
        DelegatedAttentionBlockBridge strips the block-level aliases, and
        NativeForwardAttentionBridge never creates the underlying
        attention-level HookPoints in the first place. hook_attn_out is
        untouched by either change and must still resolve, both as the
        block-level alias and as the attention component's own hook_out."""
        cfg = _make_cfg(n_layers=1)
        model = TinyPretrainModel(
            d_model=32, n_heads=4, n_layers=1, d_ff=64, vocab_size=256
        )
        bridge = build_pretrain_bridge(model, cfg)
        hook_dict = bridge.hook_dict
        for dangling in (
            "blocks.0.hook_attn_in",
            "blocks.0.hook_q_input",
            "blocks.0.hook_k_input",
            "blocks.0.hook_v_input",
            "blocks.0.attn.hook_attn_in",
            "blocks.0.attn.hook_q_input",
            "blocks.0.attn.hook_k_input",
            "blocks.0.attn.hook_v_input",
        ):
            assert dangling not in hook_dict, f"{dangling} should not resolve"
        assert "blocks.0.hook_attn_out" in hook_dict
        assert "blocks.0.attn.hook_out" in hook_dict


class TestDenseOrMoEFeedForwardBridgeDispatch:
    def test_dispatches_to_moe_delegate_for_moe_module(self) -> None:
        cfg = _make_cfg()
        bridge = DenseOrMoEFeedForwardBridge(name="mlp", config=cfg)
        moe_module = TinyMoE(d_model=32, d_ff=64, n_experts=4, top_k=2)
        bridge.set_original_component(moe_module)

        from transformer_lens.model_bridge.generalized_components import MoEBridge

        assert isinstance(bridge._delegate, MoEBridge)

    def test_dispatches_to_dense_delegate_for_dense_module(self) -> None:
        cfg = _make_cfg()
        bridge = DenseOrMoEFeedForwardBridge(name="mlp", config=cfg)
        dense_module = TinyDenseMLP(d_model=32, d_ff=64)
        bridge.set_original_component(dense_module)

        from transformer_lens.model_bridge.generalized_components import GatedMLPBridge

        assert isinstance(bridge._delegate, GatedMLPBridge)

    def test_raises_clear_error_for_malformed_module(self) -> None:
        cfg = _make_cfg()
        bridge = DenseOrMoEFeedForwardBridge(name="mlp", config=cfg)
        malformed = MalformedMLP(d_model=32)

        with pytest.raises(ValueError, match="doesn't know how to wrap it"):
            bridge.set_original_component(malformed)


class TestDenseOrMoEFeedForwardBridgeDispatchValidatesTypes:
    """hasattr alone would let a module win a branch by attribute-name
    coincidence even if the attributes are the wrong type. These tests
    exercise the basic type checks added to set_original_component so
    such a mismatch is caught here, with a clear message naming the
    actual field and type, rather than failing later inside MoEBridge/
    GatedMLPBridge or during forward."""

    def test_router_of_wrong_type_raises_type_error(self) -> None:
        class BadRouterMoE(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.router = "not_a_module"  # right name, wrong type
                self.experts = torch.nn.ModuleList(
                    [TinyDenseMLP(d_model=32, d_ff=64) for _ in range(4)]
                )

        cfg = _make_cfg()
        bridge = DenseOrMoEFeedForwardBridge(name="mlp", config=cfg)
        with pytest.raises(TypeError, match="router"):
            bridge.set_original_component(BadRouterMoE())

    def test_experts_of_wrong_type_raises_type_error(self) -> None:
        class BadExpertsMoE(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.router = torch.nn.Linear(32, 4)
                self.experts = "not_a_module_collection"  # right name, wrong type

        cfg = _make_cfg()
        bridge = DenseOrMoEFeedForwardBridge(name="mlp", config=cfg)
        with pytest.raises(TypeError, match="experts"):
            bridge.set_original_component(BadExpertsMoE())

    def test_plain_list_of_experts_is_rejected(self) -> None:
        """Modules held in an ordinary Python list are not registered as
        children -- their parameters would silently drop out of
        parameters()/state_dict()/.to(...)/train()/eval(), contradicting
        this adapter's lifecycle guarantees. Rejected even though every
        element is itself a valid nn.Module."""

        class ListExpertsMoE(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.router = torch.nn.Linear(32, 4)
                self.experts = [TinyDenseMLP(d_model=32, d_ff=64) for _ in range(4)]

        cfg = _make_cfg()
        bridge = DenseOrMoEFeedForwardBridge(name="mlp", config=cfg)
        with pytest.raises(TypeError, match="registered module collection"):
            bridge.set_original_component(ListExpertsMoE())

    def test_gate_of_wrong_type_raises_type_error(self) -> None:
        class BadGateDense(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.gate = "not_a_module"  # right name, wrong type
                self.up = torch.nn.Linear(32, 64, bias=False)
                self.down = torch.nn.Linear(64, 32, bias=False)

        cfg = _make_cfg()
        bridge = DenseOrMoEFeedForwardBridge(name="mlp", config=cfg)
        with pytest.raises(TypeError, match="gate"):
            bridge.set_original_component(BadGateDense())


class TestDenseOrMoEFeedForwardBridgeTupleOutput:
    """Exercises the dispatcher's own tuple-output handling directly, by
    patching an already-wired delegate's forward -- independent of
    whether GatedMLPBridge/MoEBridge happen to return a tuple today. The
    dispatcher explicitly supports and validates this shape, so it's
    tested as its own contract rather than assumed from what the real
    delegates currently do."""

    def _dispatcher_with_patched_delegate(self, fake_forward):
        cfg = _make_cfg(n_layers=1)
        model = TinyPretrainModel(
            d_model=16,
            n_heads=2,
            n_layers=1,
            d_ff=32,
            vocab_size=64,
            n_experts=4,
            top_k=2,
            moe_layer_indices=frozenset({0}),
        )
        bridge = build_pretrain_bridge(model, cfg)
        dispatcher = bridge.blocks[0].mlp
        dispatcher._delegate.forward = fake_forward
        return dispatcher

    def test_tensor_first_tuple_passes_through_with_hook_applied(self) -> None:
        dispatcher = self._dispatcher_with_patched_delegate(
            lambda *a, **kw: (torch.zeros(1, 3, 16), "aux")
        )
        output = dispatcher(torch.ones(1, 3, 16))
        assert isinstance(output, tuple)
        assert output[1] == "aux"
        torch.testing.assert_close(output[0], torch.zeros(1, 3, 16))

    def test_empty_tuple_raises_clear_type_error(self) -> None:
        dispatcher = self._dispatcher_with_patched_delegate(lambda *a, **kw: ())
        with pytest.raises(TypeError, match="torch.Tensor"):
            dispatcher(torch.ones(1, 3, 16))

    def test_non_tensor_first_element_raises_clear_type_error(self) -> None:
        dispatcher = self._dispatcher_with_patched_delegate(
            lambda *a, **kw: ("not_a_tensor", torch.zeros(1))
        )
        with pytest.raises(TypeError, match="torch.Tensor"):
            dispatcher(torch.ones(1, 3, 16))


class TestPretrainAdapterBridgeConstruction:
    @pytest.fixture
    def dense_bridge(self):
        cfg = _make_cfg(n_layers=2)
        model = TinyPretrainModel(
            d_model=32, n_heads=4, n_layers=2, d_ff=64, vocab_size=256
        )
        return build_pretrain_bridge(model, cfg)

    @pytest.fixture
    def moe_bridge(self):
        cfg = _make_cfg(n_layers=2)
        model = TinyPretrainModel(
            d_model=32,
            n_heads=4,
            n_layers=2,
            d_ff=64,
            vocab_size=256,
            n_experts=4,
            top_k=2,
            moe_layer_indices=frozenset({1}),
        )
        return build_pretrain_bridge(model, cfg)

    def test_bridge_has_correct_block_count(self, dense_bridge) -> None:
        assert len(dense_bridge.blocks) == 2

    def test_bridge_has_embed_unembed_and_final_norm(self, dense_bridge) -> None:
        assert hasattr(dense_bridge, "embed")
        assert hasattr(dense_bridge, "unembed")
        assert hasattr(dense_bridge, "ln_final")

    def test_forward_returns_logits_of_expected_shape(self, dense_bridge) -> None:
        tokens = torch.randint(0, 256, (1, 5))
        with torch.no_grad():
            output = dense_bridge(tokens)
        assert output.shape == (1, 5, 256)

    def test_run_with_cache_exposes_resid_hooks(self, dense_bridge) -> None:
        tokens = torch.randint(0, 256, (1, 5))
        _, cache = dense_bridge.run_with_cache(tokens)
        assert any("resid_pre" in k for k in cache.keys())
        assert any("resid_post" in k for k in cache.keys())

    def test_no_per_head_attention_hooks(self, dense_bridge) -> None:
        """This adapter deliberately does not expose hook_q/hook_k/hook_v --
        see PretrainArchitectureAdapter's class docstring."""
        tokens = torch.randint(0, 256, (1, 5))
        _, cache = dense_bridge.run_with_cache(tokens)
        assert not any("hook_q" in k for k in cache.keys())
        assert not any("hook_pattern" in k for k in cache.keys())

    def test_moe_bridge_forward_runs(self, moe_bridge) -> None:
        tokens = torch.randint(0, 256, (1, 5))
        with torch.no_grad():
            output = moe_bridge(tokens)
        assert output.shape == (1, 5, 256)
        assert not torch.isnan(output).any()

    def test_moe_block_uses_dispatcher(self, moe_bridge) -> None:
        assert isinstance(moe_bridge.blocks[1].mlp, DenseOrMoEFeedForwardBridge)

    def test_moe_layer_hooks_are_registered_at_the_dispatcher_path(self, moe_bridge) -> None:
        """Verifies actual cache keys, not just isinstance -- the claim in
        DenseOrMoEFeedForwardBridge's docstring that hooks fire at
        blocks.{i}.mlp.hook_in/hook_out regardless of dense-vs-MoE."""
        tokens = torch.randint(0, 256, (1, 5))
        _, cache = moe_bridge.run_with_cache(tokens)
        assert "blocks.1.mlp.hook_in" in cache
        assert "blocks.1.mlp.hook_out" in cache
        # Layer 0 is dense in this fixture -- same path convention applies.
        assert "blocks.0.mlp.hook_in" in cache
        assert "blocks.0.mlp.hook_out" in cache


class TestMoEConfigFieldIndependence:
    def test_bridge_behaves_identically_with_moe_config_fields_populated(self) -> None:
        """MoEBridge itself (transformer_lens/model_bridge/generalized_
        components/moe.py) never reads cfg.num_experts or
        cfg.experts_per_token. Populating them with realistic values must
        not change behavior.
        Uses an actual MoE layer (moe_layer_indices={0}) -- both models
        being dense would let this pass without ever constructing
        MoEBridge, so the dispatch type is asserted directly too."""
        from transformer_lens.model_bridge.generalized_components import MoEBridge

        torch.manual_seed(0)
        model_a = TinyPretrainModel(
            d_model=16, n_heads=2, n_layers=1, d_ff=32, vocab_size=64,
            n_experts=4, top_k=2, moe_layer_indices=frozenset({0}),
        )
        cfg_without = _make_cfg(n_layers=1)
        bridge_without = build_pretrain_bridge(model_a, cfg_without)

        torch.manual_seed(0)
        model_b = TinyPretrainModel(
            d_model=16, n_heads=2, n_layers=1, d_ff=32, vocab_size=64,
            n_experts=4, top_k=2, moe_layer_indices=frozenset({0}),
        )
        cfg_with = _make_cfg(n_layers=1)
        cfg_with.num_experts = 4
        cfg_with.experts_per_token = 2
        bridge_with = build_pretrain_bridge(model_b, cfg_with)

        assert isinstance(bridge_without.blocks[0].mlp._delegate, MoEBridge)
        assert isinstance(bridge_with.blocks[0].mlp._delegate, MoEBridge)

        tokens = torch.randint(0, 64, (1, 3))
        with torch.no_grad():
            out_without = bridge_without(tokens)
            out_with = bridge_with(tokens)
        torch.testing.assert_close(out_without, out_with, atol=0, rtol=0)


class TestPretrainAdapterTiedEmbeddings:
    def test_tied_embedding_bridge_construction_succeeds(self) -> None:
        cfg = _make_cfg(n_layers=1)
        model = TinyPretrainModel(
            d_model=16, n_heads=2, n_layers=1, d_ff=32, vocab_size=64, tie_embeddings=True
        )
        bridge = build_pretrain_bridge(model, cfg)
        tokens = torch.randint(0, 64, (1, 3))
        with torch.no_grad():
            output = bridge(tokens)
        assert output.shape == (1, 3, 64)

    def test_untied_embedding_bridge_construction_succeeds(self) -> None:
        cfg = _make_cfg(n_layers=1)
        model = TinyPretrainModel(
            d_model=16, n_heads=2, n_layers=1, d_ff=32, vocab_size=64, tie_embeddings=False
        )
        bridge = build_pretrain_bridge(model, cfg)
        tokens = torch.randint(0, 64, (1, 3))
        with torch.no_grad():
            output = bridge(tokens)
        assert output.shape == (1, 3, 64)


class TestBuildBridgeFromModuleDirectlyStillWorks:
    """Advanced/internal path -- build_pretrain_bridge is the intended
    public entry point, but direct build_bridge_from_module use (with the
    container applied manually) must keep working too."""

    def test_direct_build_bridge_from_module_with_manual_container(self) -> None:
        cfg = _make_cfg(n_layers=1)
        model = TinyPretrainModel(d_model=16, n_heads=2, n_layers=1, d_ff=32, vocab_size=64)
        bridge = build_bridge_from_module(
            PretrainModelContainer(model),
            architecture=ARCHITECTURE_NAME,
            tl_config=cfg,
        )
        tokens = torch.randint(0, 64, (1, 3))
        with torch.no_grad():
            output = bridge(tokens)
        assert output.shape == (1, 3, 64)
