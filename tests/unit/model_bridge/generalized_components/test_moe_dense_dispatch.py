"""Unit tests for MoEBridge's per-layer dense/sparse dispatch (#1645).

Interleaved and dense-prefix MoE architectures build a plain gated MLP on some
layers under the same attribute name as the sparse block. A single uniform
template then exposed MoE boundary aliases there — ``hook_pre``/``hook_post``
delivering d_model residual tensors under neuron-hook names, no hooks on the
dense projections, and no weight accessors. MoEBridge now detects the dense
binding and adopts gated-MLP semantics for that layer only.
"""

from __future__ import annotations

import copy
from pathlib import Path

import pytest
import torch
import torch.nn as nn

import transformer_lens.model_bridge.supported_architectures
from tests.unit.model_bridge.supported_architectures.helpers import make_bridge_cfg
from transformer_lens.factories.architecture_adapter_factory import (
    ArchitectureAdapterFactory,
)
from transformer_lens.model_bridge.component_setup import setup_submodules
from transformer_lens.model_bridge.generalized_components import LinearBridge, MoEBridge

D_MODEL, D_MLP = 8, 16


class _DenseMLP(nn.Module):
    """Standard SwiGLU gated MLP — the dense-prefix layer shape."""

    def __init__(self) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(D_MODEL, D_MLP, bias=False)
        self.up_proj = nn.Linear(D_MODEL, D_MLP, bias=False)
        self.down_proj = nn.Linear(D_MLP, D_MODEL, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.down_proj(
            torch.nn.functional.silu(self.gate_proj(hidden_states)) * self.up_proj(hidden_states)
        )


class _SparseMoE(nn.Module):
    """Minimal sparse block: a router plus experts, no dense projections."""

    def __init__(self) -> None:
        super().__init__()
        self.gate = nn.Linear(D_MODEL, 4, bias=False)
        self.experts = nn.ModuleList([nn.Linear(D_MODEL, D_MODEL) for _ in range(4)])

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return hidden_states + 0.0 * self.gate(hidden_states).sum(-1, keepdim=True)


def _adapter():
    """A real adapter (beartype enforces the type on setup_submodules), whose
    mlp template is the interleaved-MoE mapping under test."""
    return ArchitectureAdapterFactory.select_architecture_adapter(
        make_bridge_cfg("DeepseekV3ForCausalLM", d_model=D_MODEL, n_heads=2, d_head=4)
    )


def _template() -> MoEBridge:
    """The adapter's own mlp template, as block setup deepcopies it per layer."""
    template = _adapter().component_mapping["blocks"].submodules["mlp"]
    assert isinstance(template, MoEBridge)
    return template


def _bind(module: nn.Module) -> MoEBridge:
    """Bind through the real component-setup path, as boot does."""
    adapter = _adapter()
    bridge = copy.deepcopy(adapter.component_mapping["blocks"].submodules["mlp"])
    assert isinstance(bridge, MoEBridge)
    bridge.set_original_component(module)
    setup_submodules(bridge, adapter, module)
    return bridge


class TestDenseBinding:
    def test_dense_layer_adopts_gated_mlp_aliases(self) -> None:
        bridge = _bind(_DenseMLP())
        assert bridge._bound_dense is True
        assert bridge.hook_aliases == {
            "hook_pre": "dense_gate.hook_out",
            "hook_pre_linear": "dense_in.hook_out",
            "hook_post": "dense_out.hook_in",
        }

    def test_dense_layer_exposes_neuron_basis_weights(self) -> None:
        """The #1645 fix advertises neuron hooks; the weight API must back them
        (ActivationCache neuron-result stacking and weight collection read these,
        and a missing accessor is silently zero-filled by get_params_util)."""
        module = _DenseMLP()
        bridge = _bind(module)
        # Pin each accessor to the projection it must read. Shape alone cannot:
        # gate_proj and up_proj are both [d_model, d_mlp], so a W_gate that
        # returned the UP projection would satisfy any shape-only assertion.
        torch.testing.assert_close(bridge.W_gate, module.gate_proj.weight.T)
        torch.testing.assert_close(bridge.W_in, module.up_proj.weight.T)
        torch.testing.assert_close(bridge.W_out, module.down_proj.weight.T)
        # Negative control: the two same-shaped projections are distinguishable
        # in this fixture, so the assertions above are not trivially satisfiable.
        assert not torch.equal(module.gate_proj.weight, module.up_proj.weight)

    def test_dense_layer_drops_the_router_hook(self) -> None:
        """A dense layer has no router; advertising hook_router_scores would be a
        hook that can never fire."""
        assert not hasattr(_bind(_DenseMLP()), "hook_router_scores")

    def test_dense_gate_hook_carries_the_gate_projection_output(self) -> None:
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(0)
            module = _DenseMLP()
            x = torch.randn(2, 3, D_MODEL)
        expected = module.gate_proj(x).detach()
        bridge = _bind(module)

        seen: list[torch.Tensor] = []
        bridge.submodules["dense_gate"].hook_out.add_hook(
            lambda tensor, hook: seen.append(tensor.clone())
        )
        try:
            with torch.no_grad():
                out = bridge(x)
        finally:
            bridge.submodules["dense_gate"].hook_out.remove_hooks()

        assert len(seen) == 1
        assert seen[0].shape == (2, 3, D_MLP)  # neuron basis, not d_model
        torch.testing.assert_close(seen[0], expected)
        assert out.shape == (2, 3, D_MODEL)


class TestSparseBinding:
    def test_sparse_layer_keeps_moe_semantics(self) -> None:
        bridge = _bind(_SparseMoE())
        assert bridge._bound_dense is False
        # Pin the concrete targets rather than comparing to
        # MoEBridge.hook_aliases: that is the class constant this bind is
        # supposed to reproduce, so the comparison holds even if the constant
        # itself is wrong. hook_pre/hook_post are the MoE block boundaries.
        assert bridge.hook_aliases == {"hook_pre": "hook_in", "hook_post": "hook_out"}
        assert hasattr(bridge, "hook_router_scores")

    def test_sparse_layer_has_no_dense_weight_accessors(self) -> None:
        """Sparse layers have per-expert weights and no single W_*; hasattr must
        stay False so weight-collection helpers skip them as before."""
        bridge = _bind(_SparseMoE())
        assert not hasattr(bridge, "W_in")
        assert not hasattr(bridge, "W_gate")
        assert not hasattr(bridge, "W_out")

    def test_gate_name_is_not_overloaded_across_layers(self) -> None:
        """blocks.N.mlp.gate must not mean the router on one layer and a d_mlp
        gate projection on another — that per-layer semantic flip is the #1645
        flaw class this dispatch removes."""
        dense, sparse = _bind(_DenseMLP()), _bind(_SparseMoE())
        # The two layers must not resolve the same hook name to different
        # KINDS of tensor. Compare the resolved targets directly.
        assert dense.hook_aliases["hook_pre"] == "dense_gate.hook_out"
        assert sparse.hook_aliases["hook_pre"] == "hook_in"
        # And `gate` must never be a dense projection: on the dense layer the
        # key is absent entirely, so `blocks.N.mlp.gate.hook_out` means the
        # router on every layer that has it.
        assert "gate" not in dense.submodules
        assert "gate" in sparse.submodules


class TestDispatchRobustness:
    def test_one_template_serves_both_layer_types(self) -> None:
        """Block setup deepcopies a single template per layer; each copy must
        bind independently and leave the template untouched."""
        template = _template()
        dense = copy.deepcopy(template)
        sparse = copy.deepcopy(template)
        dense.set_original_component(_DenseMLP())
        sparse.set_original_component(_SparseMoE())
        assert dense.hook_aliases["hook_pre"] == "dense_gate.hook_out"
        assert sparse.hook_aliases == {"hook_pre": "hook_in", "hook_post": "hook_out"}
        assert template.hook_aliases == {"hook_pre": "hook_in", "hook_post": "hook_out"}

    def test_rebinding_sparse_after_dense_restores_moe_state(self) -> None:
        """The morph is symmetric: a rebinding harness must not leave a chimera
        with dense aliases and no router hook on a sparse layer."""
        bridge = _bind(_DenseMLP())
        assert bridge._bound_dense is True
        bridge.set_original_component(_SparseMoE())
        assert bridge._bound_dense is False
        assert bridge.hook_aliases == {"hook_pre": "hook_in", "hook_post": "hook_out"}
        assert hasattr(bridge, "hook_router_scores")

    def test_alias_rebind_survives_the_attribute_passthrough(self) -> None:
        """GeneralizedComponent.__setattr__ forwards unknown attributes to the
        wrapped module; hook_aliases must be exempt or the rebind vanishes
        whenever the HF module happens to expose that attribute."""
        module = _DenseMLP()
        module.hook_aliases = {"decoy": "value"}  # type: ignore[assignment]
        bridge = _bind(module)
        assert bridge.hook_aliases["hook_pre"] == "dense_gate.hook_out"

    def test_undeclared_dense_projections_leave_moe_mapping(self) -> None:
        """Detection is positive-only: without declared dense_* submodules the
        bridge must not guess a layer is dense and strip its MoE hooks."""
        bridge = MoEBridge(name="mlp", submodules={})
        bridge.set_original_component(_DenseMLP())
        assert bridge._bound_dense is False
        assert bridge.hook_aliases == {"hook_pre": "hook_in", "hook_post": "hook_out"}


# Every adapter declaring dense_* projections. Kept in sync with
# `grep -l '"dense_in"' transformer_lens/model_bridge/supported_architectures/`
# by test_roster_covers_every_dense_declaring_adapter below, so a new adapter
# cannot join the dispatch without also joining these guards.
DENSE_AWARE_ARCHS = [
    "DeepseekV2ForCausalLM",
    "DeepseekV3ForCausalLM",
    "Glm4MoeForCausalLM",
    "Glm4MoeLiteForCausalLM",
    "GlmMoeDsaForCausalLM",
    "Ernie4_5_MoeForCausalLM",
    "AfmoeForCausalLM",
    "Qwen2MoeForCausalLM",
    "Qwen3MoeForCausalLM",
    "Qwen3NextForCausalLM",
    "Qwen3VLMoeForConditionalGeneration",
    "LLaDA2MoeModelLM",
    "LagunaForCausalLM",
    "Llama4ForConditionalGeneration",
]


def test_roster_covers_every_dense_declaring_adapter() -> None:
    """The roster parametrizes the guards below; an adapter that declares dense_*
    but is missing here would be the one place the guards do not reach."""
    adapters_dir = Path(transformer_lens.model_bridge.supported_architectures.__file__).parent
    declaring = {
        path.stem for path in adapters_dir.glob("*.py") if '"dense_in"' in path.read_text()
    }
    # Adapters whose dense mapping is reached through a different template shape
    # (a per-config builder or an encoder-decoder block list) rather than
    # blocks.mlp, so the blocks-based parametrization cannot construct them.
    # jamba builds its MoEBridge only when num_experts > 1 (a per-config
    # builder) and switch_transformers maps encoder/decoder block lists, so
    # neither is reachable through a blocks.mlp template here.
    NOT_BLOCKS_MLP = {"jamba", "switch_transformers"}
    # Walk the MRO: an arch string can resolve to a subclass in another module
    # (Llama4ForConditionalGeneration -> the multimodal adapter, which inherits
    # llama4's block mapping), and the declaration lives on the base.
    covered_modules = {
        klass.__module__.rsplit(".", 1)[-1]
        for arch in DENSE_AWARE_ARCHS
        for klass in type(
            ArchitectureAdapterFactory.select_architecture_adapter(make_bridge_cfg(arch, d_head=8))
        ).__mro__
    }
    missing = declaring - covered_modules - NOT_BLOCKS_MLP
    assert not missing, f"adapters declare dense_* but are outside the guards: {sorted(missing)}"


@pytest.mark.parametrize("architecture", DENSE_AWARE_ARCHS)
def test_adapter_templates_bind_dense_layers_as_gated_mlps(architecture: str) -> None:
    """Every interleaved/dense-prefix MoE adapter must declare the dense
    projections, so its dense layers get neuron-basis hooks (#1645)."""
    cfg = make_bridge_cfg(architecture, d_head=8)
    adapter = ArchitectureAdapterFactory.select_architecture_adapter(cfg)
    blocks = adapter.component_mapping["blocks"]
    template = blocks.submodules.get("mlp") or blocks.submodules["feed_forward"]
    instance = copy.deepcopy(template)
    instance.set_original_component(_DenseMLP())
    assert instance._bound_dense is True, f"{architecture} did not declare dense projections"
    assert instance.hook_aliases["hook_pre"] == "dense_gate.hook_out"


@pytest.mark.parametrize("architecture", DENSE_AWARE_ARCHS)
def test_dense_keys_read_the_projection_they_name(architecture: str) -> None:
    """`dense_gate` must be the gate projection, not the up projection.

    Both are [d_model, d_mlp] and both bind without complaint, so a
    dense_gate/dense_in swap in an adapter survives every shape check, key-set
    check and the binding guard above — while making `hook_pre` report the
    wrong tensor under the right name. That is #1645's own confusion one level
    in, so it is checked here, once, for every adapter, rather than in the two
    that happened to have bespoke assertions.

    Hooks the concrete targets rather than the aliases; that the aliases point
    here is asserted by TestDenseBinding, and the two compose.
    """
    cfg = make_bridge_cfg(architecture, d_head=8)
    adapter = ArchitectureAdapterFactory.select_architecture_adapter(cfg)
    blocks = adapter.component_mapping["blocks"]
    template = blocks.submodules.get("mlp") or blocks.submodules["feed_forward"]
    bridge = copy.deepcopy(template)
    module = _DenseMLP()
    bridge.set_original_component(module)
    setup_submodules(bridge, adapter, module)

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)
        x = torch.randn(1, 3, D_MODEL)

    captured: dict = {}
    for key in ("dense_gate", "dense_in"):
        getattr(bridge, key).hook_out.add_hook(
            lambda t, hook, key=key: captured.__setitem__(key, t.clone())
        )
    with torch.no_grad():
        bridge(x)
        expected_gate = module.gate_proj(x)
        expected_in = module.up_proj(x)

    torch.testing.assert_close(
        captured["dense_gate"],
        expected_gate,
        msg=lambda m: f"{architecture}: dense_gate is not the gate projection\n{m}",
    )
    torch.testing.assert_close(
        captured["dense_in"],
        expected_in,
        msg=lambda m: f"{architecture}: dense_in is not the up projection\n{m}",
    )
    # The fixture must be able to tell them apart, or neither assertion means
    # anything (equal weights would satisfy both under a swap).
    assert not torch.allclose(expected_gate, expected_in)


class _UngatedDenseFF(nn.Module):
    """Switch-style ungated dense feed-forward (wi/wo, no gate projection)."""

    def __init__(self) -> None:
        super().__init__()
        self.wi = nn.Linear(D_MODEL, D_MLP, bias=False)
        self.wo = nn.Linear(D_MLP, D_MODEL, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.wo(torch.relu(self.wi(hidden_states)))


class _RenamedRouterSparseMoE(nn.Module):
    """Sparse block whose router HF renamed out from under the adapter."""

    def __init__(self) -> None:
        super().__init__()
        self.router = nn.Linear(D_MODEL, 4, bias=False)
        self.experts = nn.ModuleList([nn.Linear(D_MODEL, D_MODEL) for _ in range(4)])

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return hidden_states


class TestUngatedDenseBinding:
    """Not every dense feed-forward is gated — Switch's is wi/wo."""

    def test_ungated_dense_binds_mlp_alias_set(self) -> None:
        bridge = MoEBridge(
            name="mlp",
            submodules={
                "dense_in": LinearBridge(name="wi", optional=True),
                "dense_out": LinearBridge(name="wo", optional=True),
            },
        )
        module = _UngatedDenseFF()
        bridge.set_original_component(module)
        for key, attr in (("dense_in", "wi"), ("dense_out", "wo")):
            sub = bridge.submodules[key]
            sub.set_original_component(getattr(module, attr))
            bridge.add_module(key, sub)

        assert bridge.bound_dense is True
        assert bridge.hook_aliases == {
            "hook_pre": "dense_in.hook_out",
            "hook_post": "dense_out.hook_in",
        }
        # No gate projection exists, so no hook_pre_linear and no W_gate.
        assert "hook_pre_linear" not in bridge.hook_aliases
        assert not hasattr(bridge, "W_gate")
        assert bridge.W_in.shape == (D_MODEL, D_MLP)


class TestSparseRequiredGuard:
    """`optional` must not mean a renamed HF router silently loses its hooks."""

    def test_renamed_router_on_sparse_layer_raises(self) -> None:
        adapter = _adapter()
        bridge = copy.deepcopy(adapter.component_mapping["blocks"].submodules["mlp"])
        module = _RenamedRouterSparseMoE()
        bridge.set_original_component(module)
        with pytest.raises(ValueError, match="required submodule"):
            setup_submodules(bridge, adapter, module)

    def test_correctly_named_sparse_router_binds_clean(self) -> None:
        bridge = _bind(_SparseMoE())
        assert bridge.bound_dense is False
        assert "gate" in bridge.submodules

    def test_dense_layer_may_legitimately_lack_the_router(self) -> None:
        """Router absence is expected on dense layers — must not raise."""
        bridge = _bind(_DenseMLP())
        assert bridge.bound_dense is True

    def test_undeclared_sparse_required_key_raises_at_construction(self) -> None:
        """An unvalidated opt-in string would silently disable the guard."""
        with pytest.raises(ValueError, match="not declared submodules"):
            MoEBridge(
                name="mlp",
                submodules={"gate": LinearBridge(name="gate", optional=True)},
                sparse_required=("rooter",),
            )

    def test_opting_in_is_what_makes_a_missing_router_loud(self) -> None:
        """Differential on the opt-in alone: the SAME template shape and the SAME
        module bind silently without `sparse_required` and raise with it.

        Driven through the real setup_submodules so the skipped set is computed
        rather than hand-fed — hand-feeding it would exercise the guard's body
        while skipping the machinery that decides when the guard applies.
        """
        adapter = _adapter()
        module = _RenamedRouterSparseMoE()

        def build(**kwargs) -> MoEBridge:
            return MoEBridge(
                name="mlp",
                submodules={"gate": LinearBridge(name="gate", optional=True)},
                **kwargs,
            )

        # jamba/switch-style: no opt-in, so a skipped optional stays silent.
        opted_out = build()
        opted_out.set_original_component(module)
        setup_submodules(opted_out, adapter, module)
        assert "gate" not in opted_out.submodules  # precondition: it WAS skipped

        # Same template + same module, only the opt-in differs.
        opted_in = build(sparse_required=("gate",))
        opted_in.set_original_component(module)
        with pytest.raises(ValueError, match="required submodule"):
            setup_submodules(opted_in, adapter, module)


@pytest.mark.parametrize("architecture", DENSE_AWARE_ARCHS)
def test_dense_aware_adapters_declare_a_sparse_required_router(architecture: str) -> None:
    """Every dense-aware MoE adapter must opt into the loud-on-rename guard, so a
    new adapter cannot be added with a silently-skippable router."""
    cfg = make_bridge_cfg(architecture, d_head=8)
    adapter = ArchitectureAdapterFactory.select_architecture_adapter(cfg)
    blocks = adapter.component_mapping["blocks"]
    template = blocks.submodules.get("mlp") or blocks.submodules["feed_forward"]
    assert template._sparse_required, f"{architecture} declares no sparse_required router"
    assert set(template._sparse_required) <= set(template.submodules)


class TestDeclaredGateMustResolve:
    """A declared dense_gate that does not resolve is a RENAME, not evidence the
    MLP is ungated — binding it ungated aliases hook_pre to the up projection."""

    def test_renamed_gate_raises_instead_of_binding_ungated(self) -> None:
        class _RenamedGateDenseMLP(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.w_gate = nn.Linear(D_MODEL, D_MLP, bias=False)
                self.up_proj = nn.Linear(D_MODEL, D_MLP, bias=False)
                self.down_proj = nn.Linear(D_MLP, D_MODEL, bias=False)

        bridge = copy.deepcopy(_template())
        with pytest.raises(ValueError, match="dense MLPs are gated"):
            bridge.set_original_component(_RenamedGateDenseMLP())

    def test_undeclared_gate_still_binds_ungated(self) -> None:
        """Switch-style: no dense_gate declared, so ungated is the truth."""
        bridge = MoEBridge(
            name="mlp",
            submodules={
                "dense_in": LinearBridge(name="wi", optional=True),
                "dense_out": LinearBridge(name="wo", optional=True),
            },
        )

        class _Ungated(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.wi = nn.Linear(D_MODEL, D_MLP)
                self.wo = nn.Linear(D_MLP, D_MODEL)

        bridge.set_original_component(_Ungated())
        assert bridge.bound_dense is True
        assert bridge.hook_aliases["hook_pre"] == "dense_in.hook_out"
