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
        assert bridge.W_gate.shape == (D_MODEL, D_MLP)
        assert bridge.W_in.shape == (D_MODEL, D_MLP)
        assert bridge.W_out.shape == (D_MLP, D_MODEL)
        torch.testing.assert_close(bridge.W_in, module.up_proj.weight.T)
        torch.testing.assert_close(bridge.W_out, module.down_proj.weight.T)

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
        assert bridge.hook_aliases == MoEBridge.hook_aliases
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
        assert "gate" not in dense.submodules or "dense_gate" in dense.hook_aliases["hook_pre"]
        assert dense.hook_aliases["hook_pre"].startswith("dense_gate")
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
        assert sparse.hook_aliases == MoEBridge.hook_aliases
        assert template.hook_aliases == MoEBridge.hook_aliases

    def test_rebinding_sparse_after_dense_restores_moe_state(self) -> None:
        """The morph is symmetric: a rebinding harness must not leave a chimera
        with dense aliases and no router hook on a sparse layer."""
        bridge = _bind(_DenseMLP())
        assert bridge._bound_dense is True
        bridge.set_original_component(_SparseMoE())
        assert bridge._bound_dense is False
        assert bridge.hook_aliases == MoEBridge.hook_aliases
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
        assert bridge.hook_aliases == MoEBridge.hook_aliases


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
    NOT_BLOCKS_MLP = {"jamba", "switch_transformers", "qwen3_5_moe"}
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

    def test_templates_without_sparse_required_are_unaffected(self) -> None:
        """jamba/switch-style templates that do not opt in must bind silently."""
        bridge = MoEBridge(
            name="mlp", submodules={"gate": LinearBridge(name="gate", optional=True)}
        )
        module = _DenseMLP()
        bridge.set_original_component(module)
        bridge.validate_after_setup(["gate"])  # must not raise


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
