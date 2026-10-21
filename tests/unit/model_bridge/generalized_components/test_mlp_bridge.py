"""Unit tests for MLPBridge/GatedMLPBridge hook-fire semantics.

Pins the container-bridge contract: each projection hook fires exactly once per
forward with the projection's own tensor — no double-application of
interventions (in.hook_in pre-fire + inner fire), no stamping the module's
residual-added output over out.hook_out (MPT/BLOOM-style MLPs), and no silently
dead out.hook_out on the gated processed-weights functional path.
"""

from __future__ import annotations

import copy
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from transformer_lens.model_bridge.generalized_components import (
    GatedMLPBridge,
    LinearBridge,
    MLPBridge,
)


class _ContainerMLP(nn.Module):
    """Plain fc_in -> relu -> fc_out container MLP."""

    def __init__(self, d_model: int, d_mlp: int) -> None:
        super().__init__()
        self.fc_in = nn.Linear(d_model, d_mlp)
        self.fc_out = nn.Linear(d_mlp, d_model)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.fc_out(torch.relu(self.fc_in(hidden_states)))


class _ResidualInsideMLP(nn.Module):
    """MPT/BLOOM-style MLP that adds the residual inside its own forward."""

    def __init__(self, d_model: int, d_mlp: int) -> None:
        super().__init__()
        self.fc_in = nn.Linear(d_model, d_mlp)
        self.fc_out = nn.Linear(d_mlp, d_model)

    def forward(self, hidden_states: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
        return self.fc_out(torch.relu(self.fc_in(hidden_states))) + residual


def _wire_container_bridge(module: nn.Module) -> MLPBridge:
    """Wrap module's fc_in/fc_out with LinearBridges the way component setup does."""
    bridge = MLPBridge(
        name="mlp",
        submodules={"in": LinearBridge(name="fc_in"), "out": LinearBridge(name="fc_out")},
    )
    in_bridge = bridge.submodules["in"]
    out_bridge = bridge.submodules["out"]
    in_bridge.set_original_component(module.fc_in)
    out_bridge.set_original_component(module.fc_out)
    module.fc_in = in_bridge
    module.fc_out = out_bridge
    bridge.set_original_component(module)
    # Submodules resolve as attributes only once registered (what component
    # setup does); the bridge-level pre-fire/fallback paths need them.
    for key, sub in bridge.submodules.items():
        bridge.add_module(key, sub)
    return bridge


class TestMLPBridgeSingleFire:
    def test_projection_hooks_fire_once_and_interventions_apply_once(self) -> None:
        """in.hook_in / out.hook_out fire exactly once; an additive patch on each
        is applied exactly once (the pre-fire/re-fire paths used to double both)."""
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(0)
            reference = _ContainerMLP(8, 16)
            x = torch.randn(2, 3, 8)
        module = copy.deepcopy(reference)
        bridge = _wire_container_bridge(module)
        in_bridge, out_bridge = bridge.submodules["in"], bridge.submodules["out"]

        counts = {"in.hook_in": 0, "out.hook_out": 0}

        def _count(name: str):
            def fn(tensor: torch.Tensor, hook) -> torch.Tensor:
                counts[name] += 1
                return tensor

            return fn

        in_bridge.hook_in.add_hook(_count("in.hook_in"))
        out_bridge.hook_out.add_hook(_count("out.hook_out"))
        try:
            with torch.no_grad():
                out = bridge(x)
        finally:
            in_bridge.hook_in.remove_hooks()
            out_bridge.hook_out.remove_hooks()

        assert counts == {"in.hook_in": 1, "out.hook_out": 1}
        with torch.no_grad():
            torch.testing.assert_close(out, reference(x))

        # Additive patch on in.hook_in: must shift the effective input exactly once.
        in_bridge.hook_in.add_hook(lambda tensor, hook: tensor + 1.0)
        try:
            with torch.no_grad():
                patched = bridge(x)
        finally:
            in_bridge.hook_in.remove_hooks()
        with torch.no_grad():
            torch.testing.assert_close(patched, reference(x + 1.0))

        # Additive patch on out.hook_out: must shift the output exactly once.
        out_bridge.hook_out.add_hook(lambda tensor, hook: tensor + 2.0)
        try:
            with torch.no_grad():
                patched = bridge(x)
        finally:
            out_bridge.hook_out.remove_hooks()
        with torch.no_grad():
            torch.testing.assert_close(patched, reference(x) + 2.0)

    def test_residual_inside_mlp_out_hook_keeps_projection_output(self) -> None:
        """When the wrapped MLP adds the residual internally (MPT/BLOOM),
        out.hook_out must carry the down-projection output, not the
        residual-added module output."""
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(0)
            module = _ResidualInsideMLP(8, 16)
            x = torch.randn(2, 3, 8)
            residual = torch.randn(2, 3, 8)
        bridge = _wire_container_bridge(module)
        out_bridge = bridge.submodules["out"]

        seen: list[torch.Tensor] = []
        out_bridge.hook_out.add_hook(lambda tensor, hook: seen.append(tensor.clone()))
        try:
            with torch.no_grad():
                out = bridge(x, residual)
        finally:
            out_bridge.hook_out.remove_hooks()

        assert len(seen) == 1
        torch.testing.assert_close(seen[0], out - residual)


def _processed_gated_mlp() -> tuple[GatedMLPBridge, torch.Tensor]:
    """A GatedMLPBridge on its processed path over biased nn.Linear projections, and an input."""
    d_model, d_mlp = 8, 16
    bridge = GatedMLPBridge(
        name="mlp",
        config=SimpleNamespace(hidden_act="silu"),
        submodules={
            "gate": LinearBridge(name="gate_proj"),
            "in": LinearBridge(name="up_proj"),
            "out": LinearBridge(name="down_proj"),
        },
    )
    shapes = {"gate": (d_mlp, d_model), "in": (d_mlp, d_model), "out": (d_model, d_mlp)}
    weights = {}
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)
        for key, (d_out, d_in) in shapes.items():
            bridge.submodules[key].set_original_component(nn.Linear(d_in, d_out))
            bridge.add_module(key, bridge.submodules[key])
            weights[f"{key}.weight"] = torch.randn(d_out, d_in)
            weights[f"{key}.bias"] = torch.randn(d_out)
        x = torch.randn(2, 3, d_model)
    bridge.set_processed_weights(weights)
    return bridge, x


def _gated_reference(bridge: GatedMLPBridge, x: torch.Tensor) -> torch.Tensor:
    """down(silu(gate(x)) * up(x)) through the wrapped nn.Linear projections."""
    gate, up, down = (bridge.submodules[key].original_component for key in ("gate", "in", "out"))
    return down(torch.nn.functional.silu(gate(x)) * up(x))


class TestGatedMLPProcessedPathHooks:
    def test_processed_path_fires_out_hook_out(self) -> None:
        """The functional processed-weights path bypasses the wrapped projections;
        out.hook_out must still fire with the down-projection output."""
        bridge, x = _processed_gated_mlp()

        out_bridge = bridge.submodules["out"]
        seen: list[torch.Tensor] = []
        out_bridge.hook_out.add_hook(lambda tensor, hook: seen.append(tensor.clone()))
        try:
            with torch.no_grad():
                out = bridge(x)
        finally:
            out_bridge.hook_out.remove_hooks()

        assert len(seen) == 1
        with torch.no_grad():
            expected = _gated_reference(bridge, x)
        torch.testing.assert_close(seen[0], expected)
        torch.testing.assert_close(out, expected)


class TestGatedMLPProcessedPathWeights:
    @pytest.mark.parametrize("dtype", [torch.float16, torch.float64])
    def test_follows_dtype_change(self, dtype: torch.dtype) -> None:
        """After .to(dtype), the processed path computes with the converted projections."""
        bridge, x = _processed_gated_mlp()
        bridge.to(dtype)
        x = x.to(dtype)
        with torch.no_grad():
            out = bridge(x)
            expected = _gated_reference(bridge, x)
        assert out.dtype == dtype
        torch.testing.assert_close(out, expected)

    def test_deepcopy_reads_its_own_projections(self) -> None:
        """In-place edits to a deep copy's projections reach the copy's processed path."""
        bridge, x = _processed_gated_mlp()
        clone = copy.deepcopy(bridge)
        with torch.no_grad():
            clone.out.original_component.weight.zero_()
            clone.out.original_component.bias.zero_()
            assert torch.count_nonzero(clone(x)) == 0

    def test_projections_get_no_gradient(self) -> None:
        """The processed path reads the projections detached from autograd."""
        bridge, x = _processed_gated_mlp()
        x.requires_grad_()
        bridge(x).sum().backward()
        assert x.grad is not None
        assert all(param.grad is None for param in bridge.parameters())
