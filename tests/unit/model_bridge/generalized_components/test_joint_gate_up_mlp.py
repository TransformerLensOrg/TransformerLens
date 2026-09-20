"""Tests for JointGateUpMLPBridge split logic."""

import torch
from accelerate.hooks import attach_align_device_hook

from transformer_lens.model_bridge.generalized_components.joint_gate_up_mlp import (
    JointGateUpMLPBridge,
)


class _MockMLP(torch.nn.Module):
    """Mock MLP with fused gate_up_proj for testing split logic."""

    def __init__(self, d_model, d_mlp, bias=False):
        super().__init__()
        self.gate_up_proj = torch.nn.Linear(d_model, 2 * d_mlp, bias=bias)
        self.down_proj = torch.nn.Linear(d_mlp, d_model, bias=False)
        self.activation_fn = torch.nn.SiLU()

    def forward(self, x):
        gate_up = self.gate_up_proj(x)
        gate, up = torch.tensor_split(gate_up, 2, dim=-1)
        return self.down_proj(self.activation_fn(gate) * up)


def _splitter(fused_attr: str = "gate_up_proj"):
    """The default splitter is an instance method now (it reads fused_attr, so
    one guarded implementation serves phi-style gate_up_proj, GraniteMoeHybrid's
    input_linear and OpenELM's proj_1)."""
    bridge = JointGateUpMLPBridge(name="mlp", fused_attr=fused_attr)
    return bridge._default_split_gate_up


class TestDefaultSplitGateUp:
    def test_splits_weight_in_half(self):
        d_model, d_mlp = 32, 64
        mock_mlp = _MockMLP(d_model, d_mlp)
        gate_proj, up_proj = _splitter()(mock_mlp)

        assert gate_proj.weight.shape == (d_mlp, d_model)
        assert up_proj.weight.shape == (d_mlp, d_model)

    def test_split_reconstructs_original(self):
        d_model, d_mlp = 32, 64
        mock_mlp = _MockMLP(d_model, d_mlp)
        original_weight = mock_mlp.gate_up_proj.weight.data.clone()

        gate_proj, up_proj = _splitter()(mock_mlp)
        reconstructed = torch.cat([gate_proj.weight.data, up_proj.weight.data], dim=0)

        assert torch.equal(reconstructed, original_weight)

    def test_split_with_bias(self):
        d_model, d_mlp = 16, 32
        mock_mlp = _MockMLP(d_model, d_mlp, bias=True)
        original_bias = mock_mlp.gate_up_proj.bias.data.clone()

        gate_proj, up_proj = _splitter()(mock_mlp)

        assert gate_proj.bias is not None
        assert up_proj.bias is not None
        reconstructed_bias = torch.cat([gate_proj.bias.data, up_proj.bias.data], dim=0)
        assert torch.equal(reconstructed_bias, original_bias)

    def test_split_without_bias(self):
        d_model, d_mlp = 16, 32
        mock_mlp = _MockMLP(d_model, d_mlp, bias=False)

        gate_proj, up_proj = _splitter()(mock_mlp)

        assert gate_proj.bias is None
        assert up_proj.bias is None

    def test_split_honours_fused_attr(self):
        """The parametrization is the point: input_linear-style modules must
        split through the same guarded implementation."""

        class _InputLinearMLP(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.input_linear = torch.nn.Linear(16, 64, bias=False)

        module = _InputLinearMLP()
        gate_proj, up_proj = _splitter("input_linear")(module)
        assert torch.equal(
            torch.cat([gate_proj.weight.data, up_proj.weight.data], dim=0),
            module.input_linear.weight.data,
        )

    def test_split_does_not_advance_global_rng(self):
        """skip_init: the split Linears only carry views, so boot must not
        burn kaiming draws (two per layer) from the global RNG stream."""
        mock_mlp = _MockMLP(8, 16)
        torch.manual_seed(1234)
        before = torch.random.get_rng_state()
        _splitter()(mock_mlp)
        assert torch.equal(before, torch.random.get_rng_state())


class TestSetOriginalComponentUnderOffload:
    """set_original_component reads gate_up_proj's raw weight once, at setup, to
    build the gate/up slices - the same setup-time pattern as
    JointQKVAttentionBridge.set_original_component (#1280 review), which needs
    real (not meta) data materialized under Accelerate CPU/disk offload."""

    def test_split_materializes_offloaded_gate_up_proj(self):
        d_model, d_mlp = 8, 16
        mock_mlp = _MockMLP(d_model, d_mlp)
        weights_map = {
            k[len("gate_up_proj.") :]: v.clone()
            for k, v in mock_mlp.state_dict().items()
            if k.startswith("gate_up_proj.")
        }
        attach_align_device_hook(
            mock_mlp.gate_up_proj,
            execution_device=torch.device("cpu"),
            offload=True,
            weights_map=weights_map,
        )
        assert mock_mlp.gate_up_proj.weight.is_meta

        bridge = JointGateUpMLPBridge(name="mlp")
        bridge.set_original_component(mock_mlp)

        gate_weight = bridge.gate.original_component.weight
        up_weight = getattr(bridge, "in").original_component.weight
        assert not gate_weight.is_meta
        assert not up_weight.is_meta
        torch.testing.assert_close(
            torch.cat([gate_weight, up_weight], dim=0), weights_map["weight"]
        )
