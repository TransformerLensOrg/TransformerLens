"""Tests for JointGateUpMLPBridge split logic."""

import torch

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


class TestDefaultSplitGateUp:
    def test_splits_weight_in_half(self):
        d_model, d_mlp = 32, 64
        mock_mlp = _MockMLP(d_model, d_mlp)
        gate_proj, up_proj = JointGateUpMLPBridge._default_split_gate_up(mock_mlp)

        assert gate_proj.weight.shape == (d_mlp, d_model)
        assert up_proj.weight.shape == (d_mlp, d_model)

    def test_split_reconstructs_original(self):
        d_model, d_mlp = 32, 64
        mock_mlp = _MockMLP(d_model, d_mlp)
        original_weight = mock_mlp.gate_up_proj.weight.data.clone()

        gate_proj, up_proj = JointGateUpMLPBridge._default_split_gate_up(mock_mlp)
        reconstructed = torch.cat([gate_proj.weight.data, up_proj.weight.data], dim=0)

        assert torch.equal(reconstructed, original_weight)

    def test_split_with_bias(self):
        d_model, d_mlp = 16, 32
        mock_mlp = _MockMLP(d_model, d_mlp, bias=True)
        original_bias = mock_mlp.gate_up_proj.bias.data.clone()

        gate_proj, up_proj = JointGateUpMLPBridge._default_split_gate_up(mock_mlp)

        assert gate_proj.bias is not None
        assert up_proj.bias is not None
        reconstructed_bias = torch.cat([gate_proj.bias.data, up_proj.bias.data], dim=0)
        assert torch.equal(reconstructed_bias, original_bias)

    def test_split_without_bias(self):
        d_model, d_mlp = 16, 32
        mock_mlp = _MockMLP(d_model, d_mlp, bias=False)

        gate_proj, up_proj = JointGateUpMLPBridge._default_split_gate_up(mock_mlp)

        assert gate_proj.bias is None
        assert up_proj.bias is None
