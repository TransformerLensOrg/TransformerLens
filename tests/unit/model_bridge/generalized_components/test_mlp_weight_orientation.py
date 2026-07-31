"""Tests for MLP weight orientation (W_in/W_out/W_gate in TL convention)."""

import pytest
import torch
import torch.nn as nn

from transformer_lens.model_bridge.generalized_components.linear import LinearBridge
from transformer_lens.model_bridge.generalized_components.mlp import MLPBridge


class TestMLPWeightOrientation:
    """Test that MLP weight accessors return TL orientation regardless of backing module."""

    @pytest.fixture
    def d_model(self) -> int:
        return 64

    @pytest.fixture
    def d_mlp(self) -> int:
        return 256

    @pytest.fixture
    def linear_mlp_bridge(self, d_model: int, d_mlp: int) -> MLPBridge:
        """Create an MLPBridge backed by nn.Linear modules."""
        mlp = MLPBridge(name="mlp")

        # Create nn.Linear projections (stores weights as [out, in])
        in_proj = nn.Linear(d_model, d_mlp, bias=False)
        out_proj = nn.Linear(d_mlp, d_model, bias=False)

        # Wrap in LinearBridge
        in_bridge = LinearBridge(name="in")
        in_bridge.set_original_component(in_proj)
        out_bridge = LinearBridge(name="out")
        out_bridge.set_original_component(out_proj)

        # Register as submodules
        mlp.add_module("in", in_bridge)
        mlp.add_module("out", out_bridge)

        return mlp

    @pytest.fixture
    def conv1d_mlp_bridge(self, d_model: int, d_mlp: int) -> MLPBridge:
        """Create an MLPBridge backed by Conv1D modules (GPT-2 style)."""
        from transformers.pytorch_utils import Conv1D

        mlp = MLPBridge(name="mlp")

        # Create Conv1D projections (stores weights as [in, out])
        in_proj = Conv1D(d_mlp, d_model)
        out_proj = Conv1D(d_model, d_mlp)

        # Wrap in LinearBridge
        in_bridge = LinearBridge(name="in")
        in_bridge.set_original_component(in_proj)
        out_bridge = LinearBridge(name="out")
        out_bridge.set_original_component(out_proj)

        # Register as submodules
        mlp.add_module("in", in_bridge)
        mlp.add_module("out", out_bridge)

        return mlp

    def test_linear_w_in_shape_is_tl_convention(
        self, linear_mlp_bridge: MLPBridge, d_model: int, d_mlp: int
    ):
        """W_in from nn.Linear should be [d_model, d_mlp]."""
        w_in = linear_mlp_bridge.W_in
        assert w_in.shape == (d_model, d_mlp), f"Expected ({d_model}, {d_mlp}), got {w_in.shape}"

    def test_linear_w_out_shape_is_tl_convention(
        self, linear_mlp_bridge: MLPBridge, d_model: int, d_mlp: int
    ):
        """W_out from nn.Linear should be [d_mlp, d_model]."""
        w_out = linear_mlp_bridge.W_out
        assert w_out.shape == (d_mlp, d_model), f"Expected ({d_mlp}, {d_model}), got {w_out.shape}"

    def test_conv1d_w_in_shape_is_tl_convention(
        self, conv1d_mlp_bridge: MLPBridge, d_model: int, d_mlp: int
    ):
        """W_in from Conv1D should be [d_model, d_mlp]."""
        w_in = conv1d_mlp_bridge.W_in
        assert w_in.shape == (d_model, d_mlp), f"Expected ({d_model}, {d_mlp}), got {w_in.shape}"

    def test_conv1d_w_out_shape_is_tl_convention(
        self, conv1d_mlp_bridge: MLPBridge, d_model: int, d_mlp: int
    ):
        """W_out from Conv1D should be [d_mlp, d_model]."""
        w_out = conv1d_mlp_bridge.W_out
        assert w_out.shape == (d_mlp, d_model), f"Expected ({d_mlp}, {d_model}), got {w_out.shape}"

    def test_linear_w_in_reproduces_projection(
        self, linear_mlp_bridge: MLPBridge, d_model: int, d_mlp: int
    ):
        """resid @ W_in should match what the wrapped nn.Linear computes."""
        resid = torch.randn(1, 8, d_model)
        w_in = linear_mlp_bridge.W_in

        # Compute via TL-style matmul: resid @ W_in
        tl_result = resid @ w_in

        # Compute via the wrapped projection
        in_bridge = getattr(linear_mlp_bridge, "in")
        proj_result = in_bridge.original_component(resid)

        assert torch.allclose(tl_result, proj_result, atol=1e-5), (
            f"resid @ W_in does not match projection output. "
            f"Max diff: {(tl_result - proj_result).abs().max()}"
        )

    def test_linear_w_out_reproduces_projection(
        self, linear_mlp_bridge: MLPBridge, d_model: int, d_mlp: int
    ):
        """hidden @ W_out should match what the wrapped nn.Linear computes."""
        hidden = torch.randn(1, 8, d_mlp)
        w_out = linear_mlp_bridge.W_out

        # Compute via TL-style matmul: hidden @ W_out
        tl_result = hidden @ w_out

        # Compute via the wrapped projection
        out_bridge = linear_mlp_bridge.out
        proj_result = out_bridge.original_component(hidden)

        assert torch.allclose(tl_result, proj_result, atol=1e-5), (
            f"hidden @ W_out does not match projection output. "
            f"Max diff: {(tl_result - proj_result).abs().max()}"
        )

    def test_conv1d_w_in_reproduces_projection(
        self, conv1d_mlp_bridge: MLPBridge, d_model: int, d_mlp: int
    ):
        """resid @ W_in should match what the wrapped Conv1D computes."""
        resid = torch.randn(1, 8, d_model)
        w_in = conv1d_mlp_bridge.W_in

        # Compute via TL-style matmul: resid @ W_in
        tl_result = resid @ w_in

        # Compute via the wrapped projection
        in_bridge = getattr(conv1d_mlp_bridge, "in")
        proj_result = in_bridge.original_component(resid)

        assert torch.allclose(tl_result, proj_result, atol=1e-5), (
            f"resid @ W_in does not match projection output. "
            f"Max diff: {(tl_result - proj_result).abs().max()}"
        )

    def test_conv1d_w_out_reproduces_projection(
        self, conv1d_mlp_bridge: MLPBridge, d_model: int, d_mlp: int
    ):
        """hidden @ W_out should match what the wrapped Conv1D computes."""
        hidden = torch.randn(1, 8, d_mlp)
        w_out = conv1d_mlp_bridge.W_out

        # Compute via TL-style matmul: hidden @ W_out
        tl_result = hidden @ w_out

        # Compute via the wrapped projection
        out_bridge = conv1d_mlp_bridge.out
        proj_result = out_bridge.original_component(hidden)

        assert torch.allclose(tl_result, proj_result, atol=1e-5), (
            f"hidden @ W_out does not match projection output. "
            f"Max diff: {(tl_result - proj_result).abs().max()}"
        )

    def test_both_backends_produce_same_orientation(
        self, linear_mlp_bridge: MLPBridge, conv1d_mlp_bridge: MLPBridge, d_model: int, d_mlp: int
    ):
        """nn.Linear and Conv1D bridges should return same shape for W_in/W_out."""
        assert linear_mlp_bridge.W_in.shape == conv1d_mlp_bridge.W_in.shape
        assert linear_mlp_bridge.W_out.shape == conv1d_mlp_bridge.W_out.shape
