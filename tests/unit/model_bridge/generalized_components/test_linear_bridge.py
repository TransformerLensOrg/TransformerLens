"""Test for LinearBridge component properties."""

import torch
import torch.nn as nn

from transformer_lens.model_bridge.generalized_components.linear import LinearBridge


def test_linear_bridge_tensor_shapes():
    """Test that LinearBridge returns correct tensor shapes."""
    linear_layer = nn.Linear(4, 6, bias=True)
    bridge = LinearBridge(name="test_tensor_shapes")
    bridge.set_original_component(linear_layer)

    # Test tensor shapes are correct
    assert bridge.weight.shape == (6, 4), f"Expected weight shape (6, 4), got {bridge.weight.shape}"
    assert bridge.bias.shape == (6,), f"Expected bias shape (6,), got {bridge.bias.shape}"
    assert bridge.in_features == 4, f"Expected in_features=4, got {bridge.in_features}"
    assert bridge.out_features == 6, f"Expected out_features=6, got {bridge.out_features}"


def test_linear_bridge_parameter_modifications():
    """Test that modifications to original parameters are reflected through the bridge."""
    linear_layer = nn.Linear(4, 6, bias=True)
    bridge = LinearBridge(name="test_parameter_modifications")
    bridge.set_original_component(linear_layer)

    # Store original parameters
    original_weight_data = linear_layer.weight.data.clone()
    original_bias_data = linear_layer.bias.data.clone()

    try:
        # Modify original parameters
        linear_layer.weight.data.fill_(2.0)
        linear_layer.bias.data.fill_(1.0)

        # Check that bridge reflects the changes
        assert torch.allclose(
            bridge.weight, torch.full((6, 4), 2.0)
        ), "Bridge should reflect weight modifications"
        assert torch.allclose(
            bridge.bias, torch.ones(6)
        ), "Bridge should reflect bias modifications"

    finally:
        # Always restore original parameters
        linear_layer.weight.data.copy_(original_weight_data)
        linear_layer.bias.data.copy_(original_bias_data)
