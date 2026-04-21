"""Test for LinearBridge component properties."""

import torch
import torch.nn as nn

from transformer_lens.model_bridge.generalized_components.linear import LinearBridge


def test_linear_bridge_properties_with_bias():
    """Test that LinearBridge properly accesses all properties when bias=True."""
    # Create a linear layer with bias
    linear_layer = nn.Linear(10, 5, bias=True)

    # Create LinearBridge and set original component
    bridge = LinearBridge(name="test_linear_with_bias")
    bridge.set_original_component(linear_layer)

    # Test weight property delegation
    original_weight = linear_layer.weight
    bridge_weight = bridge.weight
    assert torch.equal(
        original_weight, bridge_weight
    ), "Bridge weight should equal original weight tensor"
    assert (
        original_weight is bridge_weight
    ), "Bridge weight should be the same object as original weight"

    # Test bias property delegation
    original_bias = linear_layer.bias
    bridge_bias = bridge.bias
    assert torch.equal(original_bias, bridge_bias), "Bridge bias should equal original bias tensor"
    assert original_bias is bridge_bias, "Bridge bias should be the same object as original bias"

    # Test in_features property
    assert bridge.in_features == 10, f"Expected in_features=10, got {bridge.in_features}"
    assert (
        bridge.in_features == linear_layer.in_features
    ), "Bridge in_features should match original"

    # Test out_features property
    assert bridge.out_features == 5, f"Expected out_features=5, got {bridge.out_features}"
    assert (
        bridge.out_features == linear_layer.out_features
    ), "Bridge out_features should match original"


def test_linear_bridge_properties_without_bias():
    """Test that LinearBridge properly accesses all properties when bias=False."""
    # Create a linear layer without bias
    linear_layer = nn.Linear(8, 3, bias=False)

    # Create LinearBridge and set original component
    bridge = LinearBridge(name="test_linear_no_bias")
    bridge.set_original_component(linear_layer)

    # Test weight property delegation (should still work without bias)
    original_weight = linear_layer.weight
    bridge_weight = bridge.weight
    assert torch.equal(
        original_weight, bridge_weight
    ), "Bridge weight should equal original weight tensor"
    assert (
        original_weight is bridge_weight
    ), "Bridge weight should be the same object as original weight"

    # Test that bias property returns None when no bias exists
    assert linear_layer.bias is None, "Original component should have no bias"
    assert bridge.bias is None, "Bridge bias should also be None"

    # Test in_features and out_features properties
    assert bridge.in_features == 8, f"Expected in_features=8, got {bridge.in_features}"
    assert bridge.out_features == 3, f"Expected out_features=3, got {bridge.out_features}"
    assert (
        bridge.in_features == linear_layer.in_features
    ), "Bridge in_features should match original"
    assert (
        bridge.out_features == linear_layer.out_features
    ), "Bridge out_features should match original"


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


def test_linear_bridge_property_accessibility():
    """Test that all properties are accessible via hasattr."""
    linear_layer = nn.Linear(3, 2, bias=True)
    bridge = LinearBridge(name="test_property_accessibility")
    bridge.set_original_component(linear_layer)

    # Test that all properties are accessible via hasattr
    assert hasattr(bridge, "weight"), "Bridge should have weight attribute"
    assert hasattr(bridge, "bias"), "Bridge should have bias attribute"
    assert hasattr(bridge, "in_features"), "Bridge should have in_features attribute"
    assert hasattr(bridge, "out_features"), "Bridge should have out_features attribute"
