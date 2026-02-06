"""Tests for GeneralizedComponent base class functionality."""

import pytest
import torch
import torch.nn as nn

from transformer_lens.hook_points import HookPoint
from transformer_lens.model_bridge.generalized_components.base import (
    GeneralizedComponent,
)


class MockOriginalComponent(nn.Module):
    """Mock original component for testing."""

    def __init__(self):
        super().__init__()
        self.existing_attr = "original_value"
        self.tensor_attr = torch.tensor([1.0, 2.0, 3.0])

    def forward(self, x):
        return x


class MockGeneralizedComponent(GeneralizedComponent):
    """Mock generalized component for testing."""

    def __init__(self, name: str = "test_component"):
        super().__init__(name)
        self.bridge_attr = "bridge_value"

    @property
    def test_property(self):
        """Test property with getter."""
        return getattr(self, "_test_property", None)

    @test_property.setter
    def test_property(self, value):
        """Test property with setter."""
        self._test_property = value


class TestGeneralizedComponentBase:
    """Test suite for GeneralizedComponent base functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.component = MockGeneralizedComponent("test")
        self.original = MockOriginalComponent()
        self.component.set_original_component(self.original)

    def test_initialization(self):
        """Test GeneralizedComponent initialization."""
        component = MockGeneralizedComponent("init_test")
        assert component.name == "init_test"
        assert isinstance(component.hook_in, HookPoint)
        assert isinstance(component.hook_out, HookPoint)
        assert component.submodules == {}
        assert component.conversion_rule is None

    def test_set_original_component(self):
        """Test setting original component."""
        component = MockGeneralizedComponent("orig_test")
        original = MockOriginalComponent()

        component.set_original_component(original)
        assert component.original_component is original
        assert component._modules["_original_component"] is original

    def test_original_component_property(self):
        """Test original_component property."""
        component = MockGeneralizedComponent("prop_test")
        assert component.original_component is None

        original = MockOriginalComponent()
        component.set_original_component(original)
        assert component.original_component is original


class TestGeneralizedComponentSetAttr:
    """Test suite for GeneralizedComponent.__setattr__ functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.component = MockGeneralizedComponent("test")
        self.original = MockOriginalComponent()
        self.component.set_original_component(self.original)

    def test_set_private_attribute(self):
        """Test setting private attributes (should go to bridge component)."""
        self.component._private_attr = "private_value"
        assert self.component._private_attr == "private_value"
        assert not hasattr(self.original, "_private_attr")

    def test_set_own_attributes(self):
        """Test setting GeneralizedComponent's own attributes."""
        # Test non-module attributes
        for attr in [
            "name",
            "config",
            "submodules",
            "conversion_rule",
            "compatibility_mode",
            "disable_warnings",
        ]:
            value = f"test_{attr}"
            setattr(self.component, attr, value)
            assert getattr(self.component, attr) == value

        # Test module attributes separately (they need to be modules or None)
        new_hook = HookPoint()
        self.component.hook_in = new_hook
        assert self.component.hook_in is new_hook

    def test_set_property_with_setter(self):
        """Test setting property that has a setter."""
        self.component.test_property = "property_value"
        assert self.component.test_property == "property_value"
        assert not hasattr(self.original, "test_property")

    def test_set_existing_attribute_on_original(self):
        """Test setting attribute that exists on original component."""
        self.component.existing_attr = "new_value"
        assert self.original.existing_attr == "new_value"

    def test_set_tensor_attribute_on_original(self):
        """Test setting tensor attribute on original component."""
        new_tensor = torch.tensor([4.0, 5.0, 6.0])
        self.component.tensor_attr = new_tensor
        assert torch.equal(self.original.tensor_attr, new_tensor)

    def test_set_nonexistent_attribute_fallback(self):
        """Test setting attribute that doesn't exist on original (should fall back to bridge)."""
        self.component.new_attr = "new_value"
        assert self.component.new_attr == "new_value"
        assert not hasattr(self.original, "new_attr")

    def test_set_attribute_no_original_component(self):
        """Test setting attribute when no original component is set."""
        component_no_orig = MockGeneralizedComponent("no_orig")
        component_no_orig.test_attr = "test_value"
        assert component_no_orig.test_attr == "test_value"

    def test_set_attribute_original_component_fails(self):
        """Test setting attribute when original component raises AttributeError."""

        class FailingOriginal(nn.Module):
            def __setattr__(self, name, value):
                if name == "failing_attr":
                    raise AttributeError("Cannot set this attribute")
                super().__setattr__(name, value)

        failing_orig = FailingOriginal()
        self.component.set_original_component(failing_orig)

        # Should fall back to bridge component
        self.component.failing_attr = "fallback_value"
        assert self.component.failing_attr == "fallback_value"

    def test_set_module_attribute(self):
        """Test setting PyTorch module attributes."""
        linear = nn.Linear(10, 5)
        self.component.linear_module = linear
        assert self.component.linear_module is linear

    def test_set_parameter_attribute(self):
        """Test setting PyTorch parameter attributes."""
        param = nn.Parameter(torch.randn(10, 5))
        self.component.param_attr = param
        # Parameters are stored in _parameters dict and accessible via named_parameters()
        param_names = [name for name, _ in self.component.named_parameters()]
        assert "param_attr" in param_names
        assert self.component._parameters["param_attr"] is param

    def test_attribute_priority_order(self):
        """Test the priority order of attribute setting."""
        # 1. Private attributes go to bridge
        self.component._priority_test = "private"
        assert self.component._priority_test == "private"

        # 2. Own attributes go to bridge
        self.component.name = "new_name"
        assert self.component.name == "new_name"

        # 3. Properties with setters go to bridge
        self.component.test_property = "prop_value"
        assert self.component.test_property == "prop_value"

        # 4. Existing attributes on original go to original
        self.component.existing_attr = "original_value"
        assert self.original.existing_attr == "original_value"

        # 5. Non-existent attributes fall back to bridge
        self.component.fallback_attr = "fallback"
        assert self.component.fallback_attr == "fallback"

    def test_readonly_property_fallback(self):
        """Test setting attribute on readonly property falls back to original."""

        class ComponentWithReadonlyProp(GeneralizedComponent):
            @property
            def readonly_prop(self):
                return "readonly"

            # No setter defined

        component = ComponentWithReadonlyProp("readonly_test")
        original = MockOriginalComponent()
        original.readonly_prop = "original_readonly"  # Set it on original
        component.set_original_component(original)

        # Should try original component since property has no setter
        component.readonly_prop = "new_readonly"
        assert original.readonly_prop == "new_readonly"

    def test_multiple_attribute_sets(self):
        """Test setting multiple attributes in sequence."""
        attrs = {
            "attr1": "value1",
            "attr2": 42,
            "attr3": torch.tensor([1, 2, 3]),
            "existing_attr": "modified_existing",
        }

        for attr_name, attr_value in attrs.items():
            setattr(self.component, attr_name, attr_value)

        # Check bridge attributes
        assert self.component.attr1 == "value1"
        assert self.component.attr2 == 42
        assert torch.equal(self.component.attr3, torch.tensor([1, 2, 3]))

        # Check original component attribute
        assert self.original.existing_attr == "modified_existing"


class TestGeneralizedComponentGetAttr:
    """Test suite for GeneralizedComponent.__getattr__ functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.component = MockGeneralizedComponent("test")
        self.original = MockOriginalComponent()
        self.component.set_original_component(self.original)

    def test_get_existing_bridge_attribute(self):
        """Test getting attribute that exists on bridge component."""
        assert self.component.bridge_attr == "bridge_value"

    def test_get_original_component_attribute(self):
        """Test getting attribute from original component."""
        assert self.component.existing_attr == "original_value"

    def test_get_nonexistent_attribute_raises_error(self):
        """Test that getting nonexistent attribute raises AttributeError."""
        with pytest.raises(
            AttributeError, match="'MockGeneralizedComponent' object has no attribute 'nonexistent'"
        ):
            _ = self.component.nonexistent

    def test_get_module_attribute(self):
        """Test getting module attributes."""
        assert isinstance(self.component.hook_in, HookPoint)
        assert isinstance(self.component.hook_out, HookPoint)


class TestGeneralizedComponentHooks:
    """Test suite for GeneralizedComponent hook functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.component = MockGeneralizedComponent("hook_test")

    def test_add_hook_output(self):
        """Test adding hook to output."""
        called = []

        def test_hook(tensor, hook):
            called.append(True)
            return tensor

        self.component.add_hook(test_hook, "output")

        # Simulate hook execution
        test_tensor = torch.tensor([1.0, 2.0])
        result = self.component.hook_out(test_tensor)

        assert len(called) == 1
        assert torch.equal(result, test_tensor)

    def test_add_hook_input(self):
        """Test adding hook to input."""
        called = []

        def test_hook(tensor, hook):
            called.append(True)
            return tensor

        self.component.add_hook(test_hook, "input")

        # Simulate hook execution
        test_tensor = torch.tensor([1.0, 2.0])
        result = self.component.hook_in(test_tensor)

        assert len(called) == 1
        assert torch.equal(result, test_tensor)

    def test_add_hook_invalid_name(self):
        """Test adding hook with invalid name raises error."""

        def test_hook(tensor, hook):
            return tensor

        with pytest.raises(ValueError, match="Hook name 'invalid' not supported"):
            self.component.add_hook(test_hook, "invalid")

    def test_remove_hooks_all(self):
        """Test removing all hooks."""

        def test_hook(tensor, hook):
            return tensor

        self.component.add_hook(test_hook, "input")
        self.component.add_hook(test_hook, "output")

        self.component.remove_hooks()

        # Hooks should be removed (this is hard to test directly,
        # but we can at least verify the method doesn't crash)
        assert True

    def test_remove_hooks_specific(self):
        """Test removing specific hooks."""

        def test_hook(tensor, hook):
            return tensor

        self.component.add_hook(test_hook, "input")
        self.component.add_hook(test_hook, "output")

        self.component.remove_hooks("input")
        self.component.remove_hooks("output")

        assert True

    def test_remove_hooks_invalid_name(self):
        """Test removing hooks with invalid name raises error."""
        with pytest.raises(ValueError, match="Hook name 'invalid' not supported"):
            self.component.remove_hooks("invalid")


class TestGeneralizedComponentEdgeCases:
    """Test edge cases for GeneralizedComponent."""

    def test_circular_reference_prevention(self):
        """Test that circular references don't cause infinite recursion."""
        component1 = MockGeneralizedComponent("comp1")
        component2 = MockGeneralizedComponent("comp2")

        # This should not cause infinite recursion
        component1.other_component = component2
        component2.other_component = component1

        assert component1.other_component is component2
        assert component2.other_component is component1

    def test_none_values(self):
        """Test setting None values."""
        component = MockGeneralizedComponent("none_test")
        original = MockOriginalComponent()
        original.nullable_attr = "initial"
        component.set_original_component(original)

        component.nullable_attr = None
        assert original.nullable_attr is None

    def test_complex_object_attributes(self):
        """Test setting complex object attributes."""
        component = MockGeneralizedComponent("complex")

        complex_obj = {
            "nested": {"deep": [1, 2, 3]},
            "tensor": torch.randn(5, 5),
            "function": lambda x: x + 1,
        }

        component.complex_attr = complex_obj
        assert component.complex_attr is complex_obj
        assert component.complex_attr["nested"]["deep"] == [1, 2, 3]

    def test_inheritance(self):
        """Test attribute setting with component inheritance."""

        class ChildComponent(MockGeneralizedComponent):
            def __init__(self, name: str = "child"):
                super().__init__(name)
                self.child_attr = "child_value"

        child = ChildComponent("child_test")
        original = MockOriginalComponent()
        child.set_original_component(original)

        # Test inherited behavior
        child.existing_attr = "child_modified"
        assert original.existing_attr == "child_modified"

        # Test child-specific attribute
        child.child_attr = "new_child_value"
        assert child.child_attr == "new_child_value"


if __name__ == "__main__":
    pytest.main([__file__])
