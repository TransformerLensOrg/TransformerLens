from unittest import mock

from transformer_lens.hook_points import HookPoint


def setup_hook_point_and_hook():
    hook_point = HookPoint()

    def hook(activation, hook):
        return activation

    return hook_point, hook


@mock.patch("torch.utils.hooks.RemovableHandle", autospec=True)
def test_add_hook_forward(mock_handle):
    mock_handle.return_value.id = 0
    hook_point, hook = setup_hook_point_and_hook()
    hook_point.add_hook(hook, dir="fwd")
    assert len(hook_point.fwd_hooks) == 1


@mock.patch("torch.utils.hooks.RemovableHandle", autospec=True)
def test_add_hook_backward(mock_handle):
    mock_handle.return_value.id = 0
    hook_point, hook = setup_hook_point_and_hook()
    hook_point.add_hook(hook, dir="bwd")
    assert len(hook_point.bwd_hooks) == 1


@mock.patch("torch.utils.hooks.RemovableHandle", autospec=True)
def test_add_hook_permanent(mock_handle):
    mock_handle.return_value.id = 0
    hook_point, hook = setup_hook_point_and_hook()
    hook_point.add_hook(hook, dir="fwd", is_permanent=True)
    assert hook_point.fwd_hooks[0].is_permanent


@mock.patch("torch.utils.hooks.RemovableHandle", autospec=True)
def test_add_hook_with_level(mock_handle):
    mock_handle.return_value.id = 0
    hook_point, hook = setup_hook_point_and_hook()
    hook_point.add_hook(hook, dir="fwd", level=5)
    assert hook_point.fwd_hooks[0].context_level == 5


@mock.patch("transformer_lens.hook_points.LensHandle")
@mock.patch("torch.utils.hooks.RemovableHandle")
def test_add_hook_prepend(mock_handle, mock_lens_handle):
    mock_handle.id = 0
    mock_handle.next_id = 1

    hook_point, _ = setup_hook_point_and_hook()

    def hook1(activation, hook):
        return activation

    def hook2(activation, hook):
        return activation

    # Make LensHandle constructor return a simple container capturing the pt_handle ('hook')
    class _LensHandleBox:
        def __init__(self, handle, is_permanent, context_level):
            self.hook = handle
            self.is_permanent = is_permanent
            self.context_level = context_level

    mock_lens_handle.side_effect = _LensHandleBox

    # Override register_forward_hook to return mocked handles with incremental ids
    next_id = {"val": 1}

    def fake_register_forward_hook(fn, prepend=False):
        handle = mock.MagicMock()
        handle.id = next_id["val"]
        next_id["val"] += 1
        return handle

    hook_point.register_forward_hook = fake_register_forward_hook  # type: ignore[assignment]

    hook_point.add_hook(hook1, dir="fwd")
    hook_point.add_hook(hook2, dir="fwd", prepend=True)

    assert len(hook_point.fwd_hooks) == 2
    assert hook_point.fwd_hooks[0].hook.id == 2
    assert hook_point.fwd_hooks[1].hook.id == 1


def test_enable_reshape():
    """Test that enable_reshape sets the hook conversion correctly."""
    from transformer_lens.conversion_utils.conversion_steps.base_tensor_conversion import (
        BaseTensorConversion,
    )

    class TestHookConversion(BaseTensorConversion):
        def handle_conversion(self, input_value, *full_context):
            return input_value * 2

        def revert(self, input_value, *full_context):
            return input_value + 1

    hook_point = HookPoint()
    conversion = TestHookConversion()

    hook_point.enable_reshape(conversion)

    assert hook_point.hook_conversion is conversion


def test_enable_reshape_with_none():
    """Test that enable_reshape works with None values."""
    hook_point = HookPoint()

    hook_point.enable_reshape(None)

    assert hook_point.hook_conversion is None


def test_reshape_functionality_integration():
    """Test that hook conversion works in an integration context."""
    import torch

    from transformer_lens.conversion_utils.conversion_steps.base_tensor_conversion import (
        BaseTensorConversion,
    )

    # Create a test hook conversion
    class TestHookConversion(BaseTensorConversion):
        def handle_conversion(self, input_value, *full_context):
            return input_value * 2  # Double the input

        def revert(self, input_value, *full_context):
            return input_value + 10  # Add 10 to the output

    # Create a simple test module that uses HookPoint
    class TestModule(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.hook_point = HookPoint()

        def forward(self, x):
            return self.hook_point(x)

    module = TestModule()

    # Set up hook conversion
    conversion = TestHookConversion()
    module.hook_point.enable_reshape(conversion)

    # Set up a hook that modifies the activation
    def test_hook(activation, hook):
        return activation + 1  # Add 1 to each element

    module.hook_point.add_hook(test_hook, dir="fwd")

    # Test the full pipeline
    test_input = torch.tensor([1.0, 2.0, 3.0])
    result = module(test_input)

    # The pipeline should be:
    # 1. conversion.convert(): [1,2,3] * 2 = [2,4,6]
    # 2. hook: [2,4,6] + 1 = [3,5,7]
    # 3. conversion.revert(): [3,5,7] + 10 = [13,15,17]
    expected = torch.tensor([13.0, 15.0, 17.0])
    assert torch.equal(result, expected)


def test_reshape_functionality_hook_returns_none_integration():
    """Test that output revert is not applied when hook returns None."""
    import torch

    from transformer_lens.conversion_utils.conversion_steps.base_tensor_conversion import (
        BaseTensorConversion,
    )

    # Create a test hook conversion
    class TestHookConversion(BaseTensorConversion):
        def handle_conversion(self, input_value, *full_context):
            return input_value * 2

        def revert(self, input_value, *full_context):
            return input_value + 10

    class TestModule(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.hook_point = HookPoint()

        def forward(self, x):
            return self.hook_point(x)

    module = TestModule()

    # Set up hook conversion
    conversion = TestHookConversion()
    module.hook_point.enable_reshape(conversion)

    # Set up a hook that returns None
    def test_hook(activation, hook):
        return None

    module.hook_point.add_hook(test_hook, dir="fwd")

    # Test the pipeline
    test_input = torch.tensor([1.0, 2.0, 3.0])
    result = module(test_input)

    # Since hook returns None, the original input should be returned
    # (HookPoint's forward method returns the input when no valid hook result)
    assert torch.equal(result, test_input)


class TestHookPointHasHooks:
    """Comprehensive test suite for HookPoint.has_hooks method."""

    def setup_method(self):
        """Set up fresh HookPoint and sample hook for each test."""
        self.hook_point = HookPoint()

        def sample_hook(activation, hook):
            return activation

        self.sample_hook = sample_hook

    def test_no_hooks_returns_false(self):
        """Test that has_hooks returns False when no hooks are present."""
        assert not self.hook_point.has_hooks()
        assert not self.hook_point.has_hooks(dir="fwd")
        assert not self.hook_point.has_hooks(dir="bwd")
        assert not self.hook_point.has_hooks(dir="both")

    def test_forward_hook_detection(self):
        """Test detection of forward hooks."""
        # Add a forward hook
        self.hook_point.add_hook(self.sample_hook, dir="fwd")

        # Should detect forward hooks
        assert self.hook_point.has_hooks()
        assert self.hook_point.has_hooks(dir="fwd")
        assert self.hook_point.has_hooks(dir="both")

        # Should not detect backward hooks
        assert not self.hook_point.has_hooks(dir="bwd")

    def test_backward_hook_detection(self):
        """Test detection of backward hooks."""
        # Add a backward hook
        self.hook_point.add_hook(self.sample_hook, dir="bwd")

        # Should detect backward hooks
        assert self.hook_point.has_hooks()
        assert self.hook_point.has_hooks(dir="bwd")
        assert self.hook_point.has_hooks(dir="both")

        # Should not detect forward hooks
        assert not self.hook_point.has_hooks(dir="fwd")

    def test_both_direction_hooks(self):
        """Test detection when both forward and backward hooks are present."""
        # Add both forward and backward hooks
        self.hook_point.add_hook(self.sample_hook, dir="fwd")
        self.hook_point.add_hook(self.sample_hook, dir="bwd")

        # All directions should detect hooks
        assert self.hook_point.has_hooks()
        assert self.hook_point.has_hooks(dir="fwd")
        assert self.hook_point.has_hooks(dir="bwd")
        assert self.hook_point.has_hooks(dir="both")

    def test_permanent_hook_detection(self):
        """Test detection of permanent hooks."""
        # Add a permanent forward hook
        self.hook_point.add_hook(self.sample_hook, dir="fwd", is_permanent=True)

        # Should detect permanent hooks by default
        assert self.hook_point.has_hooks()
        assert self.hook_point.has_hooks(including_permanent=True)

        # Should not detect when excluding permanent hooks
        assert not self.hook_point.has_hooks(including_permanent=False)

    def test_non_permanent_hook_detection(self):
        """Test detection of non-permanent hooks."""
        # Add a non-permanent forward hook
        self.hook_point.add_hook(self.sample_hook, dir="fwd", is_permanent=False)

        # Should detect non-permanent hooks regardless of including_permanent setting
        assert self.hook_point.has_hooks()
        assert self.hook_point.has_hooks(including_permanent=True)
        assert self.hook_point.has_hooks(including_permanent=False)

    def test_mixed_permanent_hooks(self):
        """Test detection with mix of permanent and non-permanent hooks."""
        # Add both permanent and non-permanent hooks
        self.hook_point.add_hook(self.sample_hook, dir="fwd", is_permanent=True)
        self.hook_point.add_hook(self.sample_hook, dir="fwd", is_permanent=False)

        # Should detect hooks in both cases
        assert self.hook_point.has_hooks(including_permanent=True)
        assert self.hook_point.has_hooks(including_permanent=False)

    def test_only_permanent_hooks(self):
        """Test detection when only permanent hooks are present."""
        # Add only permanent hooks
        self.hook_point.add_hook(self.sample_hook, dir="fwd", is_permanent=True)
        self.hook_point.add_hook(self.sample_hook, dir="bwd", is_permanent=True)

        # Should detect when including permanent
        assert self.hook_point.has_hooks(including_permanent=True)
        assert self.hook_point.has_hooks(dir="fwd", including_permanent=True)
        assert self.hook_point.has_hooks(dir="bwd", including_permanent=True)

        # Should not detect when excluding permanent
        assert not self.hook_point.has_hooks(including_permanent=False)
        assert not self.hook_point.has_hooks(dir="fwd", including_permanent=False)
        assert not self.hook_point.has_hooks(dir="bwd", including_permanent=False)

    def test_context_level_filtering(self):
        """Test context level filtering functionality."""
        # Add hooks at different context levels
        self.hook_point.add_hook(self.sample_hook, dir="fwd", level=0)
        self.hook_point.add_hook(self.sample_hook, dir="fwd", level=1)
        self.hook_point.add_hook(self.sample_hook, dir="bwd", level=2)

        # Should detect hooks at specific levels
        assert self.hook_point.has_hooks(level=0)
        assert self.hook_point.has_hooks(level=1)
        assert self.hook_point.has_hooks(level=2)

        # Should not detect hooks at non-existent levels
        assert not self.hook_point.has_hooks(level=3)
        assert not self.hook_point.has_hooks(level=-1)

        # Should detect all hooks when level is None
        assert self.hook_point.has_hooks(level=None)

    def test_context_level_with_direction(self):
        """Test context level filtering combined with direction filtering."""
        # Add hooks at different levels and directions
        self.hook_point.add_hook(self.sample_hook, dir="fwd", level=0)
        self.hook_point.add_hook(self.sample_hook, dir="bwd", level=1)

        # Test specific combinations
        assert self.hook_point.has_hooks(dir="fwd", level=0)
        assert self.hook_point.has_hooks(dir="bwd", level=1)

        # Test non-matching combinations
        assert not self.hook_point.has_hooks(dir="fwd", level=1)
        assert not self.hook_point.has_hooks(dir="bwd", level=0)

    def test_context_level_with_permanent_flags(self):
        """Test context level filtering combined with permanent hook filtering."""
        # Add permanent and non-permanent hooks at different levels
        self.hook_point.add_hook(self.sample_hook, dir="fwd", level=0, is_permanent=True)
        self.hook_point.add_hook(self.sample_hook, dir="fwd", level=1, is_permanent=False)

        # Test combinations
        assert self.hook_point.has_hooks(level=0, including_permanent=True)
        assert not self.hook_point.has_hooks(level=0, including_permanent=False)
        assert self.hook_point.has_hooks(level=1, including_permanent=True)
        assert self.hook_point.has_hooks(level=1, including_permanent=False)

    def test_all_parameters_combined(self):
        """Test all parameters combined in various ways."""
        # Create a complex setup with multiple hooks
        self.hook_point.add_hook(self.sample_hook, dir="fwd", level=0, is_permanent=True)
        self.hook_point.add_hook(self.sample_hook, dir="fwd", level=1, is_permanent=False)
        self.hook_point.add_hook(self.sample_hook, dir="bwd", level=0, is_permanent=False)
        self.hook_point.add_hook(self.sample_hook, dir="bwd", level=2, is_permanent=True)

        # Test specific combinations
        assert self.hook_point.has_hooks(dir="fwd", level=0, including_permanent=True)
        assert not self.hook_point.has_hooks(dir="fwd", level=0, including_permanent=False)
        assert self.hook_point.has_hooks(dir="fwd", level=1, including_permanent=False)
        assert self.hook_point.has_hooks(dir="bwd", level=0, including_permanent=False)
        assert not self.hook_point.has_hooks(dir="bwd", level=1, including_permanent=False)
        assert self.hook_point.has_hooks(dir="bwd", level=2, including_permanent=True)

    def test_invalid_direction_raises_error(self):
        """Test that invalid direction parameter raises error (caught by type checking)."""
        # Note: beartype catches this at the parameter level before reaching the ValueError
        import pytest
        from beartype.roar import BeartypeCallHintParamViolation

        with pytest.raises(BeartypeCallHintParamViolation):
            self.hook_point.has_hooks(dir="invalid")  # type: ignore

    def test_multiple_hooks_same_criteria(self):
        """Test detection when multiple hooks match the same criteria."""
        # Add multiple hooks with same criteria
        self.hook_point.add_hook(self.sample_hook, dir="fwd", level=0, is_permanent=False)
        self.hook_point.add_hook(self.sample_hook, dir="fwd", level=0, is_permanent=False)
        self.hook_point.add_hook(self.sample_hook, dir="fwd", level=0, is_permanent=False)

        # Should still detect hooks (method returns True on first match)
        assert self.hook_point.has_hooks(dir="fwd", level=0, including_permanent=False)

    def test_hook_removal_affects_detection(self):
        """Test that removing hooks affects detection."""
        # Add a hook
        self.hook_point.add_hook(self.sample_hook, dir="fwd")
        assert self.hook_point.has_hooks()

        # Remove all hooks
        self.hook_point.remove_hooks(dir="both")
        assert not self.hook_point.has_hooks()

    def test_default_parameter_values(self):
        """Test that default parameter values work correctly."""
        # Add hooks to test defaults
        self.hook_point.add_hook(self.sample_hook, dir="fwd", is_permanent=True, level=0)
        self.hook_point.add_hook(self.sample_hook, dir="bwd", is_permanent=False, level=1)

        # Test default behavior (dir="both", including_permanent=True, level=None)
        assert self.hook_point.has_hooks()

        # This should be equivalent to:
        assert self.hook_point.has_hooks(dir="both", including_permanent=True, level=None)

    def test_edge_case_empty_after_filtering(self):
        """Test edge case where hooks exist but are filtered out."""
        # Add hooks that will be filtered out
        self.hook_point.add_hook(self.sample_hook, dir="fwd", is_permanent=True, level=5)

        # These should not detect the hook due to filtering
        assert not self.hook_point.has_hooks(including_permanent=False)
        assert not self.hook_point.has_hooks(dir="bwd")
        assert not self.hook_point.has_hooks(level=0)
        assert not self.hook_point.has_hooks(dir="bwd", level=5, including_permanent=True)

    def test_functional_hook_execution_still_works(self):
        """Test that has_hooks doesn't interfere with actual hook functionality."""
        import torch

        results = []

        def test_hook(activation, hook):
            results.append("hook_called")
            return activation

        # Add hook and verify detection
        self.hook_point.add_hook(test_hook, dir="fwd")
        assert self.hook_point.has_hooks()

        # Execute hook and verify it still works
        test_input = torch.tensor([1.0, 2.0, 3.0])
        output = self.hook_point(test_input)

        assert torch.equal(output, test_input)
        assert "hook_called" in results

    def test_hook_point_with_conversions(self):
        """Test has_hooks with hook conversions if they exist."""
        import torch

        # This test ensures has_hooks works even when hook conversions are involved
        def simple_hook(activation, hook):
            return activation * 2

        # Add hook
        self.hook_point.add_hook(simple_hook, dir="fwd")

        # Should detect hook regardless of any internal conversions
        assert self.hook_point.has_hooks()
        assert self.hook_point.has_hooks(dir="fwd")

        # Test actual functionality still works
        test_input = torch.tensor([1.0, 2.0])
        output = self.hook_point(test_input)
        expected = torch.tensor([2.0, 4.0])
        assert torch.allclose(output, expected)
