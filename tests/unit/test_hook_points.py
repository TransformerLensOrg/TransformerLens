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


@mock.patch("torch.utils.hooks.RemovableHandle")
def test_add_hook_prepend(mock_handle):
    mock_handle.id = 0
    mock_handle.next_id = 1

    hook_point, _ = setup_hook_point_and_hook()

    def hook1(activation, hook):
        return activation

    def hook2(activation, hook):
        return activation

    hook_point.add_hook(hook1, dir="fwd")
    hook_point.add_hook(hook2, dir="fwd", prepend=True)

    assert len(hook_point.fwd_hooks) == 2
    assert hook_point.fwd_hooks[0].hook.id == 2
    assert hook_point.fwd_hooks[1].hook.id == 1


def test_enable_reshape():
    """Test that enable_reshape sets the reshape functions correctly."""
    hook_point = HookPoint()
    
    def input_reshape(x):
        return x * 2
    
    def output_reshape(x):
        return x + 1
    
    hook_point.enable_reshape(input_reshape, output_reshape)
    
    assert hook_point.input_reshape_fn is input_reshape
    assert hook_point.output_reshape_fn is output_reshape


def test_enable_reshape_with_none():
    """Test that enable_reshape works with None values."""
    hook_point = HookPoint()
    
    def input_reshape(x):
        return x * 2
    
    hook_point.enable_reshape(input_reshape, None)
    
    assert hook_point.input_reshape_fn is input_reshape
    assert hook_point.output_reshape_fn is None


def test_reshape_functionality_integration():
    """Test that reshape functions work in an integration context."""
    import torch

    # Create a simple test module that uses HookPoint
    class TestModule(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.hook_point = HookPoint()
            
        def forward(self, x):
            return self.hook_point(x)
    
    module = TestModule()
    
    # Set up reshape functions
    def input_reshape(x):
        return x * 2  # Double the input
    
    def output_reshape(x):
        return x + 10  # Add 10 to the output
    
    module.hook_point.enable_reshape(input_reshape, output_reshape)
    
    # Set up a hook that modifies the activation
    def test_hook(activation, hook):
        return activation + 1  # Add 1 to each element
    
    module.hook_point.add_hook(test_hook, dir="fwd")
    
    # Test the full pipeline
    test_input = torch.tensor([1.0, 2.0, 3.0])
    result = module(test_input)
    
    # The pipeline should be:
    # 1. input_reshape: [1,2,3] * 2 = [2,4,6]
    # 2. hook: [2,4,6] + 1 = [3,5,7] 
    # 3. output_reshape: [3,5,7] + 10 = [13,15,17]
    expected = torch.tensor([13.0, 15.0, 17.0])
    assert torch.equal(result, expected)


def test_reshape_functionality_hook_returns_none_integration():
    """Test that output reshape is not applied when hook returns None."""
    import torch
    
    class TestModule(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.hook_point = HookPoint()
            
        def forward(self, x):
            return self.hook_point(x)
    
    module = TestModule()
    
    # Set up reshape functions
    def input_reshape(x):
        return x * 2
    
    def output_reshape(x):
        return x + 10
    
    module.hook_point.enable_reshape(input_reshape, output_reshape)
    
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
