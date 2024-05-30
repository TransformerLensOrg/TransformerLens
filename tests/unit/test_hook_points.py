from unittest import mock

from torch.utils.hooks import RemovableHandle

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


@mock.patch("torch.utils.hooks.RemovableHandle", autospec=True)
def test_add_hook_prepend(mock_handle):
    mock_handle1 = mock.Mock(spec=RemovableHandle)
    mock_handle1.id = 1
    mock_handle2 = mock.Mock(spec=RemovableHandle)
    mock_handle2.id = 2

    # Set the side effect to return different mocks on subsequent calls
    mock_handle.side_effect = [mock_handle1, mock_handle2]

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
