"""Tests for the hook-introspection API added for issue #297."""

from unittest import mock

from transformer_lens.hook_points import HookIntrospectionMixin, HookPoint
from transformer_lens.HookedRootModule import HookedRootModule


class _ToyModel(HookedRootModule):
    """Minimal HookedRootModule with two hook points for testing."""

    def __init__(self):
        super().__init__()
        self.hook_a = HookPoint()
        self.hook_b = HookPoint()
        self.setup()


def _my_named_hook(activation, hook):
    return activation


def _other_hook(activation, hook):
    return activation


@mock.patch("torch.utils.hooks.RemovableHandle", autospec=True)
def test_lens_handle_stores_user_hook(mock_handle):
    mock_handle.return_value.id = 0
    hp = HookPoint()
    hp.add_hook(_my_named_hook, dir="fwd")
    assert hp.fwd_hooks[0].user_hook is _my_named_hook


@mock.patch("torch.utils.hooks.RemovableHandle", autospec=True)
def test_hookpoint_repr_includes_hook_count(mock_handle):
    mock_handle.return_value.id = 0
    hp = HookPoint()
    hp.name = "blocks.0.hook_resid_post"
    assert "blocks.0.hook_resid_post" in repr(hp)
    hp.add_hook(_my_named_hook, dir="fwd")
    hp.add_hook(_other_hook, dir="fwd")
    rep = repr(hp)
    assert "2 fwd" in rep
    assert "bwd" not in rep


def test_hookpoint_repr_with_no_name_and_no_hooks():
    assert repr(HookPoint()) == "HookPoint()"


@mock.patch("torch.utils.hooks.RemovableHandle", autospec=True)
def test_list_hooks_empty_model_returns_empty_dict(mock_handle):
    model = _ToyModel()
    assert model.list_hooks() == {}


@mock.patch("torch.utils.hooks.RemovableHandle", autospec=True)
def test_list_hooks_returns_user_callable(mock_handle):
    mock_handle.return_value.id = 0
    model = _ToyModel()
    model.hook_a.add_hook(_my_named_hook, dir="fwd")
    result = model.list_hooks()
    assert set(result.keys()) == {"hook_a"}
    handles = result["hook_a"]
    assert len(handles) == 1
    assert handles[0].user_hook is _my_named_hook


@mock.patch("torch.utils.hooks.RemovableHandle", autospec=True)
def test_list_hooks_name_filter_string(mock_handle):
    mock_handle.return_value.id = 0
    model = _ToyModel()
    model.hook_a.add_hook(_my_named_hook, dir="fwd")
    model.hook_b.add_hook(_other_hook, dir="fwd")
    assert set(model.list_hooks(name_filter="hook_a").keys()) == {"hook_a"}


@mock.patch("torch.utils.hooks.RemovableHandle", autospec=True)
def test_list_hooks_name_filter_list(mock_handle):
    mock_handle.return_value.id = 0
    model = _ToyModel()
    model.hook_a.add_hook(_my_named_hook, dir="fwd")
    model.hook_b.add_hook(_other_hook, dir="fwd")
    assert set(model.list_hooks(name_filter=["hook_a", "hook_b"]).keys()) == {"hook_a", "hook_b"}
    assert set(model.list_hooks(name_filter=["hook_b"]).keys()) == {"hook_b"}


@mock.patch("torch.utils.hooks.RemovableHandle", autospec=True)
def test_list_hooks_name_filter_callable(mock_handle):
    mock_handle.return_value.id = 0
    model = _ToyModel()
    model.hook_a.add_hook(_my_named_hook, dir="fwd")
    model.hook_b.add_hook(_other_hook, dir="fwd")
    result = model.list_hooks(name_filter=lambda n: n.endswith("a"))
    assert set(result.keys()) == {"hook_a"}


@mock.patch("torch.utils.hooks.RemovableHandle", autospec=True)
def test_list_hooks_direction_filter(mock_handle):
    mock_handle.return_value.id = 0
    model = _ToyModel()
    model.hook_a.add_hook(_my_named_hook, dir="fwd")
    model.hook_a.add_hook(_other_hook, dir="bwd")
    assert len(model.list_hooks(dir="fwd")["hook_a"]) == 1
    assert len(model.list_hooks(dir="bwd")["hook_a"]) == 1
    assert len(model.list_hooks(dir="both")["hook_a"]) == 2


@mock.patch("torch.utils.hooks.RemovableHandle", autospec=True)
def test_list_hooks_excludes_permanent_when_requested(mock_handle):
    mock_handle.return_value.id = 0
    model = _ToyModel()
    model.hook_a.add_hook(_my_named_hook, dir="fwd", is_permanent=True)
    model.hook_a.add_hook(_other_hook, dir="fwd", is_permanent=False)
    assert len(model.list_hooks(including_permanent=True)["hook_a"]) == 2
    handles = model.list_hooks(including_permanent=False)["hook_a"]
    assert len(handles) == 1
    assert handles[0].user_hook is _other_hook


@mock.patch("torch.utils.hooks.RemovableHandle", autospec=True)
def test_mixin_works_on_class_with_hook_dict_attribute(mock_handle):
    """Pin the duck-typed contract: mixin reads ``hook_dict`` off any class that exposes it."""
    mock_handle.return_value.id = 0

    class Bag(HookIntrospectionMixin):
        def __init__(self):
            hp = HookPoint()
            hp.add_hook(_my_named_hook, dir="fwd")
            self.hook_dict = {"only_hook": hp}

    result = Bag().list_hooks()
    assert set(result.keys()) == {"only_hook"}
    assert result["only_hook"][0].user_hook is _my_named_hook


@mock.patch("torch.utils.hooks.RemovableHandle", autospec=True)
def test_mixin_works_on_class_with_hook_dict_property(mock_handle):
    """``getattr`` indirection must accept a ``@property`` provider too (bridge case)."""
    mock_handle.return_value.id = 0

    class PropertyBag(HookIntrospectionMixin):
        def __init__(self):
            self._hooks = {"only_hook": HookPoint()}
            self._hooks["only_hook"].add_hook(_my_named_hook, dir="fwd")

        @property
        def hook_dict(self):
            return self._hooks

    result = PropertyBag().list_hooks()
    assert result["only_hook"][0].user_hook is _my_named_hook
