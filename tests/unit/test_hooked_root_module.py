import warnings
from unittest.mock import Mock

from transformer_lens.hook_points import HookPoint
from transformer_lens.HookedRootModule import HookedRootModule

MODEL_NAME = "solu-2l"


def test_legacy_hook_points_import_still_works():
    """Back-compat: pre-3.0 code imported HookedRootModule from transformer_lens.hook_points.

    The class moved to its own module; the shim re-exports it with a DeprecationWarning.
    """
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        from transformer_lens.hook_points import HookedRootModule as Legacy

    assert Legacy is HookedRootModule
    deprecations = [w for w in caught if issubclass(w.category, DeprecationWarning)]
    assert deprecations, "expected a DeprecationWarning from the legacy import path"
    assert "transformer_lens.HookedRootModule" in str(deprecations[0].message)


def test_legacy_hook_points_attribute_error_for_unknown_names():
    """The shim only re-exports HookedRootModule — other missing attrs should raise."""
    import transformer_lens.hook_points as hp

    try:
        hp.DefinitelyNotARealSymbol  # noqa: B018
    except AttributeError as exc:
        assert "DefinitelyNotARealSymbol" in str(exc)
    else:
        raise AssertionError("expected AttributeError for unknown attribute")


def test_enable_hook_with_name():
    model = HookedRootModule()
    model.mod_dict = {"linear": Mock(HookPoint)}
    model.context_level = 5

    hook = lambda x: False
    dir = "fwd"

    model._enable_hook_with_name("linear", hook=hook, dir=dir)

    model.mod_dict["linear"].add_hook.assert_called_with(hook, dir="fwd", level=5)


def test_enable_hooks_for_points():
    model = HookedRootModule()
    model.mod_dict = {}
    model.context_level = 5

    hook_points = {
        "linear": Mock(HookPoint),
        "attn": Mock(HookPoint),
    }

    enabled = lambda x: x == "attn"

    hook = lambda x: False
    dir = "bwd"

    model._enable_hooks_for_points(
        hook_points=hook_points.items(), enabled=enabled, hook=hook, dir=dir
    )

    hook_points["attn"].add_hook.assert_called_with(hook, dir="bwd", level=5)
    hook_points["linear"].add_hook.assert_not_called()


def test_enable_hook_with_string_param():
    model = HookedRootModule()
    model.mod_dict = {"linear": Mock(HookPoint)}
    model.context_level = 5

    hook = lambda x: False
    dir = "fwd"

    model._enable_hook("linear", hook=hook, dir=dir)

    model.mod_dict["linear"].add_hook.assert_called_with(hook, dir="fwd", level=5)


def test_enable_hook_with_callable_param():
    model = HookedRootModule()
    model.mod_dict = {"linear": Mock(HookPoint)}
    model.hook_dict = {
        "linear": Mock(HookPoint),
        "attn": Mock(HookPoint),
    }
    model.context_level = 5

    enabled = lambda x: x == "attn"

    hook = lambda x: False
    dir = "fwd"

    model._enable_hook(enabled, hook=hook, dir=dir)

    model.mod_dict["linear"].add_hook.assert_not_called()
    model.hook_dict["attn"].add_hook.assert_called_with(hook, dir="fwd", level=5)
    model.hook_dict["linear"].add_hook.assert_not_called()
