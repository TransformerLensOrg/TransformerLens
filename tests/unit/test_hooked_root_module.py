from unittest.mock import Mock

from transformer_lens.hook_points import HookedRootModule

MODEL_NAME = "solu-2l"


def test_enable_hook_with_name():
    model = HookedRootModule()
    model.mod_dict = {"linear": Mock()}
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
        "linear": Mock(),
        "attn": Mock(),
    }

    enabled = lambda x: x == "attn"

    hook = lambda x: False
    dir = "bwd"

    print(hook_points.items())
    model._enable_hooks_for_points(
        hook_points=hook_points.items(), enabled=enabled, hook=hook, dir=dir
    )

    hook_points["attn"].add_hook.assert_called_with(hook, dir="bwd", level=5)
    hook_points["linear"].add_hook.assert_not_called()


def test_enable_hook_with_string_param():
    model = HookedRootModule()
    model.mod_dict = {"linear": Mock()}
    model.context_level = 5

    hook = lambda x: False
    dir = "fwd"

    model._enable_hook("linear", hook=hook, dir=dir)

    model.mod_dict["linear"].add_hook.assert_called_with(hook, dir="fwd", level=5)


def test_enable_hook_with_callable_param():
    model = HookedRootModule()
    model.mod_dict = {"linear": Mock()}
    model.hook_dict = {
        "linear": Mock(),
        "attn": Mock(),
    }
    model.context_level = 5

    enabled = lambda x: x == "attn"

    hook = lambda x: False
    dir = "fwd"

    model._enable_hook(enabled, hook=hook, dir=dir)

    model.mod_dict["linear"].add_hook.assert_not_called()
    model.hook_dict["attn"].add_hook.assert_called_with(hook, dir="fwd", level=5)
    model.hook_dict["linear"].add_hook.assert_not_called()
