"""Teardown scope for the bridge's temporary-hook entry points.

`run_with_cache`, `run_with_hooks` and `hooks()` used to tear down with a blanket
`remove_hooks(dir=...)` on every hook point they touched, which also removed hooks the
caller had attached beforehand. Each call now tags its hooks with a context level and
removes only that level.

Assertions are relative to whatever is already attached: the bridge fixture is
session-scoped, so hooks belonging to other tests are not this file's business.
"""

import pytest

PROMPT = "The capital of France is Paris."
USER_HOOK_POINT = "blocks.0.hook_resid_post"


@pytest.fixture()
def bridge(distilgpt2_bridge):
    """Session bridge, with hooks this test adds to USER_HOOK_POINT removed afterwards.

    Removes by handle rather than calling `reset_hooks()`, which would also drop hooks
    belonging to other tests that share the session-scoped bridge.
    """
    hook_point = distilgpt2_bridge.hook_dict[USER_HOOK_POINT]
    preexisting = {
        "fwd": list(hook_point.fwd_hooks),
        "bwd": list(hook_point.bwd_hooks),
    }
    yield distilgpt2_bridge
    for direction, handles in (("fwd", hook_point.fwd_hooks), ("bwd", hook_point.bwd_hooks)):
        for handle in list(handles):
            if handle not in preexisting[direction]:
                handle.hook.remove()
                handles.remove(handle)


@pytest.fixture()
def user_hook(bridge):
    """Attach a caller-owned hook; yields the names it fired under and its handle."""
    fired: list = []

    def record(tensor, hook):
        fired.append(hook.name)
        return tensor

    hook_point = bridge.hook_dict[USER_HOOK_POINT]
    bridge.add_hook(USER_HOOK_POINT, record)
    return fired, hook_point.fwd_hooks[-1]


def assert_hook_survived(bridge, user_hook):
    """The caller's hook is still attached, and still runs on the next forward pass."""
    fired, handle = user_hook
    assert handle in bridge.hook_dict[USER_HOOK_POINT].fwd_hooks

    fired.clear()
    bridge(bridge.to_tokens(PROMPT), return_type="loss")
    # Hooks report the hook point's canonical name, which `blocks.0.hook_resid_post` aliases.
    assert fired == [bridge.hook_dict[USER_HOOK_POINT].name]


def test_run_with_cache_keeps_caller_hooks(bridge, user_hook):
    """A hook attached before `run_with_cache` survives it and still fires."""
    bridge.run_with_cache(bridge.to_tokens(PROMPT), return_type="loss")

    assert_hook_survived(bridge, user_hook)


def test_run_with_hooks_keeps_caller_hooks(bridge, user_hook):
    """Same for `run_with_hooks`, hooking the very same point it has to leave alone."""
    bridge.run_with_hooks(
        bridge.to_tokens(PROMPT),
        return_type="loss",
        fwd_hooks=[(USER_HOOK_POINT, lambda tensor, hook: tensor)],
    )

    assert_hook_survived(bridge, user_hook)


def test_hooks_context_keeps_caller_hooks(bridge, user_hook):
    """Same for the `hooks()` context manager."""
    hook_point = bridge.hook_dict[USER_HOOK_POINT]
    before = len(hook_point.fwd_hooks)

    with bridge.hooks(fwd_hooks=[(USER_HOOK_POINT, lambda tensor, hook: tensor)]):
        assert len(hook_point.fwd_hooks) == before + 1

    assert len(hook_point.fwd_hooks) == before
    assert_hook_survived(bridge, user_hook)


def test_run_with_cache_reset_hooks_end_false_leaves_hooks_attached(bridge):
    """`reset_hooks_end=False` keeps the caching hooks live for a later pass."""
    tokens = bridge.to_tokens(PROMPT)
    hook_point = bridge.hook_dict[USER_HOOK_POINT]
    before = len(hook_point.fwd_hooks)

    _, cache = bridge.run_with_cache(
        tokens, return_type="loss", names_filter=USER_HOOK_POINT, reset_hooks_end=False
    )

    assert len(hook_point.fwd_hooks) > before
    cached = cache[USER_HOOK_POINT].clone()
    bridge(bridge.to_tokens("A different prompt entirely, of another length."), return_type="loss")
    # Still caching: the second pass overwrote the entry.
    assert cache[USER_HOOK_POINT].shape != cached.shape


def test_run_with_cache_clear_contexts(bridge):
    """`clear_contexts=True` wipes the hook contexts of the points it touched."""
    tokens = bridge.to_tokens(PROMPT)
    hook_point = bridge.hook_dict[USER_HOOK_POINT]
    hook_point.ctx["marker"] = 1

    bridge.run_with_cache(tokens, return_type="loss", names_filter=USER_HOOK_POINT)
    assert hook_point.ctx.get("marker") == 1

    bridge.run_with_cache(
        tokens, return_type="loss", names_filter=USER_HOOK_POINT, clear_contexts=True
    )
    assert hook_point.ctx == {}


def test_run_with_hooks_clear_contexts(bridge):
    """`run_with_hooks(clear_contexts=True)` was accepted and ignored; now it acts."""
    tokens = bridge.to_tokens(PROMPT)
    hook_point = bridge.hook_dict[USER_HOOK_POINT]
    hook_point.ctx["marker"] = 1

    bridge.run_with_hooks(
        tokens, return_type="loss", fwd_hooks=[(USER_HOOK_POINT, lambda tensor, hook: tensor)]
    )
    assert hook_point.ctx.get("marker") == 1

    bridge.run_with_hooks(
        tokens,
        return_type="loss",
        fwd_hooks=[(USER_HOOK_POINT, lambda tensor, hook: tensor)],
        clear_contexts=True,
    )
    assert hook_point.ctx == {}


def test_context_level_returns_to_zero(bridge):
    """Nesting bookkeeping unwinds, including when the run raises."""
    tokens = bridge.to_tokens(PROMPT)
    bridge.run_with_cache(tokens, return_type="loss")
    assert bridge.context_level == 0

    with pytest.raises(ValueError):
        bridge.run_with_cache(tokens, return_type="logits", incl_bwd=True)
    assert bridge.context_level == 0

    with bridge.hooks(fwd_hooks=[(USER_HOOK_POINT, lambda tensor, hook: tensor)]):
        assert bridge.context_level == 1
    assert bridge.context_level == 0


def test_context_level_unwinds_when_hooks_raise(bridge):
    """A hook that blows up mid-run must not strand the nesting counter."""
    tokens = bridge.to_tokens(PROMPT)
    hook_point = bridge.hook_dict[USER_HOOK_POINT]
    before = len(hook_point.fwd_hooks)

    def explode(tensor, hook):
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError, match="boom"):
        bridge.run_with_hooks(tokens, return_type="loss", fwd_hooks=[(USER_HOOK_POINT, explode)])
    assert bridge.context_level == 0
    assert len(hook_point.fwd_hooks) == before

    with pytest.raises(RuntimeError, match="boom"):
        with bridge.hooks(fwd_hooks=[(USER_HOOK_POINT, explode)]):
            bridge(tokens, return_type="loss")
    assert bridge.context_level == 0
    assert len(hook_point.fwd_hooks) == before
