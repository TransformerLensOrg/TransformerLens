"""Focused hook tests for TransformerBridge.

Tests that Bridge hooks fire correctly, can modify activations,
and that run_with_cache returns populated caches.
"""

import pytest
import torch

from transformer_lens.model_bridge import TransformerBridge

MODEL = "gpt2"


@pytest.fixture(scope="module")
def bridge():
    return TransformerBridge.boot_transformers(MODEL, device="cpu")


def test_hook_fires_on_forward(bridge):
    """A registered forward hook must fire exactly once per forward pass."""
    count = 0

    def hook_fn(tensor, hook):
        nonlocal count
        count += 1
        return tensor

    bridge.run_with_hooks(
        "Hello world",
        fwd_hooks=[("blocks.0.hook_resid_pre", hook_fn)],
    )
    assert count == 1


def test_hook_receives_tensor(bridge):
    """Hook must receive a tensor with batch and sequence dimensions."""
    captured = {}

    def hook_fn(tensor, hook):
        captured["shape"] = tensor.shape
        captured["dtype"] = tensor.dtype
        return tensor

    bridge.run_with_hooks(
        "Hello",
        fwd_hooks=[("blocks.0.hook_resid_pre", hook_fn)],
    )
    assert "shape" in captured
    assert len(captured["shape"]) >= 2  # at least [batch, seq, ...]
    assert captured["shape"][0] >= 1  # batch >= 1


def test_hook_can_modify_output(bridge):
    """Zeroing a residual stream hook must change the final output."""
    with torch.no_grad():
        normal_output = bridge("Hello world")

        def zero_hook(tensor, hook):
            return torch.zeros_like(tensor)

        modified_output = bridge.run_with_hooks(
            "Hello world",
            fwd_hooks=[("blocks.0.hook_resid_pre", zero_hook)],
        )

    assert not torch.allclose(normal_output, modified_output)


def test_run_with_cache_returns_activations(bridge):
    """run_with_cache must return a non-empty cache with expected keys."""
    with torch.no_grad():
        _, cache = bridge.run_with_cache("Hello world")

    assert len(cache) > 0
    # Must contain at least residual stream hooks
    cache_keys = list(cache.keys())
    assert any(
        "hook_resid" in k for k in cache_keys
    ), f"No residual stream hooks in cache. Keys: {cache_keys[:10]}"


def test_cache_values_are_tensors_with_correct_batch(bridge):
    """All cached values must be tensors with batch dim matching input."""
    with torch.no_grad():
        _, cache = bridge.run_with_cache("Hello")

    for key, value in cache.items():
        assert isinstance(value, torch.Tensor), f"Cache[{key}] is {type(value)}, not Tensor"
        assert value.shape[0] == 1, f"Cache[{key}] batch dim is {value.shape[0]}, expected 1"


def test_multiple_hooks_fire_independently(bridge):
    """Multiple hooks on different points must each fire independently."""
    fired = set()

    def make_hook(name):
        def hook_fn(tensor, hook):
            fired.add(name)
            return tensor

        return hook_fn

    bridge.run_with_hooks(
        "Hello",
        fwd_hooks=[
            ("blocks.0.hook_resid_pre", make_hook("resid_pre_0")),
            ("blocks.0.hook_resid_post", make_hook("resid_post_0")),
        ],
    )
    assert "resid_pre_0" in fired
    assert "resid_post_0" in fired


@pytest.mark.xfail(reason="add_perma_hook not yet implemented on TransformerBridge")
def test_perma_hook_persists_across_calls(bridge):
    """A permanent hook must fire on every forward pass until explicitly removed."""
    count = 0

    def hook_fn(tensor, hook):
        nonlocal count
        count += 1
        return tensor

    bridge.add_perma_hook("blocks.0.hook_resid_pre", hook_fn)
    try:
        with torch.no_grad():
            bridge("Hello")
            assert count == 1
            bridge("World")
            assert count == 2  # still fires on second call
    finally:
        bridge.reset_hooks()

    # After reset, hook should no longer fire
    count = 0
    with torch.no_grad():
        bridge("Hello again")
    assert count == 0


def test_hook_context_manager_cleans_up(bridge):
    """Hooks added via run_with_hooks must not persist after the call returns."""
    count = 0

    def hook_fn(tensor, hook):
        nonlocal count
        count += 1
        return tensor

    # Run with hook
    with torch.no_grad():
        bridge.run_with_hooks(
            "Hello",
            fwd_hooks=[("blocks.0.hook_resid_pre", hook_fn)],
        )
    assert count == 1

    # Run again without hooks — count should NOT increase
    count = 0
    with torch.no_grad():
        bridge("Hello")
    assert count == 0, "Hook persisted after run_with_hooks returned"


def test_cache_with_names_filter(bridge):
    """run_with_cache with names_filter must return only matching keys."""
    with torch.no_grad():
        _, full_cache = bridge.run_with_cache("Hello")
        _, filtered_cache = bridge.run_with_cache(
            "Hello",
            names_filter=lambda name: "hook_resid_pre" in name,
        )

    # Filtered cache should be a strict subset
    assert len(filtered_cache) > 0
    assert len(filtered_cache) < len(full_cache)
    for key in filtered_cache:
        assert "hook_resid_pre" in key, f"Unexpected key in filtered cache: {key}"
