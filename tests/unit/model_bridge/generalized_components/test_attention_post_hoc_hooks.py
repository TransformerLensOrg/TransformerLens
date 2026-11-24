"""Post-hoc pattern/scores hook firing on delegating AttentionBridge.

When the wrapped HF module computes attention internally and merely returns the
weights (vision towers, any module without the ``accepts_pattern_fn`` seam),
the bridge fires hook_pattern/hook_attn_scores after the fact. An edit returned
from such a hook cannot reach the model's output, so the bridge must warn about
the discarded write — and stay silent for read-only (caching) hooks.
"""

from __future__ import annotations

import warnings
from types import SimpleNamespace

import pytest
import torch

from transformer_lens.config import TransformerBridgeConfig
from transformer_lens.model_bridge import TransformerBridge
from transformer_lens.model_bridge.generalized_components.attention import (
    AttentionBridge,
)


class _PatternReturningAttention(torch.nn.Module):
    """HF-style attention returning (output, attn_weights) with the weights
    already consumed internally — the post-hoc hook path."""

    def forward(self, hidden_states: torch.Tensor, **kwargs):
        batch, seq, _ = hidden_states.shape
        scores = torch.einsum("bqd,bkd->bqk", hidden_states, hidden_states)
        weights = torch.softmax(scores, dim=-1).unsqueeze(1).expand(batch, 2, seq, seq)
        return hidden_states * 2.0, weights


@pytest.fixture()
def bridge() -> AttentionBridge:
    attn = AttentionBridge(name="attn", config=SimpleNamespace())
    attn.set_original_component(_PatternReturningAttention())
    return attn


@pytest.fixture()
def hidden_states() -> torch.Tensor:
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)
        return torch.randn(2, 3, 4)


def _run(bridge: AttentionBridge, hidden_states: torch.Tensor):
    with torch.no_grad():
        return bridge(hidden_states=hidden_states)


def _discard_warnings(records) -> list:
    return [r for r in records if "edit is discarded" in str(r.message)]


def test_editing_pattern_hook_warns_and_output_is_unchanged(bridge, hidden_states):
    base_out, _ = _run(bridge, hidden_states)
    handle = bridge.hook_pattern.register_forward_hook(lambda m, i, o: torch.zeros_like(o))
    try:
        with pytest.warns(UserWarning, match="edit is discarded"):
            edited_out, _ = _run(bridge, hidden_states)
    finally:
        handle.remove()
    assert torch.equal(base_out, edited_out)


def test_editing_scores_hook_warns(bridge, hidden_states):
    handle = bridge.hook_attn_scores.register_forward_hook(lambda m, i, o: o + 1.0)
    try:
        with pytest.warns(UserWarning, match="edit is discarded"):
            _run(bridge, hidden_states)
    finally:
        handle.remove()


def test_read_only_hook_does_not_warn(bridge, hidden_states):
    seen: list[torch.Tensor] = []
    handle = bridge.hook_pattern.register_forward_hook(lambda m, i, o: seen.append(o))
    try:
        with warnings.catch_warnings(record=True) as records:
            warnings.simplefilter("always")
            _run(bridge, hidden_states)
    finally:
        handle.remove()
    assert len(seen) == 1
    assert _discard_warnings(records) == []


def test_hook_returning_equal_copy_does_not_warn(bridge, hidden_states):
    """A new-object, value-identical return (conversion round-trips, defensive
    clones) is not a write and must not trigger the warning."""
    handle = bridge.hook_pattern.register_forward_hook(lambda m, i, o: o.clone())
    try:
        with warnings.catch_warnings(record=True) as records:
            warnings.simplefilter("always")
            _run(bridge, hidden_states)
    finally:
        handle.remove()
    assert _discard_warnings(records) == []


@pytest.mark.parametrize("inference", [False, True])
@pytest.mark.parametrize("return_kind", ["same", "none", "view", "clone"])
def test_inplace_pattern_edit_warns(bridge, hidden_states, inference, return_kind):
    def edit(module, args, weights):
        weights.zero_()
        if return_kind == "none":
            return None
        if return_kind == "view":
            return weights.view_as(weights)
        if return_kind == "clone":
            return weights.clone()
        return weights

    with torch.inference_mode(inference):
        base_out, _ = _run(bridge, hidden_states)
        handle = bridge.hook_pattern.register_forward_hook(edit)
        try:
            with pytest.warns(UserWarning, match="edit is discarded"):
                actual, weights = _run(bridge, hidden_states)
        finally:
            handle.remove()
    assert torch.equal(actual, base_out)
    assert torch.count_nonzero(weights) == 0


@pytest.mark.parametrize("inference", [False, True])
def test_inplace_scores_edit_warns(bridge, hidden_states, inference):
    handle = bridge.hook_attn_scores.register_forward_hook(lambda m, i, t: t.add_(1))
    try:
        with torch.inference_mode(inference), pytest.warns(UserWarning, match="edit is discarded"):
            bridge._fire_post_hoc_hook(
                bridge.hook_attn_scores, torch.ones(2, 2, 3, 3), "hook_attn_scores"
            )
    finally:
        handle.remove()


@pytest.mark.parametrize("inference", [False, True])
def test_inplace_noop_does_not_warn(bridge, hidden_states, inference):
    handle = bridge.hook_attn_scores.register_forward_hook(lambda m, i, t: t.add_(0))
    try:
        with torch.inference_mode(inference), warnings.catch_warnings(record=True) as records:
            warnings.simplefilter("always")
            bridge._fire_post_hoc_hook(
                bridge.hook_attn_scores, torch.ones(2, 2, 3, 3), "hook_attn_scores"
            )
    finally:
        handle.remove()
    assert _discard_warnings(records) == []


@pytest.mark.parametrize("inference", [False, True])
def test_read_only_nonfinite_values_do_not_warn(bridge, inference):
    hook = bridge.hook_pattern
    handle = hook.register_forward_hook(lambda m, i, t: None)
    try:
        with torch.inference_mode(inference), warnings.catch_warnings(record=True) as records:
            warnings.simplefilter("always")
            values = torch.tensor([float("nan"), float("inf"), -float("inf")])
            result = bridge._fire_post_hoc_hook(hook, values, "hook_pattern")
    finally:
        handle.remove()
    assert result is values
    assert _discard_warnings(records) == []


def test_no_hooks_do_not_clone_tensor(bridge, monkeypatch):
    values = torch.ones(2, 3)

    def unexpected_clone(*args, **kwargs):
        pytest.fail("unhooked post-hoc tensors must not be copied")

    monkeypatch.setattr(torch.Tensor, "clone", unexpected_clone)
    assert bridge._fire_post_hoc_hook(bridge.hook_pattern, values, "hook_pattern") is values


@pytest.mark.parametrize("inference", [False, True])
def test_inplace_pre_hook_warns(bridge, inference):
    def edit(module, args):
        args[0].zero_()

    handle = bridge.hook_pattern.register_forward_pre_hook(edit)
    try:
        with torch.inference_mode(inference), pytest.warns(UserWarning, match="edit is discarded"):
            bridge._fire_post_hoc_hook(bridge.hook_pattern, torch.ones(2, 3), "hook_pattern")
    finally:
        handle.remove()


@pytest.mark.parametrize("inference", [False, True])
def test_read_only_pre_hook_stays_silent(bridge, inference):
    handle = bridge.hook_pattern.register_forward_pre_hook(lambda m, args: None)
    try:
        with torch.inference_mode(inference), warnings.catch_warnings(record=True) as records:
            warnings.simplefilter("always")
            bridge._fire_post_hoc_hook(bridge.hook_pattern, torch.ones(2, 3), "hook_pattern")
    finally:
        handle.remove()
    assert _discard_warnings(records) == []


@pytest.fixture()
def cached_bridge():
    cfg = TransformerBridgeConfig(
        d_model=4,
        d_head=2,
        n_heads=2,
        n_layers=1,
        n_ctx=8,
        d_vocab=16,
        d_mlp=8,
        act_fn="gelu",
        normalization_type="LN",
        seed=0,
    )
    model = TransformerBridge.boot_native(cfg)
    model.blocks[0].attn.set_original_component(_PatternReturningAttention())
    return model


PATTERN = "blocks.0.attn.hook_pattern"


def _cache_with(model, method, tokens):
    if method in ("run", "retained", "sweep"):
        _, cache = model.run_with_cache(
            tokens,
            names_filter=None if method == "sweep" else PATTERN,
            reset_hooks_end=method != "retained",
        )
        if method == "retained":
            model(tokens)
        return cache
    if method == "add":
        cache = model.add_caching_hooks(names_filter=PATTERN)
        model(tokens)
        return cache
    cache, fwd, _ = model.get_caching_hooks(names_filter=PATTERN)
    model.run_with_hooks(tokens, fwd_hooks=fwd, remove_batch_dim=method == "squeezed")
    return cache


@pytest.mark.parametrize("method", ["run", "get", "add", "squeezed", "retained", "sweep"])
@pytest.mark.parametrize("inference", [False, True])
def test_internal_cache_does_not_snapshot_or_compare(cached_bridge, monkeypatch, method, inference):
    tokens = torch.tensor([[1, 2, 3]])
    _, expected = cached_bridge.run_with_cache(tokens, names_filter=PATTERN)
    original_clone = torch.Tensor.clone
    original_equal = torch.equal
    original_close = torch.allclose

    def clone(tensor, *args, **kwargs):
        if tensor.ndim == 4:
            pytest.fail("read-only attention caching must not snapshot the pattern")
        return original_clone(tensor, *args, **kwargs)

    def compare(original):
        def check(left, right, *args, **kwargs):
            if left.ndim == 4:
                pytest.fail("read-only attention caching must not compare the pattern")
            return original(left, right, *args, **kwargs)

        return check

    try:
        with monkeypatch.context() as patch, torch.inference_mode(inference):
            patch.setattr(torch.Tensor, "clone", clone)
            patch.setattr(torch, "equal", compare(original_equal))
            patch.setattr(torch, "allclose", compare(original_close))
            cache = _cache_with(cached_bridge, method, tokens)
        assert PATTERN in cache
        reference = expected[PATTERN].squeeze(0) if method == "squeezed" else expected[PATTERN]
        torch.testing.assert_close(cache[PATTERN], reference, rtol=0, atol=0)
    finally:
        cached_bridge.reset_hooks()


@pytest.mark.parametrize("method", ["run", "get", "add"])
@pytest.mark.parametrize("inference", [False, True])
def test_cache_with_user_edit_still_warns(cached_bridge, method, inference):
    try:
        with torch.inference_mode(inference), cached_bridge.hooks(
            fwd_hooks=[(PATTERN, lambda t, hook: t.zero_())]
        ), pytest.warns(UserWarning, match="edit is discarded"):
            _cache_with(cached_bridge, method, torch.tensor([[1, 2, 3]]))
    finally:
        cached_bridge.reset_hooks()


@pytest.mark.parametrize("inference", [False, True])
def test_removing_cache_hooks_does_not_mark_future_user_hooks_read_only(cached_bridge, inference):
    tokens = torch.tensor([[1, 2, 3]])
    with torch.inference_mode(inference):
        cached_bridge.run_with_cache(tokens, names_filter=PATTERN, reset_hooks_end=False)
        cached_bridge.reset_hooks()
        with cached_bridge.hooks(fwd_hooks=[(PATTERN, lambda t, hook: t.zero_())]):
            with pytest.warns(UserWarning, match="edit is discarded"):
                cached_bridge(tokens)


@pytest.mark.parametrize("inference", [False, True])
@pytest.mark.parametrize("pre", [False, True])
def test_global_user_hook_is_not_bypassed_by_cache_fast_path(cached_bridge, inference, pre):
    point = cached_bridge.hook_dict[PATTERN]

    def edit_before(module, args):
        if module is point:
            args[0].zero_()

    def edit_after(module, args, output):
        if module is point:
            output.zero_()

    module_hooks = torch.nn.modules.module
    handle = (
        module_hooks.register_module_forward_pre_hook(edit_before)
        if pre
        else module_hooks.register_module_forward_hook(edit_after)
    )
    try:
        with torch.inference_mode(inference), pytest.warns(UserWarning, match="edit is discarded"):
            cached_bridge.run_with_cache(torch.tensor([[1, 2, 3]]), names_filter=PATTERN)
    finally:
        handle.remove()
