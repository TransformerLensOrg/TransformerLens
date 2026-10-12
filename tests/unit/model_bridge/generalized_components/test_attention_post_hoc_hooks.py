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
