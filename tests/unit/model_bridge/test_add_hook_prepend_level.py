"""TransformerBridge.add_hook takes the level and prepend HookedRootModule.add_hook does."""

from __future__ import annotations

import pytest
import torch

from transformer_lens.config import TransformerBridgeConfig
from transformer_lens.model_bridge import TransformerBridge

HOOK = "blocks.0.hook_out"
ALIAS = "blocks.0.hook_resid_post"


@pytest.fixture
def bridge():
    cfg = TransformerBridgeConfig(
        d_model=32,
        d_head=16,
        n_heads=2,
        n_layers=1,
        n_ctx=8,
        d_vocab=16,
        d_mlp=64,
        act_fn="gelu",
        normalization_type="LN",
        seed=0,
    )
    return TransformerBridge.boot_native(cfg)


def _tokens() -> torch.Tensor:
    return torch.randint(0, 16, (1, 6))


def _noop(tensor, hook):
    return None


def _recorder(order: list[str], label: str):
    def record(tensor, hook):
        order.append(label)

    return record


@pytest.mark.parametrize(
    "name", [HOOK, ALIAS, lambda name: name == HOOK], ids=["canonical", "alias", "filter"]
)
def test_prepended_hook_runs_before_existing_hooks(bridge, name):
    order: list[str] = []
    bridge.add_hook(HOOK, _recorder(order, "first"))
    bridge.add_hook(name, _recorder(order, "prepended"), prepend=True)

    bridge(_tokens())

    assert order == ["prepended", "first"]


def test_prepended_edit_reaches_an_earlier_hook(bridge):
    seen: list[torch.Tensor] = []
    bridge.add_hook(HOOK, lambda tensor, hook: seen.append(tensor.clone()))
    bridge.add_hook(HOOK, lambda tensor, hook: torch.zeros_like(tensor), prepend=True)

    bridge(_tokens())

    assert torch.count_nonzero(seen[0]) == 0


def test_prepend_applies_to_backward_hooks(bridge):
    order: list[str] = []
    bridge.add_hook(HOOK, _recorder(order, "first"), dir="bwd")
    bridge.add_hook(HOOK, _recorder(order, "prepended"), dir="bwd", prepend=True)

    bridge(_tokens()).sum().backward()

    assert order == ["prepended", "first"]


def test_level_scopes_reset_hooks(bridge):
    bridge.add_hook(HOOK, _noop, level=1)
    bridge.add_hook(HOOK, _noop, level=2)

    bridge.reset_hooks(level=1)

    assert [handle.context_level for handle in bridge.hook_dict[HOOK].fwd_hooks] == [2]


def test_check_hooks_to_add_sees_prepend(bridge, monkeypatch):
    seen: list[bool] = []

    def recording(
        self, hook_point, hook_point_name, hook, dir="fwd", is_permanent=False, prepend=False
    ):
        seen.append(prepend)

    monkeypatch.setattr(type(bridge), "check_hooks_to_add", recording)
    bridge.add_hook(HOOK, _noop, prepend=True)

    assert seen == [True]
