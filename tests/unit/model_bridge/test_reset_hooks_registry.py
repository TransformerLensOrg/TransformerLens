"""reset_hooks must clear every registered hook point, not just component-owned ones."""

from __future__ import annotations

import torch

from transformer_lens.config import TransformerBridgeConfig
from transformer_lens.model_bridge import TransformerBridge


def _bridge() -> TransformerBridge:
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


def _points_with_hooks(bridge: TransformerBridge) -> list[str]:
    return [name for name, hp in bridge._hook_registry.items() if hp.fwd_hooks or hp.bwd_hooks]


def test_reset_hooks_clears_every_registry_point() -> None:
    bridge = _bridge()
    for hp in bridge._hook_registry.values():
        hp.add_hook(lambda tensor, hook: tensor)
    assert _points_with_hooks(bridge), "sanity: hooks were added"

    bridge.reset_hooks()

    leaked = _points_with_hooks(bridge)
    assert leaked == [], f"reset_hooks leaked hooks on registry points: {leaked[:5]}"


def test_reset_hooks_permanent_semantics() -> None:
    bridge = _bridge()
    name, hp = next(iter(bridge._hook_registry.items()))
    hp.add_hook(lambda tensor, hook: tensor, is_permanent=True)

    bridge.reset_hooks()
    assert hp.fwd_hooks, "permanent hook must survive a default reset"

    bridge.reset_hooks(including_permanent=True)
    assert not hp.fwd_hooks, "including_permanent=True must clear permanent hooks"


def test_reset_hooks_still_functional_after_forward() -> None:
    bridge = _bridge()
    seen: list[str] = []
    for name, hp in bridge._hook_registry.items():
        hp.add_hook(lambda tensor, hook: seen.append(hook.name))
    bridge.reset_hooks()
    with torch.no_grad():
        bridge(torch.randint(0, bridge.cfg.d_vocab, (1, 4)))
    assert seen == [], "cleared hooks must not fire on forward"
