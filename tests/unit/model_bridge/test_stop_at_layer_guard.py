from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from transformer_lens.config import TransformerBridgeConfig
from transformer_lens.model_bridge import TransformerBridge


def _bare_bridge(**block_lists: nn.Module) -> TransformerBridge:
    """A TransformerBridge with only the given block lists registered (no HF model)."""
    bridge = TransformerBridge.__new__(TransformerBridge)
    nn.Module.__init__(bridge)
    bridge.cfg = TransformerBridgeConfig(
        d_model=8,
        d_head=4,
        n_layers=1,
        n_ctx=16,
        d_vocab=32,
        d_mlp=16,
        n_heads=2,
        architecture="RavenForCausalLM",
    )
    for name, module in block_lists.items():
        bridge.add_module(name, module)
    return bridge


def test_stop_at_layer_raises_without_blocks_stack() -> None:
    """Raven-style prelude/core_block/coda lists must not silently ignore stop_at_layer."""
    bridge = _bare_bridge(
        prelude=nn.ModuleList([nn.Identity()]),
        core_block=nn.ModuleList([nn.Identity()]),
        coda=nn.ModuleList([nn.Identity()]),
    )
    with pytest.raises(NotImplementedError, match="stop_at_layer requires a 'blocks' stack"):
        bridge.forward(torch.zeros(1, 3, dtype=torch.long), stop_at_layer=0)


def test_blocks_guard_ignores_unregistered_blocks_attribute() -> None:
    """A wrapped HF model exposing `.blocks` must not satisfy the guard via __getattr__."""
    bridge = _bare_bridge()
    bridge.__dict__["original_model"] = SimpleNamespace(blocks=[object()])
    assert hasattr(bridge, "blocks")  # the trap: __getattr__ falls through to the HF model
    assert not bridge._has_registered_blocks()
    with pytest.raises(NotImplementedError, match="stop_at_layer requires a 'blocks' stack"):
        bridge.forward(torch.zeros(1, 3, dtype=torch.long), stop_at_layer=0)
    with pytest.raises(NotImplementedError, match="start_at_layer requires a 'blocks' stack"):
        bridge.forward(torch.zeros(1, 3, 8), start_at_layer=0)
