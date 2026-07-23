"""Regression tests for TransformerBridge train/eval mode propagation.

`original_model` is stored outside the registered module tree, so
inherited `nn.Module.train()` does not reach it. These tests verify that
bridge mode changes propagate to the wrapped model, whose mode-dependent
layers (Dropout) rely on that flag. Architecture-independent stub bridge;
no HF Hub access, no weights.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from transformer_lens.config import TransformerBridgeConfig
from transformer_lens.model_bridge.architecture_adapter import ArchitectureAdapter
from transformer_lens.model_bridge.bridge import TransformerBridge
from transformer_lens.model_bridge.generalized_components import LinearBridge


class _StubModel(nn.Module):
    """Minimal source model with a mode-dependent layer (Dropout), so
    train/eval propagation has an observable behavioral consequence, not
    just a flag."""

    def __init__(self) -> None:
        super().__init__()
        self.proj = nn.Linear(4, 4)
        self.drop = nn.Dropout(0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.proj(x))


class _StubAdapter(ArchitectureAdapter):
    """Smallest adapter TransformerBridge will accept: a one-entry
    component_mapping. The key is `stub_proj` (not `proj` or `embed`)
    because TransformerBridge reserves several canonical attribute names
    on itself; a colliding key fails at add_module time."""

    def __init__(self, cfg: TransformerBridgeConfig) -> None:
        super().__init__(cfg)
        self.component_mapping = {"stub_proj": LinearBridge(name="proj")}


def _make_stub_bridge() -> tuple[TransformerBridge, _StubModel]:
    cfg = TransformerBridgeConfig(
        d_model=4,
        d_head=2,
        n_layers=1,
        n_ctx=8,
        n_heads=2,
        d_vocab=8,
        architecture="StubForTest",
    )
    model = _StubModel()
    bridge = TransformerBridge(model, _StubAdapter(cfg), tokenizer=None)
    return bridge, model


class TestTrainEvalModePropagation:
    """bridge.train()/.eval() must reach original_model.

    Without the TransformerBridge.train() override these fail with
    original_model.training stuck at its wrap-time value (bridge.training
    flips, the wrapped model's does not) -- i.e. bridge.eval() would leave
    dropout active."""

    def test_eval_propagates_to_original_model(self) -> None:
        bridge, model = _make_stub_bridge()
        assert model.training is True  # nn.Module default at wrap time
        bridge.eval()
        assert bridge.training is False
        assert model.training is False

    def test_train_propagates_to_original_model(self) -> None:
        bridge, model = _make_stub_bridge()
        bridge.eval()
        assert model.training is False
        bridge.train()
        assert bridge.training is True
        assert model.training is True

    def test_train_and_eval_return_the_bridge_itself(self) -> None:
        """nn.Module convention: train()/eval() return self so callers can
        chain (`bridge.eval().to(device)`, etc.)."""
        bridge, _ = _make_stub_bridge()
        assert bridge.train(False) is bridge
        assert bridge.train(True) is bridge
        assert bridge.eval() is bridge

    def test_direct_mode_on_original_model_stays_in_sync(self) -> None:
        """Setting mode via the caller's own reference to the source model
        still works -- both paths write the same underlying flags, so
        neither clobbers the other."""
        bridge, model = _make_stub_bridge()
        bridge.eval()
        model.train()
        assert model.training is True
        bridge.eval()
        assert model.training is False
