"""Capability flags must reach their consuming bridge code paths.

Exercises the real forward/loss guard and generation-caching resolver with the
real adapter classes on a bare bridge (full construction needs remote-code
models these unit tests cannot load).
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch.nn as nn

from tests.unit.model_bridge.supported_architectures.helpers import make_bridge_cfg
from transformer_lens.model_bridge.supported_architectures.dream import (
    DreamArchitectureAdapter,
)
from transformer_lens.model_bridge.supported_architectures.gidd import (
    GiddArchitectureAdapter,
)
from transformer_lens.model_bridge.transformer_bridge import TransformerBridge


class _StubModel(nn.Module):
    """original_model is beartype-hinted nn.Module; carry only .config."""

    def __init__(self) -> None:
        super().__init__()
        self.config = SimpleNamespace(is_encoder_decoder=False)


def _bare_bridge(adapter) -> TransformerBridge:
    bridge = object.__new__(TransformerBridge)
    nn.Module.__init__(bridge)
    bridge.adapter = adapter
    bridge.cfg = adapter.cfg
    # 4.x routes original_model through the driver.
    bridge.__dict__["_driver"] = SimpleNamespace(underlying_model=_StubModel())
    return bridge


class TestDiffusionLossGuard:
    """Diffusion LMs must refuse the shifted causal loss instead of silently
    computing it (bridge.forward's supports_causal_loss guard)."""

    def test_dream_loss_raises(self) -> None:
        cfg = make_bridge_cfg("DreamModel", n_key_value_heads=4)
        bridge = _bare_bridge(DreamArchitectureAdapter(cfg))
        with pytest.raises(NotImplementedError, match="shifted causal"):
            bridge.forward("hi", return_type="loss")

    def test_gidd_both_raises(self) -> None:
        cfg = make_bridge_cfg("GiddForDiffusionLM")
        bridge = _bare_bridge(GiddArchitectureAdapter(cfg))
        with pytest.raises(NotImplementedError, match="shifted causal"):
            bridge.forward("hi", return_type="both")


class TestRavenGenerationCaching:
    """Huginn's depth recurrence cannot use HF past_key_values stepping; the
    resolver must refuse the cache and reject batched generation."""

    def _bridge(self) -> TransformerBridge:
        from transformer_lens.model_bridge.supported_architectures.raven import (
            RavenArchitectureAdapter,
        )

        cfg = make_bridge_cfg("RavenForCausalLM")
        return _bare_bridge(RavenArchitectureAdapter(cfg))

    def test_kv_cache_refused(self) -> None:
        assert self._bridge()._resolve_generation_caching(True, batched=False) is False

    def test_batched_generation_rejected(self) -> None:
        with pytest.raises(NotImplementedError):
            self._bridge()._resolve_generation_caching(True, batched=True)
