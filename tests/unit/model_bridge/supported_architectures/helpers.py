"""Shared helpers for per-adapter unit tests."""

from typing import Any, ClassVar

import pytest

from transformer_lens.config import TransformerBridgeConfig


def make_bridge_cfg(architecture: str, **overrides) -> TransformerBridgeConfig:
    """Minimal TransformerBridgeConfig with the standard tiny test dims.

    Defaults: d_model=64, n_heads=8, n_layers=2, d_vocab=100, n_ctx=128,
    default_prepend_bos=False. d_head is derived from d_model/n_heads unless
    overridden. Pass any TransformerBridgeConfig field as a keyword.
    """
    cfg = dict(
        d_model=64,
        n_heads=8,
        n_layers=2,
        n_ctx=128,
        d_vocab=100,
        default_prepend_bos=False,
        architecture=architecture,
    )
    cfg.update(overrides)
    cfg.setdefault("d_head", cfg["d_model"] // cfg["n_heads"])
    return TransformerBridgeConfig(**cfg)


class DiffusionContractTests:
    """Cross-cutting contract for masked-diffusion adapters.

    Inherit in the adapter's own test file and set the class attrs; pytest
    collects the tests there, so the contract lives next to the adapter's other
    coverage and new diffusion adapters pick it up via the sibling-copy
    workflow. Completeness is enforced by the factory-derived meta-test in
    tests/unit/model_bridge/test_diffusion_generate.py.
    """

    adapter_cls: ClassVar[type]
    architecture: ClassVar[str]
    expected_sampler: ClassVar[str]
    cfg_overrides: ClassVar[dict[str, Any]] = {}

    def _make_adapter(self) -> Any:
        return self.adapter_cls(make_bridge_cfg(self.architecture, **self.cfg_overrides))

    def test_declares_native_sampler(self) -> None:
        assert self.adapter_cls.native_sampler == self.expected_sampler
        # Autoregressive generate() stays off: it is the wrong algorithm, not a gap.
        assert self.adapter_cls.supports_generation is False

    def test_forward_refuses_causal_loss(self) -> None:
        """Shifted causal CE is undefined for masked denoising; forward must refuse it."""
        from types import SimpleNamespace

        from transformer_lens.model_bridge.bridge import TransformerBridge

        adapter = self._make_adapter()
        fake_bridge = SimpleNamespace(adapter=adapter, cfg=adapter.cfg)
        with pytest.raises(NotImplementedError, match="shifted causal"):
            TransformerBridge.forward(fake_bridge, "hi", return_type="loss")
