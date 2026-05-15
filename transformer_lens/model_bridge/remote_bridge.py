"""Non-torch bridge: vLLM workers, Inspect remote providers."""
from __future__ import annotations

from typing import Any

from transformer_lens.hook_points import HookIntrospectionMixin, HookPoint
from transformer_lens.model_bridge.bridge_core import BridgeCore
from transformer_lens.model_bridge.driver_protocol import (
    ForwardResult,
    TensorLike,
    to_torch,
    validate_driver,
)


class RemoteBridge(BridgeCore, HookIntrospectionMixin):
    """Bridge for backends with no local ``nn.Module`` (vLLM, Inspect).

    No nn.Module parentage strips the torch-only surface; driver pre-declares
    ``supported_hook_points`` (no model to walk).
    """

    def __init__(
        self,
        adapter: Any,
        tokenizer: Any,
        driver: Any,
    ) -> None:
        if not driver.supported_hook_points:
            raise ValueError(
                "RemoteBridge requires driver.supported_hook_points to be "
                "non-empty: non-torch drivers own the hook namespace because "
                "there is no local model for the bridge to walk."
            )
        BridgeCore.__init__(self, adapter, tokenizer, driver)
        # No local device; tensor.to(None) is a no-op so downstream patterns degrade cleanly.
        self.cfg.device = None
        # HookPoint is nn.Module-backed but RemoteBridge isn't an nn.Module —
        # named_modules() walks don't apply; only registry lookup matters.
        for name in driver.supported_hook_points:
            hp = HookPoint()
            hp.name = name
            self._hook_registry[name] = hp
        self._hook_registry_initialized = True
        validate_driver(self._driver, after_bridge_construction=True)

    def _scan_existing_hooks(self, module: Any, prefix: str = "") -> None:
        """No-op: registry built from driver declarations in __init__."""

    def forward(
        self,
        input: Any = None,
        *,
        return_type: str | None = "logits",
        loss_per_token: bool = False,
        **kwargs: Any,
    ) -> Any:
        """Tokenize → driver.forward → replay captures → finalize per return_type."""
        if isinstance(input, str):
            assert self.tokenizer is not None, "Tokenizer must be set for string input."
            tokens = self.tokenizer.encode(input, return_tensors="pt")
            kwargs["input_ids"] = tokens
        elif input is not None:
            kwargs["input_ids"] = input

        result: ForwardResult = self._driver.forward(**kwargs)
        if result.captured:
            self._replay_captures(result.captured)

        logits: Any = result.logits
        if logits is not None and not isinstance(logits, TensorLike):
            return logits  # weird shape — let caller handle
        if logits is not None:
            logits = to_torch(logits)

        return self._finalize_return(
            return_type,
            logits,
            kwargs.get("input_ids"),
            is_audio_model=getattr(self.cfg, "is_audio_model", False),
            loss_per_token=loss_per_token,
        )

    def to_tokens(self, text: Any, *args: Any, **kwargs: Any) -> Any:
        """Tokenize via ``self.tokenizer``. BOS/padding handling lives on
        :class:`TransformerBridge`."""
        assert self.tokenizer is not None, "Tokenizer must be set."
        return self.tokenizer.encode(text, return_tensors="pt")
