"""HuggingFace transformers Driver."""
from __future__ import annotations

from typing import Any, Iterator, Mapping

import torch
from torch import nn

from transformer_lens.model_bridge.driver_protocol import (
    ForwardResult,
    Intervention,
    TensorLike,
)
from transformer_lens.model_bridge.sources._driver_base import DriverBase


class TransformersDriver(DriverBase):
    """Wraps an HF ``nn.Module``. PyTorch hooks fire via module replacement during the
    real forward; this driver just runs the engine and threads the native output back."""

    _supported_features = frozenset(
        {"gradients", "parameters", "state_dict", "weight_access", "intervention_callbacks"}
    )

    def __init__(self, model: nn.Module, adapter: Any, tokenizer: Any) -> None:
        super().__init__(adapter.cfg, tokenizer)
        self._model = model
        self._adapter = adapter

    def forward(
        self,
        input_ids: TensorLike | None = None,
        *,
        capture: tuple[str, ...] = (),
        intervene: Mapping[str, Intervention] | None = None,
        max_new_tokens: int = 1,
        return_logits: bool = True,
        **kwargs: Any,
    ) -> ForwardResult:
        if input_ids is not None:
            raw = self._model(input_ids, **kwargs)
        else:
            raw = self._model(**kwargs)

        logits = None
        if return_logits:
            if hasattr(raw, "logits"):
                logits = raw.logits
            elif isinstance(raw, tuple) and len(raw) > 0:
                logits = raw[0]
            else:
                logits = raw

        return ForwardResult(logits=logits, raw_output=raw)

    def parameters(self) -> Iterator[torch.Tensor]:
        return self._model.parameters()

    def named_parameters(
        self,
        prefix: str = "",
        recurse: bool = True,
        remove_duplicate: bool = True,
    ) -> Iterator[tuple[str, torch.Tensor]]:
        return self._model.named_parameters(prefix, recurse, remove_duplicate)

    @property
    def underlying_model(self) -> nn.Module:
        """Escape hatch for code that needs the raw HF module. Driver-specific."""
        return self._model

    def set_underlying_model(self, value: nn.Module) -> None:
        """Used by weight-processing paths that move the model to a different
        device. Non-torch drivers don't implement this."""
        self._model = value
