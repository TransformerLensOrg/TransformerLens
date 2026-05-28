"""Optional base class for :class:`Driver` implementations.

The protocol is duck-typed; inheriting is convenient, not required.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Mapping

from transformer_lens.config import TransformerBridgeConfig
from transformer_lens.model_bridge.driver_protocol import (
    ForwardResult,
    Intervention,
    TensorLike,
)


class DriverBase(ABC):
    """Defaults for the optional Driver members. Subclasses implement ``forward``
    and override the rest only when they can do better."""

    architecture: str = ""
    # The bridge overwrites this slot at construction (registry − non_fireable)
    # when it's empty. Whitelist-semantic drivers (e.g. Inspect) declare a
    # non-empty set in the subclass to keep it.
    supported_hook_points: frozenset[str] = frozenset()
    non_fireable_hook_points: frozenset[str] = frozenset()

    # Subclasses override with the capability strings they actually serve.
    _supported_features: frozenset[str] = frozenset()

    # True if forward() returns logits for every position, so loss is computable.
    # Drivers that synthesize only the final position set this False; the bridge
    # then refuses return_type=loss/both rather than return nan.
    provides_sequence_logits: bool = True

    def __init__(
        self,
        bridge_config: TransformerBridgeConfig,
        tokenizer: Any,
        *,
        architecture: str | None = None,
    ) -> None:
        self.bridge_config = bridge_config
        self.tokenizer = tokenizer
        # Resolution order: explicit kwarg > bridge_config field > class default.
        self.architecture = (
            architecture or getattr(bridge_config, "architecture", "") or self.architecture
        )

    @abstractmethod
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
        ...

    def close(self) -> None:
        """No-op default. Override when the driver owns releasable resources."""

    def supports(self, feature: str) -> bool:
        return feature in self._supported_features

    # Torch-specific surface (parameters, named_parameters, state_dict, weight
    # access) is NOT defined here. Drivers that serve those methods provide
    # them directly; callers route via supports("...") + hasattr/getattr.
