"""Driver protocol: the contract every model-execution backend satisfies."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Protocol, Union, runtime_checkable

import numpy as np
import torch

from transformer_lens.config import TransformerBridgeConfig


@runtime_checkable
class TensorLike(Protocol):
    """Quacks like a tensor: ``__array__`` + ``shape`` + ``dtype``."""

    # Attribute types stay loose — torch.Size, plain tuple, and numpy's shape
    # don't share a Protocol-strict supertype.
    @property
    def shape(self) -> Any:
        ...

    @property
    def dtype(self) -> Any:
        ...

    def __array__(self, dtype: Any = None) -> np.ndarray:
        ...


InterventionFn = Callable[[TensorLike], TensorLike]
InterventionSpec = Mapping[str, Any]
# Drivers that can dispatch Python at the engine boundary (HF) accept
# InterventionFn; drivers that can't (vLLM under compile, remote APIs) accept
# InterventionSpec only.
Intervention = Union[InterventionFn, InterventionSpec]


@dataclass(frozen=True)
class ForwardResult:
    """One forward call's outputs. Tensors are native to the driver's framework."""

    logits: TensorLike | None = None
    captured: Mapping[str, TensorLike] = field(default_factory=dict)
    new_tokens: TensorLike | None = None
    # Driver's native return value (HF CausalLMOutputWithPast, vLLM
    # RequestOutput, ...). Bridge reads driver-specific extras here.
    raw_output: Any = None


@runtime_checkable
class Driver(Protocol):
    """The forward-pass contract. Hook installation is the driver's problem."""

    architecture: str
    bridge_config: TransformerBridgeConfig
    tokenizer: Any
    supported_hook_points: frozenset[str]
    non_fireable_hook_points: frozenset[str]

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
        ...

    def supports(self, feature: str) -> bool:
        """Capability flag. Known features: gradients, parameters, state_dict,
        generate_streaming, weight_access, intervention_callbacks."""
        ...

    # Note: torch-specific surface (parameters, named_parameters, state_dict,
    # weight access) is NOT in the protocol. Drivers that can serve those
    # methods provide them as implementation details, gated by supports("...").


def to_torch(t: TensorLike, *, dtype: torch.dtype | None = None) -> torch.Tensor:
    """Convert any TensorLike to torch.Tensor at the bridge boundary.

    Order: torch passthrough → DLPack (jax/mlx/tf/cupy/numpy≥1.22, preserves
    device) → ``__array__`` + ``from_numpy`` (CPU only).
    """
    if isinstance(t, torch.Tensor):
        return t.to(dtype) if dtype is not None else t

    if hasattr(t, "__dlpack__"):
        try:
            out = torch.from_dlpack(t)
            return out.to(dtype) if dtype is not None else out
        except (BufferError, RuntimeError, ValueError, TypeError, AttributeError):
            # Fall through on stream-sync, missing __dlpack_device__, or
            # version-skew failures; the numpy path either succeeds or raises
            # informatively.
            pass

    arr = np.asarray(t)
    out = torch.from_numpy(arr)
    return out.to(dtype) if dtype is not None else out


# Parameter names a conforming driver's forward() must accept; missing names
# get silently swallowed by **kwargs and break the contract.
_DRIVER_FORWARD_REQUIRED_PARAMS = frozenset(
    ("input_ids", "capture", "intervene", "max_new_tokens", "return_logits")
)


def validate_driver(driver: Any, *, after_bridge_construction: bool = False) -> None:
    """Stronger than ``isinstance(driver, Driver)``: checks types, signatures,
    and (optionally) post-construction state.

    Args:
        after_bridge_construction: when True, also requires at least one of
            ``supported_hook_points`` / ``non_fireable_hook_points`` non-empty
            (the bridge backfills the former, so empty-on-both means the
            driver silently degrades to "supports nothing").

    Raises:
        TypeError: with a message naming the contract violation.
    """
    _expect_attr_type(driver, "architecture", str)
    _expect_attr_type(driver, "bridge_config", TransformerBridgeConfig)
    if not hasattr(driver, "tokenizer"):
        raise TypeError("Driver missing required attribute: 'tokenizer'")
    _expect_attr_type(driver, "supported_hook_points", frozenset)
    _expect_attr_type(driver, "non_fireable_hook_points", frozenset)

    overlap = driver.supported_hook_points & driver.non_fireable_hook_points
    if overlap:
        raise TypeError(
            f"Driver.supported_hook_points and Driver.non_fireable_hook_points "
            f"overlap on {sorted(overlap)[:3]}; a hook is either fireable or not."
        )
    for name in driver.supported_hook_points | driver.non_fireable_hook_points:
        if not isinstance(name, str):
            raise TypeError(f"Hook-point names must be str; got {type(name).__name__}: {name!r}")

    forward = getattr(driver, "forward", None)
    if not callable(forward):
        raise TypeError("Driver.forward must be callable")
    import inspect

    sig = inspect.signature(forward)
    params = sig.parameters
    has_var_keyword = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())
    missing = [p for p in _DRIVER_FORWARD_REQUIRED_PARAMS if p not in params]
    if missing and not has_var_keyword:
        raise TypeError(
            f"Driver.forward must accept parameters {sorted(_DRIVER_FORWARD_REQUIRED_PARAMS)}; "
            f"missing {sorted(missing)} (and no **kwargs to absorb them)."
        )

    if not callable(getattr(driver, "close", None)):
        raise TypeError("Driver.close must be callable")

    if after_bridge_construction:
        if not driver.supported_hook_points and not driver.non_fireable_hook_points:
            raise TypeError(
                "Driver has empty supported_hook_points AND non_fireable_hook_points "
                "after bridge construction. Drivers must declare at least one — the "
                "bridge backfills supported from registry minus non_fireable, but "
                "empty-on-both means there's no contract for downstream code."
            )


def _expect_attr_type(obj: Any, name: str, expected: type) -> None:
    if not hasattr(obj, name):
        raise TypeError(f"Driver missing required attribute: {name!r}")
    value = getattr(obj, name)
    if not isinstance(value, expected):
        raise TypeError(f"Driver.{name} must be {expected.__name__}; got {type(value).__name__}.")


__all__ = [
    "Driver",
    "ForwardResult",
    "Intervention",
    "InterventionFn",
    "InterventionSpec",
    "TensorLike",
    "to_torch",
    "validate_driver",
]
