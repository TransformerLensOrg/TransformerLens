"""Structural type shared by :class:`HookedTransformer` and :class:`TransformerBridge`.

The interpretability utilities (``patching``, ``head_detector``, ``ActivationCache``)
are model-agnostic at runtime — they only need ``cfg`` plus the ``run_with_*`` /
tokenization surface and a few weight-processing helpers. Historically their
signatures said ``HookedTransformer``, which made a ``TransformerBridge`` fail to
type-check even though it works. Typing the model parameter as this Protocol accepts
either (and any future structural match, e.g. ``RemoteBridge``).

Members are typed loosely on purpose: this is a compatibility shim over two concrete
classes whose method signatures differ in detail but agree in use.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from transformer_lens.config import TransformerLensConfig


# runtime_checkable so the interp utilities' beartype-decorated signatures can
# isinstance-check the parameter at runtime (presence-only; both models qualify).
@runtime_checkable
class TransformerLensModel(Protocol):
    """Minimal structural interface common to HookedTransformer and TransformerBridge."""

    # Read-only property (not a bare attribute) so it is covariant: a concrete model
    # whose cfg is a TransformerLensConfig *subclass* (HookedTransformerConfig /
    # TransformerBridgeConfig) still conforms. A mutable attribute would be invariant.
    @property
    def cfg(self) -> "TransformerLensConfig":
        ...

    def run_with_cache(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def run_with_hooks(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def to_tokens(self, *args: Any, **kwargs: Any) -> Any:
        ...


@runtime_checkable
class TransformerLensModelWithWeights(TransformerLensModel, Protocol):
    """Adds the weight-processing surface that ``ActivationCache``'s advanced helpers
    (LayerNorm folding, residual-direction projection) reach for. Both concrete models
    expose these; the bridge builds them from its adapter."""

    @property
    def blocks(self) -> Any:
        ...

    @property
    def ln_final(self) -> Any:
        ...

    def accumulated_bias(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def to_single_token(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def tokens_to_residual_directions(self, *args: Any, **kwargs: Any) -> Any:
        ...
