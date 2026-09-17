"""Structural model types for the model-agnostic interpretability utilities.

The interpretability utilities (``patching``, ``head_detector``, ``ActivationCache``)
only need ``cfg`` plus the ``run_with_*`` / tokenization surface and a few
weight-processing helpers. Typing their model parameter as this Protocol accepts
``TransformerBridge`` and any other structural match (e.g. ``RemoteBridge``).

Members are typed loosely on purpose: implementations differ in signature detail
but agree in use.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from transformer_lens.config import TransformerLensConfig


# runtime_checkable so the interp utilities' beartype-decorated signatures can
# isinstance-check the parameter at runtime (presence-only).
@runtime_checkable
class TransformerLensModel(Protocol):
    """Minimal structural interface the interpretability utilities rely on."""

    # Read-only property (not a bare attribute) so it is covariant: a concrete model
    # whose cfg is a TransformerLensConfig *subclass* (e.g. TransformerBridgeConfig)
    # still conforms. A mutable attribute would be invariant.
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
class TrainableTransformerLensModel(Protocol):
    """Exactly the surface the ``tools.training`` loop touches: callable with
    ``return_type="loss"`` plus the standard torch parameter/mode/device
    methods. Deliberately standalone (not extending TransformerLensModel):
    beartype validates protocols via ``getattr_static``, and demanding the
    full TL surface would spuriously reject plain ``nn.Module`` models that
    train fine through this loop."""

    def parameters(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def train(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def to(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def state_dict(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        ...


@runtime_checkable
class TransformerLensModelWithWeights(TransformerLensModel, Protocol):
    """Adds the weight-processing surface that ``ActivationCache``'s advanced helpers
    (LayerNorm folding, residual-direction projection) reach for. The bridge builds
    them from its adapter."""

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
