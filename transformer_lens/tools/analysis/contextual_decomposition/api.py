"""Validated single-source contextual decomposition on raw GPT-2 Bridge models."""

import math
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Mapping, Optional

import torch

from transformer_lens.model_bridge import TransformerBridge
from transformer_lens.tools.analysis.contextual_decomposition.executor import (
    _decompose_gpt2,
    _require_gpt2,
)
from transformer_lens.tools.analysis.contextual_decomposition.numerics import (
    _check_reconstruction,
    _reconstruction_error,
    _ReconstructionError,
)
from transformer_lens.tools.analysis.contextual_decomposition.rules import (
    _CDConstituents,
)
from transformer_lens.tools.analysis.contextual_decomposition.selectors import (
    _CDSource,
    _CDTarget,
    _select_target,
    _target_stage,
)


@dataclass(frozen=True)
class CDSource(_CDSource):
    """One GPT-2 residual or attention-head write, in residual-stream coordinates.

    Positions are unique absolute indices shared across examples. Selected
    padding is rejected. Head writes exclude the shared output projection bias.
    """

    @classmethod
    def attention_head(
        cls, layer: int, head: int, *, positions: Optional[tuple[int, ...]] = None
    ) -> "CDSource":
        """Select an attention head at the attention residual merge."""
        return cls(layer, kind="head", location="mid", head=head, positions=positions)

    @classmethod
    def residual(
        cls, layer: int, *, location: str = "pre", positions: Optional[tuple[int, ...]] = None
    ) -> "CDSource":
        """Select block input, attention merge, or block output residual."""
        if location not in ("pre", "mid", "post"):
            raise ValueError("residual location must be pre, mid, or post")
        if location == "pre":
            return cls(layer, location="pre", positions=positions)
        if location == "mid":
            return cls(layer, location="mid", positions=positions)
        return cls(layer, location="post", positions=positions)


@dataclass(frozen=True)
class CDTarget(_CDTarget):
    """A strictly downstream internal tensor or per-example logit projection."""

    @classmethod
    def internal(
        cls,
        hook_name: str,
        *,
        position: Optional[int | tuple[int, ...]] = None,
        head: Optional[int] = None,
    ) -> "CDTarget":
        """Read a supported Bridge-native tensor, optionally at positions/heads."""
        return cls("internal", hook_name=hook_name, position=position, head=head)

    @classmethod
    def logit(
        cls, *, token_id: int | tuple[int, ...], position: int | tuple[int, ...]
    ) -> "CDTarget":
        """Project to one token logit per example."""
        return cls("logit", token_id=token_id, position=position)

    @classmethod
    def logit_difference(
        cls,
        *,
        positive_token_id: int | tuple[int, ...],
        negative_token_id: int | tuple[int, ...],
        position: int | tuple[int, ...],
    ) -> "CDTarget":
        """Project to positive-minus-negative token logits per example."""
        return cls(
            "logit_difference",
            token_id=positive_token_id,
            negative_token_id=negative_token_id,
            position=position,
        )


@dataclass(frozen=True)
class CDBaseline:
    """A zero reference or an owned snapshot of a residual-space reference tensor.

    Reference tensors must be finite fp32 [batch, sequence, width]. Model-device
    and input-shape compatibility are checked by decompose, without implicit casts.
    """

    description: str = "zero"
    _reference: Optional[torch.Tensor] = field(default=None, repr=False)

    def __post_init__(self) -> None:
        if not isinstance(self.description, str) or not self.description.strip():
            raise ValueError("baseline description must be a nonempty string")
        if self._reference is not None:
            if self._reference.ndim != 3 or self._reference.dtype != torch.float32:
                raise ValueError("baseline reference must be fp32 [batch, sequence, width]")
            _CDConstituents(self._reference, torch.zeros_like(self._reference))
            object.__setattr__(self, "_reference", self._reference.detach().clone())

    @classmethod
    def zero(cls) -> "CDBaseline":
        """Use zero reference for the selected source write."""
        return cls()

    @classmethod
    def from_tensor(
        cls, reference: torch.Tensor, *, description: str = "supplied reference"
    ) -> "CDBaseline":
        """Snapshot a caller-supplied source reference without retaining autograd."""
        return cls(description=description, _reference=reference)


@dataclass(frozen=True)
class CDTensor:
    """Detached beta, gamma, and native total in a selected tensor space."""

    beta: torch.Tensor
    gamma: torch.Tensor
    total: torch.Tensor

    def __post_init__(self) -> None:
        parts = _CDConstituents(self.beta, self.gamma)
        _CDConstituents(self.total, torch.zeros_like(self.total))
        if (
            self.total.shape != parts.beta.shape
            or self.total.dtype != parts.beta.dtype
            or self.total.device != parts.beta.device
        ):
            raise ValueError("CD tensor total must match constituent shape, dtype, and device")
        for name in ("beta", "gamma", "total"):
            object.__setattr__(self, name, getattr(self, name).detach().clone())


@dataclass(frozen=True)
class CDReconstructionError:
    """Native reconstruction error, not a measure of causal faithfulness."""

    max_abs: float
    reference_max_abs: float
    constituent_max_abs: float = 0.0

    @property
    def tolerance(self) -> float:
        """Absolute plus tensor-relative reconstruction budget."""
        return 2e-6 + 2e-5 * self.reference_max_abs

    def __post_init__(self) -> None:
        if any(
            not math.isfinite(value) or value < 0
            for value in (self.max_abs, self.reference_max_abs, self.constituent_max_abs)
        ):
            raise ValueError("reconstruction error magnitudes must be finite and nonnegative")


@dataclass(frozen=True)
class CDResult:
    """Single-source allocation with native totals and explicitly retained tensors.

    Scalar logit projections retain one value per example. Internal targets
    preserve unselected dimensions. Diagnostics describe numerical reconstruction,
    not correctness of an allocation policy or causal faithfulness of a circuit.
    """

    target_beta: torch.Tensor
    target_gamma: torch.Tensor
    target_total: torch.Tensor
    source: CDSource
    target: CDTarget
    policy: str
    baseline_description: str
    coverage: tuple[str, ...]
    diagnostics: Mapping[str, CDReconstructionError]
    intermediates: Mapping[str, CDTensor]

    def __post_init__(self) -> None:
        if self.policy != "magnitude-affine-ordered-gelu-v1":
            raise ValueError("unsupported result allocation policy")
        if not isinstance(self.baseline_description, str) or not self.baseline_description.strip():
            raise ValueError("result baseline description must be nonempty")
        if len(set(self.coverage)) != len(self.coverage):
            raise ValueError("result coverage must contain unique hook names")
        owned = CDTensor(self.target_beta, self.target_gamma, self.target_total)
        object.__setattr__(self, "target_beta", owned.beta)
        object.__setattr__(self, "target_gamma", owned.gamma)
        object.__setattr__(self, "target_total", owned.total)
        object.__setattr__(self, "diagnostics", MappingProxyType(dict(self.diagnostics)))
        object.__setattr__(
            self,
            "intermediates",
            MappingProxyType(
                {
                    name: CDTensor(value.beta, value.gamma, value.total)
                    for name, value in self.intermediates.items()
                }
            ),
        )


class ContextualDecomposition:
    """Single-source nonlinear CD allocation on raw fp32 eager GPT-2 Bridge.

    No fitting, gradients, generation cache, joint sources, or arbitrary task
    metric is implied. The supported allocation policy is explicitly named;
    numerical reconstruction does not establish a causal explanation.
    """

    def __init__(
        self, model: TransformerBridge, *, policy: str = "magnitude-affine-ordered-gelu-v1"
    ) -> None:
        if policy != "magnitude-affine-ordered-gelu-v1":
            raise ValueError("unsupported contextual decomposition policy")
        if not isinstance(model, TransformerBridge):
            raise ValueError("ContextualDecomposition requires a TransformerBridge model")
        _require_gpt2(model)
        self._model = model

    @property
    def policy(self) -> str:
        """The implemented immutable allocation policy identity."""
        return "magnitude-affine-ordered-gelu-v1"

    @torch.no_grad()
    def decompose(
        self,
        tokens: torch.Tensor,
        *,
        source: CDSource,
        target: CDTarget,
        baseline: Optional[CDBaseline] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        retain_intermediates: tuple[str, ...] = (),
    ) -> CDResult:
        """Allocate one source to a downstream target without mutating the model.

        Args:
            tokens: Long [batch, sequence] IDs on the model device.
            source: One residual or head write, with optional absolute positions.
            target: Downstream internal tensor or logit projection.
            baseline: Zero reference by default, or a supplied reference snapshot.
            attention_mask: Aligned boolean key mask; selected padding is rejected.
            position_ids: Optional aligned long position IDs.
            retain_intermediates: Supported downstream hook names to retain.

        Returns:
            Owned detached target tensors, policy and baseline metadata, native
            reconstruction diagnostics, and only requested intermediate tensors.

        Raises:
            ValueError: On unsupported execution, invalid selectors/references,
                conflicting hooks, or native reconstruction failure.
        """
        if not isinstance(source, CDSource) or not isinstance(target, CDTarget):
            raise ValueError("source and target must be CDSource and CDTarget")
        if baseline is None:
            baseline = CDBaseline.zero()
        if not isinstance(baseline, CDBaseline):
            raise ValueError("baseline must be a CDBaseline")
        if (
            not isinstance(retain_intermediates, tuple)
            or any(not isinstance(name, str) for name in retain_intermediates)
            or len(set(retain_intermediates)) != len(retain_intermediates)
        ):
            raise ValueError("intermediate retention must be a tuple of unique hook names")
        n_layers = len(self._model.blocks)
        for name in retain_intermediates:
            if not isinstance(name, str) or _target_stage(name, n_layers) <= _target_stage(
                source.anchor, n_layers
            ):
                raise ValueError("retained intermediate must be a supported downstream hook")
        execution = _decompose_gpt2(
            self._model,
            tokens,
            source=source,
            target=target,
            reference=baseline._reference,
            attention_mask=attention_mask,
            position_ids=position_ids,
        )
        assert execution.target is not None
        target_name = target.hook_name if target.kind == "internal" else "unembed.hook_out"
        assert target_name is not None
        native = execution.native_cache[target_name]
        native_target = _select_target(
            _CDConstituents(native, torch.zeros_like(native)), target
        ).beta
        target_error = _reconstruction_error(
            execution.target.beta, execution.target.gamma, native_target
        )
        if target.kind == "logit_difference":
            # Subtraction can cancel native logits; propagate the two input error budgets.
            reference_scale = 2 * float(native.abs().max())
            target_error = _ReconstructionError(
                target_error.max_abs, reference_scale, target_error.constituent_max_abs
            )
        elif target.kind == "logit":
            target_error = _ReconstructionError(
                target_error.max_abs, float(native.abs().max()), target_error.constituent_max_abs
            )
        _check_reconstruction(target_error, "target")
        intermediates = {
            name: CDTensor(
                execution.constituents[name].beta,
                execution.constituents[name].gamma,
                execution.native_cache[name],
            )
            for name in retain_intermediates
        }
        diagnostics = {
            name: CDReconstructionError(
                error.max_abs, error.reference_max_abs, error.constituent_max_abs
            )
            for name, error in execution.errors.items()
        }
        diagnostics["target"] = CDReconstructionError(
            target_error.max_abs, target_error.reference_max_abs, target_error.constituent_max_abs
        )
        return CDResult(
            execution.target.beta,
            execution.target.gamma,
            native_target,
            source,
            target,
            self.policy,
            baseline.description,
            tuple(execution.errors),
            diagnostics,
            intermediates,
        )
