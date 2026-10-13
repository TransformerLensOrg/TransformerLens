"""Interchange interventions on linear subspaces.

An interchange intervention runs the model on a *base* input while setting the base's activation
at one hook point, inside a subspace, to its value on a *source* input::

    h' = h_base + strength * (h_source - h_base) @ B @ B.T

``B`` is an orthonormal ``[d_model, rank]`` basis, the convention of
:func:`~transformer_lens.tools.analysis.projection_kernel.orthonormal_subspace`. The base keeps its
own coordinates outside the subspace. If the model then produces the output a high-level causal
model predicts for the base with one variable taken from the source, the subspace carries that
variable at that hook point (Geiger et al., 2021).

The module provides the intervention, fixed-direction bases to intervene on (the difference of
class means, a linear probe's weights, a random basis), and a norm-matched random-subspace placebo.
An interchange accuracy is not evidence on its own: compare it with the placebo and the baselines.

Works with any model exposing the hook API (``TransformerBridge`` or a ``HookedRootModule``).
Activations at the hook must be ``[batch, d_model]``, or ``[batch, pos, d_model]`` with one
position index per example.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Hashable, Iterator, Optional, Sequence, Union

import numpy as np
import torch
from jaxtyping import Float, Int

from transformer_lens.tools.analysis.projection_kernel import orthonormal_subspace

TaskLogits = Callable[[torch.Tensor], Float[torch.Tensor, "batch n_classes"]]
"""Runs the model on a batch of inputs in one forward pass and returns task logits, e.g.
``lambda tokens: model(tokens)[:, -1, answer_token_ids]``."""

Intervention = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
"""Maps (base activations, source activations), both ``[batch, d_model]``, to new base activations."""


def random_basis(d_model: int, rank: int, seed: int) -> Float[torch.Tensor, "d_model rank"]:
    """A seeded random orthonormal basis: the baseline any aligned basis should beat.

    Args:
        d_model: Dimension of the activation.
        rank: Number of basis directions.
        seed: Seed for the Gaussian matrix the basis is drawn from.

    Returns:
        Orthonormal basis, ``[d_model, rank]``.
    """
    generator = torch.Generator().manual_seed(seed)
    return orthonormal_subspace(torch.randn(d_model, rank, generator=generator), rank=rank).basis


def mean_difference_basis(
    activations: Float[torch.Tensor, "batch d_model"], labels: Sequence[Hashable]
) -> Float[torch.Tensor, "d_model rank"]:
    """The span of the centred class means: for two classes, the difference-in-means direction.

    Args:
        activations: Activations, one row per example.
        labels: The variable's value for each example; at least two distinct values.

    Returns:
        Orthonormal basis of rank ``n_classes - 1``, ``[d_model, n_classes - 1]``.

    Raises:
        ValueError: If fewer than two classes are present.
    """
    values = np.asarray(labels)
    classes = np.unique(values)
    if len(classes) < 2:
        raise ValueError("mean_difference_basis needs at least two classes")
    acts = activations.float().cpu()
    means = torch.stack([acts[torch.as_tensor(values == c)].mean(0) for c in classes])
    return orthonormal_subspace((means - means.mean(0)).T, rank=len(classes) - 1).basis


def probe_basis(
    weights: Float[torch.Tensor, "n_rows d_model"],
) -> Float[torch.Tensor, "d_model rank"]:
    """The directions of a linear probe's weights, in activation space.

    Undo any input scaling first: for a probe fit on standardised inputs, divide each weight by its
    feature's standard deviation.

    Args:
        weights: Probe weights; one row for a binary probe, one row per class otherwise.

    Returns:
        Orthonormal basis: the weight direction for one row, the span of the centred rows (rank
        ``n_rows - 1``) for several.
    """
    w = torch.as_tensor(weights, dtype=torch.float32)
    if w.shape[0] == 1:
        return orthonormal_subspace(w.T, rank=1).basis
    return orthonormal_subspace((w - w.mean(0)).T, rank=w.shape[0] - 1).basis


@dataclass(frozen=True)
class SubspaceInterchange:
    """Set the base's coordinates inside ``basis`` to the source's.

    Attributes:
        basis: Orthonormal basis, ``[d_model, rank]``.
        strength: 1 interchanges the coordinates; other values scale the move, 0 leaves the base.
    """

    basis: torch.Tensor
    strength: float = 1.0

    def __call__(self, base: torch.Tensor, source: torch.Tensor) -> torch.Tensor:
        B = self.basis.to(base.device, base.dtype)
        return base + self.strength * ((source - base) @ B) @ B.T


@dataclass(frozen=True)
class NormMatchedPlacebo:
    """Move each base exactly as far as ``reference`` does, but inside a random subspace.

    An effect that the placebo also produces comes from displacing the activation by that much,
    not from the aligned subspace.

    Attributes:
        reference: The intervention whose displacement is matched.
        basis: Orthonormal basis of the placebo subspace, usually from :func:`random_basis` with
            the reference's rank.
    """

    reference: SubspaceInterchange
    basis: torch.Tensor

    def __call__(self, base: torch.Tensor, source: torch.Tensor) -> torch.Tensor:
        distance = (self.reference(base, source) - base).norm(dim=-1, keepdim=True)
        B = self.basis.to(base.device, base.dtype)
        move = ((source - base) @ B) @ B.T
        return base + move * (distance / move.norm(dim=-1, keepdim=True).clamp_min(1e-8))


def _select(act: torch.Tensor, positions: Optional[torch.Tensor]) -> torch.Tensor:
    if positions is None:
        if act.ndim != 2:
            raise ValueError(
                f"activation has shape {tuple(act.shape)}; pass one position index per example"
            )
        return act
    if act.ndim != 3:
        raise ValueError(
            f"with positions, the activation must be [batch, pos, d_model]; "
            f"got shape {tuple(act.shape)}"
        )
    if positions.shape != (act.shape[0],):
        raise ValueError(
            f"positions must hold one index per example, shape ({act.shape[0]},); "
            f"got {tuple(positions.shape)}"
        )
    return act[torch.arange(act.shape[0], device=act.device), positions.to(act.device)]


def _write(act: torch.Tensor, positions: Optional[torch.Tensor], new: torch.Tensor) -> torch.Tensor:
    if positions is None:
        return new
    out = act.clone()
    out[torch.arange(act.shape[0], device=act.device), positions.to(act.device)] = new
    return out


def _chunks(n: int, size: int) -> Iterator[slice]:
    for start in range(0, n, size):
        yield slice(start, min(start + size, n))


def _at(
    positions: Optional[torch.Tensor], idx: Union[slice, torch.Tensor]
) -> Optional[torch.Tensor]:
    return None if positions is None else positions[idx]


@torch.no_grad()
def collect_activations(
    model: Any,
    hook_name: str,
    inputs: torch.Tensor,
    task_logits: TaskLogits,
    *,
    positions: Optional[Int[torch.Tensor, "batch"]] = None,
    batch_size: int = 256,
) -> Float[torch.Tensor, "batch d_model"]:
    """Activations at ``hook_name``, at one position per example if given, returned on the CPU.

    Args:
        model: A model exposing the hook API.
        hook_name: The hook point to read.
        inputs: Model inputs, one row per example.
        task_logits: Runs the model on a batch of inputs.
        positions: For ``[batch, pos, d_model]`` activations, the position to read per example.
        batch_size: Rows per forward pass.

    Returns:
        Activations, ``[batch, d_model]``.

    Raises:
        ValueError: If the activation does not match the positions given (or not given).
        RuntimeError: If the hook does not fire exactly once per forward pass.
    """
    out = []
    for idx in _chunks(len(inputs), batch_size):
        store: list[torch.Tensor] = []

        def grab(
            act: torch.Tensor, hook: Any, _positions: Optional[torch.Tensor] = _at(positions, idx)
        ) -> None:
            store.append(_select(act, _positions).detach().cpu())

        with model.hooks(fwd_hooks=[(hook_name, grab)]):
            task_logits(inputs[idx])
        if len(store) != 1:
            raise RuntimeError(f"{hook_name} fired {len(store)} times in one forward pass")
        out.append(store[0])
    return torch.cat(out)


def run_with_intervention(
    model: Any,
    hook_name: str,
    intervention: Intervention,
    base_inputs: torch.Tensor,
    source_activations: Float[torch.Tensor, "batch d_model"],
    task_logits: TaskLogits,
    *,
    positions: Optional[Int[torch.Tensor, "batch"]] = None,
) -> Float[torch.Tensor, "batch n_classes"]:
    """Task logits for ``base_inputs`` with ``intervention`` applied at ``hook_name``, in one pass.

    Differentiable with respect to any parameters of ``intervention``, for training an
    intervention. To evaluate one over many pairs in bounded memory, use :func:`intervened_logits`.

    Args:
        model: A model exposing the hook API.
        hook_name: The hook point to intervene on.
        intervention: Maps base and source activations to new base activations, e.g. a
            :class:`SubspaceInterchange`.
        base_inputs: Model inputs to run.
        source_activations: Activations from the source inputs at the same hook point, one row per
            base, e.g. from :func:`collect_activations`.
        task_logits: Runs the model on a batch of inputs.
        positions: For ``[batch, pos, d_model]`` activations, the position to intervene on per base.

    Returns:
        Task logits under the intervention.

    Raises:
        ValueError: If the activation does not match the positions given (or not given).
        RuntimeError: If the hook does not fire exactly once in the forward pass.
    """
    calls = 0

    def hook(act: torch.Tensor, hook: Any) -> torch.Tensor:
        nonlocal calls
        calls += 1
        selected = _select(act, positions)
        source = source_activations.to(act.device, act.dtype)
        return _write(act, positions, intervention(selected, source))

    with model.hooks(fwd_hooks=[(hook_name, hook)]):
        logits = task_logits(base_inputs)
    # A second firing would reuse source activations collected for the first
    if calls != 1:
        raise RuntimeError(f"{hook_name} fired {calls} times in one forward pass")
    return logits


@torch.no_grad()
def intervened_logits(
    model: Any,
    hook_name: str,
    intervention: Intervention,
    base_inputs: torch.Tensor,
    source_activations: Float[torch.Tensor, "batch d_model"],
    task_logits: TaskLogits,
    *,
    positions: Optional[Int[torch.Tensor, "batch"]] = None,
    batch_size: int = 256,
) -> Float[torch.Tensor, "batch n_classes"]:
    """Task logits under ``intervention``, in batches without gradients, returned on the CPU.

    The evaluation counterpart of :func:`run_with_intervention`, with the same arguments and
    checks: memory is bounded by ``batch_size`` rather than by the number of pairs.

    Args:
        model: A model exposing the hook API.
        hook_name: The hook point to intervene on.
        intervention: Maps base and source activations to new base activations.
        base_inputs: Model inputs to run.
        source_activations: Activations from the source inputs at the same hook point, one row per
            base.
        task_logits: Runs the model on a batch of inputs.
        positions: For ``[batch, pos, d_model]`` activations, the position to intervene on per base.
        batch_size: Rows per forward pass.

    Returns:
        Task logits under the intervention, ``[batch, n_classes]``.
    """
    if len(source_activations) != len(base_inputs):
        raise ValueError("source_activations must have one row per base input")
    out = []
    for idx in _chunks(len(base_inputs), batch_size):
        logits = run_with_intervention(
            model,
            hook_name,
            intervention,
            base_inputs[idx],
            source_activations[idx],
            task_logits,
            positions=_at(positions, idx),
        )
        out.append(logits.detach().cpu())
    return torch.cat(out)
