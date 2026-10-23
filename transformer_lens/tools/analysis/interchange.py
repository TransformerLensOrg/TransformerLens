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
:func:`neuron_basis` turns a set of MLP neurons, such as the features a sparse probe selects, into a
basis, and :func:`random_neuron_basis` draws random neurons as its placebo.
:func:`fit_das` learns the basis instead, by distributed alignment search (DAS; Geiger et al.,
2024): it trains an orthonormal basis so that the interchange produces the high-level model's
predicted outputs.

An interchange accuracy is not evidence on its own. DAS fits arbitrary label mappings and finds
effects in randomly initialised models (Arora et al., 2024), so the module also provides the
controls to train the same way: :func:`arbitrary_targets` for an arbitrary-label control, and
:func:`random_init_bridge` or :func:`reinitialized` for an untrained copy of the model. Compare a
learned subspace with both, and with the placebo and the baselines.

Works with any model exposing the hook API (``TransformerBridge`` or a ``HookedRootModule``).
Activations at the hook must be ``[batch, d_model]``, or ``[batch, pos, d_model]`` with one
position index per example.
"""

from __future__ import annotations

import copy
import hashlib
from dataclasses import dataclass
from typing import Any, Callable, Hashable, Iterator, Optional, Sequence, Union, cast

import numpy as np
import torch
import torch.nn.functional as F
from jaxtyping import Float, Int
from torch import nn
from torch.nn.utils.parametrizations import orthogonal

from transformer_lens.tools.analysis._model_state import require_eval_mode
from transformer_lens.tools.analysis.projection_kernel import orthonormal_subspace

TaskLogits = Callable[[torch.Tensor], Float[torch.Tensor, "batch n_classes"]]
"""Runs the model on a batch of inputs in one forward pass and returns task logits, e.g.
``lambda tokens: model(tokens)[:, -1, answer_token_ids]``."""

Intervention = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
"""Maps (base activations, source activations), both ``[batch, d_model]``, to new base activations."""


def random_basis(d_model: int, rank: int, seed: int) -> Float[torch.Tensor, "d_model rank"]:
    """A seeded random orthonormal basis: the baseline any aligned basis should beat.

    Args:
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


Neurons = Union[Sequence[int], np.ndarray, torch.Tensor]
"""Integer neuron indices: a list, a NumPy array, or a tensor such as a sparse probe's
``selected_features``."""


def neuron_basis(d_mlp: int, neurons: Neurons) -> Float[torch.Tensor, "d_mlp k"]:
    """The coordinate basis of a set of MLP neurons, e.g. the features a sparse probe selects.

    Interchanging it sets exactly those neurons to the source's values, at a neuron hook such as
    ``blocks.N.mlp.hook_post``. Compare it with :func:`random_neuron_basis` of the same size: a random
    direction is the wrong placebo for a claim about neurons, because it spreads over every neuron.

    Args:
        d_mlp: Number of neurons at the hook.
        neurons: Integer indices of the neurons, each in ``[0, d_mlp)`` and none repeated.

    Returns:
        Orthonormal basis of unit vectors, ``[d_mlp, k]``, in the order given.

    Raises:
        ValueError: If no neuron is given, or an index is not an integer, out of range or repeated.
    """
    index = _neuron_indices(d_mlp, neurons)
    if index.numel() == 0:
        raise ValueError("neuron_basis needs at least one neuron")
    basis = torch.zeros(d_mlp, index.numel())
    basis[index, torch.arange(index.numel())] = 1.0
    return basis


def random_neuron_basis(
    d_mlp: int,
    k: int,
    seed: int,
    exclude: Optional[Neurons] = None,
) -> Float[torch.Tensor, "d_mlp k"]:
    """The coordinate basis of ``k`` neurons drawn at random: the placebo for :func:`neuron_basis`.

    Args:
        d_mlp: Number of neurons at the hook.
        seed: Seed for the draw; the caller's random stream is untouched.
        exclude: Neurons the draw must avoid, usually the ones the placebo is compared with.

    Returns:
        Orthonormal basis of unit vectors, ``[d_mlp, k]``.

    Raises:
        ValueError: If fewer than ``k`` neurons remain to draw from, or ``exclude`` holds an
            invalid index.
    """
    excluded = _neuron_indices(d_mlp, exclude if exclude is not None else [])
    allowed = torch.ones(d_mlp, dtype=torch.bool)
    allowed[excluded] = False
    pool = allowed.nonzero().flatten()
    if not 0 < k <= len(pool):
        raise ValueError(
            f"k must be in [1, {len(pool)}] with {len(excluded)} neurons excluded, got {k}"
        )
    generator = torch.Generator().manual_seed(seed)
    return neuron_basis(d_mlp, pool[torch.randperm(len(pool), generator=generator)[:k]])


def _neuron_indices(d_mlp: int, neurons: Neurons) -> Int[torch.Tensor, "k"]:
    index = torch.as_tensor(neurons)
    if index.numel() == 0:  # an empty list arrives as a float tensor
        return torch.empty(0, dtype=torch.long)
    if index.dtype == torch.bool or index.is_floating_point() or index.is_complex():
        raise ValueError(f"neuron indices must be integers, got dtype {index.dtype}")
    if index.dim() != 1:
        raise ValueError(f"neuron indices must be one-dimensional, got shape {tuple(index.shape)}")
    index = index.long().cpu()
    if ((index < 0) | (index >= d_mlp)).any():
        raise ValueError(f"neuron indices must be in [0, {d_mlp}), got {index.tolist()}")
    if index.unique().numel() != index.numel():
        raise ValueError(f"neuron indices must not repeat, got {index.tolist()}")
    return index


class LearnedSubspace(nn.Module):
    """A trainable rank-``rank`` subspace of ``d_model`` whose basis stays orthonormal.

    Args:
        rank: Number of basis directions, in ``[1, d_model]``.
    """

    def __init__(self, d_model: int, rank: int):
        super().__init__()
        if not 0 < rank <= d_model:
            raise ValueError(f"rank must be in [1, {d_model}], got {rank}")
        self.proj = orthogonal(nn.Linear(rank, d_model, bias=False))

    @property
    def basis(self) -> Float[torch.Tensor, "d_model rank"]:
        return cast(torch.Tensor, self.proj.weight)


Basis = Union[torch.Tensor, LearnedSubspace]


def _basis(basis: Basis, like: torch.Tensor) -> torch.Tensor:
    b = basis.basis if isinstance(basis, LearnedSubspace) else basis
    return b.to(like.device, like.dtype)


@dataclass(frozen=True)
class SubspaceInterchange:
    """Set the base's coordinates inside ``basis`` to the source's.

    Attributes:
        basis: Orthonormal basis, ``[d_model, rank]``, or a :class:`LearnedSubspace`.
        strength: 1 interchanges the coordinates; other values scale the move, 0 leaves the base.
    """

    basis: Basis
    strength: float = 1.0

    def __call__(self, base: torch.Tensor, source: torch.Tensor) -> torch.Tensor:
        B = _basis(self.basis, base)
        return base + self.strength * ((source - base) @ B) @ B.T


@dataclass(frozen=True)
class NormMatchedPlacebo:
    """Move each base exactly as far as ``reference`` does, but inside a random subspace.

    An effect that the placebo also produces comes from displacing the activation by that much,
    not from the aligned subspace.

    Attributes:
        reference: The intervention whose displacement is matched.
        basis: Orthonormal basis of the placebo subspace, usually from :func:`random_basis` with
            the reference's rank, or from :func:`random_neuron_basis` for a reference built by
            :func:`neuron_basis`.
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
        inputs: Model inputs, one row per example.
        task_logits: Runs the model on a batch of inputs.
        positions: For ``[batch, pos, d_model]`` activations, the position to read per example.
        batch_size: Rows per forward pass.

    Returns:
        Activations, ``[batch, d_model]``.

    Raises:
        ValueError: If any part of the model is in training mode, or the activation does not match
            the positions given (or not given).
        RuntimeError: If the hook does not fire exactly once per forward pass.
    """
    require_eval_mode(model, operation="collect_activations")
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
        intervention: Maps base and source activations to new base activations, e.g. a
            :class:`SubspaceInterchange`.
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
        intervention: Maps base and source activations to new base activations.
        source_activations: Activations from the source inputs at the same hook point, one row per
            base.
        task_logits: Runs the model on a batch of inputs.
        positions: For ``[batch, pos, d_model]`` activations, the position to intervene on per base.
        batch_size: Rows per forward pass.

    Returns:
        Task logits under the intervention, ``[batch, n_classes]``.

    Raises:
        ValueError: If any part of the model is in training mode, or the inputs do not pair up.
    """
    require_eval_mode(model, operation="intervened_logits")
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


def _model_device(model: Any) -> torch.device:
    parameter = next(iter(model.parameters()), None)
    return parameter.device if parameter is not None else torch.device("cpu")


@dataclass(frozen=True)
class DASResult:
    """A subspace learned by :func:`fit_das`.

    Attributes:
        subspace: The learned subspace; pass it to :class:`SubspaceInterchange`.
        losses: Mean training loss per epoch.
        hook_name: The hook point it was trained at.
    """

    subspace: LearnedSubspace
    losses: tuple[float, ...]
    hook_name: str
    rank: int

    @property
    def basis(self) -> Float[torch.Tensor, "d_model rank"]:
        return self.subspace.basis.detach()


def fit_das(
    model: Any,
    hook_name: str,
    base_inputs: torch.Tensor,
    source_inputs: torch.Tensor,
    targets: Int[torch.Tensor, "batch"],
    task_logits: TaskLogits,
    *,
    rank: int,
    base_positions: Optional[Int[torch.Tensor, "batch"]] = None,
    source_positions: Optional[Int[torch.Tensor, "batch"]] = None,
    epochs: int = 20,
    batch_size: int = 128,
    lr: float = 1e-2,
    strength: float = 1.0,
    seed: int = 0,
    device: Optional[Union[str, torch.device]] = None,
    micro_batch_size: Optional[int] = None,
) -> DASResult:
    """Learn the subspace whose interchange makes the model output ``targets`` (DAS).

    ``targets[i]`` is the class the high-level causal model predicts for base ``i`` with the
    aligned variable set to its value in source ``i``. The basis is trained with cross-entropy
    on the task logits under the interchange. Model parameters are frozen during the call and
    their ``requires_grad`` flags restored afterwards.

    Train on pairs where several variables differ between base and source, labelled by the
    high-level model's interchange, as well as on pairs that differ only in the aligned variable:
    otherwise the subspace can learn whatever the pairs happen to move together.

    Args:
        model: A model exposing the hook API.
        hook_name: The hook point to align.
        base_inputs: Base inputs, one row per pair.
        source_inputs: Source inputs, one row per pair.
        targets: The high-level model's output class for each pair under the interchange.
        task_logits: Runs the model on a batch of inputs and returns task logits.
        rank: Dimension of the subspace.
        base_positions: For ``[batch, pos, d_model]`` activations, the base position per pair.
        source_positions: Likewise for the sources.
        epochs: Passes over the pairs.
        batch_size: Pairs per optimisation step.
        lr: Adam learning rate.
        strength: Interchange strength during training.
        seed: Seed for the basis initialisation and the batch order.
        device: Device for the subspace's parameters; by default the device of the model's
            parameters, or the CPU for a model without any.
        micro_batch_size: Split each batch into parts this size and accumulate gradients: the same
            gradients, up to floating-point rounding, in less memory.

    Returns:
        The learned subspace with its training losses.

    Raises:
        ValueError: If any part of the model is in training mode, or the inputs and targets differ
            in length.
    """
    require_eval_mode(model, operation="fit_das")
    if not (len(base_inputs) == len(source_inputs) == len(targets)):
        raise ValueError("base_inputs, source_inputs and targets must have the same length")
    rng = np.random.default_rng(seed)
    micro = micro_batch_size or batch_size
    source_acts = collect_activations(
        model, hook_name, source_inputs, task_logits, positions=source_positions, batch_size=micro
    )
    # The subspace is built on the CPU, so seed only the CPU generator, inside a fork that restores
    # it; torch.manual_seed would also reseed every CUDA, MPS and XPU generator
    with torch.random.fork_rng(devices=[]):
        torch.default_generator.manual_seed(seed)
        subspace = LearnedSubspace(source_acts.shape[1], rank)
    subspace = subspace.to(device if device is not None else _model_device(model))
    step = SubspaceInterchange(subspace, strength)
    optimizer = torch.optim.Adam(subspace.parameters(), lr=lr)
    labels = torch.as_tensor(targets, dtype=torch.long)

    flags = [p.requires_grad for p in model.parameters()]
    model.requires_grad_(False)
    losses = []
    try:
        for _ in range(epochs):
            order = torch.as_tensor(rng.permutation(len(labels)))
            total = 0.0
            for chunk in _chunks(len(order), batch_size):
                batch = order[chunk]
                optimizer.zero_grad()
                for part in _chunks(len(batch), micro):
                    idx = batch[part]
                    logits = run_with_intervention(
                        model,
                        hook_name,
                        step,
                        base_inputs[idx],
                        source_acts[idx],
                        task_logits,
                        positions=_at(base_positions, idx),
                    )
                    # Summed, then divided by the whole batch, so split batches give equal gradients
                    loss = F.cross_entropy(
                        logits, labels[idx].to(logits.device), reduction="sum"
                    ) / len(batch)
                    loss.backward()
                    total += loss.item() * len(batch)
                optimizer.step()
            losses.append(total / len(order))
    finally:
        for p, flag in zip(model.parameters(), flags):
            p.requires_grad_(flag)
    return DASResult(subspace.eval(), tuple(losses), hook_name, rank)


def arbitrary_targets(
    keys: Sequence[Hashable], n_classes: int, seed: int
) -> Int[torch.Tensor, "batch"]:
    """A fixed pseudo-random class per key: targets for the arbitrary-label control.

    Train :func:`fit_das` on these instead of the high-level model's targets, with the same pairs,
    rank and budget. The interchange accuracy it reaches shows how much accuracy the method's
    expressivity alone produces; a learned subspace should clearly exceed it.

    Args:
        keys: One key per pair identifying its source, e.g. the source text.

    Returns:
        A class in ``[0, n_classes)`` per key; equal keys always get equal classes.
    """

    def draw(key: Hashable) -> int:
        digest = hashlib.sha256(f"{seed}:{key!r}".encode()).digest()
        return int.from_bytes(digest[:8], "big") % n_classes

    return torch.tensor([draw(k) for k in keys], dtype=torch.long)


def reinitialized(model: nn.Module, seed: int) -> nn.Module:
    """A copy of ``model`` with freshly initialised parameters: the random-init control.

    For models built from standard ``nn`` layers, such as a ``HookedRootModule``: every module with
    ``reset_parameters`` is reset, so every parameter must belong to one. Buffers, such as fixed
    architectural matrices, are kept. For HuggingFace models use :func:`random_init_bridge`, which
    applies the architecture's own initialisation scheme.

    Args:
        model: The trained model; it is not modified.

    Returns:
        The untrained copy, on the device of the model's parameters, in evaluation mode.

    Raises:
        ValueError: If a parameter belongs to no module with ``reset_parameters``, so it would keep
            its trained value.
    """
    covered = {
        id(parameter)
        for module in model.modules()
        if callable(getattr(module, "reset_parameters", None))
        for parameter in module.parameters(recurse=False)
    }
    missing = [name for name, parameter in model.named_parameters() if id(parameter) not in covered]
    if missing:
        more = f" and {len(missing) - 5} more" if len(missing) > 5 else ""
        raise ValueError(
            f"reinitialized cannot reset {', '.join(missing[:5])}{more}: no module's "
            "reset_parameters covers them, so they would keep their trained values. Reset them "
            "yourself, or use random_init_bridge for a HuggingFace model."
        )
    device = _model_device(model)
    clone = copy.deepcopy(model).cpu()
    # Reset on the CPU, seeding only the CPU generator inside a fork that restores it;
    # torch.manual_seed would also reseed every CUDA, MPS and XPU generator
    with torch.random.fork_rng(devices=[]):
        torch.default_generator.manual_seed(seed)
        for module in clone.modules():
            reset = getattr(module, "reset_parameters", None)
            if callable(reset):
                reset()
    return clone.to(device).eval()


def random_init_bridge(
    model_name: str,
    *,
    seed: int,
    dtype: torch.dtype = torch.float32,
    device: Optional[Union[str, torch.device]] = None,
    **boot_kwargs: Any,
) -> Any:
    """A ``TransformerBridge`` with the architecture's HuggingFace initialisation and no training.

    The random-init control: train :func:`fit_das` on it with the same pairs, rank and budget as
    on the trained model. Effects DAS finds here come from the method and the architecture, not
    from anything the model learned. Built with ``load_weights=False``, so the model class and
    prepared configuration match the trained bridge.

    Args:
        model_name: Model name or alias, as for ``TransformerBridge.boot_transformers``.
        device: Device for the bridge; by default the device ``boot_transformers`` would choose.
        **boot_kwargs: Passed on to ``TransformerBridge.boot_transformers``, e.g. ``revision``.

    Returns:
        The untrained ``TransformerBridge``, in evaluation mode.
    """
    from transformer_lens.model_bridge import TransformerBridge
    from transformer_lens.utilities.multi_gpu import maybe_cast_floating_params

    # The model is built on the CPU, so seed only the CPU generator, inside a fork that restores it
    with torch.random.fork_rng(devices=[]):
        torch.default_generator.manual_seed(seed)
        bridge = TransformerBridge.boot_transformers(
            model_name, load_weights=False, dtype=dtype, device=device, **boot_kwargs
        )
    # Building from a config follows the config's dtype and leaves the weights on the CPU,
    # whatever device boot recorded; HuggingFace also leaves the model in training mode
    maybe_cast_floating_params(bridge.original_model, dtype)
    return bridge.to(bridge.cfg.device).eval()
