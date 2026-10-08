"""Patching.

A module for patching activations in a transformer model, and measuring the effect of the patch on
the output. This implements the activation patching technique for a range of types of activation.
The structure is to have a single :func:`generic_activation_patch` function that does everything,
and to have a range of specialised functions for specific types of activation.

Context:

Activation Patching is technique introduced in the `ROME paper <http://rome.baulab.info/>`, which
uses a causal intervention to identify which activations in a model matter for producing some
output. It runs the model on input A, replaces (patches) an activation with that same activation on
input B, and sees how much that shifts the answer from A to B.

More details: The setup of activation patching is to take two runs of the model on two different
inputs, the clean run and the corrupted run. The clean run outputs the correct answer and the
corrupted run does not. The key idea is that we give the model the corrupted input, but then
intervene on a specific activation and patch in the corresponding activation from the clean run (ie
replace the corrupted activation with the clean activation), and then continue the run. And we then
measure how much the output has updated towards the correct answer.

- We can then iterate over many
    possible activations and look at how much they affect the corrupted run. If patching in an
    activation significantly increases the probability of the correct answer, this allows us to
    localise which activations matter.
- A key detail is that we move a single activation __from__ the clean run __to __the corrupted run.
    So if this changes the answer from incorrect to correct, we can be confident that the activation
    moved was important.

Intuition:

The ability to **localise** is a key move in mechanistic interpretability - if the computation is
diffuse and spread across the entire model, it is likely much harder to form a clean mechanistic
story for what's going on. But if we can identify precisely which parts of the model matter, we can
then zoom in and determine what they represent and how they connect up with each other, and
ultimately reverse engineer the underlying circuit that they represent. And, empirically, on at
least some tasks activation patching tends to find that computation is extremely localised:

- This technique helps us precisely identify which parts of the model matter for a certain
    part of a task. Eg, answering “The Eiffel Tower is in” with “Paris” requires figuring out that
    the Eiffel Tower is in Paris, and that it’s a factual recall task and that the output is a
    location. Patching to “The Colosseum is in” controls for everything other than the “Eiffel Tower
    is located in Paris” feature.
- It helps a lot if the corrupted prompt has the same number of tokens

This, unlike direct logit attribution, can identify meaningful parts of a circuit from anywhere
within the model, rather than just the end.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from functools import partial
from typing import (
    Callable,
    Dict,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Union,
    overload,
)

import einops
import pandas as pd
import torch
from jaxtyping import Float, Int
from tqdm.auto import tqdm
from typing_extensions import Literal

import transformer_lens.utilities as utils
from transformer_lens.ActivationCache import ActivationCache
from transformer_lens.model_protocol import TransformerLensModel
from transformer_lens.utilities.statistics import (
    bootstrap_ci,
    derive_generator,
    sign_flip_permutation_pvalue,
    standard_error,
)

# %%
Logits = torch.Tensor
AxisNames = Literal["layer", "pos", "head_index", "head", "src_pos", "dest_pos"]


# %%


def make_df_from_ranges(
    column_max_ranges: Sequence[int], column_names: Sequence[str]
) -> pd.DataFrame:
    """
    Takes in a list of column names and max ranges for each column, and returns a dataframe with the cartesian product of the range for each column (ie iterating through all combinations from zero to column_max_range - 1, in order, incrementing the final column first)
    """
    rows = list(itertools.product(*[range(axis_max_range) for axis_max_range in column_max_ranges]))
    df = pd.DataFrame(rows, columns=column_names)
    return df


# %%
CorruptedActivation = torch.Tensor
PatchedActivation = torch.Tensor
MetricFn = Callable[[Float[torch.Tensor, "batch pos d_vocab"]], torch.Tensor]
Reduce = Literal["scalar", "mean", "none"]
_REDUCE_MODES = ("scalar", "mean", "none")
_RESULT_AXIS_RENAMES = {"head_index": "head", "dest_pos": "pos"}


@dataclass
class PatchingResult:
    """Per-example (or per-metric) activation-patching results with statistics on top.

    Returned by :func:`generic_activation_patch` and every ``get_act_patch_*``
    function when ``reduce="none"``, ``baseline=True`` or several metrics are
    requested. ``values`` is laid out ``[metric?, *axes | n_rows, batch?]``: the
    metric axis exists only when a mapping of metrics was passed
    (``has_metric_axis``), the middle axes are ``axis_names`` (or one flat
    ``n_rows`` axis when an ``index_df`` was supplied), and the trailing batch
    axis exists only when ``reduced`` is ``False``.

    Attributes:
        values: The patched metric values.
        axis_names: Names of the swept axes, ``[]`` in flat ``index_df`` mode.
        index_df: One row per swept cell, in the order the cells were run.
        metric_names: Metric names, ``["metric"]`` for a single callable.
        reduced: ``True`` when the batch axis has been averaged away.
        baseline: Unpatched corrupted-run metric, ``[metric?, batch]`` (or
            ``[metric?]`` when reduced), when the sweep was run with
            ``baseline=True``.
        has_metric_axis: Whether ``values`` carries a leading metric axis.
        patch_type_names: Names of the ``patch_type`` axis entries when the
            result came from a ``*_every`` helper (``"out"``, ``"q"``, ...).
            GQA ``k``/``v`` sweeps are zero-padded to ``n_heads`` there, so the
            padded cells appear as zero-valued rows in ``index_df`` and
            :meth:`to_dataframe`.
    """

    values: torch.Tensor
    axis_names: List[str]
    index_df: pd.DataFrame
    metric_names: List[str]
    reduced: bool
    baseline: Optional[torch.Tensor] = None
    has_metric_axis: bool = False
    patch_type_names: Optional[List[str]] = None

    def __getitem__(self, metric_name: str) -> "PatchingResult":
        """One metric's result, with the metric axis dropped."""
        if metric_name not in self.metric_names:
            raise KeyError(f"unknown metric {metric_name!r}; have {self.metric_names}")
        if not self.has_metric_axis:
            return self
        i = self.metric_names.index(metric_name)
        return PatchingResult(
            values=self.values[i],
            axis_names=list(self.axis_names),
            index_df=self.index_df,
            metric_names=[metric_name],
            reduced=self.reduced,
            baseline=None if self.baseline is None else self.baseline[i],
            has_metric_axis=False,
            patch_type_names=self.patch_type_names,
        )

    def _require_per_example(self, what: str) -> None:
        if self.reduced:
            raise ValueError(f"{what} needs per-example values; run the sweep with reduce='none'")

    def mean(self) -> torch.Tensor:
        """Batch mean per cell (``values`` itself when already reduced)."""
        return self.values if self.reduced else self.values.mean(dim=-1)

    def stderr(self) -> torch.Tensor:
        """Standard error of the batch mean per cell."""
        self._require_per_example("stderr")
        return standard_error(self.values, dim=-1)

    def _broadcast_baseline(self) -> torch.Tensor:
        if self.baseline is None:
            raise ValueError("effect needs a baseline; run the sweep with baseline=True")
        lead = 1 if self.has_metric_axis else 0
        middle = self.values.ndim - lead - (0 if self.reduced else 1)
        shape = (
            tuple(self.baseline.shape[:lead]) + (1,) * middle + tuple(self.baseline.shape[lead:])
        )
        return self.baseline.reshape(shape)

    def effect(self) -> torch.Tensor:
        """``values - baseline``: the change each patch makes to the corrupted run, per example."""
        return self.values - self._broadcast_baseline()

    def normalized(self, clean_baseline: torch.Tensor) -> torch.Tensor:
        """``(values - corrupted) / (clean - corrupted)`` per example.

        ``clean_baseline`` is the same metric evaluated on the unpatched clean run,
        shaped like ``baseline``. 0 means the patch did nothing, 1 means it
        restored the clean metric. A prompt whose clean and corrupted metrics
        coincide has no recovery scale and yields ``inf`` / ``nan`` for that
        example; filter such prompts out before normalizing.
        """
        corrupted = self._broadcast_baseline()
        baseline = self.baseline
        assert baseline is not None  # _broadcast_baseline raised otherwise
        clean = torch.as_tensor(clean_baseline, dtype=self.values.dtype, device=self.values.device)
        if clean.shape != baseline.shape:
            raise ValueError(
                f"clean_baseline must match baseline shape {tuple(baseline.shape)}, "
                f"got {tuple(clean.shape)}"
            )
        clean = clean.reshape(corrupted.shape)
        return (self.values - corrupted) / (clean - corrupted)

    def bootstrap_ci(
        self,
        *,
        confidence: float = 0.95,
        n_resamples: int = 1000,
        seed: int = 0,
        statistic: str = "mean",
        of_effect: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Percentile bootstrap interval of the batch ``statistic`` per cell.

        ``of_effect=True`` bootstraps ``effect()`` instead of ``values``. Seeded
        through :func:`~transformer_lens.utilities.statistics.derive_generator`.
        """
        self._require_per_example("bootstrap_ci")
        data = self.effect() if of_effect else self.values
        return bootstrap_ci(
            data,
            dim=-1,
            statistic=statistic,
            confidence=confidence,
            n_resamples=n_resamples,
            generator=derive_generator(seed, "bootstrap", statistic),
        )

    def permutation_pvalue(
        self,
        *,
        n_permutations: int = 1000,
        seed: int = 0,
        alternative: str = "two-sided",
    ) -> torch.Tensor:
        """Sign-flip permutation p-value that each cell's mean effect is zero.

        Uses ``effect()`` (needs ``baseline=True``); exact for 12 or fewer
        examples, Monte Carlo above that.
        """
        self._require_per_example("permutation_pvalue")
        return sign_flip_permutation_pvalue(
            self.effect(),
            dim=-1,
            n_permutations=n_permutations,
            generator=derive_generator(seed, "permutation"),
            alternative=alternative,
        )

    def to_dataframe(self) -> pd.DataFrame:
        """Long form: one row per (metric, swept cell, example) with the value."""
        rows = []
        n_metrics = len(self.metric_names)
        values = self.values if self.has_metric_axis else self.values.unsqueeze(0)
        flat_cells = values.reshape(n_metrics, len(self.index_df), -1).detach().cpu()
        for m, name in enumerate(self.metric_names):
            for c, (_, index_row) in enumerate(self.index_df.iterrows()):
                for b in range(flat_cells.shape[-1]):
                    row = {"metric": name, **index_row.to_dict()}
                    if self.patch_type_names is not None and "patch_type" in row:
                        row["patch_type_name"] = self.patch_type_names[int(row["patch_type"])]
                    if not self.reduced:
                        row["example"] = b
                    row["value"] = flat_cells[m, c, b].item()
                    rows.append(row)
        return pd.DataFrame(rows)


@overload
def generic_activation_patch(
    model: TransformerLensModel,
    corrupted_tokens: Int[torch.Tensor, "batch pos"],
    clean_cache: ActivationCache,
    patching_metric: MetricFn,
    patch_setter: Callable[
        [CorruptedActivation, Sequence[int], ActivationCache], PatchedActivation
    ],
    activation_name: str,
    index_axis_names: Optional[Sequence[AxisNames]] = None,
    index_df: Optional[pd.DataFrame] = None,
    return_index_df: Literal[False] = False,
    *,
    reduce: Literal["scalar", "mean"] = "scalar",
    baseline: Literal[False] = False,
) -> torch.Tensor:
    ...


@overload
def generic_activation_patch(
    model: TransformerLensModel,
    corrupted_tokens: Int[torch.Tensor, "batch pos"],
    clean_cache: ActivationCache,
    patching_metric: MetricFn,
    patch_setter: Callable[
        [CorruptedActivation, Sequence[int], ActivationCache], PatchedActivation
    ],
    activation_name: str,
    index_axis_names: Optional[Sequence[AxisNames]],
    index_df: Optional[pd.DataFrame],
    return_index_df: Literal[True],
    *,
    reduce: Literal["scalar", "mean"] = "scalar",
    baseline: Literal[False] = False,
) -> Tuple[torch.Tensor, pd.DataFrame]:
    ...


@overload
def generic_activation_patch(
    model: TransformerLensModel,
    corrupted_tokens: Int[torch.Tensor, "batch pos"],
    clean_cache: ActivationCache,
    patching_metric: Union[MetricFn, Mapping[str, MetricFn]],
    patch_setter: Callable[
        [CorruptedActivation, Sequence[int], ActivationCache], PatchedActivation
    ],
    activation_name: str,
    index_axis_names: Optional[Sequence[AxisNames]] = None,
    index_df: Optional[pd.DataFrame] = None,
    return_index_df: bool = False,
    *,
    reduce: Literal["none"],
    baseline: bool = False,
) -> PatchingResult:
    ...


@overload
def generic_activation_patch(
    model: TransformerLensModel,
    corrupted_tokens: Int[torch.Tensor, "batch pos"],
    clean_cache: ActivationCache,
    patching_metric: MetricFn,
    patch_setter: Callable[
        [CorruptedActivation, Sequence[int], ActivationCache], PatchedActivation
    ],
    activation_name: str,
    index_axis_names: Optional[Sequence[AxisNames]] = None,
    index_df: Optional[pd.DataFrame] = None,
    return_index_df: bool = False,
    *,
    reduce: Reduce = "scalar",
    baseline: Literal[True],
) -> PatchingResult:
    ...


@overload
def generic_activation_patch(
    model: TransformerLensModel,
    corrupted_tokens: Int[torch.Tensor, "batch pos"],
    clean_cache: ActivationCache,
    patching_metric: Mapping[str, MetricFn],
    patch_setter: Callable[
        [CorruptedActivation, Sequence[int], ActivationCache], PatchedActivation
    ],
    activation_name: str,
    index_axis_names: Optional[Sequence[AxisNames]] = None,
    index_df: Optional[pd.DataFrame] = None,
    return_index_df: bool = False,
    *,
    reduce: Reduce = "scalar",
    baseline: bool = False,
) -> PatchingResult:
    ...


def generic_activation_patch(
    model: TransformerLensModel,
    corrupted_tokens: Int[torch.Tensor, "batch pos"],
    clean_cache: ActivationCache,
    patching_metric: Union[MetricFn, Mapping[str, MetricFn]],
    patch_setter: Callable[
        [CorruptedActivation, Sequence[int], ActivationCache], PatchedActivation
    ],
    activation_name: str,
    index_axis_names: Optional[Sequence[AxisNames]] = None,
    index_df: Optional[pd.DataFrame] = None,
    return_index_df: bool = False,
    *,
    reduce: str = "scalar",  # Reduce; widened so the ValueError below fires before beartype
    baseline: bool = False,
) -> Union[torch.Tensor, Tuple[torch.Tensor, pd.DataFrame], PatchingResult]:
    """
    A generic function to do activation patching, will be specialised to specific use cases.

    Activation patching is about studying the counterfactual effect of a specific activation between a clean run and a corrupted run. The idea is have two inputs, clean and corrupted, which have two different outputs, and differ in some key detail. Eg "The Eiffel Tower is in" vs "The Colosseum is in". Then to take a cached set of activations from the "clean" run, and a set of corrupted.

    Internally, the key function comes from three things: A list of tuples of indices (eg (layer, position, head_index)), a index_to_act_name function which identifies the right activation for each index, a patch_setter function which takes the corrupted activation, the index and the clean cache, and a metric for how well the patched model has recovered.

    The indices can either be given explicitly as a pandas dataframe, or by listing the relevant axis names and having them inferred from the tokens and the model config. It is assumed that the first column is always layer.

    This function then iterates over every tuple of indices, does the relevant patch, and stores it

    Every patched forward runs the whole corrupted batch, so per-example values are
    available for free: ``reduce="none"`` keeps them (the metric must then return a
    ``[batch]`` tensor) and returns a :class:`PatchingResult` with mean, standard
    error, bootstrap intervals and permutation p-values on top; ``reduce="mean"``
    averages a ``[batch]`` metric back to today's tensor; a mapping of metrics
    evaluates all of them on each patched forward. ``baseline=True`` adds one
    unpatched corrupted forward so effects and normalized recovery can be reported.

    Args:
        model: The relevant model
        corrupted_tokens: The input tokens for the corrupted run
        clean_cache: The cached activations from the clean run
        patching_metric: A function from the model's output logits to some metric (eg loss, logit diff, etc), or a mapping of names to such functions. Must return a scalar under ``reduce="scalar"`` and a ``[batch]`` tensor otherwise.
        patch_setter: A function which acts on (corrupted_activation, index, clean_cache) to edit the activation and patch in the relevant chunk of the clean activation
        activation_name: The name of the activation being patched
        index_axis_names: The names of the axes to (fully) iterate over, implicitly fills in index_df
        index_df: The dataframe of indices, columns are axis names and each row is a tuple of indices. Will be inferred from index_axis_names if not given. When this is input, the output will be a flattened tensor with an element per row of index_df
        return_index_df: A Boolean flag for whether to return the dataframe of indices too (tensor return only; a ``PatchingResult`` always carries ``index_df``)
        reduce: ``"scalar"`` (default, metric returns a scalar, tensor return), ``"mean"`` (metric returns ``[batch]``, averaged, tensor return) or ``"none"`` (metric returns ``[batch]``, kept, ``PatchingResult`` return)
        baseline: Also evaluate the metric on the unpatched corrupted run and return a ``PatchingResult`` carrying it

    Returns:
        patched_output: The tensor of the patching metric for each patch. By default it has one dimension for each index dimension, via index_df set explicitly it is flattened with one element per row.
        index_df *optional*: The dataframe of indices
        Or a :class:`PatchingResult` when ``reduce="none"``, ``baseline=True`` or a metric mapping was given.
    """
    # Lazy import: tools.analysis imports this module's neighbours at package import.
    from transformer_lens.tools.analysis._model_state import require_eval_mode

    require_eval_mode(model, operation="activation patching")
    if reduce not in _REDUCE_MODES:
        raise ValueError(f"reduce must be one of {_REDUCE_MODES}, got {reduce!r}")
    metrics: Dict[str, MetricFn]
    if isinstance(patching_metric, Mapping):
        has_metric_axis = True
        metrics = dict(patching_metric)
        if not metrics:
            raise ValueError("patching_metric mapping must contain at least one metric")
    else:
        has_metric_axis = False
        metrics = {"metric": patching_metric}
    want_result = has_metric_axis or reduce == "none" or baseline
    if want_result and return_index_df:
        raise ValueError(
            "return_index_df only applies to the tensor return; a PatchingResult carries index_df"
        )

    if index_df is None:
        assert index_axis_names is not None
        number_of_heads = model.cfg.n_heads
        # For some models, the number of key value heads is not the same as the number of attention heads
        if activation_name in ["k", "v"] and model.cfg.n_key_value_heads is not None:
            number_of_heads = model.cfg.n_key_value_heads

        # Get the max range for all possible axes
        max_axis_range = {
            "layer": model.cfg.n_layers,
            "pos": corrupted_tokens.shape[-1],
            "head_index": number_of_heads,
        }
        max_axis_range["src_pos"] = max_axis_range["pos"]
        max_axis_range["dest_pos"] = max_axis_range["pos"]
        max_axis_range["head"] = max_axis_range["head_index"]

        # Get the max range for each axis we iterate over
        index_axis_max_range = [max_axis_range[axis_name] for axis_name in index_axis_names]

        # Get the dataframe where each row is a tuple of indices
        index_df = make_df_from_ranges(index_axis_max_range, index_axis_names)

        flattened_output = False
    else:
        # A dataframe of indices was provided. Verify that we did not *also* receive index_axis_names
        assert index_axis_names is None
        index_axis_max_range = index_df.max().to_list()
        flattened_output = True

    batch_size = int(corrupted_tokens.shape[0])
    per_example = reduce != "scalar"
    cell_shape = (len(index_df),) if flattened_output else tuple(index_axis_max_range)
    store_shape = (len(metrics),) + cell_shape + ((batch_size,) if per_example else ())
    # Create an empty tensor to show the patched metric for each patch
    store = torch.zeros(store_shape, device=model.cfg.device)

    def evaluate(logits: torch.Tensor) -> torch.Tensor:
        """Every metric on one set of logits, shaped [metric, batch?] after validation."""
        rows = []
        for name, fn in metrics.items():
            out = torch.as_tensor(fn(logits))
            if per_example:
                if out.ndim != 1 or out.shape[0] != batch_size:
                    raise ValueError(
                        f"patching_metric {name!r} must return a [batch] tensor of length "
                        f"{batch_size} under reduce={reduce!r}, got shape {tuple(out.shape)}"
                    )
            elif out.ndim != 0:
                raise ValueError(
                    f"patching_metric {name!r} must return a scalar under reduce='scalar', got "
                    f"shape {tuple(out.shape)}; return a [batch] tensor and pass reduce='mean' "
                    "or reduce='none' to keep per-example values"
                )
            rows.append(out.detach())
        return torch.stack(rows).to(store.dtype)

    baseline_values: Optional[torch.Tensor] = None
    if baseline:
        # Same entry point as the patched runs, with no hooks attached.
        baseline_values = evaluate(model.run_with_hooks(corrupted_tokens, fwd_hooks=[]))

    # A generic patching hook - for each index, it applies the patch_setter appropriately to patch the activation
    def patching_hook(corrupted_activation, hook, index, clean_activation):
        if corrupted_activation.requires_grad:
            corrupted_activation = corrupted_activation.clone()
        return patch_setter(corrupted_activation, index, clean_activation)

    for c, index_row in enumerate(tqdm((list(index_df.iterrows())))):
        index = index_row[1].to_list()

        # The current activation name is just the activation name plus the layer (assumed to be the first element of the input)
        current_activation_name = utils.get_act_name(activation_name, layer=index[0])

        # The hook function cannot receive additional inputs, so we use partial to include the specific index and the corresponding clean activation
        current_hook = partial(
            patching_hook,
            index=index,
            clean_activation=clean_cache[current_activation_name],
        )

        patched_logits = model.run_with_hooks(
            corrupted_tokens, fwd_hooks=[(current_activation_name, current_hook)]
        )
        values = evaluate(patched_logits)
        if flattened_output:
            store[:, c] = values
        else:
            store[(slice(None), *index)] = values

    if reduce == "mean":
        store = store.mean(dim=-1)
        if baseline_values is not None:
            baseline_values = baseline_values.mean(dim=-1)

    if not want_result:
        patched_metric_output = store[0]
        if return_index_df:
            return patched_metric_output, index_df
        return patched_metric_output

    return PatchingResult(
        values=store if has_metric_axis else store[0],
        axis_names=[] if flattened_output else list(index_axis_names or []),
        index_df=index_df,
        metric_names=list(metrics),
        reduced=reduce != "none",
        baseline=None
        if baseline_values is None
        else (baseline_values if has_metric_axis else baseline_values[0]),
        has_metric_axis=has_metric_axis,
    )


# %%
# Defining patch setters for various shapes of activations
def layer_pos_patch_setter(corrupted_activation, index, clean_activation):
    """
    Applies the activation patch where index = [layer, pos]

    Implicitly assumes that the activation axis order is [batch, pos, ...], which is true of everything that is not an attention pattern shaped tensor.
    """
    assert len(index) == 2
    layer, pos = index
    corrupted_activation[:, pos, ...] = clean_activation[:, pos, ...]
    return corrupted_activation


def layer_pos_head_vector_patch_setter(
    corrupted_activation,
    index,
    clean_activation,
):
    """
    Applies the activation patch where index = [layer, pos, head_index]

    Implicitly assumes that the activation axis order is [batch, pos, head_index, ...], which is true of all attention head vector activations (q, k, v, z, result) but *not* of attention patterns.
    """
    assert len(index) == 3
    layer, pos, head_index = index
    corrupted_activation[:, pos, head_index] = clean_activation[:, pos, head_index]
    return corrupted_activation


def layer_head_vector_patch_setter(
    corrupted_activation,
    index,
    clean_activation,
):
    """
    Applies the activation patch where index = [layer,  head_index]

    Implicitly assumes that the activation axis order is [batch, pos, head_index, ...], which is true of all attention head vector activations (q, k, v, z, result) but *not* of attention patterns.
    """
    assert len(index) == 2
    layer, head_index = index
    corrupted_activation[:, :, head_index] = clean_activation[:, :, head_index]

    return corrupted_activation


def layer_head_pattern_patch_setter(
    corrupted_activation,
    index,
    clean_activation,
):
    """
    Applies the activation patch where index = [layer,  head_index]

    Implicitly assumes that the activation axis order is [batch, head_index, dest_pos, src_pos], which is true of attention scores and patterns.
    """
    assert len(index) == 2
    layer, head_index = index
    corrupted_activation[:, head_index, :, :] = clean_activation[:, head_index, :, :]

    return corrupted_activation


def layer_head_pos_pattern_patch_setter(
    corrupted_activation,
    index,
    clean_activation,
):
    """
    Applies the activation patch where index = [layer,  head_index, dest_pos]

    Implicitly assumes that the activation axis order is [batch, head_index, dest_pos, src_pos], which is true of attention scores and patterns.
    """
    assert len(index) == 3
    layer, head_index, dest_pos = index
    corrupted_activation[:, head_index, dest_pos, :] = clean_activation[:, head_index, dest_pos, :]

    return corrupted_activation


def layer_head_dest_src_pos_pattern_patch_setter(
    corrupted_activation,
    index,
    clean_activation,
):
    """
    Applies the activation patch where index = [layer,  head_index, dest_pos, src_pos]

    Implicitly assumes that the activation axis order is [batch, head_index, dest_pos, src_pos], which is true of attention scores and patterns.
    """
    assert len(index) == 4
    layer, head_index, dest_pos, src_pos = index
    corrupted_activation[:, head_index, dest_pos, src_pos] = clean_activation[
        :, head_index, dest_pos, src_pos
    ]

    return corrupted_activation


# %%
# Defining activation patching functions for a range of common activation patches.
get_act_patch_resid_pre = partial(
    generic_activation_patch,
    patch_setter=layer_pos_patch_setter,
    activation_name="resid_pre",
    index_axis_names=("layer", "pos"),
)
get_act_patch_resid_pre.__doc__ = """
    Function to get activation patching results for the residual stream (at the start of each block) (by position). Returns a tensor of shape [n_layers, pos]

    See generic_activation_patch for a more detailed explanation of activation patching 

    Args:
        model: The relevant model
        corrupted_tokens (torch.Tensor): The input tokens for the corrupted run. Has shape [batch, pos]
        clean_cache (ActivationCache): The cached activations from the clean run
        patching_metric: A function from the model's output logits to some metric (eg loss, logit diff, etc)

    Returns:
        patched_output (torch.Tensor): The tensor of the patching metric for each resid_pre patch. Has shape [n_layers, pos]
    """

get_act_patch_resid_mid = partial(
    generic_activation_patch,
    patch_setter=layer_pos_patch_setter,
    activation_name="resid_mid",
    index_axis_names=("layer", "pos"),
)
get_act_patch_resid_mid.__doc__ = """
    Function to get activation patching results for the residual stream (between the attn and MLP layer of each block) (by position). Returns a tensor of shape [n_layers, pos]

    See generic_activation_patch for a more detailed explanation of activation patching 

    Args:
        model: The relevant model
        corrupted_tokens (torch.Tensor): The input tokens for the corrupted run. Has shape [batch, pos]
        clean_cache (ActivationCache): The cached activations from the clean run
        patching_metric: A function from the model's output logits to some metric (eg loss, logit diff, etc)

    Returns:
        patched_output (torch.Tensor): The tensor of the patching metric for each patch. Has shape [n_layers, pos]
    """

get_act_patch_attn_out = partial(
    generic_activation_patch,
    patch_setter=layer_pos_patch_setter,
    activation_name="attn_out",
    index_axis_names=("layer", "pos"),
)
get_act_patch_attn_out.__doc__ = """
    Function to get activation patching results for the output of each Attention layer (by position). Returns a tensor of shape [n_layers, pos]

    See generic_activation_patch for a more detailed explanation of activation patching 

    Args:
        model: The relevant model
        corrupted_tokens (torch.Tensor): The input tokens for the corrupted run. Has shape [batch, pos]
        clean_cache (ActivationCache): The cached activations from the clean run
        patching_metric: A function from the model's output logits to some metric (eg loss, logit diff, etc)

    Returns:
        patched_output (torch.Tensor): The tensor of the patching metric for each patch. Has shape [n_layers, pos]
    """

get_act_patch_mlp_out = partial(
    generic_activation_patch,
    patch_setter=layer_pos_patch_setter,
    activation_name="mlp_out",
    index_axis_names=("layer", "pos"),
)
get_act_patch_mlp_out.__doc__ = """
    Function to get activation patching results for the output of each MLP layer (by position). Returns a tensor of shape [n_layers, pos]

    See generic_activation_patch for a more detailed explanation of activation patching 

    Args:
        model: The relevant model
        corrupted_tokens (torch.Tensor): The input tokens for the corrupted run. Has shape [batch, pos]
        clean_cache (ActivationCache): The cached activations from the clean run
        patching_metric: A function from the model's output logits to some metric (eg loss, logit diff, etc)

    Returns:
        patched_output (torch.Tensor): The tensor of the patching metric for each patch. Has shape [n_layers, pos]
    """
# %%
get_act_patch_attn_head_out_by_pos = partial(
    generic_activation_patch,
    patch_setter=layer_pos_head_vector_patch_setter,
    activation_name="z",
    index_axis_names=("layer", "pos", "head"),
)
get_act_patch_attn_head_out_by_pos.__doc__ = """
    Function to get activation patching results for the output of each Attention Head (by position). Returns a tensor of shape [n_layers, pos, n_heads]

    See generic_activation_patch for a more detailed explanation of activation patching 

    Args:
        model: The relevant model
        corrupted_tokens (torch.Tensor): The input tokens for the corrupted run. Has shape [batch, pos]
        clean_cache (ActivationCache): The cached activations from the clean run
        patching_metric: A function from the model's output logits to some metric (eg loss, logit diff, etc)

    Returns:
        patched_output (torch.Tensor): The tensor of the patching metric for each patch. Has shape [n_layers, pos, n_heads]
    """

get_act_patch_attn_head_q_by_pos = partial(
    generic_activation_patch,
    patch_setter=layer_pos_head_vector_patch_setter,
    activation_name="q",
    index_axis_names=("layer", "pos", "head"),
)
get_act_patch_attn_head_q_by_pos.__doc__ = """
    Function to get activation patching results for the queries of each Attention Head (by position). Returns a tensor of shape [n_layers, pos, n_heads]

    See generic_activation_patch for a more detailed explanation of activation patching 

    Args:
        model: The relevant model
        corrupted_tokens (torch.Tensor): The input tokens for the corrupted run. Has shape [batch, pos]
        clean_cache (ActivationCache): The cached activations from the clean run
        patching_metric: A function from the model's output logits to some metric (eg loss, logit diff, etc)

    Returns:
        patched_output (torch.Tensor): The tensor of the patching metric for each patch. Has shape [n_layers, pos, n_heads]
    """

get_act_patch_attn_head_k_by_pos = partial(
    generic_activation_patch,
    patch_setter=layer_pos_head_vector_patch_setter,
    activation_name="k",
    index_axis_names=("layer", "pos", "head"),
)
get_act_patch_attn_head_k_by_pos.__doc__ = """
    Function to get activation patching results for the keys of each Attention Head (by position). Returns a tensor of shape [n_layers, pos, n_heads] or [n_layers, pos, n_key_value_heads] if the model has a different number of key value heads than attention heads.

    See generic_activation_patch for a more detailed explanation of activation patching 

    Args:
        model: The relevant model
        corrupted_tokens (torch.Tensor): The input tokens for the corrupted run. Has shape [batch, pos]
        clean_cache (ActivationCache): The cached activations from the clean run
        patching_metric: A function from the model's output logits to some metric (eg loss, logit diff, etc)

    Returns:
        patched_output (torch.Tensor): The tensor of the patching metric for each patch. Has shape [n_layers, pos, n_heads] or [n_layers, pos, n_key_value_heads] if the model has a different number of key value heads than attention heads.
    """

get_act_patch_attn_head_v_by_pos = partial(
    generic_activation_patch,
    patch_setter=layer_pos_head_vector_patch_setter,
    activation_name="v",
    index_axis_names=("layer", "pos", "head"),
)
get_act_patch_attn_head_v_by_pos.__doc__ = """
    Function to get activation patching results for the values of each Attention Head (by position). Returns a tensor of shape [n_layers, pos, n_heads] or [n_layers, pos, n_key_value_heads] if the model has a different number of key value heads than attention heads.

    See generic_activation_patch for a more detailed explanation of activation patching 

    Args:
        model: The relevant model
        corrupted_tokens (torch.Tensor): The input tokens for the corrupted run. Has shape [batch, pos]
        clean_cache (ActivationCache): The cached activations from the clean run
        patching_metric: A function from the model's output logits to some metric (eg loss, logit diff, etc)

    Returns:
        patched_output (torch.Tensor): The tensor of the patching metric for each patch. Has shape [n_layers, pos, n_heads] or [n_layers, pos, n_key_value_heads] if the model has a different number of key value heads than attention heads.
    """
# %%
get_act_patch_attn_head_pattern_by_pos = partial(
    generic_activation_patch,
    patch_setter=layer_head_pos_pattern_patch_setter,
    activation_name="pattern",
    index_axis_names=("layer", "head_index", "dest_pos"),
)
get_act_patch_attn_head_pattern_by_pos.__doc__ = """
    Function to get activation patching results for the attention pattern of each Attention Head (by destination position). Returns a tensor of shape [n_layers, n_heads, dest_pos]

    See generic_activation_patch for a more detailed explanation of activation patching 

    Args:
        model: The relevant model
        corrupted_tokens (torch.Tensor): The input tokens for the corrupted run. Has shape [batch, pos]
        clean_cache (ActivationCache): The cached activations from the clean run
        patching_metric: A function from the model's output logits to some metric (eg loss, logit diff, etc)

    Returns:
        patched_output (torch.Tensor): The tensor of the patching metric for each patch. Has shape [n_layers, n_heads, dest_pos]
    """

get_act_patch_attn_head_pattern_dest_src_pos = partial(
    generic_activation_patch,
    patch_setter=layer_head_dest_src_pos_pattern_patch_setter,
    activation_name="pattern",
    index_axis_names=("layer", "head_index", "dest_pos", "src_pos"),
)
get_act_patch_attn_head_pattern_dest_src_pos.__doc__ = """
    Function to get activation patching results for each destination, source entry of the attention pattern for each Attention Head. Returns a tensor of shape [n_layers, n_heads, dest_pos, src_pos]

    See generic_activation_patch for a more detailed explanation of activation patching 

    Args:
        model: The relevant model
        corrupted_tokens (torch.Tensor): The input tokens for the corrupted run. Has shape [batch, pos]
        clean_cache (ActivationCache): The cached activations from the clean run
        patching_metric: A function from the model's output logits to some metric (eg loss, logit diff, etc)

    Returns:
        patched_output (torch.Tensor): The tensor of the patching metric for each patch. Has shape [n_layers, n_heads, dest_pos, src_pos]
    """

# %%
get_act_patch_attn_head_out_all_pos = partial(
    generic_activation_patch,
    patch_setter=layer_head_vector_patch_setter,
    activation_name="z",
    index_axis_names=("layer", "head"),
)
get_act_patch_attn_head_out_all_pos.__doc__ = """
    Function to get activation patching results for the outputs of each Attention Head (across all positions). Returns a tensor of shape [n_layers, n_heads]

    See generic_activation_patch for a more detailed explanation of activation patching 

    Args:
        model: The relevant model
        corrupted_tokens (torch.Tensor): The input tokens for the corrupted run. Has shape [batch, pos]
        clean_cache (ActivationCache): The cached activations from the clean run
        patching_metric: A function from the model's output logits to some metric (eg loss, logit diff, etc)

    Returns:
        patched_output (torch.Tensor): The tensor of the patching metric for each patch. Has shape [n_layers, n_heads]
    """

get_act_patch_attn_head_q_all_pos = partial(
    generic_activation_patch,
    patch_setter=layer_head_vector_patch_setter,
    activation_name="q",
    index_axis_names=("layer", "head"),
)
get_act_patch_attn_head_q_all_pos.__doc__ = """
    Function to get activation patching results for the queries of each Attention Head (across all positions). Returns a tensor of shape [n_layers, n_heads]

    See generic_activation_patch for a more detailed explanation of activation patching 

    Args:
        model: The relevant model
        corrupted_tokens (torch.Tensor): The input tokens for the corrupted run. Has shape [batch, pos]
        clean_cache (ActivationCache): The cached activations from the clean run
        patching_metric: A function from the model's output logits to some metric (eg loss, logit diff, etc)

    Returns:
        patched_output (torch.Tensor): The tensor of the patching metric for each patch. Has shape [n_layers, n_heads]
    """

get_act_patch_attn_head_k_all_pos = partial(
    generic_activation_patch,
    patch_setter=layer_head_vector_patch_setter,
    activation_name="k",
    index_axis_names=("layer", "head"),
)
get_act_patch_attn_head_k_all_pos.__doc__ = """
    Function to get activation patching results for the keys of each Attention Head (across all positions). Returns a tensor of shape [n_layers, n_heads] or [n_layers, n_key_value_heads] if the model has a different number of key value heads than attention heads.

    See generic_activation_patch for a more detailed explanation of activation patching 

    Args:
        model: The relevant model
        corrupted_tokens (torch.Tensor): The input tokens for the corrupted run. Has shape [batch, pos]
        clean_cache (ActivationCache): The cached activations from the clean run
        patching_metric: A function from the model's output logits to some metric (eg loss, logit diff, etc)

    Returns:
        patched_output (torch.Tensor): The tensor of the patching metric for each patch. Has shape [n_layers, n_heads] or [n_layers, n_key_value_heads] if the model has a different number of key value heads than attention heads.
    """

get_act_patch_attn_head_v_all_pos = partial(
    generic_activation_patch,
    patch_setter=layer_head_vector_patch_setter,
    activation_name="v",
    index_axis_names=("layer", "head"),
)
get_act_patch_attn_head_v_all_pos.__doc__ = """
    Function to get activation patching results for the values of each Attention Head (across all positions). Returns a tensor of shape [n_layers, n_heads] or [n_layers, n_key_value_heads] if the model has a different number of key value heads than attention heads.

    See generic_activation_patch for a more detailed explanation of activation patching 

    Args:
        model: The relevant model
        corrupted_tokens (torch.Tensor): The input tokens for the corrupted run. Has shape [batch, pos]
        clean_cache (ActivationCache): The cached activations from the clean run
        patching_metric: A function from the model's output logits to some metric (eg loss, logit diff, etc)

    Returns:
        patched_output (torch.Tensor): The tensor of the patching metric for each patch. Has shape [n_layers, n_heads] or [n_layers, n_key_value_heads] if the model has a different number of key value heads than attention heads.
    """

get_act_patch_attn_head_pattern_all_pos = partial(
    generic_activation_patch,
    patch_setter=layer_head_pattern_patch_setter,
    activation_name="pattern",
    index_axis_names=("layer", "head_index"),
)
get_act_patch_attn_head_pattern_all_pos.__doc__ = """
    Function to get activation patching results for the attention pattern of each Attention Head (across all positions). Returns a tensor of shape [n_layers, n_heads]

    See generic_activation_patch for a more detailed explanation of activation patching 

    Args:
        model: The relevant model
        corrupted_tokens (torch.Tensor): The input tokens for the corrupted run. Has shape [batch, pos]
        clean_cache (ActivationCache): The cached activations from the clean run
        patching_metric: A function from the model's output logits to some metric (eg loss, logit diff, etc)

    Returns:
        patched_output (torch.Tensor): The tensor of the patching metric for each patch. Has shape [n_layers, n_heads]
    """

# %%


_HEAD_TYPES = ("out", "q", "k", "v", "pattern")
_BLOCK_TYPES = ("resid_pre", "attn_out", "mlp_out")
_PER_EXAMPLE_NOTE = """
    Keyword-only ``reduce`` ("scalar" | "mean" | "none") and ``baseline`` are forwarded to
    generic_activation_patch; with reduce="none", baseline=True or a mapping of metrics the
    return is a PatchingResult carrying per-example values and statistics instead of a tensor.
"""
for _fn in (
    get_act_patch_resid_pre,
    get_act_patch_resid_mid,
    get_act_patch_attn_out,
    get_act_patch_mlp_out,
    get_act_patch_attn_head_out_by_pos,
    get_act_patch_attn_head_q_by_pos,
    get_act_patch_attn_head_k_by_pos,
    get_act_patch_attn_head_v_by_pos,
    get_act_patch_attn_head_pattern_by_pos,
    get_act_patch_attn_head_pattern_dest_src_pos,
    get_act_patch_attn_head_out_all_pos,
    get_act_patch_attn_head_q_all_pos,
    get_act_patch_attn_head_k_all_pos,
    get_act_patch_attn_head_v_all_pos,
    get_act_patch_attn_head_pattern_all_pos,
):
    _fn.__doc__ = (_fn.__doc__ or "") + _PER_EXAMPLE_NOTE


def _wrap_tensor_result(tensor: torch.Tensor, axis_names: Sequence[str]) -> PatchingResult:
    """Give a default tensor sweep the result shape so it can be stacked with typed ones."""
    return PatchingResult(
        values=tensor,
        axis_names=list(axis_names),
        index_df=make_df_from_ranges(list(tensor.shape), list(axis_names)),
        metric_names=["metric"],
        reduced=True,
    )


def _stack_patch_types(
    results: Sequence[Union[torch.Tensor, PatchingResult]],
    *,
    axis_order: Sequence[str],
    n_heads: int,
    type_names: Sequence[str],
) -> Union[torch.Tensor, PatchingResult]:
    """Stack per-activation sweeps on a leading ``patch_type`` axis.

    Tensors (the default return) are stacked as-is after the caller padded and
    rearranged them. ``PatchingResult`` values are first permuted to
    ``axis_order`` (``head_index`` / ``dest_pos`` renamed to ``head`` / ``pos``)
    and the head axis zero-padded to ``n_heads`` for GQA ``k``/``v`` sweeps,
    keeping the metric axis first and the batch axis last. ``index_df`` is
    rebuilt from the padded grid so it stays one row per cell of ``values``;
    padded GQA cells therefore appear as zero-valued rows.
    """
    if all(isinstance(r, torch.Tensor) for r in results):
        return torch.stack([r for r in results if isinstance(r, torch.Tensor)], dim=0)
    typed = [r for r in results if isinstance(r, PatchingResult)]
    if len(typed) != len(results):
        raise TypeError("cannot stack tensor and PatchingResult sweeps together")
    first = typed[0]
    for r in typed[1:]:
        if r.metric_names != first.metric_names or r.reduced != first.reduced:
            raise ValueError("every sweep must use the same metrics and reduce mode to be stacked")
    head_axis = list(axis_order).index("head") if "head" in axis_order else None
    values = []
    for r in typed:
        names = [_RESULT_AXIS_RENAMES.get(n, n) for n in r.axis_names]
        if sorted(names) != sorted(axis_order):
            raise ValueError(f"sweep axes {names} do not match {list(axis_order)}")
        lead = 1 if r.has_metric_axis else 0
        perm = list(range(lead)) + [lead + names.index(n) for n in axis_order]
        tail = list(range(lead + len(names), r.values.ndim))
        v = r.values.permute(*perm, *tail)
        pad = 0 if head_axis is None else n_heads - v.shape[lead + head_axis]
        if head_axis is not None and pad > 0:
            pad_shape = list(v.shape)
            pad_shape[lead + head_axis] = pad
            v = torch.cat([v, v.new_zeros(pad_shape)], dim=lead + head_axis)
        values.append(v)
    stacked = torch.stack(values, dim=1 if first.has_metric_axis else 0)
    lead = 1 if first.has_metric_axis else 0
    cell_shape = list(stacked.shape[lead : lead + 1 + len(axis_order)])
    return PatchingResult(
        values=stacked,
        axis_names=["patch_type", *axis_order],
        index_df=make_df_from_ranges(cell_shape, ["patch_type", *axis_order]),
        metric_names=list(first.metric_names),
        reduced=first.reduced,
        baseline=first.baseline,
        has_metric_axis=first.has_metric_axis,
        patch_type_names=list(type_names),
    )


def _run_sweeps(sweeps, axis_names_per_sweep, model, corrupted_tokens, clean_cache, metric, kwargs):
    """Run a helper's sweeps, computing a requested baseline once rather than once per sweep."""
    baseline = bool(kwargs.pop("baseline", False))
    results: List[Union[torch.Tensor, PatchingResult]] = []
    for i, (sweep, axis_names) in enumerate(zip(sweeps, axis_names_per_sweep)):
        out = sweep(
            model, corrupted_tokens, clean_cache, metric, baseline=baseline and i == 0, **kwargs
        )
        if baseline and isinstance(out, torch.Tensor):
            out = _wrap_tensor_result(out, axis_names)
        results.append(out)
    return results


def get_act_patch_attn_head_all_pos_every(
    model, corrupted_tokens, clean_cache, metric, **kwargs
) -> Union[Float[torch.Tensor, "patch_type layer head"], PatchingResult]:
    """Helper function to get activation patching results for every head (across all positions) for every act type (output, query, key, value, pattern). Wrapper around each's patching function, returns a stacked tensor of shape [5, n_layers, n_heads]

    Args:
        model: The relevant model
        corrupted_tokens (torch.Tensor): The input tokens for the corrupted run. Has shape [batch, pos]
        clean_cache (ActivationCache): The cached activations from the clean run
        metric: A function from the model's output logits to some metric (eg loss, logit diff, etc)
        **kwargs: ``reduce`` / ``baseline`` forwarded to each sweep; a ``PatchingResult`` is then returned with a leading ``patch_type`` axis (``patch_type_names`` = out, q, k, v, pattern)

    Returns:
        patched_output (torch.Tensor): The tensor of the patching metric for each patch. Has shape [5, n_layers, n_heads]
    """
    results = _run_sweeps(
        (
            get_act_patch_attn_head_out_all_pos,
            get_act_patch_attn_head_q_all_pos,
            get_act_patch_attn_head_k_all_pos,
            get_act_patch_attn_head_v_all_pos,
            get_act_patch_attn_head_pattern_all_pos,
        ),
        (("layer", "head"),) * 4 + (("layer", "head_index"),),
        model,
        corrupted_tokens,
        clean_cache,
        metric,
        kwargs,
    )
    if all(isinstance(r, torch.Tensor) for r in results):
        tensors = [r for r in results if isinstance(r, torch.Tensor)]
        n_heads = tensors[0].size(-1)
        # Reshape k and v to be compatible with the rest of the results in case of n_key_value_heads != n_heads
        for i in (2, 3):
            tensors[i] = torch.nn.functional.pad(tensors[i], (0, n_heads - tensors[i].size(-1)))
        return torch.stack(tensors, dim=0)
    return _stack_patch_types(
        results, axis_order=("layer", "head"), n_heads=model.cfg.n_heads, type_names=_HEAD_TYPES
    )


def get_act_patch_attn_head_by_pos_every(
    model, corrupted_tokens, clean_cache, metric, **kwargs
) -> Union[Float[torch.Tensor, "patch_type layer pos head"], PatchingResult]:
    """Helper function to get activation patching results for every head (by position) for every act type (output, query, key, value, pattern). Wrapper around each's patching function, returns a stacked tensor of shape [5, n_layers, pos, n_heads]

    Args:
        model: The relevant model
        corrupted_tokens (torch.Tensor): The input tokens for the corrupted run. Has shape [batch, pos]
        clean_cache (ActivationCache): The cached activations from the clean run
        metric: A function from the model's output logits to some metric (eg loss, logit diff, etc)
        **kwargs: ``reduce`` / ``baseline`` forwarded to each sweep; a ``PatchingResult`` is then returned with a leading ``patch_type`` axis (``patch_type_names`` = out, q, k, v, pattern)

    Returns:
        patched_output (torch.Tensor): The tensor of the patching metric for each patch. Has shape [5, n_layers, pos, n_heads]
    """
    results = _run_sweeps(
        (
            get_act_patch_attn_head_out_by_pos,
            get_act_patch_attn_head_q_by_pos,
            get_act_patch_attn_head_k_by_pos,
            get_act_patch_attn_head_v_by_pos,
            get_act_patch_attn_head_pattern_by_pos,
        ),
        (("layer", "pos", "head"),) * 4 + (("layer", "head_index", "dest_pos"),),
        model,
        corrupted_tokens,
        clean_cache,
        metric,
        kwargs,
    )
    if all(isinstance(r, torch.Tensor) for r in results):
        tensors = [r for r in results if isinstance(r, torch.Tensor)]
        n_heads = tensors[0].size(-1)
        # Reshape k and v to be compatible with the rest of the results in case of n_key_value_heads != n_heads
        for i in (2, 3):
            tensors[i] = torch.nn.functional.pad(tensors[i], (0, n_heads - tensors[i].size(-1)))
        # Reshape pattern to be compatible with the rest of the results
        tensors[4] = einops.rearrange(tensors[4], "batch head pos -> batch pos head")
        return torch.stack(tensors, dim=0)
    return _stack_patch_types(
        results,
        axis_order=("layer", "pos", "head"),
        n_heads=model.cfg.n_heads,
        type_names=_HEAD_TYPES,
    )


def get_act_patch_block_every(
    model, corrupted_tokens, clean_cache, metric, **kwargs
) -> Union[Float[torch.Tensor, "patch_type layer pos"], PatchingResult]:
    """Helper function to get activation patching results for the residual stream (at the start of each block), output of each Attention layer and output of each MLP layer. Wrapper around each's patching function, returns a stacked tensor of shape [3, n_layers, pos]

    Args:
        model: The relevant model
        corrupted_tokens (torch.Tensor): The input tokens for the corrupted run. Has shape [batch, pos]
        clean_cache (ActivationCache): The cached activations from the clean run
        metric: A function from the model's output logits to some metric (eg loss, logit diff, etc)
        **kwargs: ``reduce`` / ``baseline`` forwarded to each sweep; a ``PatchingResult`` is then returned with a leading ``patch_type`` axis (``patch_type_names`` = resid_pre, attn_out, mlp_out)

    Returns:
        patched_output (torch.Tensor): The tensor of the patching metric for each patch. Has shape [3, n_layers, pos]
    """
    results = _run_sweeps(
        (get_act_patch_resid_pre, get_act_patch_attn_out, get_act_patch_mlp_out),
        (("layer", "pos"),) * 3,
        model,
        corrupted_tokens,
        clean_cache,
        metric,
        kwargs,
    )
    if all(isinstance(r, torch.Tensor) for r in results):
        return torch.stack([r for r in results if isinstance(r, torch.Tensor)], dim=0)
    return _stack_patch_types(
        results, axis_order=("layer", "pos"), n_heads=0, type_names=_BLOCK_TYPES
    )
