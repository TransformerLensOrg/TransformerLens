"""Causal coordinate-swap benchmark for ``JacobianLens.coordinate_patch_hooks``.

Measures whether an anchored J-space coordinate edit installed live inside a forward pass
via ``coordinate_patch_hooks`` causes a directional change in model output, under three
controls: baseline-capability filtering (only intervene on prompts the model already
answers correctly), norm-matched random-atom controls (isolate "this concept mattered"
from "any edit of similar magnitude would have mattered"), and bootstrap uncertainty on
every reported rate.

This module is layered bottom-up and built out across several stages: the model-free prompt
corpus and rank/margin metric, baseline-capability filtering, norm-matched control-token
selection, and this stage's per-trial runner, which wires the first three together with real
``coordinate_patch_hooks`` calls against a live model.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import (
    Any,
    Container,
    Dict,
    Iterator,
    List,
    Literal,
    MutableMapping,
    Optional,
    Sequence,
    Tuple,
)

import torch

from transformer_lens.tools.analysis.jacobian_lens import DEFAULT_K, JacobianLens
from transformer_lens.tools.analysis.jacobian_lens_decomposition import (
    JSpaceDecomposition,
)


@dataclass(frozen=True)
class FunctionSpec:
    """A templated prompt function evaluated over a shared set of concepts.

    ``template`` takes a single ``{arg}`` placeholder, e.g. ``"The capital of {arg} is"``.
    ``answers`` maps each concept to its answer word under this function, e.g.
    ``{"France": "Paris"}``.
    """

    name: str
    template: str
    answers: Dict[str, str]


@dataclass(frozen=True)
class BenchmarkCorpus:
    """A named set of concepts and the prompt functions evaluated over them."""

    name: str
    concepts: Sequence[str]
    functions: Sequence[FunctionSpec]


@dataclass(frozen=True)
class PromptTrialSpec:
    """One (function, ordered source/target concept pair) prompt instance."""

    function: str
    source: str
    target: str
    prompt: str
    source_answer: str
    target_answer: str


def iter_prompt_trials(corpus: BenchmarkCorpus) -> Iterator[PromptTrialSpec]:
    """Yields one spec per (function, ordered source/target concept pair).

    Ordered pairs are every element of ``itertools.permutations(corpus.concepts, 2)``, the
    same cross product Jacobian_Lens_Demo.ipynb's country benchmark already uses, now as
    tested library code instead of a notebook cell.
    """
    for function in corpus.functions:
        for source, target in itertools.permutations(corpus.concepts, 2):
            yield PromptTrialSpec(
                function=function.name,
                source=source,
                target=target,
                prompt=function.template.format(arg=source),
                source_answer=function.answers[source],
                target_answer=function.answers[target],
            )


@dataclass(frozen=True)
class AnswerMetrics:
    """Rank/margin/tie metrics for one target token against one next-token logit vector."""

    top1_token_id: int
    target_rank: int
    target_is_top1: bool
    target_tied_for_top: bool
    target_logit_margin: float


def compute_answer_metrics(logits: torch.Tensor, target_token_id: int) -> AnswerMetrics:
    """Computes rank/margin/tie metrics for ``target_token_id`` against ``logits``.

    Ports Jacobian_Lens_Demo.ipynb's ``_target_metrics`` cell verbatim (arithmetic
    unchanged, only renamed and restructured into a dataclass), so results stay directly
    comparable with that notebook's already-reviewed success/rank definitions.
    """
    if logits.ndim != 1 or logits.numel() < 2:
        raise ValueError("expected one-dimensional next-token logits")
    if not 0 <= target_token_id < logits.shape[0]:
        raise ValueError("target token id is outside the vocabulary")
    if not torch.isfinite(logits).all():
        raise ValueError("logits must be finite")

    target_logit = logits[target_token_id]
    top_logit = logits.max()
    top1_token_id = int(logits.argmax().item())
    top_logit_tie_count = int((logits == top_logit).sum().item())
    competitors = torch.cat((logits[:target_token_id], logits[target_token_id + 1 :]))
    return AnswerMetrics(
        top1_token_id=top1_token_id,
        target_rank=int((logits > target_logit).sum().item()) + 1,
        target_is_top1=top1_token_id == target_token_id,
        target_tied_for_top=bool(target_logit == top_logit) and top_logit_tie_count > 1,
        target_logit_margin=float((target_logit - competitors.max()).item()),
    )


@dataclass(frozen=True)
class BaselineRecord:
    """A source prompt's own-answer metrics under the unperturbed baseline forward pass."""

    function: str
    source: str
    prompt: str
    metrics: AnswerMetrics


def filter_baseline_capable(
    baselines: Sequence[BaselineRecord],
) -> Tuple[List[BaselineRecord], List[BaselineRecord]]:
    """Splits baseline records into (capable, excluded) prompts.

    A prompt is baseline-capable when the model's own deterministic argmax already matches
    the source's answer (``metrics.target_is_top1``); only such prompts are eligible for
    later intervention trials, so an edit's effect is never measured against a prompt the
    unperturbed model already gets wrong. Order-preserving in both outputs; never mutates
    ``baselines``.
    """
    capable = [record for record in baselines if record.metrics.target_is_top1]
    excluded = [record for record in baselines if not record.metrics.target_is_top1]
    return capable, excluded


def select_norm_matched_control_token(
    dictionary: torch.Tensor,
    target_token_id: int,
    excluded_ids: Container[int],
    *,
    tolerance: float = 0.1,
    seed: int = 0,
) -> int:
    """Deterministically selects a norm-matched control token id.

    A candidate token id ``t`` qualifies when its ``dictionary`` atom norm is within
    ``tolerance`` (relative to the target token's atom norm) and ``t`` is neither
    ``target_token_id`` nor a member of ``excluded_ids``. One qualifying candidate is picked
    with a seeded ``torch.Generator`` so the same ``seed`` always yields the same control
    token. Raises ``ValueError`` if no candidate qualifies; the tolerance is never silently
    widened and selection never falls back to the globally nearest atom.
    """
    if dictionary.ndim != 2:
        raise ValueError(
            f"dictionary must be 2-D [num_atoms, d_model], got shape {tuple(dictionary.shape)}"
        )
    atom_norms = dictionary.float().norm(dim=1)
    target_norm = atom_norms[target_token_id]
    within_tolerance = (atom_norms - target_norm).abs() <= tolerance * target_norm
    candidates = [
        token_id
        for token_id in range(dictionary.shape[0])
        if token_id != target_token_id
        and token_id not in excluded_ids
        and bool(within_tolerance[token_id])
    ]
    if not candidates:
        raise ValueError(
            "no candidate token within relative tolerance "
            f"{tolerance} of target_token_id={target_token_id}'s atom norm "
            f"({float(target_norm):.4f})"
        )
    generator = torch.Generator(device=dictionary.device).manual_seed(seed)
    pick = int(torch.randint(len(candidates), (1,), generator=generator).item())
    return candidates[pick]


def _resolve_answer_token_id(model: Any, word: str) -> int:
    """Resolves a concept or answer word to the token id it maps to as a continuation.

    Prepends a leading space so the id matches how the word tokenizes when it follows
    other prompt text (e.g. " Paris", not "Paris"), the same convention the corpus
    templates themselves rely on.
    """
    return int(model.to_single_token(f" {word}"))


@dataclass(frozen=True)
class TrialResult:
    """One ``(function, source, target, layer)`` causal-swap trial's full record."""

    function: str
    source: str
    target: str
    layer: int
    status: Literal["ok", "skipped_source_inactive"]
    baseline: AnswerMetrics
    real_target_metrics: Optional[AnswerMetrics]
    control_token_id: Optional[int]
    control_target_metrics: Optional[AnswerMetrics]
    error: Optional[str]


def run_causal_swap_trial(
    lens: JacobianLens,
    model: Any,
    trial_spec: PromptTrialSpec,
    layer: int,
    *,
    decomposition_cache: Optional[MutableMapping[Tuple[int, int, int], JSpaceDecomposition]] = None,
    control_tolerance: float = 0.1,
    control_seed: int = 0,
    alpha: float = 1.0,
    k: int = DEFAULT_K,
) -> TrialResult:
    """Runs one causal-swap trial: baseline, then real and control coordinate-patch conditions.

    A baseline forward pass scores the prompt's own (unperturbed) source answer. A
    norm-matched control token is then selected from ``layer``'s lens-vector dictionary,
    excluding the source token, the real target token, and both prompts' answer tokens. The
    real and control conditions each install ``coordinate_patch_hooks`` at ``layer`` and
    position ``-1``, sharing one ``decomposition_cache`` so only the first of the two performs
    the vocabulary-scale decomposition; both are scored against the target's own answer token,
    so a real-vs-control gap isolates "swapping toward this concept mattered" from "any
    edit of this magnitude would have mattered."

    If either condition's ``coordinate_patch_hooks`` call raises ``ValueError`` (the source is
    not in the active support at this layer), the trial is recorded with
    ``status="skipped_source_inactive"`` and the caught message in ``error``, rather than
    propagating the exception -- ``coordinate_patch_hooks`` itself stays fail-fast; only this
    harness catches the failure. ``coordinate_patch_hooks``'s own ``UserWarning``s (both the
    per-call install notice and any solver-side conditioning warning) are not suppressed here
    and propagate to the caller unchanged.

    Args:
        lens: The fitted lens.
        model: The model to run trials against.
        trial_spec: The prompt, source/target concepts, and their answer words.
        layer: The single layer to patch at.
        decomposition_cache: Shared cache passed to both the real and control
            ``coordinate_patch_hooks`` calls. A fresh cache is used if omitted.
        control_tolerance: Relative tolerance for the norm-matched control token.
        control_seed: Seed for the control token's deterministic selection.
        alpha: Interpolation strength forwarded to ``coordinate_patch_hooks``.
        k: Sparse-solver upper bound forwarded to ``coordinate_patch_hooks``.

    Returns:
        The trial's :class:`TrialResult`.
    """
    if decomposition_cache is None:
        decomposition_cache = {}
    tokens = model.to_tokens(trial_spec.prompt)
    source_id = _resolve_answer_token_id(model, trial_spec.source)
    target_id = _resolve_answer_token_id(model, trial_spec.target)
    source_answer_id = _resolve_answer_token_id(model, trial_spec.source_answer)
    target_answer_id = _resolve_answer_token_id(model, trial_spec.target_answer)

    with torch.no_grad():
        baseline_logits = model(tokens)[0, -1].float()
    baseline_metrics = compute_answer_metrics(baseline_logits, source_answer_id)

    dictionary = lens.lens_vector_dictionary(model, layer)
    control_token_id = select_norm_matched_control_token(
        dictionary,
        target_id,
        excluded_ids={source_id, source_answer_id, target_answer_id},
        tolerance=control_tolerance,
        seed=control_seed,
    )

    def _condition_metrics(condition_target_id: int) -> AnswerMetrics:
        hooks = lens.coordinate_patch_hooks(
            model,
            source_id,
            condition_target_id,
            layers=[layer],
            positions=[-1],
            decomposition_cache=decomposition_cache,
            k=k,
            alpha=alpha,
        )
        with model.hooks(fwd_hooks=hooks), torch.no_grad():
            condition_logits = model(tokens)[0, -1].float()
        return compute_answer_metrics(condition_logits, target_answer_id)

    try:
        real_metrics = _condition_metrics(target_id)
        control_metrics = _condition_metrics(control_token_id)
    except ValueError as exc:
        return TrialResult(
            function=trial_spec.function,
            source=trial_spec.source,
            target=trial_spec.target,
            layer=layer,
            status="skipped_source_inactive",
            baseline=baseline_metrics,
            real_target_metrics=None,
            control_token_id=control_token_id,
            control_target_metrics=None,
            error=str(exc),
        )

    return TrialResult(
        function=trial_spec.function,
        source=trial_spec.source,
        target=trial_spec.target,
        layer=layer,
        status="ok",
        baseline=baseline_metrics,
        real_target_metrics=real_metrics,
        control_token_id=control_token_id,
        control_target_metrics=control_metrics,
        error=None,
    )


def run_causal_swap_benchmark(
    lens: JacobianLens,
    model: Any,
    corpus: BenchmarkCorpus,
    layers: Sequence[int],
    **trial_kwargs: Any,
) -> Tuple[List[TrialResult], List[BaselineRecord]]:
    """Runs the full causal-swap sweep: baseline filtering, then every surviving trial.

    Each ``(function, source)`` prompt's baseline is computed once -- it does not depend on
    ``layer`` -- and only prompts that survive :func:`filter_baseline_capable` proceed to
    :func:`run_causal_swap_trial`, once per remaining ``layer``. Each trial gets its own fresh
    ``decomposition_cache``: the cache key is ``(layer, batch_idx, position)``, which collides
    across different prompts run as independent single-example forward passes, so a cache may
    only be reused within one trial's real/control pair, never across trials.

    Args:
        lens: The fitted lens.
        model: The model to run trials against.
        corpus: The prompt corpus to sweep.
        layers: Layers to sweep as an independent trial dimension.
        **trial_kwargs: Forwarded to :func:`run_causal_swap_trial` (``control_tolerance``,
            ``control_seed``, ``alpha``, ``k``; ``decomposition_cache`` is not accepted here
            since each trial always uses its own).

    Returns:
        ``(trials, excluded_baselines)``.
    """
    all_specs = list(iter_prompt_trials(corpus))
    specs_by_prompt: Dict[Tuple[str, str], List[PromptTrialSpec]] = {}
    baselines: List[BaselineRecord] = []
    for spec in all_specs:
        prompt_key = (spec.function, spec.source)
        if prompt_key not in specs_by_prompt:
            specs_by_prompt[prompt_key] = []
            tokens = model.to_tokens(spec.prompt)
            with torch.no_grad():
                baseline_logits = model(tokens)[0, -1].float()
            source_answer_id = _resolve_answer_token_id(model, spec.source_answer)
            baselines.append(
                BaselineRecord(
                    function=spec.function,
                    source=spec.source,
                    prompt=spec.prompt,
                    metrics=compute_answer_metrics(baseline_logits, source_answer_id),
                )
            )
        specs_by_prompt[prompt_key].append(spec)

    capable, excluded = filter_baseline_capable(baselines)
    capable_prompt_keys = {(record.function, record.source) for record in capable}

    trials: List[TrialResult] = []
    for layer in layers:
        for prompt_key, specs in specs_by_prompt.items():
            if prompt_key not in capable_prompt_keys:
                continue
            for spec in specs:
                trials.append(
                    run_causal_swap_trial(
                        lens,
                        model,
                        spec,
                        layer,
                        **trial_kwargs,
                    )
                )
    return trials, excluded
