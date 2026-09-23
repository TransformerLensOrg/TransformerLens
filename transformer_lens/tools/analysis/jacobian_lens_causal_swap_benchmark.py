"""Causal coordinate-swap benchmark for ``JacobianLens.coordinate_patch_hooks``.

Measures whether an anchored J-space coordinate edit installed live inside a forward pass
via ``coordinate_patch_hooks`` causes a directional change in model output, under three
controls: baseline-capability filtering (only intervene on prompts the model already
answers correctly), displacement-matched random-atom controls (isolate "this concept
mattered" from "any edit of similar magnitude would have mattered"), and exact
Clopper-Pearson uncertainty on every reported rate. Each trial draws its control arm under
several seeds and records every draw, so the control arm's own spread is visible rather
than folded into a single number.

This module is layered bottom-up: the model-free prompt corpus and rank/margin metric,
baseline-capability filtering, displacement-matched control-token selection, a per-trial
runner that wires the first three together with real ``coordinate_patch_hooks`` calls
against a live model, and exact Clopper-Pearson confidence intervals plus a versioned,
fingerprinted JSON artifact schema. Running this module as a script (``python -m
transformer_lens.tools.analysis.jacobian_lens_causal_swap_benchmark``) generates the frozen
artifact consumed by ``demos/Jacobian_Lens_Coordinate_Patch_Benchmark_Demo.ipynb``; see
``main()`` below for the exact invocation.
"""

from __future__ import annotations

import hashlib
import itertools
import json
from dataclasses import asdict, dataclass
from pathlib import Path
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
    the source's answer (``metrics.target_is_top1``) and that answer is not tied for the
    maximum (``metrics.target_tied_for_top``); only such prompts are eligible for later
    intervention trials, so an edit's effect is never measured against a prompt the
    unperturbed model already gets wrong. A tied-for-top baseline is treated as not capable:
    ``argmax`` returns the lowest index on an exact tie, so admitting ties would let token-id
    order decide whether a prompt enters the trial set. Order-preserving in both outputs;
    never mutates ``baselines``.
    """
    capable = [
        record
        for record in baselines
        if record.metrics.target_is_top1 and not record.metrics.target_tied_for_top
    ]
    excluded = [
        record
        for record in baselines
        if not record.metrics.target_is_top1 or record.metrics.target_tied_for_top
    ]
    return capable, excluded


def _control_generator(seed: int) -> torch.Generator:
    """Builds the CPU generator used to pick a control token.

    The generator is pinned to CPU rather than the dictionary's device. ``torch.randint``
    infers its output device from the generator, so a device-local generator would make the
    same seed select a different control token per device and the frozen artifact would stop
    being reproducible. A CPU generator also avoids the device mismatch that raises on
    accelerators, where ``torch.randint`` allocates on CPU while a device-local generator
    expects its own device.
    """
    return torch.Generator(device="cpu").manual_seed(seed)


def select_displacement_matched_control_token(
    dictionary: torch.Tensor,
    source_token_id: int,
    target_token_id: int,
    excluded_ids: Container[int],
    active_support: Container[int],
    *,
    tolerance: float = 0.1,
    seed: int = 0,
) -> Optional[int]:
    """Deterministically selects a displacement-matched control token id.

    The real condition perturbs the activation by ``c_src * (a_target - a_source)``, so the
    edit's size is set by ``||a_target - a_source||``. Matching atom norms leaves that size
    free: two atoms of equal norm can sit at very different distances from ``a_source``, and
    the control edit then runs at a different magnitude from the real one. A candidate token
    id ``t`` therefore qualifies when
    ``abs(||a_t - a_source|| - ||a_target - a_source||) <= tolerance * ||a_target - a_source||``
    and ``t`` is not ``source_token_id``, not ``target_token_id``, not a member of
    ``excluded_ids``, and not a member of ``active_support`` -- an atom already carrying the
    source coordinate is not a clean control.

    One qualifying candidate is picked with a seeded CPU ``torch.Generator`` so the same
    ``seed`` always yields the same control token on every device. Returns ``None`` when no
    candidate qualifies; the tolerance is never silently widened and selection never falls
    back to the globally nearest atom. Returning ``None`` rather than raising lets the caller
    record an empty pool as its own skip status instead of aborting the sweep.
    """
    if dictionary.ndim != 2:
        raise ValueError(
            f"dictionary must be 2-D [num_atoms, d_model], got shape {tuple(dictionary.shape)}"
        )
    atoms = dictionary.float()
    source_atom = atoms[source_token_id]
    target_displacement = float((atoms[target_token_id] - source_atom).norm())
    displacements = (atoms - source_atom).norm(dim=1)
    within_tolerance = (displacements - target_displacement).abs() <= (
        tolerance * target_displacement
    )
    candidates = [
        token_id
        for token_id in range(dictionary.shape[0])
        if token_id != source_token_id
        and token_id != target_token_id
        and token_id not in excluded_ids
        and token_id not in active_support
        and bool(within_tolerance[token_id])
    ]
    if not candidates:
        return None
    generator = _control_generator(seed)
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
class ControlDraw:
    """One seeded control-arm draw: the token it selected and how that condition scored."""

    seed: int
    token_id: int
    metrics: AnswerMetrics


@dataclass(frozen=True)
class TrialResult:
    """One ``(function, source, target, layer)`` causal-swap trial's full record.

    ``control_token_id`` and ``control_target_metrics`` mirror the first entry of
    ``control_draws`` so a single-draw reader sees the same fields it always did. The full
    list carries the control arm's own spread across seeds, which one draw cannot measure.
    """

    function: str
    source: str
    target: str
    layer: int
    status: Literal["ok", "skipped_source_inactive", "skipped_no_control_token"]
    baseline: AnswerMetrics
    real_target_metrics: Optional[AnswerMetrics]
    control_token_id: Optional[int]
    control_target_metrics: Optional[AnswerMetrics]
    control_draws: List[ControlDraw]
    error: Optional[str]


def run_causal_swap_trial(
    lens: JacobianLens,
    model: Any,
    trial_spec: PromptTrialSpec,
    layer: int,
    *,
    decomposition_cache: Optional[MutableMapping[Tuple[int, int, int], JSpaceDecomposition]] = None,
    control_tolerance: float = 0.1,
    control_seeds: Sequence[int] = (0,),
    alpha: float = 1.0,
    k: int = DEFAULT_K,
) -> TrialResult:
    """Runs one causal-swap trial: baseline, then real and control coordinate-patch conditions.

    A baseline forward pass scores the prompt's own (unperturbed) source answer. A
    displacement-matched control token is then selected from ``layer``'s lens-vector
    dictionary, excluding the source token, the real target token, both prompts' answer
    tokens, and every atom in the layer's active support. The real and control conditions
    each install ``coordinate_patch_hooks`` at ``layer`` and position ``-1``, sharing one
    ``decomposition_cache`` so only the first of the two performs the vocabulary-scale
    decomposition; both are scored against the target's own answer token, so a real-vs-control
    gap isolates "swapping toward this concept mattered" from "any edit of this magnitude
    would have mattered."

    The control arm is drawn once per entry in ``control_seeds``. A single draw leaves the
    control arm's own variance unmeasured, so a reader cannot tell a real-vs-control gap from
    draw variation; the per-seed results are carried on ``control_draws``. Every draw reuses
    the same ``decomposition_cache``, so only the first pays the vocabulary-scale solve.

    Two conditions produce a skip rather than a result. When the source is not in the layer's
    active support, the trial is recorded with ``status="skipped_source_inactive"``. When no
    displacement-matched control token survives the exclusions, the trial is recorded with
    ``status="skipped_no_control_token"``. Both carry a diagnostic message in ``error``.

    The active-support precondition is checked up front, against the same decomposition the
    hooks consume, rather than inferred from a caught exception. ``coordinate_patch_hooks``
    raises plain ``ValueError`` for several unrelated protocol faults (a same-id source and
    target, an out-of-vocabulary answer, non-finite logits) and defines no narrower subclass
    to discriminate on, so catching ``ValueError`` would file those as skips and drop them
    from both denominators. Every other error therefore propagates unchanged.
    ``coordinate_patch_hooks``'s own ``UserWarning``s (both the per-call install notice and
    any solver-side conditioning warning) are not suppressed here and propagate to the caller
    unchanged.

    Args:
        lens: The fitted lens.
        model: The model to run trials against.
        trial_spec: The prompt, source/target concepts, and their answer words.
        layer: The single layer to patch at.
        decomposition_cache: Shared cache passed to both the real and control
            ``coordinate_patch_hooks`` calls. A fresh cache is used if omitted.
        control_tolerance: Relative tolerance for the displacement-matched control token.
        control_seeds: Seeds to draw the control arm under. Each seed selects its own
            control token and contributes one :class:`ControlDraw`; the first seed's draw
            is mirrored onto ``control_token_id`` and ``control_target_metrics``.
        alpha: Interpolation strength forwarded to ``coordinate_patch_hooks``.
        k: Sparse-solver upper bound forwarded to ``coordinate_patch_hooks``.

    Returns:
        The trial's :class:`TrialResult`.

    Raises:
        ValueError: If the source and target concepts resolve to the same token id, which
            would make the coordinate patch a silent no-op, or if ``control_seeds`` is
            empty.
    """
    if decomposition_cache is None:
        decomposition_cache = {}
    if not control_seeds:
        raise ValueError("control_seeds must contain at least one seed")
    tokens = model.to_tokens(trial_spec.prompt)
    source_id = _resolve_answer_token_id(model, trial_spec.source)
    target_id = _resolve_answer_token_id(model, trial_spec.target)
    source_answer_id = _resolve_answer_token_id(model, trial_spec.source_answer)
    target_answer_id = _resolve_answer_token_id(model, trial_spec.target_answer)
    if source_id == target_id:
        raise ValueError(
            f"source and target resolve to the same token id {source_id}; a coordinate "
            "patch would be a silent no-op"
        )

    with torch.no_grad():
        baseline_logits = model(tokens)[0, -1].float()
    baseline_metrics = compute_answer_metrics(baseline_logits, source_answer_id)

    dictionary = lens.lens_vector_dictionary(model, layer)
    decomposition = lens.decompose(model, trial_spec.prompt, layer=layer, position=-1, k=k)
    # Seed the shared cache under the hook's own key so the first firing is a cache hit and
    # the up-front solve is not repeated. The hook normalizes -1 against the activation's
    # sequence length, which equals the tokenized prompt length for a single-example pass.
    decomposition_cache[(layer, 0, tokens.shape[1] - 1)] = decomposition
    if source_id not in decomposition.support.tolist():
        return TrialResult(
            function=trial_spec.function,
            source=trial_spec.source,
            target=trial_spec.target,
            layer=layer,
            status="skipped_source_inactive",
            baseline=baseline_metrics,
            real_target_metrics=None,
            control_token_id=None,
            control_target_metrics=None,
            control_draws=[],
            error=(
                f"source token id {source_id} is not in layer {layer}'s active support "
                f"for this prompt"
            ),
        )
    control_token_ids: List[int] = []
    for seed in control_seeds:
        token_id = select_displacement_matched_control_token(
            dictionary,
            source_id,
            target_id,
            excluded_ids={source_answer_id, target_answer_id},
            active_support=decomposition.support,
            tolerance=control_tolerance,
            seed=seed,
        )
        if token_id is None:
            return TrialResult(
                function=trial_spec.function,
                source=trial_spec.source,
                target=trial_spec.target,
                layer=layer,
                status="skipped_no_control_token",
                baseline=baseline_metrics,
                real_target_metrics=None,
                control_token_id=None,
                control_target_metrics=None,
                control_draws=[],
                error=(
                    "no displacement-matched control token within relative tolerance "
                    f"{control_tolerance} of ||a_target - a_source|| for seed {seed}"
                ),
            )
        control_token_ids.append(token_id)

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

    real_metrics = _condition_metrics(target_id)
    control_draws = [
        ControlDraw(
            seed=seed,
            token_id=token_id,
            metrics=_condition_metrics(token_id),
        )
        for seed, token_id in zip(control_seeds, control_token_ids)
    ]

    return TrialResult(
        function=trial_spec.function,
        source=trial_spec.source,
        target=trial_spec.target,
        layer=layer,
        status="ok",
        baseline=baseline_metrics,
        real_target_metrics=real_metrics,
        control_token_id=control_draws[0].token_id,
        control_target_metrics=control_draws[0].metrics,
        control_draws=control_draws,
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
            ``control_seeds``, ``alpha``, ``k``; ``decomposition_cache`` is not accepted here
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


@dataclass(frozen=True)
class SuccessRateInterval:
    """An exact Clopper-Pearson confidence interval around a success rate.

    ``n_trials`` and ``n_successes`` are the counts the interval was computed from, so a
    reader can see the denominator behind the bounds rather than inferring it.
    """

    point_estimate: float
    ci_low: float
    ci_high: float
    n_trials: int
    n_successes: int
    confidence: float


# Bisection depth for the Clopper-Pearson bounds. Each step halves the bracket, so 200 steps
# resolve the bound far below the precision any reported rate needs.
_CLOPPER_PEARSON_STEPS = 200


def _binomial_cdf(k: int, n: int, p: float) -> float:
    """Exact ``P(X <= k)`` for ``X ~ Binomial(n, p)``.

    Accumulates the pmf by recurrence (``term_{i} = term_{i-1} * (n - i + 1) / i * p / (1 - p)``)
    rather than forming ``comb(n, i)`` and ``p ** i`` separately, which overflow for the
    trial counts this benchmark reports.
    """
    if k < 0:
        return 0.0
    if k >= n:
        return 1.0
    if p <= 0.0:
        return 1.0
    if p >= 1.0:
        return 0.0
    term = (1.0 - p) ** n
    total = term
    odds = p / (1.0 - p)
    for i in range(1, k + 1):
        term *= (n - i + 1) / i * odds
        total += term
    return total


def _clopper_pearson_bound(n_successes: int, n_trials: int, tail: float, *, upper: bool) -> float:
    """Solves one Clopper-Pearson bound by bisecting the exact binomial CDF.

    The lower bound solves ``P(X >= k) = tail`` and the upper bound solves ``P(X <= k) =
    tail``. ``P(X >= k)`` rises with ``p`` while ``P(X <= k)`` falls, so the two bounds
    bracket from opposite sides; reversing that collapses every interval to ``[0, 1]``.
    """
    if upper and n_successes == n_trials:
        return 1.0
    if not upper and n_successes == 0:
        return 0.0
    low, high = 0.0, 1.0
    for _ in range(_CLOPPER_PEARSON_STEPS):
        midpoint = (low + high) / 2.0
        if upper:
            # P(X <= k) decreases in p, so a probability above the tail puts the bound below
            # the midpoint.
            if _binomial_cdf(n_successes, n_trials, midpoint) > tail:
                low = midpoint
            else:
                high = midpoint
        else:
            # P(X >= k) increases in p, so a probability above the tail puts the bound above
            # the midpoint.
            if 1.0 - _binomial_cdf(n_successes - 1, n_trials, midpoint) > tail:
                high = midpoint
            else:
                low = midpoint
    return (low + high) / 2.0


def success_rate_ci(
    successes: Sequence[bool],
    *,
    confidence: float = 0.95,
) -> SuccessRateInterval:
    """Computes an exact Clopper-Pearson confidence interval for a success rate.

    A percentile bootstrap cannot express uncertainty about an all-failure or all-success
    sample: every resample of an all-zero vector is all zeros, so the interval collapses to
    ``[0, 0]`` and reports the same certainty for one trial as for a thousand. The exact
    interval instead widens as the trial count shrinks, giving ``[0.0, 0.459]`` for zero
    successes out of six.

    Deterministic by construction: there is no resampling and no seed.

    Raises:
        ValueError: If ``successes`` is empty.
    """
    if len(successes) == 0:
        raise ValueError("successes must be a non-empty sequence")
    n_trials = len(successes)
    n_successes = sum(1 for success in successes if success)
    tail = (1.0 - confidence) / 2.0
    return SuccessRateInterval(
        point_estimate=n_successes / n_trials,
        ci_low=_clopper_pearson_bound(n_successes, n_trials, tail, upper=False),
        ci_high=_clopper_pearson_bound(n_successes, n_trials, tail, upper=True),
        n_trials=n_trials,
        n_successes=n_successes,
        confidence=confidence,
    )


_REQUIRED_MANIFEST_FIELDS = (
    "model_id",
    "model_revision",
    "lens_repo",
    "lens_file",
    "lens_revision",
    "corpus_name",
    "layers",
    "alpha",
    "k",
    "control_tolerance",
    "control_seeds",
    "success_definition",
    "baseline_definition",
    "rank_definition",
)


def build_protocol_manifest(**fields: Any) -> Dict[str, Any]:
    """Assembles a protocol manifest, requiring the benchmark's fixed field set.

    Requires at least :data:`_REQUIRED_MANIFEST_FIELDS`, the same key set (and, where they
    overlap, the same string values) as ``Jacobian_Lens_Demo.ipynb``'s existing
    ``protocol_manifest`` cell.

    Raises:
        ValueError: If any required field is missing, naming the missing field(s).
    """
    missing = [name for name in _REQUIRED_MANIFEST_FIELDS if name not in fields]
    if missing:
        raise ValueError(f"protocol manifest is missing required field(s): {', '.join(missing)}")
    return dict(fields)


def fingerprint_manifest(manifest: Dict[str, Any]) -> str:
    """Fingerprints a protocol manifest with the recipe ``Jacobian_Lens_Demo.ipynb`` uses."""
    return hashlib.sha256(
        json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


SCHEMA_VERSION = 1


def serialize_artifact(
    manifest: Dict[str, Any],
    trials: Sequence[TrialResult],
    excluded_baselines: Sequence[BaselineRecord],
    real_ci: SuccessRateInterval,
    control_ci: SuccessRateInterval,
) -> Dict[str, Any]:
    """Assembles a JSON-serializable artifact dict from a benchmark run's results.

    ``n_independent_prompts`` records how many distinct ``(function, source)`` prompts the
    pooled rate rests on. Pooling trials as independent draws overstates the evidence when
    several trials share one prompt, so the count travels with the rate.
    """
    return {
        "schema_version": SCHEMA_VERSION,
        "protocol_manifest": manifest,
        "protocol_fingerprint": fingerprint_manifest(manifest),
        "trials": [asdict(trial) for trial in trials],
        "excluded_baselines": [asdict(record) for record in excluded_baselines],
        "real_success_ci": asdict(real_ci),
        "control_success_ci": asdict(control_ci),
        "n_independent_prompts": len(
            {(trial.function, trial.source) for trial in trials if trial.status == "ok"}
        ),
    }


_REQUIRED_ARTIFACT_FIELDS = (
    "schema_version",
    "protocol_manifest",
    "protocol_fingerprint",
    "trials",
    "excluded_baselines",
    "real_success_ci",
    "control_success_ci",
    "n_independent_prompts",
)


def load_artifact(path: Path) -> Dict[str, Any]:
    """Reads and validates a frozen artifact produced by :func:`serialize_artifact`.

    Validates that every required top-level key is present, that ``schema_version``
    matches :data:`SCHEMA_VERSION`, and that ``protocol_fingerprint`` matches a fresh
    :func:`fingerprint_manifest` of the loaded ``protocol_manifest`` -- catching a
    hand-edited or corrupted artifact rather than trusting the stored fingerprint blindly.

    Raises:
        ValueError: Naming the missing or mismatched field.
    """
    artifact = json.loads(Path(path).read_text())
    missing = [name for name in _REQUIRED_ARTIFACT_FIELDS if name not in artifact]
    if missing:
        raise ValueError(f"artifact is missing required field(s): {', '.join(missing)}")
    if artifact["schema_version"] != SCHEMA_VERSION:
        raise ValueError(
            f"artifact schema_version {artifact['schema_version']!r} does not match the "
            f"expected {SCHEMA_VERSION!r}"
        )
    expected_fingerprint = fingerprint_manifest(artifact["protocol_manifest"])
    if artifact["protocol_fingerprint"] != expected_fingerprint:
        raise ValueError(
            "artifact protocol_fingerprint does not match a freshly computed fingerprint of "
            "its protocol_manifest (the manifest may have been hand-edited or corrupted)"
        )
    return artifact


_COUNTRY_CORPUS = BenchmarkCorpus(
    name="countries",
    concepts=("France", "Canada", "China", "Egypt"),
    functions=(
        FunctionSpec(
            name="capital",
            template="The capital of {arg} is the city of",
            answers={"France": "Paris", "Canada": "Ottawa", "China": "Beijing", "Egypt": "Cairo"},
        ),
        FunctionSpec(
            name="language",
            template="Most people in {arg} speak",
            answers={
                "France": "French",
                "Canada": "English",
                "China": "Chinese",
                "Egypt": "Arabic",
            },
        ),
        FunctionSpec(
            name="continent",
            template="{arg} is a country on the continent of",
            answers={"France": "Europe", "Canada": "North", "China": "Asia", "Egypt": "Africa"},
        ),
        FunctionSpec(
            name="currency",
            template="The single-word name for the currency now used in {arg} is the",
            answers={"France": "Euro", "Canada": "Dollar", "China": "Yuan", "Egypt": "Pound"},
        ),
    ),
)

_GPT2_LENS_REPO = "neuronpedia/jacobian-lens"
_GPT2_LENS_FILE = "gpt2-small/jlens/Salesforce-wikitext/gpt2_jacobian_lens.pt"
_GPT2_LENS_REVISION = "a4114d7752d11eb546e6cf372213d7e75526d3a1"
_DEFAULT_CONTROL_SEEDS = (0, 1, 2, 3, 4)
_DEFAULT_ARTIFACT_PATH = (
    Path(__file__).resolve().parents[3]
    / "demos"
    / "data"
    / "jacobian_lens_causal_swap_benchmark_gpt2.json"
)


def _parse_seed_list(value: str) -> Tuple[int, ...]:
    """Parses a comma-separated seed list for ``--control-seeds``."""
    seeds = tuple(int(part) for part in value.split(",") if part.strip())
    if not seeds:
        raise ValueError("at least one control seed is required")
    return seeds


def main(argv: Optional[Sequence[str]] = None) -> None:
    """Generates the frozen GPT-2 causal-swap benchmark artifact.

    Reproduce with (no ``HF_TOKEN`` needed -- GPT-2 is not gated)::

        uv run python -m transformer_lens.tools.analysis.jacobian_lens_causal_swap_benchmark

    Loads the published GPT-2-small lens (the same artifact
    ``tests/integration/test_jacobian_lens.py`` uses), runs the reused country corpus over
    every one of the lens's fitted source layers, and writes the versioned, fingerprinted JSON
    artifact consumed by ``demos/Jacobian_Lens_Coordinate_Patch_Benchmark_Demo.ipynb``.
    """
    import argparse

    import torch

    from transformer_lens.model_bridge import TransformerBridge

    parser = argparse.ArgumentParser(description=main.__doc__)
    parser.add_argument("--output", type=Path, default=_DEFAULT_ARTIFACT_PATH)
    parser.add_argument("--control-tolerance", type=float, default=0.1)
    parser.add_argument(
        "--control-seeds",
        type=_parse_seed_list,
        default=_DEFAULT_CONTROL_SEEDS,
        help="comma-separated seeds to draw the control arm under (default: %(default)s)",
    )
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--k", type=int, default=DEFAULT_K)
    args = parser.parse_args(argv)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = TransformerBridge.boot_transformers("gpt2", dtype=torch.float32, device=device)
    lens = JacobianLens.from_pretrained(
        _GPT2_LENS_REPO,
        filename=_GPT2_LENS_FILE,
        revision=_GPT2_LENS_REVISION,
        model=model,
    )
    layers = list(lens.source_layers)

    trials, excluded = run_causal_swap_benchmark(
        lens,
        model,
        _COUNTRY_CORPUS,
        layers,
        control_tolerance=args.control_tolerance,
        control_seeds=args.control_seeds,
        alpha=args.alpha,
        k=args.k,
    )
    ok_trials = [trial for trial in trials if trial.status == "ok"]
    if not ok_trials:
        raise RuntimeError(
            "no trial survived baseline filtering, the active-support check, and "
            "control-token selection"
        )

    real_successes: List[bool] = []
    control_successes: List[bool] = []
    for trial in ok_trials:
        assert trial.real_target_metrics is not None
        assert trial.control_target_metrics is not None
        real_successes.append(trial.real_target_metrics.target_is_top1)
        control_successes.append(trial.control_target_metrics.target_is_top1)

    manifest = build_protocol_manifest(
        model_id="gpt2",
        model_revision="n/a",
        lens_repo=_GPT2_LENS_REPO,
        lens_file=_GPT2_LENS_FILE,
        lens_revision=_GPT2_LENS_REVISION,
        corpus_name=_COUNTRY_CORPUS.name,
        layers=layers,
        alpha=args.alpha,
        k=args.k,
        control_tolerance=args.control_tolerance,
        control_seeds=list(args.control_seeds),
        success_definition="target token id equals deterministic argmax token id",
        baseline_definition="source answer token id equals deterministic argmax token id",
        rank_definition="1 + count(logits strictly greater than target logit)",
    )
    real_ci = success_rate_ci(real_successes)
    control_ci = success_rate_ci(control_successes)
    artifact = serialize_artifact(manifest, trials, excluded, real_ci, control_ci)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n")
    inactive = sum(1 for trial in trials if trial.status == "skipped_source_inactive")
    no_control = sum(1 for trial in trials if trial.status == "skipped_no_control_token")
    n_prompts = artifact["n_independent_prompts"]
    print(
        f"wrote {len(trials)} trials ({len(ok_trials)} ok over {n_prompts} independent "
        f"prompt(s), {inactive} source-inactive, {no_control} no-control-token, "
        f"{len(excluded)} excluded prompts) to {args.output}"
    )


if __name__ == "__main__":
    main()
