"""Causal-swap benchmark for ``JacobianLens.coordinate_patch_hooks`` on GPT-2-small.

Regenerates ``demos/data/jacobian_lens_causal_swap_benchmark_gpt2.json``, which
``demos/Jacobian_Lens_Coordinate_Patch_Benchmark_Demo.ipynb`` loads without running a model.
Each trial edits one layer's J-space coordinates toward a target concept and checks that the
target's answer becomes top-1. Only prompts the model already answers are used, each edit is
paired with displacement-matched random-atom controls under several seeds, and every rate
carries an exact Clopper-Pearson interval. Lens and corpus revisions are pinned for reproduction.

Run:  uv run python scripts/jacobian_lens_causal_swap_benchmark.py
      (no HF_TOKEN needed, GPT-2 is not gated; ``--output`` redirects the artifact)
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import (
    Any,
    Container,
    Dict,
    Iterator,
    List,
    Literal,
    Mapping,
    MutableMapping,
    Optional,
    Sequence,
    Tuple,
)

import torch

from transformer_lens.model_bridge import TransformerBridge
from transformer_lens.tools.analysis._model_state import require_eval_mode
from transformer_lens.tools.analysis.jacobian_lens import DEFAULT_K, JacobianLens
from transformer_lens.tools.analysis.jacobian_lens_decomposition import (
    JSpaceDecomposition,
)
from transformer_lens.utilities.statistics import SuccessRateInterval, success_rate_ci

SCHEMA_VERSION = 2

_GPT2_LENS_REPO = "neuronpedia/jacobian-lens"
_GPT2_LENS_FILE = "gpt2-small/jlens/Salesforce-wikitext/gpt2_jacobian_lens.pt"
_GPT2_LENS_REVISION = "a4114d7752d11eb546e6cf372213d7e75526d3a1"
# Matches the triple Jacobian_Lens_Demo.ipynb pins for its country config.
_CORPUS_REPO = "anthropics/jacobian-lens"
_CORPUS_PATH = "data/experiments/flexible-generalization.json"
_CORPUS_REVISION = "581d398613e5602a5af361e1c34d3a92ea82ba8e"
_DEFAULT_CONTROL_SEEDS = (0, 1, 2, 3, 4)
_DEFAULT_ARTIFACT_PATH = (
    Path(__file__).resolve().parents[1]
    / "demos"
    / "data"
    / "jacobian_lens_causal_swap_benchmark_gpt2.json"
)


@dataclass(frozen=True)
class FunctionSpec:
    """A prompt template with one ``{arg}`` placeholder and each concept's answer word."""

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


def iter_prompt_trials(corpus: BenchmarkCorpus) -> Iterator[PromptTrialSpec]:
    """Yields one spec per (function, ordered source/target concept pair)."""
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


def corpus_definition(corpus: BenchmarkCorpus) -> Dict[str, Any]:
    """Serializes a corpus for the manifest, so prompts and answers enter the fingerprint."""
    return {
        "name": corpus.name,
        "concepts": list(corpus.concepts),
        "functions": [
            {
                "name": function.name,
                "template": function.template,
                "answers": dict(function.answers),
            }
            for function in corpus.functions
        ],
    }


@dataclass(frozen=True)
class AnswerMetrics:
    """Rank/margin/tie metrics for one target token against one next-token logit vector."""

    top1_token_id: int
    target_rank: int
    target_is_top1: bool
    target_tied_for_top: bool
    target_logit_margin: float


def compute_answer_metrics(logits: torch.Tensor, target_token_id: int) -> AnswerMetrics:
    """Computes rank/margin/tie metrics for ``target_token_id`` against 1-D ``logits``."""
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
    """A source prompt's own-answer metrics under the unperturbed forward pass."""

    function: str
    source: str
    prompt: str
    metrics: AnswerMetrics


def filter_baseline_capable(
    baselines: Sequence[BaselineRecord],
) -> Tuple[List[BaselineRecord], List[BaselineRecord]]:
    """Splits baseline records into (capable, excluded), preserving order."""
    # A tied-for-top answer is excluded: argmax returns the lowest index on an exact tie, so
    # admitting ties would let token-id order decide whether a prompt enters the trial set.
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
    """Builds a CPU generator so the same seed picks the same control token on every device."""
    # torch.randint allocates on CPU, so a device-local generator would also raise on accelerators.
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
    """Picks a seeded control token whose atom sits as far from the source as the target's.

    Returns ``None`` when no candidate qualifies; the tolerance is never widened.
    """
    # Match ``||a_t - a_source||``, not ``||a_t||``: the real edit is ``c_src * (a_target -
    # a_source)``, so a control at a different distance from the source runs at another magnitude.
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
    # Atoms in the active support carry the source coordinate, so they are not clean controls.
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
    """Resolves a word to its token id as a continuation (leading space)."""
    return int(model.to_single_token(f" {word}"))


@dataclass(frozen=True)
class ControlDraw:
    """One seeded control-arm draw: the token it selected and how that condition scored."""

    seed: int
    token_id: int
    metrics: AnswerMetrics


@dataclass(frozen=True)
class TrialResult:
    """One ``(function, source, target, layer)`` trial.

    ``control_token_id`` and ``control_target_metrics`` mirror the first entry of
    ``control_draws``; the full list carries the control arm's spread across seeds.
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
    """Runs one trial: baseline, then the real and per-seed control coordinate patches."""
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
    require_eval_mode(model, operation="run_causal_swap_trial()")

    with torch.no_grad():
        baseline_logits = model(tokens)[0, -1].float()
    baseline_metrics = compute_answer_metrics(baseline_logits, source_answer_id)

    dictionary = lens.lens_vector_dictionary(model, layer)
    decomposition = lens.decompose(model, trial_spec.prompt, layer=layer, position=-1, k=k)
    # Pre-seed the hook's key (it normalizes -1 to the prompt length) so the vocabulary-scale
    # solve is not repeated on its first firing.
    decomposition_cache[(layer, 0, tokens.shape[1] - 1)] = decomposition
    # Checked up front: ``coordinate_patch_hooks`` raises ValueError for unrelated faults too,
    # and filing those as skips would drop them from both denominators.
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
            active_support=decomposition.support.tolist(),
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
    """Filters prompts by baseline capability, then runs every surviving trial per layer."""
    all_specs = list(iter_prompt_trials(corpus))
    require_eval_mode(model, operation="run_causal_swap_benchmark()")
    specs_by_prompt: Dict[Tuple[str, str], List[PromptTrialSpec]] = {}
    baselines: List[BaselineRecord] = []
    # Baselines are layer-independent, so each prompt's is computed once.
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
                # Fresh cache per trial: its key is (layer, batch_idx, position), which
                # collides across different prompts run as single-example passes.
                trials.append(run_causal_swap_trial(lens, model, spec, layer, **trial_kwargs))
    return trials, excluded


_REQUIRED_MANIFEST_FIELDS = (
    "model_id",
    "model_revision",
    "lens_repo",
    "lens_file",
    "lens_revision",
    "corpus_name",
    "corpus",
    "corpus_repo",
    "corpus_path",
    "corpus_revision",
    "layers_swept",
    "alpha",
    "k",
    "control_tolerance",
    "control_seeds",
    "success_definition",
    "baseline_definition",
    "rank_definition",
)


def build_protocol_manifest(**fields: Any) -> Dict[str, Any]:
    """Assembles a protocol manifest, requiring the benchmark's fixed field set."""
    missing = [name for name in _REQUIRED_MANIFEST_FIELDS if name not in fields]
    if missing:
        raise ValueError(f"protocol manifest is missing required field(s): {', '.join(missing)}")
    return dict(fields)


def fingerprint_manifest(manifest: Dict[str, Any]) -> str:
    """SHA-256 over the canonical JSON of a protocol manifest."""
    return hashlib.sha256(
        json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def results_fingerprint(artifact: Mapping[str, Any]) -> str:
    """SHA-256 over the result blocks, which ``protocol_fingerprint`` does not cover."""
    return hashlib.sha256(
        json.dumps(
            {
                "trials": artifact["trials"],
                "excluded_baselines": artifact["excluded_baselines"],
                "real_success_ci": artifact["real_success_ci"],
                "control_success_ci": artifact["control_success_ci"],
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()


def serialize_artifact(
    manifest: Dict[str, Any],
    trials: Sequence[TrialResult],
    excluded_baselines: Sequence[BaselineRecord],
    real_ci: SuccessRateInterval,
    control_ci: SuccessRateInterval,
) -> Dict[str, Any]:
    """Assembles the JSON-serializable artifact from a benchmark run's results."""
    # ``n_independent_prompts`` travels with the pooled rate because trials sharing a prompt
    # are not independent draws.
    artifact: Dict[str, Any] = {
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
        "layers_executed": sorted({trial.layer for trial in trials if trial.status == "ok"}),
        "trial_status_counts": dict(Counter(trial.status for trial in trials)),
    }
    artifact["results_fingerprint"] = results_fingerprint(artifact)
    return artifact


def _parse_seed_list(value: str) -> Tuple[int, ...]:
    """Parses a comma-separated seed list for ``--control-seeds``."""
    seeds = tuple(int(part) for part in value.split(",") if part.strip())
    if not seeds:
        raise ValueError("at least one control seed is required")
    return seeds


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generates the frozen GPT-2 causal-swap benchmark artifact."
    )
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
    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    """Sweeps the country corpus over every fitted lens layer and writes the artifact."""
    args = _build_parser().parse_args(argv)

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
        corpus=corpus_definition(_COUNTRY_CORPUS),
        corpus_repo=_CORPUS_REPO,
        corpus_path=_CORPUS_PATH,
        corpus_revision=_CORPUS_REVISION,
        layers_swept=layers,
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
    status_counts = artifact["trial_status_counts"]
    n_prompts = artifact["n_independent_prompts"]
    print(
        f"wrote {len(trials)} trials ({len(ok_trials)} ok over {n_prompts} independent "
        f"prompt(s) at layers {artifact['layers_executed']}, "
        f"{status_counts.get('skipped_source_inactive', 0)} source-inactive, "
        f"{status_counts.get('skipped_no_control_token', 0)} no-control-token, "
        f"{len(excluded)} excluded prompts) to {args.output}"
    )


if __name__ == "__main__":
    main()
