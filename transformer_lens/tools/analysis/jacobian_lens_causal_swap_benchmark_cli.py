"""Artifact-generation entry point for the causal coordinate-swap benchmark.

Kept separate from
``transformer_lens.tools.analysis.jacobian_lens_causal_swap_benchmark`` because the analysis
package's ``__init__`` imports that module: running it as a script would execute it twice and
``runpy`` would warn. This module is not imported by the package, so ``python -m`` runs it
once.

Reproduce the frozen artifact with (no ``HF_TOKEN`` needed -- GPT-2 is not gated)::

    uv run python -m transformer_lens.tools.analysis.jacobian_lens_causal_swap_benchmark_cli
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

from transformer_lens.tools.analysis.jacobian_lens import DEFAULT_K, JacobianLens
from transformer_lens.tools.analysis.jacobian_lens_causal_swap_benchmark import (
    BenchmarkCorpus,
    FunctionSpec,
    build_protocol_manifest,
    corpus_definition,
    run_causal_swap_benchmark,
    serialize_artifact,
    success_rate_ci,
)

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
# The corpus mirrors the country config ``Jacobian_Lens_Demo.ipynb`` pins, so it reuses that
# notebook's repo/path/revision triple rather than inventing a second provenance scheme.
_CORPUS_REPO = "anthropics/jacobian-lens"
_CORPUS_PATH = "data/experiments/flexible-generalization.json"
_CORPUS_REVISION = "581d398613e5602a5af361e1c34d3a92ea82ba8e"
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


def _build_parser() -> argparse.ArgumentParser:
    """Builds the CLI argument parser."""
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
    """Generates the frozen GPT-2 causal-swap benchmark artifact.

    Loads the published GPT-2-small lens (the same artifact
    ``tests/integration/test_jacobian_lens.py`` uses), runs the reused country corpus over
    every one of the lens's fitted source layers, and writes the versioned, fingerprinted JSON
    artifact consumed by ``demos/Jacobian_Lens_Coordinate_Patch_Benchmark_Demo.ipynb``.
    """
    import torch

    from transformer_lens.model_bridge import TransformerBridge

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
