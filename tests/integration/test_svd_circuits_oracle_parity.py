"""Slow qualitative OV sweep on GPT-2-small layer 9 head 9.

Reports prompt-specific gate counts without requiring a minimum or claiming a causal
subfunction split. There is no pinned external numerical oracle: the Beyond Components
paper publishes no numeric table for this head. The repository's ``SVDInterpreter``
cross-check is covered separately in ``test_svd_circuits.py``.

Keep-mode gates compare the metric change with a sampled mean random in-span control
magnitude, not a statistical significance threshold. Counts may vary with seed and draw
count, including zero. The checks cover eligibility, finite results, gate polarity, and
fixed-seed reproducibility. Each sweep costs ``k * (n_baseline + 2)`` forward passes,
so the model tests are marked ``slow`` and excluded from the default tiers.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, List, Sequence, Tuple

import pytest
import torch

from transformer_lens.model_bridge import TransformerBridge
from transformer_lens.tools.analysis.svd_circuits import (
    HeadSVD,
    decompose_head,
    patch_along_directions,
)

CLEAN_PROMPT = "When Mary and John went to the store, John gave a drink to"

# Wang et al. (2022), arXiv 2211.00593, identifies L9H9 as a name mover in the IOI circuit.
LAYER, HEAD = 9, 9

TOP_K_DIRECTIONS = 8
BASELINE_DRAWS = 32
BASELINE_SEED = 0


@pytest.fixture(scope="module")
def gpt2_bridge():
    model = TransformerBridge.boot_transformers("gpt2", device="cpu", dtype=torch.float32)
    model.enable_compatibility_mode()
    # Forward-pass tools require evaluation mode.
    model.eval()
    return model


def _logit_diff_metric(model) -> Callable[[torch.Tensor], float]:
    mary_token = model.to_single_token(" Mary")
    john_token = model.to_single_token(" John")

    def metric(logits: torch.Tensor) -> float:
        return float(logits[0, -1, mary_token] - logits[0, -1, john_token])

    return metric


@dataclass(frozen=True)
class _GateRow:
    """One swept direction's gate verdict."""

    idx: int
    sigma: float
    delta_metric: float
    baseline_delta_metric: float
    gated: bool


def _eligible_directions(head_svd: HeadSVD, k: int) -> List[int]:
    """The first ``k`` directions that are attributable on their own.

    Fewer than ``k`` may qualify: the degeneracy guard is a feature, so a short sweep
    is used as-is rather than treated as an error.
    """
    eligible = [
        row.idx for row in head_svd.rank_report if not row.is_degenerate and not row.is_null
    ]
    return eligible[:k]


def _gated_directions(
    model,
    head_svd: HeadSVD,
    prompt: torch.Tensor,
    metric: Callable[[torch.Tensor], float],
    *,
    k: int,
    n_baseline: int,
    seed: int,
) -> List[_GateRow]:
    """Report keep-mode changes against mean same-width random in-span control magnitudes."""
    rows: List[_GateRow] = []
    for idx in _eligible_directions(head_svd, k):
        result = patch_along_directions(
            model,
            head_svd,
            prompt,
            metric,
            keep=[idx],
            rng=torch.Generator().manual_seed(seed),
            n_baseline=n_baseline,
        )
        rows.append(
            _GateRow(
                idx=idx,
                sigma=head_svd.rank_report[idx].sigma,
                delta_metric=result.delta_metric,
                baseline_delta_metric=result.baseline_delta_metric,
                gated=result.gated,
            )
        )
    return rows


def _format_rows(rows: Sequence[_GateRow], *, seed: int, n_baseline: int) -> str:
    lines = [
        f"seed={seed}, draws={n_baseline}, swept={len(rows)}, "
        f"gated={sum(row.gated for row in rows)} (descriptive count)",
        f"{'dir':>4} {'sigma':>10} {'delta':>12} {'baseline':>12} {'gated':>6}",
    ]
    for row in rows:
        lines.append(
            f"{row.idx:>4} {row.sigma:>10.4f} {row.delta_metric:>12.6f} "
            f"{row.baseline_delta_metric:>12.6f} {str(row.gated):>6}"
        )
    return "\n".join(lines)


def _sweep(
    gpt2_bridge, *, seed: int = BASELINE_SEED, n_baseline: int = BASELINE_DRAWS
) -> Tuple[_GateRow, ...]:
    decomposition = decompose_head(gpt2_bridge, layer=LAYER, head=HEAD, which=("OV",))
    ov = decomposition.OV
    assert ov is not None
    return tuple(
        _gated_directions(
            gpt2_bridge,
            ov,
            gpt2_bridge.to_tokens(CLEAN_PROMPT),
            _logit_diff_metric(gpt2_bridge),
            k=TOP_K_DIRECTIONS,
            n_baseline=n_baseline,
            seed=seed,
        )
    )


@pytest.fixture(scope="module")
def default_sweep(gpt2_bridge) -> Tuple[_GateRow, ...]:
    """Share immutable default rows without replacing the fresh reproducibility sweep."""
    return _sweep(gpt2_bridge)


@pytest.mark.slow
@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize("n_baseline", [16, BASELINE_DRAWS])
def test_qualitative_sweep_has_finite_results_and_consistent_gates(
    gpt2_bridge, default_sweep, seed: int, n_baseline: int
) -> None:
    rows = (
        default_sweep
        if (seed, n_baseline) == (BASELINE_SEED, BASELINE_DRAWS)
        else _sweep(gpt2_bridge, seed=seed, n_baseline=n_baseline)
    )
    report = _format_rows(rows, seed=seed, n_baseline=n_baseline)
    print(report)

    ov = decompose_head(gpt2_bridge, layer=LAYER, head=HEAD, which=("OV",)).OV
    assert ov is not None
    expected_ids = [row.idx for row in ov.rank_report if not row.is_degenerate and not row.is_null][
        :TOP_K_DIRECTIONS
    ]
    actual_ids = [row.idx for row in rows]
    assert rows, "no attributable OV directions to sweep"
    assert actual_ids == expected_ids, report
    assert len(set(actual_ids)) == len(actual_ids), report
    for row in rows:
        assert math.isfinite(row.delta_metric), report
        assert math.isfinite(row.baseline_delta_metric), report
        assert row.baseline_delta_metric >= 0, report
        assert row.gated == (abs(row.delta_metric) < row.baseline_delta_metric), report


@pytest.mark.slow
def test_sweep_is_reproducible_under_a_fixed_seed(gpt2_bridge, default_sweep) -> None:
    second = _sweep(gpt2_bridge)

    assert default_sweep == second
