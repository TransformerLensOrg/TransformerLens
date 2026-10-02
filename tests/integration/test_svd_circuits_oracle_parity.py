"""Slow qualitative parity check: causally-gated OV subfunctions on a name-mover head.

Checks that ``svd_circuits`` surfaces at least two causally-gated OV subfunctions on
``gpt2-small`` layer 9 head 9, the paper's canonical name-mover head, and that the
surfaced directions behave like the paper's name-mover split.

This is a qualitative sanity check, not a pinned-threshold parity test. Unlike
``test_jacobian_lens_oracle_parity.py`` there is no external reference implementation to
pin: the Beyond Components authors' repository is research-grade and unpinned, and the
paper publishes no numeric table for this head. The only numeric cross-check available is
the repository's own ``SVDInterpreter``, which ``test_svd_circuits.py`` already exercises
for one direction; this file does not repeat it.

Cost: marked ``slow``. Boots ``gpt2-small`` on CPU and spends ``k * (n_baseline + 2)``
forward passes per swept head, so it is excluded from the default tiers.

Seed sensitivity: the gate compares ``abs(delta_metric)`` against an averaged random
in-span control, and the per-draw control magnitudes are heavy-tailed, so a direction
whose delta sits near the threshold can gate either way across seeds. Every sweep here
passes an explicit generator seed, and one test asserts the sweep reproduces under it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, List

import pytest
import torch

from transformer_lens.model_bridge import TransformerBridge
from transformer_lens.tools.analysis.svd_circuits import (
    HeadSVD,
    decompose_head,
    patch_along_directions,
)

CLEAN_PROMPT = "When Mary and John went to the store, John gave a drink to"

# Wang et al. (2022), "Interpretability in the Wild", arXiv 2211.00593, Table 1 and
# Figure 2, latest revision (which adds 9.0 as a backup name mover and removes 11.3).
# L9H9 is the canonical name mover and the head the paper's OV split is about. The
# paper's taxonomy also lists backup name movers ((9, 0), (10, 10)) and negative name
# movers ((10, 7), (11, 10)); sweeping those is left to the wider separation study,
# since the claim checked here is about a single head's internal split.
LAYER, HEAD = 9, 9

TOP_K_DIRECTIONS = 8
BASELINE_DRAWS = 16
BASELINE_SEED = 0
MIN_GATED_SUBFUNCTIONS = 2


@pytest.fixture(scope="module")
def gpt2_bridge():
    model = TransformerBridge.boot_transformers("gpt2", device="cpu", dtype=torch.float32)
    model.enable_compatibility_mode()
    # require_eval_mode() guards both forward-pass tools, so pin eval mode here rather
    # than inherit whatever boot_transformers happens to return.
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
    """Return per-direction gate rows for the top-k non-degenerate OV directions.

    Uses ``keep=[i]`` (retain only direction i), where ``gated`` means the metric moved
    less than an arbitrary same-width in-span control: the direction alone reconstructs
    the head's behavior. See :attr:`PatchResult.gated`.
    """
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


def _format_rows(rows: List[_GateRow]) -> str:
    lines = [f"{'dir':>4} {'sigma':>10} {'delta':>12} {'baseline':>12} {'gated':>6}"]
    for row in rows:
        lines.append(
            f"{row.idx:>4} {row.sigma:>10.4f} {row.delta_metric:>12.6f} "
            f"{row.baseline_delta_metric:>12.6f} {str(row.gated):>6}"
        )
    return "\n".join(lines)


def _sweep(gpt2_bridge, *, seed: int = BASELINE_SEED) -> List[_GateRow]:
    decomposition = decompose_head(gpt2_bridge, layer=LAYER, head=HEAD, which=("OV",))
    ov = decomposition.OV
    assert ov is not None
    return _gated_directions(
        gpt2_bridge,
        ov,
        gpt2_bridge.to_tokens(CLEAN_PROMPT),
        _logit_diff_metric(gpt2_bridge),
        k=TOP_K_DIRECTIONS,
        n_baseline=BASELINE_DRAWS,
        seed=seed,
    )


@pytest.mark.slow
def test_name_mover_head_has_multiple_gated_ov_subfunctions(gpt2_bridge) -> None:
    rows = _sweep(gpt2_bridge)
    print(_format_rows(rows))

    assert rows, "no attributable OV directions to sweep"
    assert all(math.isfinite(row.delta_metric) for row in rows)
    assert all(math.isfinite(row.baseline_delta_metric) for row in rows)
    assert sum(row.gated for row in rows) >= MIN_GATED_SUBFUNCTIONS


@pytest.mark.slow
def test_gated_directions_beat_their_baseline(gpt2_bridge) -> None:
    rows = _sweep(gpt2_bridge)
    gated = [row for row in rows if row.gated]
    assert gated, "expected at least one gated direction"

    for row in gated:
        assert abs(row.delta_metric) < row.baseline_delta_metric, _format_rows(rows)


@pytest.mark.slow
def test_ungated_directions_do_not_beat_their_baseline(gpt2_bridge) -> None:
    rows = _sweep(gpt2_bridge)

    for row in rows:
        if not row.gated:
            assert abs(row.delta_metric) >= row.baseline_delta_metric, _format_rows(rows)


@pytest.mark.slow
def test_sweep_is_reproducible_under_a_fixed_seed(gpt2_bridge) -> None:
    first = _sweep(gpt2_bridge)
    second = _sweep(gpt2_bridge)

    assert [row.idx for row in first] == [row.idx for row in second]
    assert [row.gated for row in first] == [row.gated for row in second]
    assert [row.delta_metric for row in first] == [row.delta_metric for row in second]
    assert [row.baseline_delta_metric for row in first] == [
        row.baseline_delta_metric for row in second
    ]
