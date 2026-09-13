"""Unit tests for the causal coordinate-swap benchmark's corpus schema and answer metrics.

Model-free: these exercise the prompt-corpus cross product and the rank/margin metric
directly on plain tensors, so no model is loaded.
"""

import pytest
import torch

from transformer_lens.tools.analysis.jacobian_lens_causal_swap_benchmark import (
    BenchmarkCorpus,
    FunctionSpec,
    compute_answer_metrics,
    iter_prompt_trials,
)


def test_iter_prompt_trials_yields_ordered_pairs_per_function() -> None:
    corpus = BenchmarkCorpus(
        name="toy",
        concepts=("A", "B", "C"),
        functions=(
            FunctionSpec(
                name="f",
                template="{arg} implies",
                answers={"A": "a", "B": "b", "C": "c"},
            ),
        ),
    )
    trials = list(iter_prompt_trials(corpus))
    assert len(trials) == 6  # 3 * 2 ordered pairs, one function
    assert all(t.source != t.target for t in trials)
    assert {(t.source, t.target) for t in trials} == {
        ("A", "B"),
        ("A", "C"),
        ("B", "A"),
        ("B", "C"),
        ("C", "A"),
        ("C", "B"),
    }
    sample = next(t for t in trials if t.source == "A" and t.target == "B")
    assert sample.prompt == "A implies"
    assert sample.source_answer == "a" and sample.target_answer == "b"


def test_compute_answer_metrics_matches_hand_computed_rank_and_margin() -> None:
    logits = torch.tensor([1.0, 5.0, 3.0, 3.0])  # target id 2 ties with id 3 for second place
    metrics = compute_answer_metrics(logits, target_token_id=2)
    assert metrics.top1_token_id == 1
    assert metrics.target_rank == 2  # exactly one logit (id 1) strictly greater
    assert metrics.target_is_top1 is False
    assert metrics.target_tied_for_top is False  # not tied for the *maximum*, only for 2nd place
    # Margin is against the best competitor overall (id 1, the actual top1), not the nearer tie.
    assert metrics.target_logit_margin == pytest.approx(3.0 - 5.0)


def test_compute_answer_metrics_rejects_non_finite_or_out_of_range_target() -> None:
    with pytest.raises(ValueError, match="finite"):
        compute_answer_metrics(torch.tensor([1.0, float("nan")]), target_token_id=0)
    with pytest.raises(ValueError, match="one-dimensional"):
        compute_answer_metrics(torch.zeros(2, 2), target_token_id=0)
    with pytest.raises(ValueError, match="vocabulary"):
        compute_answer_metrics(torch.tensor([1.0, 2.0]), target_token_id=5)


def test_compute_answer_metrics_pins_deterministic_argmax_tie_semantics() -> None:
    # A target tied for the global maximum: target_is_top1 follows argmax's own tie-break
    # (first index), target_tied_for_top is true, and margin against itself is zero.
    metrics = compute_answer_metrics(torch.tensor([3.0, 1.0, 3.0]), target_token_id=2)
    assert metrics.target_rank == 1
    assert metrics.target_is_top1 is False
    assert metrics.target_tied_for_top is True
