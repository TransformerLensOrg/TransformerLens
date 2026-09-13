"""Unit tests for the causal coordinate-swap benchmark's corpus schema and answer metrics.

Model-free: these exercise the prompt-corpus cross product and the rank/margin metric
directly on plain tensors, so no model is loaded.
"""

import pytest
import torch

from transformer_lens.tools.analysis.jacobian_lens_causal_swap_benchmark import (
    AnswerMetrics,
    BaselineRecord,
    BenchmarkCorpus,
    FunctionSpec,
    compute_answer_metrics,
    filter_baseline_capable,
    iter_prompt_trials,
    select_norm_matched_control_token,
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


def test_filter_baseline_capable_splits_by_deterministic_argmax() -> None:
    correct = BaselineRecord("f", "A", "p1", AnswerMetrics(0, 1, True, False, 2.0))
    wrong = BaselineRecord("f", "B", "p2", AnswerMetrics(3, 5, False, False, -1.0))
    capable, excluded = filter_baseline_capable([correct, wrong])
    assert capable == [correct]
    assert excluded == [wrong]


def test_filter_baseline_capable_all_wrong_excludes_every_record() -> None:
    wrong = BaselineRecord("currency", "France", "p", AnswerMetrics(9, 4, False, False, -3.0))
    capable, excluded = filter_baseline_capable([wrong, wrong])
    assert capable == []
    assert len(excluded) == 2


def test_filter_baseline_capable_preserves_order_and_does_not_mutate_input() -> None:
    records = [
        BaselineRecord("f", "A", "p1", AnswerMetrics(0, 1, True, False, 2.0)),
        BaselineRecord("f", "B", "p2", AnswerMetrics(3, 5, False, False, -1.0)),
        BaselineRecord("f", "C", "p3", AnswerMetrics(0, 1, True, False, 1.0)),
    ]
    original = list(records)
    capable, excluded = filter_baseline_capable(records)
    assert [r.source for r in capable] == ["A", "C"]
    assert [r.source for r in excluded] == ["B"]
    assert records == original


def test_select_norm_matched_control_token_is_deterministic_given_seed() -> None:
    torch.manual_seed(0)
    dictionary = torch.randn(20, 4)
    first = select_norm_matched_control_token(
        dictionary, target_token_id=5, excluded_ids=set(), seed=1
    )
    second = select_norm_matched_control_token(
        dictionary, target_token_id=5, excluded_ids=set(), seed=1
    )
    assert first == second


def test_select_norm_matched_control_token_respects_tolerance_and_exclusions() -> None:
    dictionary = torch.zeros(5, 3)
    dictionary[0] = torch.tensor([1.0, 0.0, 0.0])  # norm 1, target
    dictionary[1] = torch.tensor([1.05, 0.0, 0.0])  # norm 1.05, within 10%
    dictionary[2] = torch.tensor([2.0, 0.0, 0.0])  # norm 2, outside tolerance
    dictionary[3] = torch.tensor([0.98, 0.0, 0.0])  # norm 0.98, within 10%, but excluded
    dictionary[4] = torch.tensor([5.0, 0.0, 0.0])  # far outside tolerance
    chosen = select_norm_matched_control_token(
        dictionary, target_token_id=0, excluded_ids={3}, tolerance=0.1, seed=0
    )
    assert chosen == 1


def test_select_norm_matched_control_token_raises_when_no_candidate_survives() -> None:
    dictionary = torch.eye(3) * torch.tensor([1.0, 10.0, 100.0]).unsqueeze(1)
    with pytest.raises(ValueError, match="no candidate token"):
        select_norm_matched_control_token(
            dictionary, target_token_id=0, excluded_ids=set(), tolerance=0.01, seed=0
        )


def test_select_norm_matched_control_token_always_excludes_the_target_itself() -> None:
    dictionary = torch.ones(
        3, 2
    )  # every atom has an identical norm -- target would trivially "match" itself
    chosen = select_norm_matched_control_token(
        dictionary, target_token_id=1, excluded_ids=set(), seed=0
    )
    assert chosen != 1


def test_select_norm_matched_control_token_rejects_non_2d_dictionary() -> None:
    with pytest.raises(ValueError, match="2-D"):
        select_norm_matched_control_token(torch.ones(3), target_token_id=0, excluded_ids=set())
