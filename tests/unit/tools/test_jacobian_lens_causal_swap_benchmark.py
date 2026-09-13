"""Unit tests for the causal coordinate-swap benchmark's corpus schema and answer metrics.

Model-free: these exercise the prompt-corpus cross product and the rank/margin metric
directly on plain tensors, so no model is loaded.
"""

import dataclasses
import hashlib
import json
from typing import Any, Dict, List

import pytest
import torch

from transformer_lens.tools.analysis.jacobian_lens_causal_swap_benchmark import (
    SCHEMA_VERSION,
    AnswerMetrics,
    BaselineRecord,
    BenchmarkCorpus,
    FunctionSpec,
    TrialResult,
    bootstrap_success_rate_ci,
    build_protocol_manifest,
    compute_answer_metrics,
    filter_baseline_capable,
    fingerprint_manifest,
    iter_prompt_trials,
    load_artifact,
    select_norm_matched_control_token,
    serialize_artifact,
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


def test_bootstrap_success_rate_ci_bounds_bracket_point_estimate_and_lie_in_unit_interval() -> None:
    successes = [True, True, False, True, False, True, True, False]
    result = bootstrap_success_rate_ci(successes, n_resamples=2000, seed=0)
    assert result.ci_low <= result.point_estimate <= result.ci_high
    assert 0.0 <= result.ci_low and result.ci_high <= 1.0
    assert result.point_estimate == pytest.approx(sum(successes) / len(successes))


def test_bootstrap_success_rate_ci_is_deterministic_given_seed() -> None:
    successes = [True, False, True]
    first = bootstrap_success_rate_ci(successes, seed=3)
    second = bootstrap_success_rate_ci(successes, seed=3)
    assert first == second


def test_bootstrap_success_rate_ci_rejects_empty_input() -> None:
    with pytest.raises(ValueError, match="empty"):
        bootstrap_success_rate_ci([])


def test_fingerprint_manifest_matches_the_notebook_recipe() -> None:
    manifest = {"b": 2, "a": 1}
    expected = hashlib.sha256(
        json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    assert fingerprint_manifest(manifest) == expected


def test_build_protocol_manifest_rejects_missing_required_field() -> None:
    with pytest.raises(ValueError, match="model_id"):
        build_protocol_manifest(model_revision="x")


def _full_manifest_fields(**overrides: Any) -> Dict[str, Any]:
    fields: Dict[str, Any] = dict(
        model_id="gpt2",
        model_revision="x",
        lens_repo="r",
        lens_file="f",
        lens_revision="y",
        corpus_name="toy",
        layers=[1],
        alpha=1.0,
        k=8,
        control_tolerance=0.1,
        control_seed=0,
        success_definition="target token id equals deterministic argmax token id",
        baseline_definition="source answer token id equals deterministic argmax token id",
        rank_definition="1 + count(logits strictly greater than target logit)",
    )
    fields.update(overrides)
    return fields


def test_serialize_then_load_artifact_round_trips(tmp_path) -> None:
    manifest = build_protocol_manifest(**_full_manifest_fields())
    trials: List[TrialResult] = []
    excluded: List[BaselineRecord] = []
    real_ci = bootstrap_success_rate_ci([True, False])
    control_ci = bootstrap_success_rate_ci([False, False])
    artifact = serialize_artifact(manifest, trials, excluded, real_ci, control_ci)
    path = tmp_path / "artifact.json"
    path.write_text(json.dumps(artifact))
    loaded = load_artifact(path)
    assert loaded["protocol_fingerprint"] == artifact["protocol_fingerprint"]
    assert loaded["schema_version"] == SCHEMA_VERSION


def test_serialize_artifact_round_trips_trial_and_baseline_records(tmp_path) -> None:
    manifest = build_protocol_manifest(**_full_manifest_fields(layers=[6]))
    trial = TrialResult(
        function="capital",
        source="France",
        target="China",
        layer=6,
        status="ok",
        baseline=AnswerMetrics(0, 1, True, False, 2.0),
        real_target_metrics=AnswerMetrics(1, 1, True, False, 0.5),
        control_token_id=42,
        control_target_metrics=AnswerMetrics(2, 3, False, False, -0.1),
        error=None,
    )
    excluded = [
        BaselineRecord("currency", "Egypt", "prompt", AnswerMetrics(9, 4, False, False, -3.0))
    ]
    real_ci = bootstrap_success_rate_ci([True])
    control_ci = bootstrap_success_rate_ci([False])
    artifact = serialize_artifact(manifest, [trial], excluded, real_ci, control_ci)
    path = tmp_path / "artifact.json"
    path.write_text(json.dumps(artifact))
    loaded = load_artifact(path)
    assert loaded["trials"] == [dataclasses.asdict(trial)]
    assert loaded["excluded_baselines"] == [dataclasses.asdict(excluded[0])]


def test_load_artifact_rejects_tampered_fingerprint(tmp_path) -> None:
    manifest = build_protocol_manifest(**_full_manifest_fields(layers=[1]))
    artifact = serialize_artifact(
        manifest, [], [], bootstrap_success_rate_ci([True]), bootstrap_success_rate_ci([False])
    )
    artifact["protocol_manifest"]["layers"] = [2]
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(artifact))
    with pytest.raises(ValueError, match="fingerprint"):
        load_artifact(path)


def test_load_artifact_rejects_wrong_schema_version(tmp_path) -> None:
    manifest = build_protocol_manifest(**_full_manifest_fields())
    artifact = serialize_artifact(
        manifest, [], [], bootstrap_success_rate_ci([True]), bootstrap_success_rate_ci([False])
    )
    artifact["schema_version"] = SCHEMA_VERSION + 1
    artifact["protocol_fingerprint"] = fingerprint_manifest(artifact["protocol_manifest"])
    path = tmp_path / "bad_version.json"
    path.write_text(json.dumps(artifact))
    with pytest.raises(ValueError, match="schema_version"):
        load_artifact(path)


def test_load_artifact_rejects_missing_required_key(tmp_path) -> None:
    manifest = build_protocol_manifest(**_full_manifest_fields())
    artifact = serialize_artifact(
        manifest, [], [], bootstrap_success_rate_ci([True]), bootstrap_success_rate_ci([False])
    )
    del artifact["excluded_baselines"]
    path = tmp_path / "bad_missing.json"
    path.write_text(json.dumps(artifact))
    with pytest.raises(ValueError, match="excluded_baselines"):
        load_artifact(path)
