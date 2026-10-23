"""Model-free tests for the causal-swap benchmark's corpus, metrics, controls and artifact."""

import dataclasses
import hashlib
import json
from typing import Any, Dict, List

import pytest
import torch

from scripts.jacobian_lens_causal_swap_benchmark import (
    SCHEMA_VERSION,
    AnswerMetrics,
    BaselineRecord,
    BenchmarkCorpus,
    ControlDraw,
    FunctionSpec,
    TrialResult,
    _control_generator,
    build_protocol_manifest,
    compute_answer_metrics,
    corpus_definition,
    filter_baseline_capable,
    fingerprint_manifest,
    iter_prompt_trials,
    results_fingerprint,
    select_displacement_matched_control_token,
    serialize_artifact,
)
from transformer_lens.utilities import success_rate_ci


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
    # argmax breaks the tie toward the first index, so a target tied for the max is not top-1.
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


def test_filter_baseline_capable_treats_tied_for_top_as_incapable() -> None:
    # Otherwise argmax's lowest-index tie-break would decide whether a prompt enters the trial set.
    tied = BaselineRecord("f", "A", "p1", AnswerMetrics(0, 1, True, True, 0.0))
    capable, excluded = filter_baseline_capable([tied])
    assert capable == []
    assert excluded == [tied]


def test_filter_baseline_capable_partitions_every_record_exactly_once() -> None:
    records = [
        BaselineRecord("f", "A", "p1", AnswerMetrics(0, 1, True, False, 2.0)),
        BaselineRecord("f", "B", "p2", AnswerMetrics(0, 1, True, True, 0.0)),
        BaselineRecord("f", "C", "p3", AnswerMetrics(3, 5, False, False, -1.0)),
        BaselineRecord("f", "D", "p4", AnswerMetrics(4, 2, False, True, -0.5)),
    ]
    capable, excluded = filter_baseline_capable(records)
    assert [r.source for r in capable] == ["A"]
    assert [r.source for r in excluded] == ["B", "C", "D"]
    assert len(capable) + len(excluded) == len(records)


def test_select_displacement_matched_control_token_is_deterministic_given_seed() -> None:
    torch.manual_seed(0)
    dictionary = torch.randn(20, 4)
    first = select_displacement_matched_control_token(
        dictionary,
        source_token_id=0,
        target_token_id=5,
        excluded_ids=set(),
        active_support=set(),
        seed=1,
    )
    second = select_displacement_matched_control_token(
        dictionary,
        source_token_id=0,
        target_token_id=5,
        excluded_ids=set(),
        active_support=set(),
        seed=1,
    )
    assert first == second


def test_select_displacement_matched_control_token_respects_tolerance_and_exclusions() -> None:
    dictionary = torch.zeros(5, 3)
    dictionary[0] = torch.tensor([1.0, 0.0, 0.0])  # source
    dictionary[1] = torch.tensor([2.0, 0.0, 0.0])  # target, displacement 1.0
    dictionary[2] = torch.tensor([2.04, 0.0, 0.0])  # displacement 1.04, within 10%
    dictionary[3] = torch.tensor([2.005, 0.0, 0.0])  # displacement 1.005, within 10%, but excluded
    dictionary[4] = torch.tensor([5.0, 0.0, 0.0])  # displacement 4.0, far outside tolerance
    chosen = select_displacement_matched_control_token(
        dictionary,
        source_token_id=0,
        target_token_id=1,
        excluded_ids={3},
        active_support=set(),
        tolerance=0.1,
        seed=0,
    )
    assert chosen == 2


def test_select_displacement_matched_control_token_matches_displacement_not_atom_norm() -> None:
    # Atom 2 matches the target's norm but not its displacement; atom 3 the reverse. A
    # norm-matching selector would pick atom 2.
    dictionary = torch.zeros(4, 2)
    dictionary[0] = torch.tensor([1.0, 0.0])  # source
    dictionary[1] = torch.tensor([3.0, 0.0])  # target: norm 3, displacement 2.0
    dictionary[2] = torch.tensor([0.0, 3.0])  # norm 3, displacement sqrt(10) ~ 3.162
    dictionary[3] = torch.tensor([-1.0, 0.0])  # norm 1, displacement 2.0
    chosen = select_displacement_matched_control_token(
        dictionary,
        source_token_id=0,
        target_token_id=1,
        excluded_ids=set(),
        active_support=set(),
        tolerance=0.1,
        seed=0,
    )
    assert chosen == 3


def test_select_displacement_matched_control_token_excludes_active_support() -> None:
    dictionary = torch.zeros(4, 2)
    dictionary[0] = torch.tensor([0.0, 0.0])  # source
    dictionary[1] = torch.tensor([1.0, 0.0])  # target, displacement 1.0
    dictionary[2] = torch.tensor([1.0, 0.0])  # displacement 1.0, but in the active support
    dictionary[3] = torch.tensor([1.02, 0.0])  # displacement 1.02, within 10%
    chosen = select_displacement_matched_control_token(
        dictionary,
        source_token_id=0,
        target_token_id=1,
        excluded_ids=set(),
        active_support={2},
        tolerance=0.1,
        seed=0,
    )
    assert chosen == 3


def test_select_displacement_matched_control_token_returns_none_on_empty_pool() -> None:
    dictionary = torch.eye(3) * torch.tensor([1.0, 10.0, 100.0]).unsqueeze(1)
    chosen = select_displacement_matched_control_token(
        dictionary,
        source_token_id=0,
        target_token_id=1,
        excluded_ids=set(),
        active_support=set(),
        tolerance=0.01,
        seed=0,
    )
    assert chosen is None


def test_select_displacement_matched_control_token_always_excludes_the_target_itself() -> None:
    # Every atom shares one displacement, so the target would trivially "match" itself.
    dictionary = torch.ones(3, 2)
    chosen = select_displacement_matched_control_token(
        dictionary,
        source_token_id=0,
        target_token_id=1,
        excluded_ids=set(),
        active_support=set(),
        seed=0,
    )
    assert chosen != 1


def test_select_displacement_matched_control_token_rejects_non_2d_dictionary() -> None:
    with pytest.raises(ValueError, match="2-D"):
        select_displacement_matched_control_token(
            torch.ones(3),
            source_token_id=0,
            target_token_id=1,
            excluded_ids=set(),
            active_support=set(),
        )


def test_control_generator_is_always_cpu() -> None:
    # torch.randint draws on CPU, so a generator following the dictionary's device would raise.
    assert _control_generator(0).device == torch.device("cpu")


def _accelerator_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return ""


@pytest.mark.skipif(not _accelerator_device(), reason="no CUDA or MPS device available")
def test_select_displacement_matched_control_token_is_device_independent() -> None:
    # The same seed must pick the same control token on an accelerator as on CPU.
    torch.manual_seed(0)
    dictionary = torch.randn(20, 4)
    cpu_choice = select_displacement_matched_control_token(
        dictionary,
        source_token_id=0,
        target_token_id=5,
        excluded_ids=set(),
        active_support=set(),
        seed=1,
    )
    device = _accelerator_device()
    moved = dictionary.to(device)
    device_choice = select_displacement_matched_control_token(
        moved,
        source_token_id=0,
        target_token_id=5,
        excluded_ids=set(),
        active_support=set(),
        seed=1,
    )
    assert device_choice == cpu_choice


def test_fingerprint_manifest_matches_the_notebook_recipe() -> None:
    manifest = {"b": 2, "a": 1}
    expected = hashlib.sha256(
        json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    assert fingerprint_manifest(manifest) == expected


def test_build_protocol_manifest_rejects_missing_required_field() -> None:
    with pytest.raises(ValueError, match="model_id"):
        build_protocol_manifest(model_revision="x")


def test_build_protocol_manifest_requires_corpus_fields() -> None:
    fields = _full_manifest_fields()
    del fields["corpus_repo"]
    with pytest.raises(ValueError, match="corpus_repo"):
        build_protocol_manifest(**fields)


def test_corpus_definition_embeds_concepts_functions_and_answers() -> None:
    corpus = BenchmarkCorpus(
        name="toy",
        concepts=("France", "China"),
        functions=(
            FunctionSpec(
                name="capital",
                template="The capital of {arg} is the city of",
                answers={"France": "Paris", "China": "Beijing"},
            ),
        ),
    )
    definition = corpus_definition(corpus)
    assert definition == {
        "name": "toy",
        "concepts": ["France", "China"],
        "functions": [
            {
                "name": "capital",
                "template": "The capital of {arg} is the city of",
                "answers": {"France": "Paris", "China": "Beijing"},
            }
        ],
    }


def _full_manifest_fields(**overrides: Any) -> Dict[str, Any]:
    fields: Dict[str, Any] = dict(
        model_id="gpt2",
        model_revision="x",
        lens_repo="r",
        lens_file="f",
        lens_revision="y",
        corpus_name="toy",
        corpus={"name": "toy", "concepts": ["France"], "functions": []},
        corpus_repo="owner/repo",
        corpus_path="data/config.json",
        corpus_revision="abc123",
        layers_swept=[1],
        alpha=1.0,
        k=8,
        control_tolerance=0.1,
        control_seeds=[0],
        success_definition="target token id equals deterministic argmax token id",
        baseline_definition="source answer token id equals deterministic argmax token id",
        rank_definition="1 + count(logits strictly greater than target logit)",
    )
    fields.update(overrides)
    return fields


def _ok_trial(*, target: str, layer: int) -> TrialResult:
    metrics = AnswerMetrics(0, 1, True, False, 2.0)
    return TrialResult(
        function="continent",
        source="Egypt",
        target=target,
        layer=layer,
        status="ok",
        baseline=metrics,
        real_target_metrics=metrics,
        control_token_id=1,
        control_target_metrics=metrics,
        control_draws=[ControlDraw(seed=0, token_id=1, metrics=metrics)],
        error=None,
    )


def _skipped_trial(*, layer: int) -> TrialResult:
    return TrialResult(
        function="continent",
        source="Egypt",
        target="France",
        layer=layer,
        status="skipped_source_inactive",
        baseline=AnswerMetrics(0, 1, True, False, 2.0),
        real_target_metrics=None,
        control_token_id=None,
        control_target_metrics=None,
        control_draws=[],
        error="inactive",
    )


def test_serialized_fingerprints_reproduce_after_a_json_round_trip() -> None:
    # The demo notebook recomputes both fingerprints from the reloaded JSON, so they must
    # survive serialization, not just match the in-memory artifact.
    manifest = build_protocol_manifest(**_full_manifest_fields())
    trials: List[TrialResult] = [_ok_trial(target="France", layer=9)]
    excluded: List[BaselineRecord] = []
    real_ci = success_rate_ci([True, False])
    control_ci = success_rate_ci([False, False])
    artifact = serialize_artifact(manifest, trials, excluded, real_ci, control_ci)
    loaded = json.loads(json.dumps(artifact, indent=2, sort_keys=True))
    assert loaded["schema_version"] == SCHEMA_VERSION
    assert fingerprint_manifest(loaded["protocol_manifest"]) == loaded["protocol_fingerprint"]
    assert results_fingerprint(loaded) == loaded["results_fingerprint"]


def test_serialize_artifact_round_trips_trial_and_baseline_records() -> None:
    manifest = build_protocol_manifest(**_full_manifest_fields(layers_swept=[6]))
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
        control_draws=[
            ControlDraw(seed=0, token_id=42, metrics=AnswerMetrics(2, 3, False, False, -0.1))
        ],
        error=None,
    )
    excluded = [
        BaselineRecord("currency", "Egypt", "prompt", AnswerMetrics(9, 4, False, False, -3.0))
    ]
    real_ci = success_rate_ci([True])
    control_ci = success_rate_ci([False])
    artifact = serialize_artifact(manifest, [trial], excluded, real_ci, control_ci)
    loaded = json.loads(json.dumps(artifact))
    assert loaded["trials"] == [dataclasses.asdict(trial)]
    assert loaded["excluded_baselines"] == [dataclasses.asdict(excluded[0])]


def test_protocol_fingerprint_detects_a_tampered_manifest() -> None:
    manifest = build_protocol_manifest(**_full_manifest_fields(layers_swept=[1]))
    artifact = serialize_artifact(
        manifest, [], [], success_rate_ci([True]), success_rate_ci([False])
    )
    artifact["protocol_manifest"]["layers_swept"] = [2]
    assert fingerprint_manifest(artifact["protocol_manifest"]) != artifact["protocol_fingerprint"]


def test_results_fingerprint_detects_tampered_trials() -> None:
    manifest = build_protocol_manifest(**_full_manifest_fields())
    artifact = serialize_artifact(
        manifest,
        [_ok_trial(target="France", layer=9)],
        [],
        success_rate_ci([True]),
        success_rate_ci([False]),
    )
    artifact["trials"][0]["status"] = "skipped_source_inactive"
    assert results_fingerprint(artifact) != artifact["results_fingerprint"]


def test_results_fingerprint_detects_a_tampered_ci_block() -> None:
    manifest = build_protocol_manifest(**_full_manifest_fields())
    artifact = serialize_artifact(
        manifest, [], [], success_rate_ci([True, False]), success_rate_ci([False, False])
    )
    artifact["real_success_ci"]["n_trials"] = 99
    assert results_fingerprint(artifact) != artifact["results_fingerprint"]


def test_results_fingerprint_is_stable_across_key_order() -> None:
    manifest = build_protocol_manifest(**_full_manifest_fields())
    artifact = serialize_artifact(
        manifest, [], [], success_rate_ci([True]), success_rate_ci([False])
    )
    reordered = {
        "control_success_ci": artifact["control_success_ci"],
        "real_success_ci": artifact["real_success_ci"],
        "excluded_baselines": artifact["excluded_baselines"],
        "trials": artifact["trials"],
    }
    assert results_fingerprint(reordered) == artifact["results_fingerprint"]


def test_serialize_artifact_records_executed_layers_and_status_counts() -> None:
    manifest = build_protocol_manifest(**_full_manifest_fields(layers_swept=[0, 9, 10]))
    trials = [
        _ok_trial(target="France", layer=9),
        _ok_trial(target="China", layer=10),
        _skipped_trial(layer=0),
    ]
    artifact = serialize_artifact(
        manifest, trials, [], success_rate_ci([True, False]), success_rate_ci([False, False])
    )
    assert artifact["layers_executed"] == [9, 10]
    assert artifact["trial_status_counts"] == {"ok": 2, "skipped_source_inactive": 1}


def test_serialize_artifact_counts_independent_prompts(tmp_path) -> None:
    # Trials can share a prompt, so the pooled rate rests on fewer independent prompts than trials.
    manifest = build_protocol_manifest(**_full_manifest_fields())
    trials = [
        _ok_trial(target=target, layer=layer)
        for target, layer in (("France", 9), ("France", 10), ("China", 9))
    ]
    artifact = serialize_artifact(
        manifest, trials, [], success_rate_ci([True]), success_rate_ci([False])
    )
    assert artifact["n_independent_prompts"] == 1


def test_serialize_artifact_ignores_skipped_trials_when_counting_prompts() -> None:
    manifest = build_protocol_manifest(**_full_manifest_fields())
    artifact = serialize_artifact(
        manifest, [_skipped_trial(layer=6)], [], success_rate_ci([True]), success_rate_ci([False])
    )
    assert artifact["n_independent_prompts"] == 0
