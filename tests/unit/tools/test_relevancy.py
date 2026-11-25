"""Unit tests for transformer_lens.tools.model_registry.relevancy.

Tests cover:
- Individual component scoring functions
- Composite relevancy score computation
- The authorship component and how far its boost reaches
- Batch scoring and sorting via compute_scores_for_gaps
- Edge cases (zero values, missing data)
"""

import pytest

from transformer_lens.tools.model_registry.relevancy import (
    WEIGHT_AUTHORSHIP,
    _normalize_demand,
    _normalize_usage,
    _score_benchmarkability,
    compute_relevancy_score,
    compute_scores_for_gaps,
)

# ============================================================
# _normalize_demand
# ============================================================


class TestNormalizeDemand:
    def test_max_count_returns_100(self):
        assert _normalize_demand(50, 50) == 100.0

    def test_half_of_max(self):
        assert _normalize_demand(25, 50) == 50.0

    def test_zero_count(self):
        assert _normalize_demand(0, 50) == 0.0

    def test_zero_max(self):
        assert _normalize_demand(10, 0) == 0.0

    def test_caps_at_100(self):
        assert _normalize_demand(100, 50) == 100.0


# ============================================================
# _normalize_usage
# ============================================================


class TestNormalizeUsage:
    def test_max_downloads_returns_100(self):
        assert _normalize_usage(1_000_000, 1_000_000) == 100.0

    def test_zero_downloads(self):
        assert _normalize_usage(0, 1_000_000) == 0.0

    def test_zero_max(self):
        assert _normalize_usage(100, 0) == 0.0

    def test_log_scale_compresses_range(self):
        # 10x fewer downloads should NOT produce 10x lower score (log compression)
        score_high = _normalize_usage(1_000_000, 1_000_000)
        score_low = _normalize_usage(100_000, 1_000_000)
        assert score_low > score_high * 0.5  # Log scale keeps it above 50% of max

    def test_small_downloads_still_score(self):
        score = _normalize_usage(100, 10_000_000)
        assert score > 0


# ============================================================
# _score_benchmarkability
# ============================================================


class TestScoreBenchmarkability:
    def test_none_returns_zero(self):
        assert _score_benchmarkability(None) == 0.0

    def test_under_1b(self):
        assert _score_benchmarkability(350_000_000) == 100.0

    def test_exactly_1b(self):
        assert _score_benchmarkability(1_000_000_000) == 100.0

    def test_under_3b(self):
        assert _score_benchmarkability(2_000_000_000) == 80.0

    def test_under_7b(self):
        assert _score_benchmarkability(6_000_000_000) == 60.0

    def test_under_14b(self):
        assert _score_benchmarkability(13_000_000_000) == 40.0

    def test_under_30b(self):
        assert _score_benchmarkability(27_000_000_000) == 20.0

    def test_over_30b(self):
        assert _score_benchmarkability(70_000_000_000) == 0.0


# ============================================================
# compute_relevancy_score
# ============================================================


class TestComputeRelevancyScore:
    def test_perfect_scores_all_components(self):
        score = compute_relevancy_score(
            model_count=100,
            total_downloads=1_000_000,
            min_param_count=350_000_000,
            max_model_count=100,
            max_downloads=1_000_000,
            preferred_author=True,
        )
        assert score == 100.0

    def test_perfect_scores_without_preferred_author(self):
        score = compute_relevancy_score(
            model_count=100,
            total_downloads=1_000_000,
            min_param_count=350_000_000,
            max_model_count=100,
            max_downloads=1_000_000,
        )
        assert score == round(100 * (1 - WEIGHT_AUTHORSHIP), 1)

    def test_preferred_author_adds_its_weight(self):
        base_kwargs = dict(
            model_count=10,
            total_downloads=100_000,
            min_param_count=1_000_000_000,
            max_model_count=50,
            max_downloads=10_000_000,
        )
        plain = compute_relevancy_score(**base_kwargs)
        boosted = compute_relevancy_score(preferred_author=True, **base_kwargs)
        assert boosted - plain == pytest.approx(WEIGHT_AUTHORSHIP * 100, abs=0.1)

    def test_zero_everything(self):
        score = compute_relevancy_score(
            model_count=0,
            total_downloads=0,
            min_param_count=None,
            max_model_count=100,
            max_downloads=1_000_000,
        )
        assert score == 0.0

    def test_score_in_valid_range(self):
        score = compute_relevancy_score(
            model_count=25,
            total_downloads=500_000,
            min_param_count=7_000_000_000,
            max_model_count=100,
            max_downloads=10_000_000,
        )
        assert 0 <= score <= 100

    def test_higher_downloads_increase_score(self):
        base_kwargs = dict(
            model_count=10,
            min_param_count=1_000_000_000,
            max_model_count=50,
            max_downloads=10_000_000,
        )
        score_low = compute_relevancy_score(total_downloads=1_000, **base_kwargs)
        score_high = compute_relevancy_score(total_downloads=5_000_000, **base_kwargs)
        assert score_high > score_low

    def test_smaller_model_increases_score(self):
        base_kwargs = dict(
            model_count=10,
            total_downloads=100_000,
            max_model_count=50,
            max_downloads=10_000_000,
        )
        score_big = compute_relevancy_score(min_param_count=70_000_000_000, **base_kwargs)
        score_small = compute_relevancy_score(min_param_count=350_000_000, **base_kwargs)
        assert score_small > score_big

    def test_higher_model_count_increases_score(self):
        base_kwargs = dict(
            total_downloads=100_000,
            min_param_count=1_000_000_000,
            max_model_count=100,
            max_downloads=10_000_000,
        )
        score_few = compute_relevancy_score(model_count=5, **base_kwargs)
        score_many = compute_relevancy_score(model_count=80, **base_kwargs)
        assert score_many > score_few


# ============================================================
# compute_scores_for_gaps
# ============================================================


class TestComputeScoresForGaps:
    def test_adds_relevancy_score_field(self):
        gaps = [
            {
                "architecture_id": "CodeGenForCausalLM",
                "total_models": 29,
                "total_downloads": 100_000,
                "min_param_count": 350_000_000,
            },
        ]
        result = compute_scores_for_gaps(gaps)
        assert "relevancy_score" in result[0]
        assert isinstance(result[0]["relevancy_score"], float)

    def test_sorts_by_score_descending(self):
        gaps = [
            {
                "architecture_id": "ArchLow",
                "total_models": 2,
                "total_downloads": 100,
                "min_param_count": None,
            },
            {
                "architecture_id": "ArchHigh",
                "total_models": 50,
                "total_downloads": 4_000_000,
                "min_param_count": 350_000_000,
            },
        ]
        result = compute_scores_for_gaps(gaps)
        assert result[0]["architecture_id"] == "ArchHigh"
        assert result[1]["architecture_id"] == "ArchLow"
        assert result[0]["relevancy_score"] >= result[1]["relevancy_score"]

    def test_empty_list(self):
        assert compute_scores_for_gaps([]) == []

    def test_missing_optional_fields_use_defaults(self):
        gaps = [
            {"architecture_id": "SomeNewArch", "total_models": 5},
        ]
        result = compute_scores_for_gaps(gaps)
        assert "relevancy_score" in result[0]
        assert result[0]["relevancy_score"] >= 0

    def test_tiebreaker_uses_model_count(self):
        # Two architectures with same downloads and no params — demand decides
        gaps = [
            {
                "architecture_id": "ArchA",
                "total_models": 10,
                "total_downloads": 0,
                "min_param_count": None,
            },
            {
                "architecture_id": "ArchB",
                "total_models": 20,
                "total_downloads": 0,
                "min_param_count": None,
            },
        ]
        result = compute_scores_for_gaps(gaps)
        assert result[0]["architecture_id"] == "ArchB"


# ============================================================
# Authorship in compute_scores_for_gaps
# ============================================================


def _gap(arch: str, publisher: str, downloads: int) -> dict:
    return {
        "architecture_id": arch,
        "total_models": 5,
        "total_downloads": downloads,
        "min_param_count": 1_000_000_000,
        "publishers": [
            {
                "name": publisher,
                "models": 5,
                "downloads": downloads,
                "likes": 10,
                "first_published": "2026-01-01T00:00:00Z",
            }
        ],
    }


class TestAuthorshipBoost:
    def test_preferred_author_breaks_a_tie(self):
        gaps = [_gap("CommunityArch", "somebody", 50_000), _gap("LabArch", "google", 50_000)]
        result = compute_scores_for_gaps(gaps)
        assert [g["architecture_id"] for g in result] == ["LabArch", "CommunityArch"]
        assert result[0]["authors"] == ["google"] and result[0]["preferred_author"] is True
        assert result[1]["authors"] == ["somebody"] and result[1]["preferred_author"] is False

    def test_boost_outweighs_three_times_the_downloads(self):
        gaps = [
            _gap("CommunityArch", "somebody", 30_000),
            _gap("LabArch", "google", 10_000),
            _gap("Ceiling", "somebody-else", 1_000_000),
        ]
        order = [g["architecture_id"] for g in compute_scores_for_gaps(gaps)]
        assert order.index("LabArch") < order.index("CommunityArch")

    def test_boost_does_not_outweigh_a_hundred_times_the_downloads(self):
        gaps = [_gap("CommunityArch", "somebody", 1_000_000), _gap("LabArch", "google", 10_000)]
        order = [g["architecture_id"] for g in compute_scores_for_gaps(gaps)]
        assert order == ["CommunityArch", "LabArch"]

    def test_namespace_match_ignores_case(self):
        result = compute_scores_for_gaps([_gap("LabArch", "GOOGLE", 10_000)])
        assert result[0]["preferred_author"] is True

    def test_entry_without_publishers_uses_top_sample_model(self):
        gaps = [
            {
                "architecture_id": "LegacyEntry",
                "total_models": 3,
                "sample_models": ["google/legacy-base", "somebody/legacy-finetune"],
            }
        ]
        result = compute_scores_for_gaps(gaps)
        assert result[0]["authors"] == ["google"]
        assert result[0]["preferred_author"] is True
