"""Unit tests for transformer_lens.tools.model_registry.authorship and its scraper wiring.

Tests cover:
- Publisher tallies, trimming and cross-pass merging
- Crediting an architecture's authors
- The candidate-organization criteria
- The scraper writing both gap reports, resuming from a checkpoint, and --rescore
  after a promotion
- Schema, validation and API access for the organization gaps report
- Consistency of the committed gap data with PREFERRED_AUTHORS
"""

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

from transformer_lens.tools.model_registry import (
    HF_SUPPORTED_ARCHITECTURES,
    PREFERRED_AUTHORS,
    api,
    authorship,
    hf_scraper,
)
from transformer_lens.tools.model_registry.authorship import (
    MAX_PUBLISHERS,
    architecture_authors,
    build_organization_gaps_report,
    is_preferred_author,
    merge_publishers,
    namespace_of,
    record_publisher,
    top_publishers,
)
from transformer_lens.tools.model_registry.schemas import (
    ArchitectureGap,
    OrganizationGapsReport,
)
from transformer_lens.tools.model_registry.validate import (
    validate_json_schema,
    validate_organization_gaps_report,
)

DATA_DIR = Path(hf_scraper.__file__).parent / "data"


def _publisher(name, downloads, likes=0, models=1, first_published="2026-06-01T00:00:00Z"):
    return {
        "name": name,
        "models": models,
        "downloads": downloads,
        "likes": likes,
        "first_published": first_published,
    }


def _gap(arch, *publishers):
    return {"architecture_id": arch, "total_models": 1, "publishers": list(publishers)}


class TestPublishers:
    def test_namespace_of(self):
        assert namespace_of("deepseek-ai/DeepSeek-V3") == "deepseek-ai"
        assert namespace_of("gpt2") is None

    def test_record_publisher_tallies_and_keeps_earliest_upload(self):
        publishers: dict[str, dict] = {}
        later = datetime(2026, 5, 2, 12, 0, tzinfo=timezone.utc)
        # Same instant as 2026-05-01T23:00Z, written in a +02:00 offset.
        earlier = datetime(2026, 5, 2, 1, 0, tzinfo=timezone(timedelta(hours=2)))
        record_publisher(publishers, "lab/model-a", 1000, 7, later)
        record_publisher(publishers, "lab/model-b", 500, 3, earlier)
        record_publisher(publishers, "unnamespaced", 9999, 99, later)
        assert publishers == {
            "lab": {
                "name": "lab",
                "models": 2,
                "downloads": 1500,
                "likes": 10,
                "first_published": "2026-05-01T23:00:00Z",
            }
        }

    def test_top_publishers_keeps_the_earliest_outside_the_cap(self):
        records = [_publisher(f"org{i}", 1000 - i) for i in range(MAX_PUBLISHERS + 3)]
        records.append(_publisher("first-mover", 1, first_published="2025-01-01T00:00:00Z"))
        kept = [p["name"] for p in top_publishers(records)]
        assert kept == [f"org{i}" for i in range(MAX_PUBLISHERS)] + ["first-mover"]

    def test_merge_publishers_sums_and_takes_earliest(self):
        old = [_publisher("lab", 100, likes=5, models=2, first_published="2026-03-01T00:00:00Z")]
        new = [
            _publisher("lab", 50, likes=1, models=1, first_published="2026-02-01T00:00:00Z"),
            _publisher("other", 500),
        ]
        merged = {p["name"]: p for p in merge_publishers(old, new)}
        assert merged["lab"] == _publisher(
            "lab", 150, likes=6, models=3, first_published="2026-02-01T00:00:00Z"
        )
        assert merged["other"]["downloads"] == 500
        # Inputs are left alone so a retried merge cannot double count.
        assert old[0]["downloads"] == 100


class TestArchitectureAuthors:
    def test_single_author_when_top_publisher_was_also_first(self):
        gap = _gap(
            "Arch",
            _publisher("lab", 1000, first_published="2026-01-01T00:00:00Z"),
            _publisher("finetuner", 10, first_published="2026-02-01T00:00:00Z"),
        )
        assert architecture_authors(gap) == ["lab"]

    def test_popular_reupload_does_not_hide_the_original_publisher(self):
        gap = _gap(
            "Arch",
            _publisher("reuploader", 5000, first_published="2026-02-01T00:00:00Z"),
            _publisher("lab", 1000, first_published="2026-01-01T00:00:00Z"),
        )
        assert architecture_authors(gap) == ["reuploader", "lab"]

    def test_no_data_means_no_authors(self):
        assert architecture_authors({"architecture_id": "Arch", "total_models": 1}) == []

    def test_is_preferred_author_ignores_case(self):
        assert is_preferred_author("LAB", preferred={"lab"})
        assert not is_preferred_author("laboratory", preferred={"lab"})


class TestOrganizationGaps:
    PREFERRED = {"lab-small", "lab-mid", "lab-big"}

    def _reference_gaps(self):
        # Preferred footprints: downloads 100 / 1,000 / 10,000 and likes 10 / 100 / 1,000,
        # so the bar (their median) is 1,000 downloads and 100 likes.
        return [
            _gap("SmallArch", _publisher("lab-small", 100, likes=10)),
            _gap("MidArch", _publisher("lab-mid", 1000, likes=100)),
            _gap("BigArch", _publisher("lab-big", 10_000, likes=1000)),
        ]

    def _report(self, *candidate_gaps):
        return build_organization_gaps_report(
            self._reference_gaps() + list(candidate_gaps), preferred=self.PREFERRED
        )

    def test_bar_is_the_median_preferred_footprint(self):
        report = self._report()
        assert report["criteria"] == {
            "reference_authors": 3,
            "reference_statistic": "median",
            "min_downloads": 1000,
            "min_likes": 100,
        }
        assert report["gaps"] == [] and report["total_candidate_organizations"] == 0

    def test_candidate_must_clear_both_downloads_and_likes(self):
        report = self._report(
            _gap("PulledArch", _publisher("bot-pulled", 50_000, likes=99)),
            _gap("HypedArch", _publisher("hyped", 999, likes=5000)),
            _gap("SolidArch", _publisher("solid", 1000, likes=100)),
        )
        assert [g["organization"] for g in report["gaps"]] == ["solid"]

    def test_candidate_entry_reports_footprint_and_percentiles(self):
        report = self._report(
            _gap("ArchOne", _publisher("solid", 4000, likes=150, models=2)),
            _gap("ArchTwo", _publisher("solid", 6000, likes=50, models=1)),
        )
        assert report["gaps"] == [
            {
                "organization": "solid",
                # Matches lab-small, lab-mid and lab-big on downloads; all but lab-big on likes.
                "relevancy_score": 83.3,
                "total_architectures": 2,
                "total_models": 3,
                "total_downloads": 10_000,
                "total_likes": 200,
                "downloads_percentile": 100.0,
                "likes_percentile": 66.7,
                "architectures": ["ArchOne", "ArchTwo"],
            }
        ]

    def test_preferred_authors_are_never_candidates(self):
        report = self._report()
        assert "lab-big" not in [g["organization"] for g in report["gaps"]]

    def test_footprint_counts_only_the_authors_own_checkpoints(self):
        # "tiny-lab" was first, but nearly all the downloads belong to a re-uploader's
        # copies of other work; neither clears the bar on what it published itself here.
        report = self._report(
            _gap(
                "SharedArch",
                _publisher("reuploader", 900, likes=99, first_published="2026-02-01T00:00:00Z"),
                _publisher("tiny-lab", 800, likes=5000, first_published="2026-01-01T00:00:00Z"),
            )
        )
        assert report["gaps"] == []

    def test_candidates_sorted_by_score(self):
        report = self._report(
            _gap("ArchA", _publisher("modest", 1000, likes=100)),
            _gap("ArchB", _publisher("strong", 20_000, likes=2000)),
        )
        assert [g["organization"] for g in report["gaps"]] == ["strong", "modest"]
        assert report["gaps"][0]["relevancy_score"] == 100.0

    def test_no_preferred_reference_means_no_candidates(self):
        report = build_organization_gaps_report(
            [_gap("ArchA", _publisher("anyone", 10**9, likes=10**6))], preferred={"absent-lab"}
        )
        assert report["gaps"] == []
        assert report["criteria"]["min_downloads"] is None


def _hub_model(model_id, arch, downloads, likes, created_day):
    return SimpleNamespace(
        id=model_id,
        config={"architectures": [arch]},
        safetensors={"total": 1_000_000},
        downloads=downloads,
        likes=likes,
        created_at=datetime(2026, 1, created_day, tzinfo=timezone.utc),
        pipeline_tag="text-generation",
        tags=[],
        card_data=None,
    )


class _FakeHub:
    """Stands in for HfApi: one listing per task tag, already sorted by downloads."""

    listings: dict[str, list] = {}

    def __init__(self, token=None):
        pass

    def list_models(self, **kwargs):
        return iter(self.listings[kwargs["filter"]])


@pytest.fixture
def fake_hub(monkeypatch):
    import huggingface_hub

    monkeypatch.setattr(huggingface_hub, "HfApi", _FakeHub)
    monkeypatch.setattr(authorship, "PREFERRED_AUTHORS", {"biglab"})
    _FakeHub.listings = {
        "text-generation": [
            _hub_model("biglab/alpha-base", "AlphaForCausalLM", 9000, 400, 1),
            _hub_model("newlab/beta-base", "BetaForCausalLM", 8000, 500, 2),
            _hub_model("hobbyist/gamma", "GammaForCausalLM", 7000, 1, 3),
            _hub_model("fan/beta-finetune", "BetaForCausalLM", 600, 2, 9),
        ],
        "text2text-generation": [
            _hub_model("newlab/beta-seq2seq", "BetaForCausalLM", 3000, 50, 5),
        ],
    }
    return _FakeHub


class TestScraperGapReports:
    def _scrape(self, out_dir, task="text-generation"):
        hf_scraper.scrape_all_models(output_dir=out_dir, task=task, canonical_sweep=False)
        gaps = json.loads((out_dir / "architecture_gaps.json").read_text())
        organizations = json.loads((out_dir / "organization_gaps.json").read_text())
        return {g["architecture_id"]: g for g in gaps["gaps"]}, organizations

    def test_scan_credits_authors_and_lists_candidate_organizations(self, fake_hub, tmp_path):
        gaps, organizations = self._scrape(tmp_path)

        assert gaps["AlphaForCausalLM"]["authors"] == ["biglab"]
        assert gaps["AlphaForCausalLM"]["preferred_author"] is True
        assert gaps["BetaForCausalLM"]["authors"] == ["newlab"]
        assert gaps["BetaForCausalLM"]["preferred_author"] is False
        assert gaps["BetaForCausalLM"]["publishers"] == [
            _publisher("newlab", 8000, likes=500, first_published="2026-01-02T00:00:00Z"),
            _publisher("fan", 600, likes=2, first_published="2026-01-09T00:00:00Z"),
        ]

        # biglab is the only preferred author, so its footprint (9,000 downloads, 400
        # likes) is the bar: newlab misses on downloads, hobbyist on likes.
        assert organizations["criteria"]["min_downloads"] == 9000
        assert organizations["gaps"] == []
        assert validate_json_schema(tmp_path / "organization_gaps.json").valid
        assert validate_json_schema(tmp_path / "architecture_gaps.json").valid

    def test_second_task_pass_merges_publishers(self, fake_hub, tmp_path):
        self._scrape(tmp_path)
        gaps, organizations = self._scrape(tmp_path, task="text2text-generation")

        newlab = gaps["BetaForCausalLM"]["publishers"][0]
        assert newlab == _publisher(
            "newlab", 11_000, likes=550, models=2, first_published="2026-01-02T00:00:00Z"
        )
        # With the seq2seq checkpoint counted, newlab now clears biglab's bar.
        assert [g["organization"] for g in organizations["gaps"]] == ["newlab"]
        assert organizations["gaps"][0]["architectures"] == ["BetaForCausalLM"]

    def test_rescore_applies_a_promotion_without_rescanning(self, fake_hub, tmp_path, monkeypatch):
        self._scrape(tmp_path)
        self._scrape(tmp_path, task="text2text-generation")
        before = json.loads((tmp_path / "architecture_gaps.json").read_text())
        beta_before = next(g for g in before["gaps"] if g["architecture_id"] == "BetaForCausalLM")
        # A re-score must not need the Hub.
        fake_hub.listings = {}
        (tmp_path / "architecture_gaps.json").write_text(
            json.dumps({**before, "generated_at": "2026-02-02"})
        )

        monkeypatch.setattr(authorship, "PREFERRED_AUTHORS", {"biglab", "newlab"})
        gaps_report, organizations = hf_scraper.rescore_gaps(tmp_path)

        beta = next(g for g in gaps_report["gaps"] if g["architecture_id"] == "BetaForCausalLM")
        assert beta["preferred_author"] is True
        assert beta["relevancy_score"] > beta_before["relevancy_score"]
        assert "newlab" not in [g["organization"] for g in organizations["gaps"]]
        assert organizations["criteria"]["reference_authors"] == 2
        on_disk = json.loads((tmp_path / "organization_gaps.json").read_text())
        assert on_disk["generated_at"] == gaps_report["generated_at"] == "2026-02-02"

    def test_resumed_scan_keeps_publishers_from_before_the_interruption(self, fake_hub, tmp_path):
        checkpoint = {
            "supported_models": [],
            "unsupported_arch_counts": {"BetaForCausalLM": 1},
            "unsupported_arch_samples": {"BetaForCausalLM": ["newlab/beta-base"]},
            "unsupported_arch_downloads": {"BetaForCausalLM": 8000},
            "unsupported_arch_min_params": {},
            "unsupported_arch_publishers": {
                "BetaForCausalLM": {
                    "newlab": _publisher(
                        "newlab", 8000, likes=500, first_published="2026-01-02T00:00:00Z"
                    )
                }
            },
            "seen_models": ["biglab/alpha-base", "newlab/beta-base"],
            "scanned": 2,
        }
        (tmp_path / "scrape_checkpoint.json").write_text(json.dumps(checkpoint))

        gaps, _ = self._scrape(tmp_path)

        assert gaps["BetaForCausalLM"]["authors"] == ["newlab"]
        assert [p["name"] for p in gaps["BetaForCausalLM"]["publishers"]] == ["newlab", "fan"]


class TestOrganizationGapsSchema:
    def _report(self):
        gaps = [
            _gap("LabArch", _publisher("lab", 1000, likes=100)),
            _gap("ArchA", _publisher("solid", 5000, likes=500, models=2)),
        ]
        return build_organization_gaps_report(
            gaps,
            scan_info={"total_scanned": 3, "task_filter": "text-generation"},
            preferred={"lab"},
        )

    def test_report_round_trips_through_the_schema(self):
        data = self._report()
        assert OrganizationGapsReport.from_dict(data).to_dict() == data

    def test_architecture_gap_round_trips_authorship_fields(self):
        gap = _gap("ArchA", _publisher("solid", 5000))
        gap.update(authors=["solid"], preferred_author=True, sample_models=["solid/a"])
        restored = ArchitectureGap.from_dict(gap)
        assert restored.authors == ["solid"] and restored.preferred_author is True
        assert restored.to_dict()["publishers"] == gap["publishers"]

    def test_architecture_gap_without_authorship_fields_loads(self):
        legacy = ArchitectureGap.from_dict({"architecture_id": "Old", "total_models": 2})
        assert legacy.authors == [] and legacy.preferred_author is False

    def test_validation_rejects_malformed_entries(self):
        data = self._report()
        assert validate_organization_gaps_report(data).valid
        data["gaps"][0]["relevancy_score"] = 140
        del data["gaps"][0]["organization"]
        result = validate_organization_gaps_report(data)
        assert {e.path for e in result.errors} == {
            "gaps[0].organization",
            "gaps[0].relevancy_score",
        }

    def test_api_returns_candidates_in_score_order(self, tmp_path, monkeypatch):
        gaps = [
            _gap("LabArch", _publisher("lab", 1000, likes=100)),
            _gap("ArchA", _publisher("modest", 1000, likes=100)),
            _gap("ArchB", _publisher("strong", 9000, likes=900)),
        ]
        report = build_organization_gaps_report(gaps, preferred={"lab"})
        (tmp_path / "organization_gaps.json").write_text(json.dumps(report))
        monkeypatch.setattr(api, "_DATA_DIR", tmp_path)
        api.clear_cache()
        try:
            assert [g.organization for g in api.get_organization_gaps()] == ["strong", "modest"]
            assert [g.organization for g in api.get_organization_gaps(top_n=1)] == ["strong"]
        finally:
            api.clear_cache()


class TestCommittedGapData:
    """Editing PREFERRED_AUTHORS without `hf_scraper --rescore` leaves these files stale."""

    RESCORE = "run `python -m transformer_lens.tools.model_registry.hf_scraper --rescore`"

    def test_gap_files_validate(self):
        assert validate_json_schema(DATA_DIR / "architecture_gaps.json").valid
        assert validate_json_schema(DATA_DIR / "organization_gaps.json").valid

    def test_no_preferred_author_is_listed_as_a_candidate(self):
        organizations = json.loads((DATA_DIR / "organization_gaps.json").read_text())
        promoted = [
            g["organization"]
            for g in organizations["gaps"]
            if is_preferred_author(g["organization"])
        ]
        assert not promoted, f"{promoted} are preferred authors now; {self.RESCORE}"

    def test_preferred_author_flags_match_the_list(self):
        gaps = json.loads((DATA_DIR / "architecture_gaps.json").read_text())["gaps"]
        stale = [
            g["architecture_id"]
            for g in gaps
            if g["preferred_author"] != any(is_preferred_author(a) for a in g["authors"])
        ]
        assert not stale, f"{len(stale)} gaps disagree with PREFERRED_AUTHORS; {self.RESCORE}"

    def test_preferred_authors_has_no_case_duplicates(self):
        folded = [name.casefold() for name in PREFERRED_AUTHORS]
        assert len(folded) == len(set(folded))

    def test_candidates_author_only_unsupported_architectures(self):
        organizations = json.loads((DATA_DIR / "organization_gaps.json").read_text())
        listed = {a for g in organizations["gaps"] for a in g["architectures"]}
        assert not listed & HF_SUPPORTED_ARCHITECTURES
