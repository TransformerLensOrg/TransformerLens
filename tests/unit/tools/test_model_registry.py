"""Unit tests for transformer_lens.tools.model_registry.

Tests cover:
- schemas.py: Round-trip serialization, backwards compat
- verification.py: add_record, invalidate, is_verified, get_record
- api.py: get_supported_models, is_model_supported, get_registry_stats
- validate.py: validate_json_schema with valid and invalid data
- registry_io.py: update_model_status, add_verification_record
- alias_drift.py: check_drift, DriftReport
- verify_models.py: _sanitize_note
"""

import json
from datetime import date, datetime

import pytest

# ============================================================
# Fixture: temp data directory with minimal valid JSON files
# ============================================================


@pytest.fixture
def registry_data_dir(temp_dir):
    """Create a temp directory with minimal valid registry JSON files."""
    supported = {
        "generated_at": "2026-01-01",
        "scan_info": {"total_scanned": 100, "task_filter": "text-generation"},
        "total_architectures": 2,
        "total_models": 3,
        "total_verified": 1,
        "models": [
            {
                "architecture_id": "GPT2LMHeadModel",
                "model_id": "openai-community/gpt2",
                "status": 1,
                "verified_date": "2026-01-01",
                "metadata": None,
                "note": None,
                "phase1_score": 100.0,
                "phase2_score": 95.0,
                "phase3_score": 90.0,
            },
            {
                "architecture_id": "GPT2LMHeadModel",
                "model_id": "sshleifer/tiny-gpt2",
                "status": 0,
                "verified_date": None,
                "metadata": None,
                "note": None,
                "phase1_score": None,
                "phase2_score": None,
                "phase3_score": None,
            },
            {
                "architecture_id": "LlamaForCausalLM",
                "model_id": "meta-llama/Llama-2-7b-hf",
                "status": 1,
                "verified_date": "2026-01-01",
                "metadata": None,
                "note": None,
                "phase1_score": 100.0,
                "phase2_score": 100.0,
                "phase3_score": 100.0,
            },
        ],
    }
    (temp_dir / "supported_models.json").write_text(json.dumps(supported, indent=2))

    gaps = {
        "generated_at": "2026-01-01",
        "scan_info": {"total_scanned": 100, "task_filter": "text-generation"},
        "total_unsupported_architectures": 1,
        "total_unsupported_models": 50,
        "gaps": [
            {
                "architecture_id": "FalconForCausalLM",
                "total_models": 50,
                "sample_models": ["tiiuae/falcon-7b"],
            },
        ],
    }
    (temp_dir / "architecture_gaps.json").write_text(json.dumps(gaps, indent=2))

    history = {
        "last_updated": "2026-01-01T12:00:00",
        "records": [
            {
                "model_id": "openai-community/gpt2",
                "architecture_id": "GPT2LMHeadModel",
                "verified_date": "2026-01-01",
                "verified_by": "test",
                "transformerlens_version": "3.0.0",
                "notes": "Test verification",
                "invalidated": False,
                "invalidation_reason": None,
            },
        ],
    }
    (temp_dir / "verification_history.json").write_text(json.dumps(history, indent=2))

    return temp_dir


# ============================================================
# Test schemas.py: Round-trip serialization
# ============================================================


class TestModelEntry:
    """Tests for ModelEntry serialization."""

    def test_round_trip(self):
        from transformer_lens.tools.model_registry.schemas import ModelEntry

        entry = ModelEntry(
            architecture_id="GPT2LMHeadModel",
            model_id="openai-community/gpt2",
            status=1,
            verified_date=date(2026, 1, 1),
            phase1_score=100.0,
            phase2_score=95.5,
            phase3_score=None,
        )
        d = entry.to_dict()
        restored = ModelEntry.from_dict(d)
        assert restored.architecture_id == entry.architecture_id
        assert restored.model_id == entry.model_id
        assert restored.status == entry.status
        assert restored.verified_date == entry.verified_date
        assert restored.phase1_score == entry.phase1_score
        assert restored.phase2_score == entry.phase2_score
        assert restored.phase3_score is None

    def test_backwards_compat_verified_bool(self):
        """Old format had 'verified: true' instead of 'status: 1'."""
        from transformer_lens.tools.model_registry.schemas import ModelEntry

        old_data = {
            "architecture_id": "GPT2LMHeadModel",
            "model_id": "gpt2",
            "verified": True,
        }
        entry = ModelEntry.from_dict(old_data)
        assert entry.status == 1

    def test_backwards_compat_verified_false(self):
        from transformer_lens.tools.model_registry.schemas import ModelEntry

        old_data = {
            "architecture_id": "GPT2LMHeadModel",
            "model_id": "gpt2",
            "verified": False,
        }
        entry = ModelEntry.from_dict(old_data)
        assert entry.status == 0

    def test_status_takes_precedence_over_verified(self):
        """When both 'status' and 'verified' are present, status wins."""
        from transformer_lens.tools.model_registry.schemas import ModelEntry

        data = {
            "architecture_id": "GPT2LMHeadModel",
            "model_id": "gpt2",
            "status": 3,
            "verified": True,
        }
        entry = ModelEntry.from_dict(data)
        assert entry.status == 3


class TestModelMetadata:
    def test_round_trip(self):
        from transformer_lens.tools.model_registry.schemas import ModelMetadata

        meta = ModelMetadata(
            downloads=1000,
            likes=50,
            last_modified=datetime(2026, 1, 15, 10, 30),
            tags=["text-generation", "en"],
            parameter_count=125000000,
        )
        d = meta.to_dict()
        restored = ModelMetadata.from_dict(d)
        assert restored.downloads == 1000
        assert restored.likes == 50
        assert restored.last_modified == datetime(2026, 1, 15, 10, 30)
        assert restored.tags == ["text-generation", "en"]
        assert restored.parameter_count == 125000000

    def test_from_dict_defaults(self):
        from transformer_lens.tools.model_registry.schemas import ModelMetadata

        meta = ModelMetadata.from_dict({})
        assert meta.downloads == 0
        assert meta.likes == 0
        assert meta.last_modified is None
        assert meta.tags == []
        assert meta.parameter_count is None


class TestSupportedModelsReport:
    def test_round_trip(self):
        from transformer_lens.tools.model_registry.schemas import (
            ModelEntry,
            ScanInfo,
            SupportedModelsReport,
        )

        report = SupportedModelsReport(
            generated_at=date(2026, 1, 1),
            scan_info=ScanInfo(total_scanned=100, task_filter="text-generation"),
            total_architectures=1,
            total_models=1,
            total_verified=1,
            models=[
                ModelEntry(
                    architecture_id="GPT2LMHeadModel",
                    model_id="gpt2",
                    status=1,
                ),
            ],
        )
        d = report.to_dict()
        restored = SupportedModelsReport.from_dict(d)
        assert restored.total_models == 1
        assert len(restored.models) == 1
        assert restored.models[0].model_id == "gpt2"


class TestArchitectureGapsReport:
    def test_round_trip(self):
        from transformer_lens.tools.model_registry.schemas import (
            ArchitectureGap,
            ArchitectureGapsReport,
            ScanInfo,
        )

        report = ArchitectureGapsReport(
            generated_at=date(2026, 1, 1),
            scan_info=ScanInfo(total_scanned=100, task_filter="text-generation"),
            total_unsupported_architectures=1,
            total_unsupported_models=50,
            gaps=[ArchitectureGap("FalconForCausalLM", 50, ["tiiuae/falcon-7b"])],
        )
        d = report.to_dict()
        restored = ArchitectureGapsReport.from_dict(d)
        assert restored.total_unsupported_architectures == 1
        assert len(restored.gaps) == 1


# ============================================================
# Test verification.py
# ============================================================


class TestVerificationHistory:
    def test_add_record(self):
        from transformer_lens.tools.model_registry.verification import (
            VerificationHistory,
            VerificationRecord,
        )

        history = VerificationHistory()
        record = VerificationRecord(
            model_id="gpt2",
            architecture_id="GPT2LMHeadModel",
            verified_date=date(2026, 1, 1),
            verified_by="test",
        )
        history.add_record(record)
        assert len(history.records) == 1
        assert history.last_updated is not None

    def test_is_verified(self):
        from transformer_lens.tools.model_registry.verification import (
            VerificationHistory,
            VerificationRecord,
        )

        history = VerificationHistory()
        assert not history.is_verified("gpt2")

        record = VerificationRecord(
            model_id="gpt2",
            architecture_id="GPT2LMHeadModel",
            verified_date=date(2026, 1, 1),
        )
        history.add_record(record)
        assert history.is_verified("gpt2")

    def test_get_record_returns_most_recent_valid(self):
        from transformer_lens.tools.model_registry.verification import (
            VerificationHistory,
            VerificationRecord,
        )

        history = VerificationHistory()
        r1 = VerificationRecord(
            model_id="gpt2",
            architecture_id="GPT2LMHeadModel",
            verified_date=date(2026, 1, 1),
            notes="first",
        )
        r2 = VerificationRecord(
            model_id="gpt2",
            architecture_id="GPT2LMHeadModel",
            verified_date=date(2026, 2, 1),
            notes="second",
        )
        history.add_record(r1)
        history.add_record(r2)
        result = history.get_record("gpt2")
        assert result is not None
        assert result.notes == "second"

    def test_invalidate(self):
        from transformer_lens.tools.model_registry.verification import (
            VerificationHistory,
            VerificationRecord,
        )

        history = VerificationHistory()
        record = VerificationRecord(
            model_id="gpt2",
            architecture_id="GPT2LMHeadModel",
            verified_date=date(2026, 1, 1),
        )
        history.add_record(record)
        result = history.invalidate("gpt2", "outdated")
        assert result is True
        assert not history.is_verified("gpt2")

    def test_invalidate_nonexistent(self):
        from transformer_lens.tools.model_registry.verification import (
            VerificationHistory,
        )

        history = VerificationHistory()
        result = history.invalidate("nonexistent", "reason")
        assert result is False

    def test_round_trip(self):
        from transformer_lens.tools.model_registry.verification import (
            VerificationHistory,
            VerificationRecord,
        )

        history = VerificationHistory()
        history.add_record(
            VerificationRecord(
                model_id="gpt2",
                architecture_id="GPT2LMHeadModel",
                verified_date=date(2026, 1, 1),
                verified_by="test",
                transformerlens_version="3.0.0",
                notes="ok",
            )
        )
        d = history.to_dict()
        restored = VerificationHistory.from_dict(d)
        assert len(restored.records) == 1
        assert restored.records[0].model_id == "gpt2"


# ============================================================
# Test api.py (with fixture data dir)
# ============================================================


class TestApi:
    def test_get_supported_models(self, registry_data_dir, monkeypatch):
        from transformer_lens.tools.model_registry import api

        monkeypatch.setattr(api, "_DATA_DIR", registry_data_dir)
        api.clear_cache()

        models = api.get_supported_models()
        assert len(models) == 3

    def test_get_supported_models_filter_arch(self, registry_data_dir, monkeypatch):
        from transformer_lens.tools.model_registry import api

        monkeypatch.setattr(api, "_DATA_DIR", registry_data_dir)
        api.clear_cache()

        models = api.get_supported_models(architecture="LlamaForCausalLM")
        assert len(models) == 1
        assert models[0].model_id == "meta-llama/Llama-2-7b-hf"

    def test_get_supported_models_verified_only(self, registry_data_dir, monkeypatch):
        from transformer_lens.tools.model_registry import api

        monkeypatch.setattr(api, "_DATA_DIR", registry_data_dir)
        api.clear_cache()

        models = api.get_supported_models(verified_only=True)
        assert len(models) == 2

    def test_is_model_supported(self, registry_data_dir, monkeypatch):
        from transformer_lens.tools.model_registry import api

        monkeypatch.setattr(api, "_DATA_DIR", registry_data_dir)
        api.clear_cache()

        assert api.is_model_supported("openai-community/gpt2")
        assert not api.is_model_supported("nonexistent/model")

    def test_get_registry_stats(self, registry_data_dir, monkeypatch):
        from transformer_lens.tools.model_registry import api

        monkeypatch.setattr(api, "_DATA_DIR", registry_data_dir)
        api.clear_cache()

        stats = api.get_registry_stats()
        assert stats["total_supported_models"] == 3
        assert stats["total_verified"] == 1
        assert stats["total_unsupported_architectures"] == 1

    def test_get_model_info_not_found(self, registry_data_dir, monkeypatch):
        from transformer_lens.tools.model_registry import api
        from transformer_lens.tools.model_registry.exceptions import ModelNotFoundError

        monkeypatch.setattr(api, "_DATA_DIR", registry_data_dir)
        api.clear_cache()

        with pytest.raises(ModelNotFoundError):
            api.get_model_info("nonexistent/model")

    def test_get_architecture_models(self, registry_data_dir, monkeypatch):
        from transformer_lens.tools.model_registry import api

        monkeypatch.setattr(api, "_DATA_DIR", registry_data_dir)
        api.clear_cache()

        models = api.get_architecture_models("GPT2LMHeadModel")
        assert len(models) == 2
        assert "openai-community/gpt2" in models

    def test_get_supported_architectures(self, registry_data_dir, monkeypatch):
        from transformer_lens.tools.model_registry import api

        monkeypatch.setattr(api, "_DATA_DIR", registry_data_dir)
        api.clear_cache()

        archs = api.get_supported_architectures()
        assert "GPT2LMHeadModel" in archs
        assert "LlamaForCausalLM" in archs
        assert len(archs) == 2


# ============================================================
# Test validate.py
# ============================================================


class TestValidate:
    def test_validate_valid_supported_models(self, registry_data_dir):
        from transformer_lens.tools.model_registry.validate import validate_json_schema

        result = validate_json_schema(
            registry_data_dir / "supported_models.json", "supported_models"
        )
        assert result.valid
        assert result.error_count == 0

    def test_validate_valid_architecture_gaps(self, registry_data_dir):
        from transformer_lens.tools.model_registry.validate import validate_json_schema

        result = validate_json_schema(
            registry_data_dir / "architecture_gaps.json", "architecture_gaps"
        )
        assert result.valid

    def test_validate_valid_verification_history(self, registry_data_dir):
        from transformer_lens.tools.model_registry.validate import validate_json_schema

        result = validate_json_schema(
            registry_data_dir / "verification_history.json", "verification_history"
        )
        assert result.valid

    def test_validate_invalid_missing_required_field(self):
        from transformer_lens.tools.model_registry.validate import (
            validate_supported_models_report,
        )

        invalid_data = {"models": []}  # missing generated_at, totals
        result = validate_supported_models_report(invalid_data)
        assert not result.valid
        assert result.error_count > 0

    def test_validate_invalid_model_entry(self):
        from transformer_lens.tools.model_registry.validate import _validate_model_entry

        errors = _validate_model_entry(
            {"architecture_id": "", "model_id": "ok", "status": 5},
            "test",
        )
        # architecture_id too short and status > 3
        assert len(errors) >= 2


# ============================================================
# Test verify_models.py: _sanitize_note
# ============================================================


class TestSanitizeNote:
    def test_none_input(self):
        from transformer_lens.tools.model_registry.verify_models import _sanitize_note

        assert _sanitize_note(None) is None

    def test_strips_hf_token(self):
        from transformer_lens.tools.model_registry.verify_models import _sanitize_note

        note = "Error with token hf_abcdefghijklmnopqrstuvwx in request"
        result = _sanitize_note(note)
        assert "hf_abcdefghijklmnopqrstuvwx" not in result
        assert "HF_TOKEN" in result

    def test_gated_repo_message(self):
        from transformer_lens.tools.model_registry.verify_models import _sanitize_note

        note = (
            "Access denied: gated repo at "
            "https://huggingface.co/meta-llama/Llama-2-7b-hf please accept terms"
        )
        result = _sanitize_note(note)
        assert result == "Config unavailable: Gated repo (meta-llama/Llama-2-7b-hf)"

    def test_plain_note_unchanged(self):
        from transformer_lens.tools.model_registry.verify_models import _sanitize_note

        note = "Estimated 48 GB exceeds 16 GB limit"
        assert _sanitize_note(note) == note


# ============================================================
# Test registry_io.py
# ============================================================


class TestRegistryIO:
    def test_update_model_status_existing(self, registry_data_dir, monkeypatch):
        from transformer_lens.tools.model_registry import registry_io

        monkeypatch.setattr(
            registry_io,
            "_SUPPORTED_MODELS_PATH",
            registry_data_dir / "supported_models.json",
        )

        result = registry_io.update_model_status(
            model_id="sshleifer/tiny-gpt2",
            arch_id="GPT2LMHeadModel",
            status=1,
            phase_scores={1: 100.0, 2: 95.0, 3: 90.0},
        )
        assert result is True

        with open(registry_data_dir / "supported_models.json") as f:
            data = json.load(f)
        entry = next(m for m in data["models"] if m["model_id"] == "sshleifer/tiny-gpt2")
        assert entry["status"] == 1
        assert entry["phase1_score"] == 100.0
        assert data["total_verified"] == 3  # was 2 verified, now 3

    def test_update_model_status_not_found_non_verified(self, registry_data_dir, monkeypatch):
        from transformer_lens.tools.model_registry import registry_io

        monkeypatch.setattr(
            registry_io,
            "_SUPPORTED_MODELS_PATH",
            registry_data_dir / "supported_models.json",
        )

        result = registry_io.update_model_status(
            model_id="nonexistent/model",
            arch_id="UnknownArch",
            status=3,  # FAILED -- should not add
        )
        assert result is False

    def test_update_model_status_adds_verified_if_missing(self, registry_data_dir, monkeypatch):
        from transformer_lens.tools.model_registry import registry_io

        monkeypatch.setattr(
            registry_io,
            "_SUPPORTED_MODELS_PATH",
            registry_data_dir / "supported_models.json",
        )

        result = registry_io.update_model_status(
            model_id="brand-new/model",
            arch_id="GPT2LMHeadModel",
            status=1,  # VERIFIED -- should add
            phase_scores={1: 100.0},
        )
        assert result is True
        with open(registry_data_dir / "supported_models.json") as f:
            data = json.load(f)
        assert data["total_models"] == 4

    def test_add_verification_record(self, registry_data_dir, monkeypatch):
        from transformer_lens.tools.model_registry import registry_io

        monkeypatch.setattr(
            registry_io,
            "_VERIFICATION_HISTORY_PATH",
            registry_data_dir / "verification_history.json",
        )

        registry_io.add_verification_record(
            model_id="sshleifer/tiny-gpt2",
            arch_id="GPT2LMHeadModel",
            notes="Test verification",
            verified_by="unit_test",
        )

        with open(registry_data_dir / "verification_history.json") as f:
            data = json.load(f)
        assert len(data["records"]) == 2  # was 1, now 2
        assert data["records"][-1]["model_id"] == "sshleifer/tiny-gpt2"
        assert data["records"][-1]["verified_by"] == "unit_test"

    def test_update_model_status_with_sanitize(self, registry_data_dir, monkeypatch):
        from transformer_lens.tools.model_registry import registry_io

        monkeypatch.setattr(
            registry_io,
            "_SUPPORTED_MODELS_PATH",
            registry_data_dir / "supported_models.json",
        )

        def fake_sanitize(note):
            return "SANITIZED"

        registry_io.update_model_status(
            model_id="openai-community/gpt2",
            arch_id="GPT2LMHeadModel",
            status=3,
            note="raw note with hf_secrettoken12345678901",
            sanitize_fn=fake_sanitize,
        )
        with open(registry_data_dir / "supported_models.json") as f:
            data = json.load(f)
        entry = next(m for m in data["models"] if m["model_id"] == "openai-community/gpt2")
        assert entry["note"] == "SANITIZED"


# ============================================================
# Test alias_drift.py
# ============================================================


class TestAliasDrift:
    def test_check_drift_finds_aliases_not_in_registry(self, registry_data_dir, monkeypatch):
        from transformer_lens.tools.model_registry import registry_io
        from transformer_lens.tools.model_registry.alias_drift import check_drift

        monkeypatch.setattr(
            registry_io,
            "_SUPPORTED_MODELS_PATH",
            registry_data_dir / "supported_models.json",
        )

        report = check_drift()
        # MODEL_ALIASES has ~258 entries; the fixture registry has 3 models
        # So most aliases should show up as "in aliases not in registry"
        assert len(report.in_aliases_not_registry) > 0
        assert report.has_drift

    def test_drift_report_serialization(self):
        from transformer_lens.tools.model_registry.alias_drift import DriftReport

        report = DriftReport(
            in_aliases_not_registry=["model-a"],
            in_registry_not_aliases=["model-b"],
        )
        d = report.to_dict()
        assert d["summary"]["aliases_only"] == 1
        assert d["summary"]["registry_only"] == 1
        assert d["has_drift"] is True

    def test_no_drift_report(self):
        from transformer_lens.tools.model_registry.alias_drift import DriftReport

        report = DriftReport()
        assert not report.has_drift
        d = report.to_dict()
        assert d["has_drift"] is False
