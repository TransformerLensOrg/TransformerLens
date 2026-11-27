"""Report assembly and adapter loading for the adapter coverage runner (no model loads)."""

import subprocess
from pathlib import Path
from typing import Iterator

import pytest

from transformer_lens.benchmarks import adapter_coverage_runner as runner
from transformer_lens.benchmarks.utils import BenchmarkResult, BenchmarkSeverity
from transformer_lens.factories import architecture_adapter_factory as factory_module

_ADAPTER_SOURCE = """
from transformer_lens.model_bridge.architecture_adapter import ArchitectureAdapter


class _SharedBase(ArchitectureAdapter):
    pass


class DraftArchitectureAdapter(_SharedBase):
    pass
"""


def _result(
    name: str,
    blocking: tuple[tuple[str, bool], ...] = (),
    gaps: tuple[tuple[str, str], ...] = (),
    skipped: bool = False,
    **details: object,
) -> BenchmarkResult:
    failed = any(not ok for _, ok in blocking)
    if skipped:
        severity = BenchmarkSeverity.SKIPPED
    elif failed:
        severity = BenchmarkSeverity.DANGER
    else:
        severity = BenchmarkSeverity.WARNING if gaps else BenchmarkSeverity.INFO
    return BenchmarkResult(
        name=name,
        severity=severity,
        message=name,
        passed=not failed,
        details={
            **details,
            "blocking": [
                {"part": part, "status": "pass" if ok else "fail", "message": f"{part} is off"}
                for part, ok in blocking
            ],
            "gaps": [{"kind": kind, "where": where} for kind, where in gaps],
        },
    )


@pytest.fixture
def restore_factory() -> Iterator[None]:
    factory = factory_module.ArchitectureAdapterFactory
    adapters = dict(factory._adapters)
    supported = dict(factory_module.SUPPORTED_ARCHITECTURES)
    yield
    factory._adapters.clear()
    factory._adapters.update(adapters)
    factory_module.SUPPORTED_ARCHITECTURES.clear()
    factory_module.SUPPORTED_ARCHITECTURES.update(supported)


def test_summarize_separates_blocking_findings_from_gaps() -> None:
    report = runner.summarize(
        [
            _result("hook_consistency", blocking=(("logit_parity", True), ("residual_sum", False))),
            _result(
                "structural_coverage",
                gaps=(("unmapped", "model.tower"), ("generic", "model.layers.{i}.mlp.fc")),
                pattern_coverage={"share": 0.5},
            ),
            _result("residual_floor", blocking=(("floor", True),)),
            _result("reference_comparison", skipped=True),
        ],
        {"model.tower": "vision input is out of scope", "model.audio": "never matched"},
        checkpoint="org/model",
    )
    assert [check["name"] for check in report["checks"]] == [
        "structural_coverage",
        "residual_floor",
        "hook_consistency",
        "reference_comparison",
    ]
    assert [check["status"] for check in report["checks"]] == ["gaps", "pass", "fail", "skip"]
    assert report["passed"] is False
    assert report["blocking_findings"] == [
        {"check": "hook_consistency", "part": "residual_sum", "message": "residual_sum is off"}
    ]
    assert (report["n_blocking_parts"], report["n_blocking_failed"]) == (3, 1)
    assert (report["coverage"], report["coverage_basis"]) == (0.5, "ownership")
    assert report["gaps"] == [
        {
            "kind": "unmapped",
            "where": "model.tower",
            "declared": True,
            "reason": "vision input is out of scope",
        },
        {"kind": "generic", "where": "model.layers.{i}.mlp.fc", "declared": False, "reason": None},
    ]
    assert report["stale_declarations"] == ["model.audio"]
    assert report["checkpoint"] == "org/model"
    assert "coverage_score" not in report


def test_gaps_alone_do_not_fail_and_a_reference_sets_the_coverage_basis() -> None:
    report = runner.summarize(
        [
            _result("structural_coverage", pattern_coverage={"share": 0.25}),
            _result(
                "reference_comparison",
                gaps=(("missing_from_reference", "blocks.{i}.attn.hook_out"),),
                coverage=0.8,
            ),
        ]
    )
    assert report["passed"] is True and report["blocking_findings"] == []
    assert (report["coverage"], report["coverage_basis"]) == (0.8, "reference")


def test_report_details_survive_strict_json() -> None:
    import json

    result = _result("hook_consistency")
    result.details = {"error": float("nan"), "names": {"b", "a"}, "nested": {1: (2.0, None)}}
    details = runner.summarize([result])["checks"][0]["details"]
    assert details == {"error": "nan", "names": ["a", "b"], "nested": {"1": [2.0, None]}}
    json.dumps(details, allow_nan=False)


def test_load_adapter_class_picks_the_leaf_class(tmp_path: Path) -> None:
    source = tmp_path / "draft.py"
    source.write_text(_ADAPTER_SOURCE)
    assert runner.load_adapter_class(str(source)).__name__ == "DraftArchitectureAdapter"


def test_load_adapter_class_rejects_a_file_with_two_adapters(tmp_path: Path) -> None:
    source = tmp_path / "draft.py"
    source.write_text(_ADAPTER_SOURCE + "\n\nclass Second(ArchitectureAdapter):\n    pass\n")
    with pytest.raises(ValueError, match="exactly one adapter class"):
        runner.load_adapter_class(str(source))


def test_load_adapter_class_accepts_an_importable_module() -> None:
    loaded = runner.load_adapter_class("transformer_lens.model_bridge.supported_architectures.gpt2")
    assert loaded.__name__ == "GPT2ArchitectureAdapter"


def test_register_adapter_replaces_the_selected_adapter(
    tmp_path: Path, restore_factory: None
) -> None:
    source = tmp_path / "draft.py"
    source.write_text(_ADAPTER_SOURCE)
    draft = runner.load_adapter_class(str(source))
    runner.register_adapter("GPT2LMHeadModel", draft)
    assert factory_module.ArchitectureAdapterFactory._adapters["GPT2LMHeadModel"] is draft
    assert factory_module.SUPPORTED_ARCHITECTURES["GPT2LMHeadModel"] is draft


def test_an_adapter_that_cannot_load_is_one_blocking_finding(tmp_path: Path) -> None:
    broken = tmp_path / "draft.py"
    broken.write_text("raise RuntimeError('boom')\n")
    report = runner.run_checks_in_process(
        "org/model", adapter_path=str(broken), hf_class="GPT2LMHeadModel"
    )
    assert report["passed"] is False
    assert [finding["check"] for finding in report["blocking_findings"]] == ["boot"]
    assert "boom" in report["blocking_findings"][0]["message"]
    assert (report["n_blocking_parts"], report["n_blocking_failed"]) == (1, 1)
    assert report["coverage"] is None and report["coverage_basis"] is None


def test_the_hf_class_is_read_from_the_checkpoint_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from transformers import GPT2Config

    named = tmp_path / "named"
    GPT2Config(n_layer=1, n_head=1, n_embd=8, architectures=["GPT2Model"]).save_pretrained(named)
    assert runner._checkpoint_class(str(named), trust_remote_code=False) == "GPT2Model"

    # A config that names no class falls back to the library's model_type mapping.
    unnamed = tmp_path / "unnamed"
    GPT2Config(n_layer=1, n_head=1, n_embd=8).save_pretrained(unnamed)
    assert runner._checkpoint_class(str(unnamed), trust_remote_code=False) == "GPT2LMHeadModel"

    def unknown(config: object) -> str:
        raise ValueError("unknown model_type")

    monkeypatch.setattr(runner, "determine_architecture_from_hf_config", unknown)
    with pytest.raises(ValueError, match="pass the HuggingFace class explicitly"):
        runner._checkpoint_class(str(unnamed), trust_remote_code=False)


def test_a_run_that_times_out_is_reported_as_a_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    def time_out(command: list, **kwargs: object) -> None:
        raise subprocess.TimeoutExpired(command, 1.0)

    monkeypatch.setattr(subprocess, "run", time_out)
    output = tmp_path / "report.json"
    report = runner.run_adapter_coverage("org/model", output_path=str(output), timeout=1.0)
    assert report["passed"] is False
    assert "timed out" in report["blocking_findings"][0]["message"]
    assert output.exists()
