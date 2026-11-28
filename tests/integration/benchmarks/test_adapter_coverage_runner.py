"""End-to-end adapter coverage runs on a tiny Hub checkpoint, each in its own subprocess."""

import json
from pathlib import Path

from tests.tiny_checkpoints import TINY_CHECKPOINTS
from transformer_lens.benchmarks import adapter_coverage_runner as runner
from transformer_lens.factories import architecture_adapter_factory as factory_module
from transformer_lens.model_bridge.supported_architectures import gpt2

CHECKPOINT = str(TINY_CHECKPOINTS["gpt2"]["name"])
HF_CLASS = "GPT2LMHeadModel"
REFERENCE = "transformer_lens.model_bridge.supported_architectures.gpt2"

_SHELL_ADAPTER = """
from transformer_lens.model_bridge.generalized_components import BlockBridge
from transformer_lens.model_bridge.supported_architectures.gpt2 import GPT2ArchitectureAdapter


class ShellGPT2ArchitectureAdapter(GPT2ArchitectureAdapter):
    def __init__(self, cfg):
        super().__init__(cfg)
        blocks = self.component_mapping["blocks"]
        self.component_mapping["blocks"] = BlockBridge(name=blocks.name, config=self.cfg)
"""


def _statuses(report: dict) -> dict:
    return {check["name"]: check["status"] for check in report["checks"]}


def test_the_repo_adapter_passes_and_matches_itself_as_reference(tmp_path: Path) -> None:
    output = tmp_path / "report.json"
    exit_code = runner.main(
        [
            "--checkpoint",
            CHECKPOINT,
            "--adapter",
            str(Path(gpt2.__file__)),
            "--reference-adapter",
            REFERENCE,
            "--output",
            str(output),
        ]
    )
    report = json.loads(output.read_text())
    assert exit_code == 0, report["blocking_findings"]
    assert report["passed"] is True and report["n_blocking_failed"] == 0
    statuses = _statuses(report)
    for check in ("structural_coverage", "residual_floor", "hook_consistency"):
        assert statuses[check] == "pass"
    assert statuses["reference_comparison"] == "pass"
    assert (report["coverage"], report["coverage_basis"]) == (1.0, "reference")
    assert set(report["checks"][0]) == {"name", "status", "message", "details"}
    # --hf-class was omitted; it was read from the checkpoint's config.
    assert report["hf_class"] == HF_CLASS


def test_a_shell_adapter_fails_and_leaves_the_registry_alone(tmp_path: Path) -> None:
    shell = tmp_path / "shell_gpt2.py"
    shell.write_text(_SHELL_ADAPTER)
    registered = factory_module.SUPPORTED_ARCHITECTURES[HF_CLASS]
    output = tmp_path / "report.json"
    exit_code = runner.main(
        [
            "--checkpoint",
            CHECKPOINT,
            "--adapter",
            str(shell),
            "--hf-class",
            HF_CLASS,
            "--reference-adapter",
            REFERENCE,
            "--output",
            str(output),
        ]
    )
    report = json.loads(output.read_text())
    assert exit_code == 1
    statuses = _statuses(report)
    # Leaving the blocks' insides unmapped is a gap; having no hooked account of what
    # each block adds to the residual stream is what blocks.
    assert statuses["structural_coverage"] == "gaps"
    assert statuses["residual_floor"] == "fail"
    assert statuses["reference_comparison"] == "gaps"
    assert "residual_floor" in {finding["check"] for finding in report["blocking_findings"]}
    assert report["coverage_basis"] == "reference" and 0 < report["coverage"] < 1
    kinds = {gap["kind"] for gap in report["gaps"]}
    assert {"unmapped", "absent_hook", "missing_from_reference"} <= kinds
    missing = [g["where"] for g in report["gaps"] if g["kind"] == "missing_from_reference"]
    assert "blocks.{i}.attn.hook_out" in missing
    # The shell adapter was registered in the subprocess only.
    assert factory_module.SUPPORTED_ARCHITECTURES[HF_CLASS] is registered
