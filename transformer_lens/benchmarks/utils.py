"""Utility types and functions for benchmarking."""

from dataclasses import dataclass
from enum import Enum
from typing import Any, Collection, Dict, List, Optional, Union

import torch

# Hook patterns that bridge models inherently don't have because they use HF's
# native implementation rather than reimplementing attention/MLP internals.
BRIDGE_EXPECTED_MISSING_PATTERNS = [
    "mlp.hook_pre",
    "mlp.hook_post",
    "hook_mlp_in",
    "hook_mlp_out",
    "attn.hook_rot_q",
    "attn.hook_rot_k",
    "hook_pos_embed",
    "embed.ln.hook_scale",
    "embed.ln.hook_normalized",
    "attn.hook_q",
    "attn.hook_k",
    "attn.hook_v",
    "hook_q_input",
    "hook_k_input",
    "hook_v_input",
    "attn.hook_attn_scores",
    "attn.hook_pattern",
    # MoE per-expert hooks: Bridge uses HF's batched MoE forward pass via MoEBridge,
    # which wraps the entire MoE module. HookedTransformer creates individual expert
    # modules with per-expert hooks (e.g., blocks.0.mlp.experts.3.hook_pre).
    "mlp.experts.",
    "mlp.hook_experts",
    "mlp.hook_expert_indices",
    "mlp.hook_expert_weights",
]


def filter_expected_missing_hooks(hook_names: Collection[str]) -> list[str]:
    """Filter out hook names that bridge models are expected to be missing."""
    return [
        h
        for h in hook_names
        if not any(pattern in h for pattern in BRIDGE_EXPECTED_MISSING_PATTERNS)
    ]


def safe_allclose(
    tensor1: torch.Tensor,
    tensor2: torch.Tensor,
    atol: float = 1e-5,
    rtol: float = 1e-5,
) -> bool:
    """torch.allclose that handles dtype and device mismatches."""
    if tensor1.device != tensor2.device:
        tensor1 = tensor1.cpu()
        tensor2 = tensor2.cpu()
    if tensor1.dtype != tensor2.dtype:
        tensor1 = tensor1.to(torch.float32)
        tensor2 = tensor2.to(torch.float32)
    return torch.allclose(tensor1, tensor2, atol=atol, rtol=rtol)


class BenchmarkSeverity(Enum):
    """Severity levels for benchmark results."""

    INFO = "info"  # ✅ PASS - Model working perfectly, all checks passed
    WARNING = "warning"  # ⚠️ PASS with notes - Acceptable differences worth noting
    DANGER = "danger"  # ❌ FAIL - Significant mismatches or failures
    ERROR = "error"  # ❌ ERROR - Test crashed or couldn't run
    SKIPPED = "skipped"  # ⏭️ SKIPPED - Test skipped (e.g., no reference model available)


@dataclass
class BenchmarkResult:
    """Result of a benchmark test."""

    name: str
    severity: BenchmarkSeverity
    message: str
    details: Optional[Dict[str, Any]] = None
    passed: bool = True
    phase: Optional[int] = None  # Phase number (1, 2, 3, etc.)

    def __str__(self) -> str:
        """Format result for console output."""
        severity_icons = {
            BenchmarkSeverity.INFO: "🟢",
            BenchmarkSeverity.WARNING: "🟡",
            BenchmarkSeverity.DANGER: "🔴",
            BenchmarkSeverity.ERROR: "❌",
            BenchmarkSeverity.SKIPPED: "⏭️",
        }
        icon = severity_icons[self.severity]

        if self.severity == BenchmarkSeverity.SKIPPED:
            status = "SKIPPED"
        else:
            status = "PASS" if self.passed else "FAIL"

        result = f"{icon} [{status}] {self.name}: {self.message}"

        if self.details:
            detail_lines = []
            for key, value in self.details.items():
                detail_lines.append(f"  {key}: {value}")
            result += "\n" + "\n".join(detail_lines)

        return result

    def print_immediate(self) -> None:
        """Print this result immediately to console."""
        print(str(self))


@dataclass
class PhaseReferenceData:
    """Float32 reference data from Phase 1 for Phase 3 equivalence comparison."""

    hf_logits: Optional[torch.Tensor] = None
    hf_loss: Optional[float] = None
    test_text: Optional[str] = None


def compare_tensors(
    tensor1: torch.Tensor,
    tensor2: torch.Tensor,
    atol: float = 1e-5,
    rtol: float = 1e-5,
    name: str = "tensors",
) -> BenchmarkResult:
    """Compare two tensors and return a benchmark result.

    Args:
        tensor1: First tensor
        tensor2: Second tensor
        atol: Absolute tolerance
        rtol: Relative tolerance
        name: Name of the comparison

    Returns:
        BenchmarkResult with comparison details
    """
    # Check shapes
    if tensor1.shape != tensor2.shape:
        return BenchmarkResult(
            name=name,
            severity=BenchmarkSeverity.DANGER,
            message=f"Shape mismatch: {tensor1.shape} vs {tensor2.shape}",
            passed=False,
        )

    if tensor1.device != tensor2.device:
        tensor1 = tensor1.cpu()
        tensor2 = tensor2.cpu()

    if tensor1.dtype != tensor2.dtype:
        tensor1 = tensor1.to(torch.float32)
        tensor2 = tensor2.to(torch.float32)

    if torch.allclose(tensor1, tensor2, atol=atol, rtol=rtol):
        return BenchmarkResult(
            name=name,
            severity=BenchmarkSeverity.INFO,
            message="Tensors match within tolerance",
            details={"atol": atol, "rtol": rtol},
        )

    diff = torch.abs(tensor1 - tensor2)
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    rel_diff = diff / (torch.abs(tensor1) + 1e-10)
    mean_rel = rel_diff.mean().item()

    return BenchmarkResult(
        name=name,
        severity=BenchmarkSeverity.DANGER,
        message=f"Tensors differ: max_diff={max_diff:.6f}, mean_rel={mean_rel:.6f}",
        details={
            "max_diff": max_diff,
            "mean_diff": mean_diff,
            "mean_rel": mean_rel,
            "atol": atol,
            "rtol": rtol,
        },
        passed=False,
    )


def compare_scalars(
    scalar1: Union[float, int],
    scalar2: Union[float, int],
    atol: float = 1e-5,
    name: str = "scalars",
) -> BenchmarkResult:
    """Compare two scalar values and return a benchmark result.

    Args:
        scalar1: First scalar
        scalar2: Second scalar
        atol: Absolute tolerance
        name: Name of the comparison

    Returns:
        BenchmarkResult with comparison details
    """
    diff = abs(float(scalar1) - float(scalar2))

    if diff < atol:
        return BenchmarkResult(
            name=name,
            severity=BenchmarkSeverity.INFO,
            message=f"Scalars match: {scalar1:.6f} ≈ {scalar2:.6f}",
            details={"diff": diff, "atol": atol},
        )
    else:
        return BenchmarkResult(
            name=name,
            severity=BenchmarkSeverity.DANGER,
            message=f"Scalars differ: {scalar1:.6f} vs {scalar2:.6f}",
            details={"diff": diff, "atol": atol},
            passed=False,
        )


def format_results(results: List[BenchmarkResult]) -> str:
    """Format a list of benchmark results for console output.

    Args:
        results: List of benchmark results

    Returns:
        Formatted string for console output
    """
    output = []
    output.append("=" * 80)
    output.append("BENCHMARK RESULTS")
    output.append("=" * 80)

    # Count by severity
    severity_counts = {
        BenchmarkSeverity.INFO: 0,
        BenchmarkSeverity.WARNING: 0,
        BenchmarkSeverity.DANGER: 0,
        BenchmarkSeverity.ERROR: 0,
        BenchmarkSeverity.SKIPPED: 0,
    }

    passed = 0
    failed = 0
    skipped = 0

    for result in results:
        severity_counts[result.severity] += 1
        if result.severity == BenchmarkSeverity.SKIPPED:
            skipped += 1
        elif result.passed:
            passed += 1
        else:
            failed += 1

    # Summary
    total = len(results)
    run_tests = total - skipped
    output.append(f"\nTotal: {total} tests")
    if skipped > 0:
        output.append(f"Run: {run_tests} tests")
        output.append(f"Skipped: {skipped} tests")
    if run_tests > 0:
        output.append(f"Passed: {passed} ({passed/run_tests*100:.1f}%)")
        output.append(f"Failed: {failed} ({failed/run_tests*100:.1f}%)")
    output.append("")
    output.append(f"🟢 INFO: {severity_counts[BenchmarkSeverity.INFO]}")
    output.append(f"🟡 WARNING: {severity_counts[BenchmarkSeverity.WARNING]}")
    output.append(f"🔴 DANGER: {severity_counts[BenchmarkSeverity.DANGER]}")
    output.append(f"❌ ERROR: {severity_counts[BenchmarkSeverity.ERROR]}")
    if skipped > 0:
        output.append(f"⏭️ SKIPPED: {severity_counts[BenchmarkSeverity.SKIPPED]}")
    output.append("")
    output.append("-" * 80)

    # Individual results
    for result in results:
        output.append(str(result))
        output.append("")

    output.append("=" * 80)

    return "\n".join(output)
