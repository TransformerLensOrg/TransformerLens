"""Component-level benchmarks to compare individual model pieces.

This module provides benchmarks for comparing individual model components
(attention, MLP, embedding, etc.) between HuggingFace and TransformerBridge.
"""

from typing import Any, Optional

from transformer_lens.benchmarks.component_outputs import ComponentBenchmarker
from transformer_lens.benchmarks.utils import BenchmarkResult, BenchmarkSeverity


def benchmark_all_components(
    bridge,
    hf_model,
    atol: float = 1e-4,
    rtol: float = 1e-4,
    reference_model: Optional[Any] = None,
) -> BenchmarkResult:
    """Comprehensive benchmark of all model components.

    This function systematically tests every component in the model using the
    architecture adapter to find and compare equivalent components.

    Args:
        bridge: The TransformerBridge model
        hf_model: The HuggingFace model to compare against
        atol: Absolute tolerance for comparison
        rtol: Relative tolerance for comparison
        reference_model: Optional reference model (unused, for API consistency)

    Returns:
        BenchmarkResult summarizing all component tests
    """
    try:
        # Set up component testing (e.g., sync rotary_emb references for Gemma models, eager attention)
        # This must be called before creating the ComponentBenchmarker
        bridge.adapter.setup_component_testing(hf_model, bridge_model=bridge)

        # Create benchmarker
        benchmarker = ComponentBenchmarker(
            bridge_model=bridge,
            hf_model=hf_model,
            adapter=bridge.adapter,
            cfg=bridge.cfg,
            atol=atol,
            rtol=rtol,
        )

        # Skip vision components for multimodal models — they require image
        # inputs that isolated text-based component testing cannot provide.
        # Vision components are validated separately in Phase 7.
        skip_components = []
        if getattr(bridge.cfg, "is_multimodal", False):
            skip_components = ["vision_encoder", "vision_projector"]

        # Run comprehensive benchmark
        report = benchmarker.benchmark_all_components(skip_components=skip_components)

        # Convert to BenchmarkResult format
        if report.failed_components == 0:
            return BenchmarkResult(
                name="all_components",
                severity=BenchmarkSeverity.INFO,
                passed=True,
                message=f"All {report.total_components} components produce equivalent outputs",
                details={
                    "total_components": report.total_components,
                    "pass_rate": report.pass_rate,
                    "component_types": report.get_component_type_summary(),
                },
            )
        else:
            # Get failure details
            failures_by_severity = report.get_failure_by_severity()

            # Determine overall severity
            if failures_by_severity["critical"]:
                severity = BenchmarkSeverity.ERROR
            elif failures_by_severity["high"]:
                severity = BenchmarkSeverity.DANGER
            else:
                severity = BenchmarkSeverity.WARNING

            # Create failure message
            failure_summary = []
            for sev in ["critical", "high", "medium", "low"]:
                count = len(failures_by_severity[sev])
                if count > 0:
                    failure_summary.append(f"{count} {sev}")

            message = (
                f"{report.failed_components}/{report.total_components} components failed "
                f"({', '.join(failure_summary)})"
            )

            # Collect failed component details
            failed_details = {}
            for result in report.component_results:
                if not result.passed:
                    failed_details[result.component_path] = {
                        "max_diff": result.max_diff,
                        "mean_diff": result.mean_diff,
                        "severity": result.get_failure_severity(),
                        "error": result.error_message,
                    }

            return BenchmarkResult(
                name="all_components",
                passed=False,
                severity=severity,
                message=message,
                details={
                    "total_components": report.total_components,
                    "passed_components": report.passed_components,
                    "failed_components": report.failed_components,
                    "pass_rate": report.pass_rate,
                    "failures": failed_details,
                },
            )

    except Exception as e:
        return BenchmarkResult(
            name="all_components",
            passed=False,
            severity=BenchmarkSeverity.ERROR,
            message=f"Error running comprehensive component benchmark: {str(e)}",
            details={"exception": str(e)},
        )
