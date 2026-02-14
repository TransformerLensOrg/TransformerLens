"""Component-level benchmarks to compare individual model pieces.

This module provides benchmarks for comparing individual model components
(attention, MLP, embedding, etc.) between HuggingFace and TransformerBridge.
"""

from typing import Any, Optional

import torch

from transformer_lens.benchmarks.component_outputs import ComponentBenchmarker
from transformer_lens.benchmarks.utils import BenchmarkResult, BenchmarkSeverity


def benchmark_component_forward(
    bridge_component: Any,
    hf_component: Any,
    test_input: torch.Tensor,
    component_name: str,
    atol: float = 1e-4,
    rtol: float = 1e-4,
) -> BenchmarkResult:
    """Benchmark forward pass equivalence for a single component.

    Args:
        bridge_component: The bridge component to test
        hf_component: The HuggingFace component to compare against
        test_input: Input tensor for the component
        component_name: Name of the component being tested
        atol: Absolute tolerance for comparison
        rtol: Relative tolerance for comparison

    Returns:
        BenchmarkResult with the comparison results
    """
    try:
        # Run both components
        with torch.no_grad():
            bridge_output = bridge_component(test_input)
            hf_output = hf_component(test_input)

        # Extract tensors from outputs (handle both tensor and tuple outputs)
        if isinstance(bridge_output, tuple):
            bridge_tensor = bridge_output[0]
        else:
            bridge_tensor = bridge_output

        if isinstance(hf_output, tuple):
            hf_tensor = hf_output[0]
        else:
            hf_tensor = hf_output

        # Compare outputs
        if not torch.allclose(bridge_tensor, hf_tensor, atol=atol, rtol=rtol):
            max_diff = (bridge_tensor - hf_tensor).abs().max().item()
            mean_diff = (bridge_tensor - hf_tensor).abs().mean().item()

            return BenchmarkResult(
                name=f"{component_name}_forward",
                passed=False,
                severity=BenchmarkSeverity.DANGER,
                message=f"Component {component_name} outputs differ: max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f}",
                details={
                    "max_diff": max_diff,
                    "mean_diff": mean_diff,
                    "bridge_mean": bridge_tensor.mean().item(),
                    "hf_mean": hf_tensor.mean().item(),
                },
            )

        return BenchmarkResult(
            name=f"{component_name}_forward",
            passed=True,
            severity=BenchmarkSeverity.INFO,
            message=f"Component {component_name} produces equivalent outputs",
        )

    except Exception as e:
        return BenchmarkResult(
            name=f"{component_name}_forward",
            passed=False,
            severity=BenchmarkSeverity.ERROR,
            message=f"Error testing component {component_name}: {str(e)}",
            details={"exception": str(e)},
        )


def benchmark_block_components(
    bridge,
    hf_model,
    block_idx: int = 0,
    atol: float = 1e-4,
) -> list[BenchmarkResult]:
    """Benchmark all components within a transformer block.

    Args:
        bridge: The TransformerBridge model
        hf_model: The HuggingFace model
        block_idx: Which block to test (default: 0)
        atol: Absolute tolerance for comparison

    Returns:
        List of BenchmarkResult for each component in the block
    """
    results = []

    try:
        # Create test input
        batch_size = 1
        seq_len = 4
        d_model = bridge.cfg.d_model
        test_input = torch.randn(batch_size, seq_len, d_model, device=bridge.cfg.device)

        # Get the blocks
        bridge_block = bridge.blocks[block_idx]

        # Get HF block using adapter
        hf_blocks_path = bridge.adapter.component_mapping.blocks.name
        hf_blocks = hf_model
        for part in hf_blocks_path.split("."):
            hf_blocks = getattr(hf_blocks, part)
        hf_block = hf_blocks[block_idx]

        # Test attention component
        if hasattr(bridge_block, "attn"):
            bridge_attn = bridge_block.attn

            # Get HF attention
            attn_path = bridge.adapter.component_mapping.blocks.submodules["attn"].name
            if attn_path:
                hf_attn = hf_block
                for part in attn_path.split("."):
                    hf_attn = getattr(hf_attn, part)

                results.append(
                    benchmark_component_forward(
                        bridge_attn,
                        hf_attn,
                        test_input,
                        f"block_{block_idx}_attn",
                        atol=atol,
                    )
                )

        # Test MLP component
        if hasattr(bridge_block, "mlp"):
            bridge_mlp = bridge_block.mlp

            # Get HF MLP
            mlp_path = bridge.adapter.component_mapping.blocks.submodules["mlp"].name
            if mlp_path:
                hf_mlp = hf_block
                for part in mlp_path.split("."):
                    hf_mlp = getattr(hf_mlp, part)

                results.append(
                    benchmark_component_forward(
                        bridge_mlp,
                        hf_mlp,
                        test_input,
                        f"block_{block_idx}_mlp",
                        atol=atol,
                    )
                )

        # Test layer norms if present
        if hasattr(bridge_block, "ln1"):
            bridge_ln1 = bridge_block.ln1

            # Get HF ln1
            ln1_path = bridge.adapter.component_mapping.blocks.submodules["ln1"].name
            if ln1_path:
                hf_ln1 = hf_block
                for part in ln1_path.split("."):
                    hf_ln1 = getattr(hf_ln1, part)

                results.append(
                    benchmark_component_forward(
                        bridge_ln1,
                        hf_ln1,
                        test_input,
                        f"block_{block_idx}_ln1",
                        atol=atol,
                    )
                )

        if hasattr(bridge_block, "ln2"):
            bridge_ln2 = bridge_block.ln2

            # Get HF ln2
            ln2_path = bridge.adapter.component_mapping.blocks.submodules["ln2"].name
            if ln2_path:
                hf_ln2 = hf_block
                for part in ln2_path.split("."):
                    hf_ln2 = getattr(hf_ln2, part)

                results.append(
                    benchmark_component_forward(
                        bridge_ln2,
                        hf_ln2,
                        test_input,
                        f"block_{block_idx}_ln2",
                        atol=atol,
                    )
                )

    except Exception as e:
        results.append(
            BenchmarkResult(
                name=f"block_{block_idx}_components",
                passed=False,
                severity=BenchmarkSeverity.ERROR,
                message=f"Error benchmarking block {block_idx} components: {str(e)}",
                details={"exception": str(e)},
            )
        )

    return results


def benchmark_attention_subcomponents(
    bridge,
    hf_model,
    block_idx: int = 0,
    atol: float = 1e-4,
) -> list[BenchmarkResult]:
    """Benchmark attention subcomponents (Q, K, V, O projections).

    Args:
        bridge: The TransformerBridge model
        hf_model: The HuggingFace model
        block_idx: Which block to test (default: 0)
        atol: Absolute tolerance for comparison

    Returns:
        List of BenchmarkResult for each attention subcomponent
    """
    results = []

    try:
        # Create test input
        batch_size = 1
        seq_len = 4
        d_model = bridge.cfg.d_model
        test_input = torch.randn(batch_size, seq_len, d_model, device=bridge.cfg.device)

        # Get the attention components
        bridge_attn = bridge.blocks[block_idx].attn

        # Get HF block
        hf_blocks_path = bridge.adapter.component_mapping.blocks.name
        hf_blocks = hf_model
        for part in hf_blocks_path.split("."):
            hf_blocks = getattr(hf_blocks, part)
        hf_block = hf_blocks[block_idx]

        # Get HF attention
        attn_path = bridge.adapter.component_mapping.blocks.submodules["attn"].name
        hf_attn = hf_block
        for part in attn_path.split("."):
            hf_attn = getattr(hf_attn, part)

        # Test Q, K, V projections if they exist
        for proj_name in ["q", "k", "v", "o"]:
            if hasattr(bridge_attn, proj_name):
                bridge_proj = getattr(bridge_attn, proj_name)

                # Try to get corresponding HF projection
                hf_proj_name = f"{proj_name}_proj"
                if hasattr(hf_attn, hf_proj_name):
                    hf_proj = getattr(hf_attn, hf_proj_name)

                    results.append(
                        benchmark_component_forward(
                            bridge_proj,
                            hf_proj,
                            test_input,
                            f"block_{block_idx}_attn_{proj_name}",
                            atol=atol,
                        )
                    )

    except Exception as e:
        results.append(
            BenchmarkResult(
                name=f"block_{block_idx}_attn_subcomponents",
                passed=False,
                severity=BenchmarkSeverity.ERROR,
                message=f"Error benchmarking attention subcomponents: {str(e)}",
                details={"exception": str(e)},
            )
        )

    return results


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

        # Run comprehensive benchmark
        report = benchmarker.benchmark_all_components()

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
