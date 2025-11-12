"""Comprehensive component benchmarking utility for TransformerBridge.

This module provides utilities to benchmark all standard components in a TransformerBridge
model against their HuggingFace equivalents, ensuring output parity.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, cast

import torch
from torch import nn

from transformer_lens.config import TransformerBridgeConfig
from transformer_lens.model_bridge.architecture_adapter import ArchitectureAdapter
from transformer_lens.model_bridge.generalized_components.base import (
    GeneralizedComponent,
)


@dataclass
class ComponentTestResult:
    """Result of testing a single component."""

    component_path: str
    component_type: str
    passed: bool
    max_diff: float
    mean_diff: float
    output_shape: Tuple[int, ...]
    error_message: Optional[str] = None
    percentile_diffs: Optional[Dict[str, float]] = None  # 50th, 90th, 99th percentile diffs

    def get_failure_severity(self) -> str:
        """Categorize the severity of a failure.

        Returns:
            Severity level: "critical", "high", "medium", "low", or "pass"
        """
        if self.passed:
            return "pass"
        if self.error_message:
            return "critical"
        if self.max_diff > 1e-1:
            return "critical"
        elif self.max_diff > 1e-3:
            return "high"
        elif self.max_diff > 1e-4:
            return "medium"
        else:
            return "low"


@dataclass
class BenchmarkReport:
    """Complete benchmark report for all components."""

    model_name: str
    total_components: int
    passed_components: int
    failed_components: int
    component_results: List[ComponentTestResult] = field(default_factory=list)

    @property
    def pass_rate(self) -> float:
        """Calculate the pass rate as a percentage."""
        if self.total_components == 0:
            return 0.0
        return (self.passed_components / self.total_components) * 100

    def print_summary(self, verbose: bool = False) -> None:
        """Print a summary of the benchmark results.

        Args:
            verbose: If True, print details for all components. If False, only print failures.
        """
        print("\n" + "=" * 80)
        print(f"Component Benchmark Report: {self.model_name}")
        print("=" * 80)
        print(f"Total components tested: {self.total_components}")
        print(f"Passed: {self.passed_components} ({self.pass_rate:.1f}%)")
        print(f"Failed: {self.failed_components}")
        print("=" * 80)

        if verbose:
            print("\nAll Component Results:")
            print("-" * 80)
            for result in self.component_results:
                self._print_component_result(result)
        elif self.failed_components > 0:
            print("\nFailed Components:")
            print("-" * 80)
            for result in self.component_results:
                if not result.passed:
                    self._print_component_result(result)

        print("=" * 80 + "\n")

    def _print_component_result(self, result: ComponentTestResult) -> None:
        """Print details of a single component result."""
        status = "✓ PASS" if result.passed else "✗ FAIL"
        severity = result.get_failure_severity()

        # Add severity indicator for failures
        if not result.passed and severity != "critical":
            status = f"{status} [{severity.upper()}]"

        print(f"{status} | {result.component_path}")
        print(f"  Type: {result.component_type}")
        print(f"  Shape: {result.output_shape}")
        print(f"  Max diff: {result.max_diff:.6e}")
        print(f"  Mean diff: {result.mean_diff:.6e}")

        if result.percentile_diffs:
            print(f"  Percentile diffs:")
            for percentile, diff in sorted(result.percentile_diffs.items()):
                print(f"    {percentile}: {diff:.6e}")

        if result.error_message:
            print(f"  Error: {result.error_message}")
        print()

    def get_component_type_summary(self) -> Dict[str, Dict[str, int]]:
        """Get a summary of results grouped by component type.

        Returns:
            Dictionary mapping component types to their pass/fail counts
        """
        summary: Dict[str, Dict[str, int]] = {}

        for result in self.component_results:
            comp_type = result.component_type
            if comp_type not in summary:
                summary[comp_type] = {"passed": 0, "failed": 0, "total": 0}

            summary[comp_type]["total"] += 1
            if result.passed:
                summary[comp_type]["passed"] += 1
            else:
                summary[comp_type]["failed"] += 1

        return summary

    def get_failure_by_severity(self) -> Dict[str, List[ComponentTestResult]]:
        """Group failures by severity level.

        Returns:
            Dictionary mapping severity levels to lists of failed components
        """
        failures: Dict[str, List[ComponentTestResult]] = {
            "critical": [],
            "high": [],
            "medium": [],
            "low": [],
        }

        for result in self.component_results:
            if not result.passed:
                severity = result.get_failure_severity()
                if severity in failures:
                    failures[severity].append(result)

        return failures

    def print_detailed_analysis(self) -> None:
        """Print detailed analysis of benchmark results."""
        print("\n" + "=" * 80)
        print("Detailed Benchmark Analysis")
        print("=" * 80)

        # Component type summary
        print("\nResults by Component Type:")
        print("-" * 80)
        type_summary = self.get_component_type_summary()
        for comp_type, stats in sorted(type_summary.items()):
            pass_rate = (stats["passed"] / stats["total"]) * 100 if stats["total"] > 0 else 0
            print(
                f"{comp_type:30s}: {stats['passed']:3d}/{stats['total']:3d} passed ({pass_rate:5.1f}%)"
            )

        # Failure severity analysis
        if self.failed_components > 0:
            print("\nFailures by Severity:")
            print("-" * 80)
            failures_by_severity = self.get_failure_by_severity()
            for severity in ["critical", "high", "medium", "low"]:
                count = len(failures_by_severity[severity])
                if count > 0:
                    print(f"{severity.upper():10s}: {count} component(s)")
                    for result in failures_by_severity[severity][:3]:  # Show first 3
                        print(f"  - {result.component_path} (max_diff: {result.max_diff:.2e})")
                    if count > 3:
                        print(f"  ... and {count - 3} more")

        print("=" * 80 + "\n")


class ComponentBenchmarker:
    """Benchmarking utility for testing TransformerBridge components against HuggingFace."""

    def __init__(
        self,
        bridge_model: nn.Module,
        hf_model: nn.Module,
        adapter: ArchitectureAdapter,
        cfg: TransformerBridgeConfig,
        atol: float = 1e-4,
        rtol: float = 1e-4,
    ):
        """Initialize the component benchmarker.

        Args:
            bridge_model: The TransformerBridge model
            hf_model: The HuggingFace model
            adapter: The architecture adapter for mapping components
            cfg: The model configuration
            atol: Absolute tolerance for comparing outputs
            rtol: Relative tolerance for comparing outputs
        """
        self.bridge_model = bridge_model
        self.hf_model = hf_model
        self.adapter = adapter
        self.cfg = cfg

        # Adjust tolerances based on dtype for reduced precision formats
        model_dtype = getattr(cfg, "dtype", torch.float32)
        if model_dtype == torch.bfloat16:
            # bfloat16 has ~7 bits of precision (3 decimal digits)
            # Use more lenient tolerance
            # Normalization layers (RMSNorm/LayerNorm) can have larger errors due to
            # square roots and divisions, so use 0.3 tolerance
            self.atol = max(atol, 0.3)
            self.rtol = max(rtol, 0.3)
        elif model_dtype == torch.float16:
            # float16 has ~10 bits of precision (3-4 decimal digits)
            self.atol = max(atol, 5e-3)
            self.rtol = max(rtol, 5e-3)
        else:
            # float32 or float64 - use provided tolerances
            self.atol = atol
            self.rtol = rtol

    def benchmark_all_components(
        self,
        test_inputs: Optional[Dict[str, torch.Tensor]] = None,
        skip_components: Optional[List[str]] = None,
    ) -> BenchmarkReport:
        """Benchmark all components in the model.

        Args:
            test_inputs: Optional dictionary of pre-generated test inputs.
                        If None, will generate default inputs.
            skip_components: Optional list of component paths to skip

        Returns:
            BenchmarkReport with results for all tested components
        """
        skip_components = skip_components or []
        component_mapping = self.adapter.get_component_mapping()

        # Generate test inputs if not provided
        if test_inputs is None:
            test_inputs = self._generate_test_inputs()

        results: List[ComponentTestResult] = []

        # Test top-level components (embed, pos_embed, ln_final, unembed)
        for comp_name, component in component_mapping.items():
            if comp_name in skip_components:
                continue

            if comp_name == "blocks":
                # Handle blocks separately
                continue

            result = self._test_component(comp_name, component, test_inputs)
            if result is not None:
                results.append(result)

        # Test block components
        if "blocks" in component_mapping and "blocks" not in skip_components:
            blocks_component = component_mapping["blocks"]
            n_layers = self.cfg.n_layers

            for layer_idx in range(n_layers):
                # Test each subcomponent in the block
                for subcomp_name, subcomponent in blocks_component.submodules.items():
                    comp_path = f"blocks.{layer_idx}.{subcomp_name}"
                    if comp_path in skip_components:
                        continue

                    result = self._test_component(comp_path, subcomponent, test_inputs)
                    if result is not None:
                        results.append(result)

        # Create report
        passed = sum(1 for r in results if r.passed)
        failed = sum(1 for r in results if not r.passed)

        return BenchmarkReport(
            model_name=getattr(self.cfg, "model_name", "unknown"),
            total_components=len(results),
            passed_components=passed,
            failed_components=failed,
            component_results=results,
        )

    def _test_component(
        self,
        component_path: str,
        component: GeneralizedComponent,
        test_inputs: Dict[str, torch.Tensor],
    ) -> Optional[ComponentTestResult]:
        """Test a single component.

        Args:
            component_path: Path to the component (e.g., "embed", "blocks.0.attn")
            component: The generalized component bridge
            test_inputs: Dictionary of test inputs

        Returns:
            ComponentTestResult or None if the component cannot be tested
        """
        try:
            # Get bridge component
            bridge_component = self.adapter.get_component(self.bridge_model, component_path)

            # Get HuggingFace component
            hf_component = self.adapter.get_component(self.hf_model, component_path)

            # Determine appropriate test input based on component type
            test_input = self._get_test_input_for_component(component_path, test_inputs)
            if test_input is None:
                return None

            # For embedding components, generate token indices once to use for both
            shared_token_indices = None
            if component_path == "embed":
                batch, seq_len, _ = test_input.shape
                shared_token_indices = torch.randint(0, self.cfg.d_vocab, (batch, seq_len))

            # Generate shared inputs for attention/MLP/rotary components that have get_random_inputs()
            # This is needed for model-specific inputs like position_embeddings or attention_mask
            shared_inputs = None
            if (
                ("attn" in component_path or "mlp" in component_path or "rotary" in component_path)
                and hasattr(bridge_component, "get_random_inputs")
                and callable(getattr(bridge_component, "get_random_inputs"))
            ):
                batch_size, seq_len = test_input.shape[:2]
                # Cast to callable to satisfy mypy - we've already verified it exists and is callable
                get_random_inputs_fn = cast(
                    Callable[..., Dict[str, Any]], bridge_component.get_random_inputs
                )
                shared_inputs = get_random_inputs_fn(
                    batch_size=batch_size,
                    seq_len=seq_len,
                    device=test_input.device,
                    dtype=test_input.dtype,
                )

                # Override position_embeddings with correct values from HF model's rotary_emb
                # This is needed for models with partial RoPE or non-standard rotary dims
                if (
                    "attn" in component_path
                    and "position_embeddings" in shared_inputs
                    and hasattr(self.hf_model, "model")
                ):
                    rotary_attr = getattr(self.hf_model.model, "rotary_emb", None)
                    if callable(rotary_attr):
                        try:
                            position_ids = (
                                torch.arange(seq_len, device=test_input.device)
                                .unsqueeze(0)
                                .expand(batch_size, -1)
                            )
                            position_embeddings = rotary_attr(test_input, position_ids)
                            shared_inputs["position_embeddings"] = position_embeddings
                        except Exception:
                            # If rotary_emb fails, keep the fallback position_embeddings from get_random_inputs()
                            pass

            # Run through both components with shared inputs (for attention) or standard inputs (for others)
            bridge_output = self._run_component(
                bridge_component, test_input, component_path, shared_token_indices, shared_inputs
            )
            hf_output = self._run_component(
                hf_component, test_input, component_path, shared_token_indices, shared_inputs
            )

            # Extract tensors if outputs are tuples
            bridge_tensor = bridge_output[0] if isinstance(bridge_output, tuple) else bridge_output
            hf_tensor = hf_output[0] if isinstance(hf_output, tuple) else hf_output

            # Ensure both are tensors
            if not isinstance(bridge_tensor, torch.Tensor) or not isinstance(
                hf_tensor, torch.Tensor
            ):
                return ComponentTestResult(
                    component_path=component_path,
                    component_type=type(component).__name__,
                    passed=False,
                    max_diff=float("inf"),
                    mean_diff=float("inf"),
                    output_shape=(),
                    error_message=f"Outputs are not tensors: bridge={type(bridge_tensor)}, hf={type(hf_tensor)}",
                )

            # Compare outputs
            passed, max_diff, mean_diff, percentile_diffs = self._compare_outputs(
                bridge_tensor, hf_tensor
            )

            return ComponentTestResult(
                component_path=component_path,
                component_type=type(component).__name__,
                passed=passed,
                max_diff=max_diff,
                mean_diff=mean_diff,
                output_shape=tuple(bridge_tensor.shape),
                percentile_diffs=percentile_diffs,
            )

        except Exception as e:
            return ComponentTestResult(
                component_path=component_path,
                component_type=type(component).__name__,
                passed=False,
                max_diff=float("inf"),
                mean_diff=float("inf"),
                output_shape=(),
                error_message=str(e),
            )

    def _run_component(
        self,
        component: nn.Module,
        test_input: torch.Tensor,
        component_path: str,
        shared_token_indices: Optional[torch.Tensor] = None,
        shared_inputs: Optional[dict] = None,
    ) -> Any:
        """Run a component with appropriate arguments.

        Args:
            component: The component to run
            test_input: The test input tensor
            component_path: Path to the component for debugging
            shared_token_indices: Pre-generated token indices for embedding components
            shared_inputs: Pre-generated inputs from get_random_inputs() to use for both bridge and HF components

        Returns:
            The component output
        """
        # Use shared inputs if provided (generated from bridge component's get_random_inputs())
        if shared_inputs is not None:
            # Check if shared_inputs contains positional args
            if "args" in shared_inputs:
                # Call with positional args (e.g., for rotary embeddings)
                return component(*shared_inputs["args"])
            else:
                # Call with keyword args (e.g., for attention)
                return component(**shared_inputs)

        # Fallback: Use legacy calling conventions for components without get_random_inputs()
        if "attn" in component_path and "attn" == component_path.split(".")[-1]:
            # Attention components (legacy fallback)
            try:
                # Try TransformerLens-style attention
                return component(
                    query_input=test_input,
                    key_input=test_input,
                    value_input=test_input,
                    past_kv_cache_entry=None,
                    attention_mask=None,
                )
            except TypeError:
                try:
                    # Try HuggingFace-style attention
                    return component(hidden_states=test_input)
                except TypeError:
                    # Try simple call
                    return component(test_input)
        elif component_path == "embed":
            # Main embedding component expects integer indices
            # Use shared token indices if provided, otherwise generate new ones
            if shared_token_indices is not None:
                token_indices = shared_token_indices
            else:
                batch, seq_len, _ = test_input.shape
                token_indices = torch.randint(0, self.cfg.d_vocab, (batch, seq_len))
            return component(token_indices)
        elif component_path == "pos_embed" or "pos_embed" in component_path:
            # Position embedding expects integer position indices
            batch, seq_len, _ = test_input.shape
            # For positional embeddings, we need position indices
            pos_indices = torch.arange(seq_len).unsqueeze(0).expand(batch, -1)
            try:
                return component(pos_indices)
            except (TypeError, IndexError):
                # Some pos embeds just return their embeddings directly
                # or may not take inputs
                try:
                    if hasattr(component, "weight") and isinstance(component.weight, torch.Tensor):
                        return component.weight[:seq_len]
                    else:
                        raise AttributeError("Component has no weight attribute")
                except AttributeError:
                    # Skip this component
                    raise ValueError("Cannot test pos_embed - unclear interface")
        elif (
            component_path == "unembed"
            or "unembed" in component_path
            or "lm_head" in component_path
        ):
            # Unembedding expects [batch, seq, d_model] input
            return component(test_input)
        else:
            # Standard components (MLP, LayerNorm, etc.)
            try:
                return component(test_input)
            except TypeError:
                # Try with hidden_states kwarg
                return component(hidden_states=test_input)

    def _get_test_input_for_component(
        self, component_path: str, test_inputs: Dict[str, torch.Tensor]
    ) -> Optional[torch.Tensor]:
        """Get the appropriate test input for a component.

        Args:
            component_path: Path to the component
            test_inputs: Dictionary of available test inputs

        Returns:
            The appropriate test input tensor, or None if not applicable
        """
        # Use standard hidden state input for most components
        return test_inputs.get("hidden_states")

    def _generate_test_inputs(self) -> Dict[str, torch.Tensor]:
        """Generate default test inputs for benchmarking.

        Returns:
            Dictionary of test input tensors
        """
        batch_size = 2
        seq_len = 8
        d_model = self.cfg.d_model

        # Use dtype from config (matches HF model's dtype)
        dtype = getattr(self.cfg, "dtype", torch.float32)
        device = next(self.hf_model.parameters()).device

        return {
            "hidden_states": torch.randn(batch_size, seq_len, d_model, dtype=dtype, device=device),
            "token_ids": torch.randint(0, self.cfg.d_vocab, (batch_size, seq_len), device=device),
        }

    def _compare_outputs(
        self, bridge_output: torch.Tensor, hf_output: torch.Tensor
    ) -> Tuple[bool, float, float, Dict[str, float]]:
        """Compare two output tensors.

        Args:
            bridge_output: Output from TransformerBridge component
            hf_output: Output from HuggingFace component

        Returns:
            Tuple of (passed, max_diff, mean_diff, percentile_diffs)
        """
        # Check shapes match
        if bridge_output.shape != hf_output.shape:
            return False, float("inf"), float("inf"), {}

        # Compute differences
        diff = torch.abs(bridge_output - hf_output)
        max_diff = diff.max().item()
        mean_diff = diff.mean().item()

        # Compute percentile differences
        # Convert to float32 for quantile computation (bfloat16 not supported)
        flat_diff = diff.flatten()
        if flat_diff.dtype == torch.bfloat16 or flat_diff.dtype == torch.float16:
            flat_diff = flat_diff.float()
        percentile_diffs = {
            "50th": torch.quantile(flat_diff, 0.5).item(),
            "90th": torch.quantile(flat_diff, 0.9).item(),
            "99th": torch.quantile(flat_diff, 0.99).item(),
        }

        # Check if within tolerance
        passed = torch.allclose(bridge_output, hf_output, atol=self.atol, rtol=self.rtol)

        return passed, max_diff, mean_diff, percentile_diffs


def benchmark_model(
    model_name: str,
    device: str = "cpu",
    atol: float = 1e-4,
    rtol: float = 1e-4,
    skip_components: Optional[List[str]] = None,
    verbose: bool = False,
) -> BenchmarkReport:
    """Benchmark all components in a model.

    Args:
        model_name: Name of the HuggingFace model to benchmark
        device: Device to run on
        atol: Absolute tolerance for comparisons
        rtol: Relative tolerance for comparisons
        skip_components: Optional list of component paths to skip
        verbose: If True, print detailed results for all components

    Returns:
        BenchmarkReport with results for all components
    """
    from transformers import AutoModelForCausalLM

    from transformer_lens.model_bridge import TransformerBridge

    # Load models
    print(f"Loading models: {model_name}")
    bridge_model = TransformerBridge.boot_transformers(model_name, device=device)  # type: ignore[attr-defined]

    # Load HF model with same attn_implementation as bridge model (if specified)
    # This ensures numerical consistency between bridge and HF models
    hf_kwargs = {"device_map": device}
    if (
        hasattr(bridge_model.adapter.cfg, "attn_implementation")
        and bridge_model.adapter.cfg.attn_implementation is not None
    ):
        hf_kwargs["attn_implementation"] = bridge_model.adapter.cfg.attn_implementation

    hf_model = AutoModelForCausalLM.from_pretrained(model_name, **hf_kwargs)

    # Set models to eval mode (disable dropout, etc.)
    bridge_model.eval()
    hf_model.eval()

    # Get adapter
    adapter = bridge_model.adapter

    # Set up component testing (e.g., sync rotary_emb references for Gemma-3)
    # Pass bridge_model so adapter can set up actual bridge instances, not just templates
    adapter.setup_component_testing(hf_model, bridge_model=bridge_model)

    # Create benchmarker
    benchmarker = ComponentBenchmarker(
        bridge_model=bridge_model,
        hf_model=hf_model,
        adapter=adapter,
        cfg=bridge_model.cfg,
        atol=atol,
        rtol=rtol,
    )

    # Run benchmark
    print("Running component benchmark...")
    report = benchmarker.benchmark_all_components(skip_components=skip_components)

    # Print report
    report.print_summary(verbose=verbose)

    return report
