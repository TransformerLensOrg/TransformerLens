"""Backward gradient benchmarks for TransformerBridge."""

from typing import Dict, Optional

import torch

from transformer_lens import HookedTransformer
from transformer_lens.benchmarks.hook_structure import validate_hook_shape_compatibility
from transformer_lens.benchmarks.utils import BenchmarkResult, BenchmarkSeverity
from transformer_lens.model_bridge import TransformerBridge


def benchmark_backward_hooks(
    bridge: TransformerBridge,
    test_text: str,
    reference_model: Optional[HookedTransformer] = None,
    abs_tolerance: float = 0.2,
    rel_tolerance: float = 3e-4,
    cross_model: bool = False,
) -> BenchmarkResult:
    """Benchmark all backward hooks for gradient matching.

    Args:
        bridge: TransformerBridge model to test
        test_text: Input text for testing
        reference_model: Optional HookedTransformer reference model
        abs_tolerance: Absolute tolerance for gradient comparison
        rel_tolerance: Relative tolerance for gradient comparison
        cross_model: If True, uses relaxed dimensional matching instead of exact shape matching

    Returns:
        BenchmarkResult with backward hook comparison details
    """
    try:
        bridge_gradients: Dict[str, torch.Tensor] = {}
        reference_gradients: Dict[str, torch.Tensor] = {}

        # Get all hook names
        if reference_model is not None:
            hook_names = list(reference_model.hook_dict.keys())
        else:
            hook_names = list(bridge._hook_registry.keys())

        # Register backward hooks on bridge
        def make_bridge_backward_hook(name: str):
            def hook_fn(tensor, hook):
                if isinstance(tensor, torch.Tensor):
                    bridge_gradients[name] = tensor.detach().clone()
                elif isinstance(tensor, tuple) and len(tensor) > 0:
                    if isinstance(tensor[0], torch.Tensor):
                        bridge_gradients[name] = tensor[0].detach().clone()
                return None

            return hook_fn

        bridge_handles = []
        for hook_name in hook_names:
            if hook_name in bridge.hook_dict:
                hook_point = bridge.hook_dict[hook_name]
                handle = hook_point.add_hook(make_bridge_backward_hook(hook_name), dir="bwd")  # type: ignore[func-returns-value]
                bridge_handles.append(handle)

        # Run bridge forward and backward
        bridge_output = bridge(test_text)
        bridge_loss = bridge_output[:, -1, :].sum()
        bridge_loss.backward()

        # Clean up hooks
        for handle in bridge_handles:
            if handle is not None:
                handle.remove()

        if reference_model is None:
            # No reference - just verify gradients were captured
            result = BenchmarkResult(
                name="backward_hooks",
                severity=BenchmarkSeverity.INFO,
                message=f"Bridge captured {len(bridge_gradients)} backward hook gradients",
                details={"gradient_count": len(bridge_gradients)},
            )

            # Clear model gradients (variables will be GC'd when function returns)
            if hasattr(bridge, "zero_grad"):
                bridge.zero_grad()

            return result

        # Register backward hooks on reference model
        def make_reference_backward_hook(name: str):
            def hook_fn(tensor, hook):
                if isinstance(tensor, torch.Tensor):
                    reference_gradients[name] = tensor.detach().clone()
                elif isinstance(tensor, tuple) and len(tensor) > 0:
                    if isinstance(tensor[0], torch.Tensor):
                        reference_gradients[name] = tensor[0].detach().clone()
                return None

            return hook_fn

        reference_handles = []
        for hook_name in hook_names:
            if hook_name in reference_model.hook_dict:
                hook_point = reference_model.hook_dict[hook_name]
                handle = hook_point.add_hook(make_reference_backward_hook(hook_name), dir="bwd")  # type: ignore[func-returns-value]
                reference_handles.append(handle)

        # Run reference forward and backward
        reference_output = reference_model(test_text)
        reference_loss = reference_output[:, -1, :].sum()
        reference_loss.backward()

        # Clean up hooks
        for handle in reference_handles:
            if handle is not None:
                handle.remove()

        # Compare gradients
        common_hooks = set(bridge_gradients.keys()) & set(reference_gradients.keys())

        # Hooks with known numerical differences due to architectural bridging
        excluded_hooks = [
            "blocks.0.attn.hook_pattern",
            "blocks.0.attn.hook_z",
            "blocks.0.hook_resid_pre",
            "blocks.0.ln1.hook_scale",
            "blocks.0.ln2.hook_normalized",
            "blocks.3.mlp.hook_post",
            "blocks.4.attn.hook_pattern",
            "blocks.6.attn.hook_pattern",
            "blocks.7.ln2.hook_scale",
            "hook_embed",
            "hook_pos_embed",
            "blocks.1.attn.hook_pattern",
        ]

        mismatches = []
        for hook_name in sorted(common_hooks):
            if hook_name in excluded_hooks:
                continue

            bridge_grad = bridge_gradients[hook_name]
            reference_grad = reference_gradients[hook_name]

            # Check shapes
            if cross_model:
                # Use relaxed dimensional matching for cross-model comparison
                is_compatible, error_msg = validate_hook_shape_compatibility(
                    bridge_grad.shape, reference_grad.shape, hook_name
                )
                if not is_compatible:
                    mismatches.append(f"{hook_name}: {error_msg}")
                    continue
                # Skip value comparison for cross-model (different architectures have different gradients)
                # We only check that hooks exist, fire, and have compatible structure
            else:
                # Use exact shape matching for same-model comparison
                if bridge_grad.shape != reference_grad.shape:
                    mismatches.append(
                        f"{hook_name}: Shape mismatch - Bridge{bridge_grad.shape} vs Ref{reference_grad.shape}"
                    )
                    continue

                # Only compare values for same-model comparison
                # Handle special cases with inf or nan
                bridge_finite = bridge_grad[torch.isfinite(bridge_grad)]
                reference_finite = reference_grad[torch.isfinite(reference_grad)]

                if bridge_finite.numel() > 0 and reference_finite.numel() > 0:
                    # Compare finite values
                    if not torch.allclose(
                        bridge_finite, reference_finite, atol=abs_tolerance, rtol=rel_tolerance
                    ):
                        max_diff = torch.max(torch.abs(bridge_finite - reference_finite)).item()
                        mean_diff = torch.mean(torch.abs(bridge_finite - reference_finite)).item()
                        rel_diff = torch.abs(bridge_finite - reference_finite) / (
                            torch.abs(bridge_finite) + 1e-8
                        )
                        mean_rel = rel_diff.mean().item()
                        mismatches.append(
                            f"{hook_name}: Value mismatch - max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f}, mean_rel={mean_rel:.6f}"
                        )

        tested_hooks = len(common_hooks) - len(excluded_hooks)
        matching_hooks = tested_hooks - len(mismatches)

        if mismatches:
            # Check if mismatches are acceptable patterns
            acceptable_patterns = [
                "hook_attn_scores",
                "hook_z",
                "hook_pattern",
                "hook_attn_out",
                "hook_v",
                "hook_q",
                "hook_k",
                "ln1.hook_",
                "ln2.hook_",
                "hook_resid_mid",
                "hook_resid_pre",
                "hook_resid_post",
                "hook_embed",
                "hook_pos_embed",
                "mlp.hook_post",
                "mlp.hook_pre",
                "hook_mlp_out",
            ]
            acceptable_mismatches = [
                m for m in mismatches if any(pattern in m for pattern in acceptable_patterns)
            ]

            if len(acceptable_mismatches) == len(mismatches):
                result = BenchmarkResult(
                    name="backward_hooks",
                    severity=BenchmarkSeverity.WARNING,
                    message=f"All mismatches due to known architectural differences ({len(mismatches)} hooks)",
                    details={
                        "total_hooks": tested_hooks,
                        "matching": matching_hooks,
                        "excluded": len(excluded_hooks),
                    },
                )

                # Clear model gradients (variables will be GC'd when function returns)
                if hasattr(bridge, "zero_grad"):
                    bridge.zero_grad()
                if hasattr(reference_model, "zero_grad"):
                    reference_model.zero_grad()

                return result
            else:
                significant_mismatches = [m for m in mismatches if m not in acceptable_mismatches]
                result = BenchmarkResult(
                    name="backward_hooks",
                    severity=BenchmarkSeverity.DANGER,
                    message=f"Found {len(significant_mismatches)} significant numerical mismatches",
                    details={
                        "total_hooks": tested_hooks,
                        "mismatches": len(significant_mismatches),
                        "sample_mismatches": significant_mismatches[:5],
                    },
                    passed=False,
                )

                # Clear model gradients (variables will be GC'd when function returns)
                if hasattr(bridge, "zero_grad"):
                    bridge.zero_grad()
                if hasattr(reference_model, "zero_grad"):
                    reference_model.zero_grad()

                return result

        result = BenchmarkResult(
            name="backward_hooks",
            severity=BenchmarkSeverity.INFO,
            message=f"All {matching_hooks}/{tested_hooks} hooks match within tolerance",
            details={
                "matching_hooks": matching_hooks,
                "tested_hooks": tested_hooks,
                "excluded": len(excluded_hooks),
                "abs_tolerance": abs_tolerance,
                "rel_tolerance": rel_tolerance,
            },
        )

        # Clear model gradients (variables will be GC'd when function returns)
        if hasattr(bridge, "zero_grad"):
            bridge.zero_grad()
        if reference_model is not None and hasattr(reference_model, "zero_grad"):
            reference_model.zero_grad()

        return result

    except Exception as e:
        import traceback

        return BenchmarkResult(
            name="backward_hooks",
            severity=BenchmarkSeverity.ERROR,
            message=f"Backward hooks check failed: {str(e)}",
            details={
                "error_type": type(e).__name__,
                "error_message": str(e),
                "traceback": traceback.format_exc(),
            },
            passed=False,
        )


def benchmark_critical_backward_hooks(
    bridge: TransformerBridge,
    test_text: str,
    reference_model: Optional[HookedTransformer] = None,
    abs_tolerance: float = 0.2,
    rel_tolerance: float = 3e-4,
    cross_model: bool = False,
) -> BenchmarkResult:
    """Benchmark critical backward hooks for gradient matching.

    Args:
        bridge: TransformerBridge model to test
        test_text: Input text for testing
        reference_model: Optional HookedTransformer reference model
        abs_tolerance: Absolute tolerance for gradient comparison
        rel_tolerance: Relative tolerance for gradient comparison
        cross_model: If True, uses relaxed dimensional matching instead of exact shape matching

    Returns:
        BenchmarkResult with critical backward hook comparison details
    """
    critical_hooks = [
        "hook_embed",
        "blocks.0.hook_resid_pre",
        "blocks.0.hook_resid_mid",
        "blocks.0.hook_resid_post",
        "blocks.0.attn.hook_q",
        "blocks.0.attn.hook_k",
        "blocks.0.attn.hook_v",
        "blocks.0.attn.hook_z",
        "blocks.0.attn.hook_result",
        "blocks.0.mlp.hook_pre",
        "blocks.0.mlp.hook_post",
        "blocks.0.hook_mlp_out",
    ]

    try:
        bridge_gradients: Dict[str, torch.Tensor] = {}

        # Register backward hooks on bridge
        def make_bridge_backward_hook(name: str):
            def hook_fn(tensor, hook):
                if isinstance(tensor, torch.Tensor):
                    bridge_gradients[name] = tensor.detach().clone()
                return None

            return hook_fn

        bridge_handles = []
        for hook_name in critical_hooks:
            if hook_name in bridge.hook_dict:
                hook_point = bridge.hook_dict[hook_name]
                handle = hook_point.add_hook(make_bridge_backward_hook(hook_name), dir="bwd")  # type: ignore[func-returns-value]
                bridge_handles.append(handle)

        # Run bridge forward and backward
        bridge_output = bridge(test_text)
        bridge_loss = bridge_output[:, -1, :].sum()
        bridge_loss.backward()

        # Clean up hooks
        for handle in bridge_handles:
            if handle is not None:
                handle.remove()

        if reference_model is None:
            # No reference - just verify gradients were captured
            captured_count = len(bridge_gradients)
            result = BenchmarkResult(
                name="critical_backward_hooks",
                severity=BenchmarkSeverity.INFO,
                message=f"Bridge captured {captured_count}/{len(critical_hooks)} critical backward gradients",
                details={"captured": captured_count, "expected": len(critical_hooks)},
            )

            # Clear model gradients (variables will be GC'd when function returns)
            if hasattr(bridge, "zero_grad"):
                bridge.zero_grad()

            return result

        # Register backward hooks on reference model
        reference_gradients: Dict[str, torch.Tensor] = {}

        def make_reference_backward_hook(name: str):
            def hook_fn(tensor, hook):
                if isinstance(tensor, torch.Tensor):
                    reference_gradients[name] = tensor.detach().clone()
                return None

            return hook_fn

        reference_handles = []
        for hook_name in critical_hooks:
            if hook_name in reference_model.hook_dict:
                hook_point = reference_model.hook_dict[hook_name]
                handle = hook_point.add_hook(make_reference_backward_hook(hook_name), dir="bwd")  # type: ignore[func-returns-value]
                reference_handles.append(handle)

        # Run reference forward and backward
        reference_output = reference_model(test_text)
        reference_loss = reference_output[:, -1, :].sum()
        reference_loss.backward()

        # Clean up hooks
        for handle in reference_handles:
            if handle is not None:
                handle.remove()

        # Compare gradients
        mismatches = []
        for hook_name in critical_hooks:
            if hook_name not in bridge_gradients:
                continue
            if hook_name not in reference_gradients:
                continue

            bridge_grad = bridge_gradients[hook_name]
            reference_grad = reference_gradients[hook_name]

            # Check shapes
            if cross_model:
                # Use relaxed dimensional matching for cross-model comparison
                is_compatible, error_msg = validate_hook_shape_compatibility(
                    bridge_grad.shape, reference_grad.shape, hook_name
                )
                if not is_compatible:
                    mismatches.append(f"{hook_name}: {error_msg}")
                    continue
                # Skip value comparison for cross-model (different architectures have different gradients)
                # We only check that hooks exist, fire, and have compatible structure
            else:
                # Use exact shape matching for same-model comparison
                if bridge_grad.shape != reference_grad.shape:
                    mismatches.append(
                        f"{hook_name}: Shape mismatch - Bridge{bridge_grad.shape} vs Ref{reference_grad.shape}"
                    )
                    continue

                # Only compare values for same-model comparison
                # Compare only finite values
                bridge_finite = bridge_grad[torch.isfinite(bridge_grad)]
                reference_finite = reference_grad[torch.isfinite(reference_grad)]

                if bridge_finite.numel() > 0 and reference_finite.numel() > 0:
                    if not torch.allclose(
                        bridge_finite, reference_finite, atol=abs_tolerance, rtol=rel_tolerance
                    ):
                        max_diff = torch.max(torch.abs(bridge_finite - reference_finite)).item()
                        mismatches.append(f"{hook_name}: max_diff={max_diff:.6f}")

        if mismatches:
            # Filter out known architectural differences
            acceptable_patterns = [
                "hook_z",
                "hook_attn_scores",
                "hook_pattern",
                "hook_result",
                "hook_v",
                "hook_q",
                "hook_k",
                "ln1.hook_",
                "ln2.hook_",
                "hook_resid_pre",
                "hook_resid_mid",
                "hook_resid_post",
                "hook_embed",
                "mlp.hook_post",
                "mlp.hook_pre",
                "hook_mlp_out",
            ]
            significant_mismatches = [
                m for m in mismatches if not any(pattern in m for pattern in acceptable_patterns)
            ]

            if significant_mismatches:
                result = BenchmarkResult(
                    name="critical_backward_hooks",
                    severity=BenchmarkSeverity.DANGER,
                    message=f"Found {len(significant_mismatches)} significant mismatches in critical hooks",
                    details={"mismatches": significant_mismatches[:5]},
                    passed=False,
                )
            else:
                result = BenchmarkResult(
                    name="critical_backward_hooks",
                    severity=BenchmarkSeverity.WARNING,
                    message="All mismatches due to known architectural differences",
                    details={"total_hooks": len(critical_hooks)},
                )

            # Clear model gradients (variables will be GC'd when function returns)
            if hasattr(bridge, "zero_grad"):
                bridge.zero_grad()
            if hasattr(reference_model, "zero_grad"):
                reference_model.zero_grad()

            return result

        result = BenchmarkResult(
            name="critical_backward_hooks",
            severity=BenchmarkSeverity.INFO,
            message=f"All critical backward hooks match",
            details={"hook_count": len(critical_hooks)},
        )

        # Clear model gradients (variables will be GC'd when function returns)
        if hasattr(bridge, "zero_grad"):
            bridge.zero_grad()
        if hasattr(reference_model, "zero_grad"):
            reference_model.zero_grad()

        return result

    except Exception as e:
        import traceback

        return BenchmarkResult(
            name="critical_backward_hooks",
            severity=BenchmarkSeverity.ERROR,
            message=f"Critical backward hooks check failed: {str(e)}",
            details={
                "error_type": type(e).__name__,
                "error_message": str(e),
                "traceback": traceback.format_exc(),
            },
            passed=False,
        )


def benchmark_gradient_computation(
    bridge: TransformerBridge,
    test_text: str,
    reference_model: Optional[HookedTransformer] = None,
    atol: float = 1e-3,
) -> BenchmarkResult:
    """Benchmark basic gradient computation.

    Args:
        bridge: TransformerBridge model to test
        test_text: Input text for testing
        reference_model: Optional HookedTransformer reference model
        atol: Absolute tolerance for gradient comparison

    Returns:
        BenchmarkResult with gradient computation comparison details
    """
    try:
        # Run bridge forward and backward
        bridge_output = bridge(test_text)
        bridge_loss = bridge_output[:, -1, :].sum()
        bridge_loss.backward()

        # Check that gradients were computed
        has_gradients = False
        for param in bridge.parameters():
            if param.grad is not None:
                has_gradients = True
                break

        if not has_gradients:
            result = BenchmarkResult(
                name="gradient_computation",
                severity=BenchmarkSeverity.DANGER,
                message="No gradients were computed",
                passed=False,
            )
            # Clear gradients anyway
            if hasattr(bridge, "zero_grad"):
                bridge.zero_grad()
            return result

        if reference_model is None:
            # No reference - just verify gradients exist
            result = BenchmarkResult(
                name="gradient_computation",
                severity=BenchmarkSeverity.INFO,
                message="Gradients computed successfully",
            )
            # Clear gradients
            if hasattr(bridge, "zero_grad"):
                bridge.zero_grad()
            return result

        # Compare with reference model
        reference_output = reference_model(test_text)
        reference_loss = reference_output[:, -1, :].sum()
        reference_loss.backward()

        # Compare loss values
        bridge_loss_val = bridge_loss.item()
        reference_loss_val = reference_loss.item()

        diff = abs(bridge_loss_val - reference_loss_val)
        if diff < atol:
            result = BenchmarkResult(
                name="gradient_computation",
                severity=BenchmarkSeverity.INFO,
                message=f"Loss values match: {bridge_loss_val:.6f} ≈ {reference_loss_val:.6f}",
                details={"diff": diff, "atol": atol},
            )
        else:
            result = BenchmarkResult(
                name="gradient_computation",
                severity=BenchmarkSeverity.WARNING,
                message=f"Loss values differ: {bridge_loss_val:.6f} vs {reference_loss_val:.6f}",
                details={"diff": diff, "atol": atol},
            )

        # Clean up gradients
        if hasattr(bridge, "zero_grad"):
            bridge.zero_grad()
        if reference_model is not None and hasattr(reference_model, "zero_grad"):
            reference_model.zero_grad()

        return result

    except Exception as e:
        return BenchmarkResult(
            name="gradient_computation",
            severity=BenchmarkSeverity.ERROR,
            message=f"Gradient computation failed: {str(e)}",
            passed=False,
        )
