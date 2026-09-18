"""Backward gradient benchmarks for TransformerBridge."""

from typing import Dict, Optional

import torch

from transformer_lens.benchmarks.utils import (
    BenchmarkResult,
    BenchmarkSeverity,
    make_grad_capture_hook,
    safe_allclose,
)
from transformer_lens.hook_points import HookPoint
from transformer_lens.model_bridge import TransformerBridge

# Grading band for numerical (non-convention) gradient mismatches. Registering
# backward hooks forces normalization off HF's native autograd onto the python
# norm, which shifts results at float-rounding scale; measured noise is ~1e-5
# rel_l2 with a single over-tolerance element, while injected bugs start at
# ~1e-3 rel_l2 with 60+ elements over. Valid for fp32 gradients only — the
# gradient section upcasts reduced-precision models before comparing.
REL_L2_TOLERANCE = 1e-4
OVER_TOLERANCE_MAX_ELEMENTS = 3


def needs_fp32_gradients(dtype: Optional[torch.dtype]) -> bool:
    """Reduced-precision gradients cannot be graded against the fp32-calibrated
    band — bf16's rounding floor alone is ~2e-3 rel_l2, inside the bug band."""
    return dtype is not None and dtype not in (torch.float32, torch.float64)


def gradient_mismatch_stats(
    bridge_finite: torch.Tensor,
    reference_finite: torch.Tensor,
    abs_tolerance: float,
    rel_tolerance: float,
) -> dict:
    """Scale-aware statistics for grading one recorded gradient mismatch.

    A zero reference with a nonzero bridge gradient is the maximally divergent
    case, not perfect agreement, so rel_l2 is inf there rather than 0.
    """
    bf, rf = bridge_finite.float(), reference_finite.float()
    ref_norm = torch.norm(rf)
    diff_norm = torch.norm(bf - rf)
    if ref_norm > 0:
        rel_l2 = (diff_norm / ref_norm).item()
    else:
        rel_l2 = 0.0 if diff_norm == 0 else float("inf")
    over_count = int(
        (torch.abs(bf - rf) > abs_tolerance + rel_tolerance * torch.abs(rf)).sum().item()
    )
    return {"rel_l2": rel_l2, "over_count": over_count}


def gradient_mismatch_is_numerical_noise(rel_l2: float, over_count: int) -> bool:
    """True when a gradient mismatch is diffuse and tiny rather than a divergence.

    Elementwise worst-case cannot separate the two: one element of 55k crossing
    the tolerance scores the same as a head scaled by 1%. rel_l2 separates them
    by 58x or more, and the element COUNT guards the localized case rel_l2 would
    dilute. A count (not a fraction) keeps the band reachable on small tensors:
    detection guarantees count >= 1, so a fractional guard of 1e-4 was
    arithmetically unsatisfiable below 10,000 elements (gemma-3-270m's MQA
    hook_rot_k is 6,912).
    """
    return rel_l2 <= REL_L2_TOLERANCE and over_count <= OVER_TOLERANCE_MAX_ELEMENTS


def benchmark_backward_hooks(
    bridge: TransformerBridge,
    test_text: str,
    reference_gradients: Optional[Dict[str, torch.Tensor]] = None,
    abs_tolerance: float = 0.2,
    rel_tolerance: float = 3e-4,
) -> BenchmarkResult:
    """Benchmark all backward hooks for gradient matching.

    Args:
        bridge: TransformerBridge model to test
        test_text: Input text for testing (must match the snapshot's prompt)
        reference_gradients: Optional reference gradients keyed by hook name
            (e.g. a golden fixture snapshot). Capture-only self-check if None.
        abs_tolerance: Absolute tolerance for gradient comparison
        rel_tolerance: Relative tolerance for gradient comparison

    Returns:
        BenchmarkResult with backward hook comparison details
    """
    try:
        bridge_gradients: Dict[str, torch.Tensor] = {}

        # Reference hook names come from the snapshot when provided
        if reference_gradients is not None:
            hook_names = list(reference_gradients.keys())
        else:
            hook_names = list(bridge._hook_registry.keys())

        # Register backward hooks on bridge
        bridge_hook_points: list[HookPoint] = []
        for hook_name in hook_names:
            if hook_name in bridge.hook_dict:
                hook_point = bridge.hook_dict[hook_name]
                hook_point.add_hook(
                    make_grad_capture_hook(bridge_gradients, hook_name, return_none=True),
                    dir="bwd",
                )
                bridge_hook_points.append(hook_point)

        # Run bridge forward and backward
        bridge_output = bridge(test_text)
        bridge_loss = bridge_output[:, -1, :].sum()
        bridge_loss.backward()

        # Clean up hooks
        for hook_point in bridge_hook_points:
            hook_point.remove_hooks(dir="bwd")

        if reference_gradients is None:
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
        mismatch_stats: dict = {}
        for hook_name in sorted(common_hooks):
            if hook_name in excluded_hooks:
                continue

            bridge_grad = bridge_gradients[hook_name]
            reference_grad = reference_gradients[hook_name]

            # Check shapes
            if bridge_grad.shape != reference_grad.shape:
                mismatches.append(
                    f"{hook_name}: Shape mismatch - Bridge{bridge_grad.shape} vs Ref{reference_grad.shape}"
                )
                continue

            # Handle special cases with inf or nan
            bridge_finite = bridge_grad[torch.isfinite(bridge_grad)]
            reference_finite = reference_grad[torch.isfinite(reference_grad)]

            if bridge_finite.numel() > 0 and reference_finite.numel() > 0:
                # Compare finite values
                if not safe_allclose(
                    bridge_finite, reference_finite, atol=abs_tolerance, rtol=rel_tolerance
                ):
                    bf = bridge_finite.float()
                    rf = reference_finite.float()
                    max_diff = torch.max(torch.abs(bf - rf)).item()
                    mean_diff = torch.mean(torch.abs(bf - rf)).item()
                    rel_diff = torch.abs(bf - rf) / (torch.abs(bf) + 1e-8)
                    mean_rel = rel_diff.mean().item()
                    # Scale-aware stats for grading. Elementwise worst-case alone
                    # cannot separate a real divergence from the float-rounding
                    # shift the python-norm fallback introduces when backward
                    # hooks force normalization off HF's native autograd path.
                    stats = gradient_mismatch_stats(bf, rf, abs_tolerance, rel_tolerance)
                    mismatch_stats[hook_name] = stats
                    mismatches.append(
                        f"{hook_name}: Value mismatch - max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f}, "
                        f"mean_rel={mean_rel:.6f}, rel_l2={stats['rel_l2']:.3e}, "
                        f"over_count={stats['over_count']}"
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
                "q_norm",  # QK norm: Bridge uses 4D, HT uses 2D (shape convention)
                "k_norm",  # QK norm: Bridge uses 4D, HT uses 2D (shape convention)
                "ln1.hook_",
                "ln2.hook_",
                # Sandwich norms (gemma-2/3): same class as ln1/ln2 above, which
                # predate them.
                "ln1_post.hook_",
                "ln2_post.hook_",
                "ln_final.hook_",
                "hook_resid_mid",
                "hook_resid_pre",
                "hook_resid_post",
                "hook_embed",
                "hook_pos_embed",
                "unembed.hook_",
                "mlp.hook_post",
                "mlp.hook_pre",
                "hook_mlp_out",
            ]

            def within_noise_band(entry: str) -> bool:
                """Diffuse, tiny deviation — the fallback's rounding, not a divergence.

                Measured noise across architectures is rel_l2 ~1e-5 with a single
                over-tolerance element on the rotary hooks (the only ones outside
                the pattern list); injected bugs of a 1% head scale or a 0.1%
                uniform scale land at rel_l2 1e-3+ with 60+ elements over.
                """
                name = entry.split(":")[0]
                stats = mismatch_stats.get(name)
                if stats is None:
                    return False
                return gradient_mismatch_is_numerical_noise(stats["rel_l2"], stats["over_count"])

            acceptable_mismatches = [
                m
                for m in mismatches
                if any(pattern in m for pattern in acceptable_patterns) or within_noise_band(m)
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
    reference_gradients: Optional[Dict[str, torch.Tensor]] = None,
    abs_tolerance: float = 0.2,
    rel_tolerance: float = 3e-4,
) -> BenchmarkResult:
    """Benchmark critical backward hooks for gradient matching.

    Args:
        bridge: TransformerBridge model to test
        test_text: Input text for testing (must match the snapshot's prompt)
        reference_gradients: Optional reference gradients keyed by hook name
            (e.g. a golden fixture snapshot). Capture-only self-check if None.
        abs_tolerance: Absolute tolerance for gradient comparison
        rel_tolerance: Relative tolerance for gradient comparison

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
        bridge_hook_points: list[HookPoint] = []
        for hook_name in critical_hooks:
            if hook_name in bridge.hook_dict:
                hook_point = bridge.hook_dict[hook_name]
                hook_point.add_hook(
                    make_grad_capture_hook(bridge_gradients, hook_name, return_none=True),
                    dir="bwd",
                )
                bridge_hook_points.append(hook_point)

        # Run bridge forward and backward
        bridge_output = bridge(test_text)
        bridge_loss = bridge_output[:, -1, :].sum()
        bridge_loss.backward()

        # Clean up hooks
        for hook_point in bridge_hook_points:
            hook_point.remove_hooks(dir="bwd")

        if reference_gradients is None:
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
            if bridge_grad.shape != reference_grad.shape:
                mismatches.append(
                    f"{hook_name}: Shape mismatch - Bridge{bridge_grad.shape} vs Ref{reference_grad.shape}"
                )
                continue

            # Compare only finite values
            bridge_finite = bridge_grad[torch.isfinite(bridge_grad)]
            reference_finite = reference_grad[torch.isfinite(reference_grad)]

            if bridge_finite.numel() > 0 and reference_finite.numel() > 0:
                if not safe_allclose(
                    bridge_finite, reference_finite, atol=abs_tolerance, rtol=rel_tolerance
                ):
                    max_diff = torch.max(
                        torch.abs(bridge_finite.float() - reference_finite.float())
                    ).item()
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
                "q_norm",  # QK norm: Bridge uses 4D, HT uses 2D (shape convention)
                "k_norm",  # QK norm: Bridge uses 4D, HT uses 2D (shape convention)
                "ln1.hook_",
                "ln2.hook_",
                # Sandwich norms (gemma-2/3): same class as ln1/ln2 above.
                "ln1_post.hook_",
                "ln2_post.hook_",
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
    reference_loss: Optional[float] = None,
    atol: float = 1e-3,
) -> BenchmarkResult:
    """Benchmark basic gradient computation.

    Args:
        bridge: TransformerBridge model to test
        test_text: Input text for testing (must match the reference's prompt)
        reference_loss: Optional reference last-position summed-logit value
            (e.g. from a golden fixture or an HF forward). Self-check only if None.
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

        if reference_loss is None:
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

        # Compare loss values against the reference scalar
        bridge_loss_val = bridge_loss.item()
        reference_loss_val = reference_loss

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

        return result

    except Exception as e:
        return BenchmarkResult(
            name="gradient_computation",
            severity=BenchmarkSeverity.ERROR,
            message=f"Gradient computation failed: {str(e)}",
            passed=False,
        )
