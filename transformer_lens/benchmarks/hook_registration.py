"""Hook registration and behavior benchmarks for TransformerBridge."""

from typing import Dict, Optional

import torch

from transformer_lens import HookedTransformer
from transformer_lens.benchmarks.hook_structure import validate_hook_shape_compatibility
from transformer_lens.benchmarks.utils import (
    BenchmarkResult,
    BenchmarkSeverity,
    compare_scalars,
    safe_allclose,
)
from transformer_lens.model_bridge import TransformerBridge

# Hook patterns that bridge models inherently don't have because they wrap HF's
# native implementation. Used to filter expected missing/non-firing hooks.
_BRIDGE_EXPECTED_MISSING_PATTERNS = [
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
]


def _filter_expected_missing(hook_names):
    """Filter out hook names that bridge models are expected to be missing."""
    return [
        h
        for h in hook_names
        if not any(pattern in h for pattern in _BRIDGE_EXPECTED_MISSING_PATTERNS)
    ]


def benchmark_hook_registry(
    bridge: TransformerBridge,
    reference_model: Optional[HookedTransformer] = None,
    cross_model: bool = False,
) -> BenchmarkResult:
    """Benchmark hook registry completeness.

    Args:
        bridge: TransformerBridge model to test
        reference_model: Optional HookedTransformer reference model
        cross_model: If True, filter out expected architectural differences

    Returns:
        BenchmarkResult with registry comparison details
    """
    try:
        if reference_model is None:
            # No reference - just verify hooks exist
            if not hasattr(bridge, "_hook_registry"):
                return BenchmarkResult(
                    name="hook_registry",
                    severity=BenchmarkSeverity.DANGER,
                    message="Bridge does not have _hook_registry attribute",
                    passed=False,
                )

            hook_count = len(bridge._hook_registry)
            if hook_count == 0:
                return BenchmarkResult(
                    name="hook_registry",
                    severity=BenchmarkSeverity.WARNING,
                    message="Bridge hook registry is empty",
                )

            return BenchmarkResult(
                name="hook_registry",
                severity=BenchmarkSeverity.INFO,
                message=f"Bridge has {hook_count} registered hooks",
                details={"hook_count": hook_count},
            )

        # Compare with reference model
        bridge_hooks = set(bridge.hook_dict.keys())
        reference_hooks = set(reference_model.hook_dict.keys())

        common_hooks = bridge_hooks & reference_hooks
        missing_hooks = reference_hooks - bridge_hooks
        extra_hooks = bridge_hooks - reference_hooks

        # Filter out hooks that are expected to differ due to architectural differences.
        if missing_hooks:
            missing_hooks = set(_filter_expected_missing(missing_hooks))

        if missing_hooks:
            return BenchmarkResult(
                name="hook_registry",
                severity=BenchmarkSeverity.DANGER,
                message=f"Bridge is missing {len(missing_hooks)} hooks from reference model",
                details={
                    "missing_hooks": len(missing_hooks),
                    "extra_hooks": len(extra_hooks),
                    "common_hooks": len(common_hooks),
                    "sample_missing": list(missing_hooks)[:5],
                },
                passed=False,
            )

        # Bridge having extra hooks is fine - it just means Bridge has more granular hooks
        # What matters is that all HookedTransformer hooks are present in Bridge
        return BenchmarkResult(
            name="hook_registry",
            severity=BenchmarkSeverity.INFO,
            message=f"All {len(reference_hooks)} reference hooks present in Bridge"
            + (f" (Bridge has {len(extra_hooks)} additional hooks)" if extra_hooks else ""),
            details={
                "reference_hooks": len(reference_hooks),
                "bridge_hooks": len(bridge_hooks),
                "extra_hooks": len(extra_hooks) if extra_hooks else 0,
            },
        )

    except Exception as e:
        return BenchmarkResult(
            name="hook_registry",
            severity=BenchmarkSeverity.ERROR,
            message=f"Hook registry check failed: {str(e)}",
            passed=False,
        )


def benchmark_forward_hooks(
    bridge: TransformerBridge,
    test_text: str,
    reference_model: Optional[HookedTransformer] = None,
    tolerance: float = 0.5,
    prepend_bos: Optional[bool] = None,
    cross_model: bool = False,
) -> BenchmarkResult:
    """Benchmark all forward hooks for activation matching.

    Args:
        bridge: TransformerBridge model to test
        test_text: Input text for testing
        reference_model: Optional HookedTransformer for comparison
        tolerance: Tolerance for activation matching (fraction of mismatches allowed)
        prepend_bos: Whether to prepend BOS token. If None, uses model default.
        cross_model: If True, uses relaxed dimensional matching instead of exact shape matching

    Returns:
        BenchmarkResult with hook activation comparison details
    """
    try:
        bridge_activations: Dict[str, torch.Tensor] = {}
        reference_activations: Dict[str, torch.Tensor] = {}

        # Get all hook names
        if reference_model is not None:
            hook_names = list(reference_model.hook_dict.keys())
        else:
            hook_names = list(bridge.hook_dict.keys())

        # Register hooks on bridge and track missing hooks
        def make_bridge_hook(name: str):
            def hook_fn(tensor, hook):
                if isinstance(tensor, torch.Tensor):
                    bridge_activations[name] = tensor.detach().clone()
                elif isinstance(tensor, tuple) and len(tensor) > 0:
                    if isinstance(tensor[0], torch.Tensor):
                        bridge_activations[name] = tensor[0].detach().clone()
                return tensor

            return hook_fn

        bridge_handles = []
        missing_from_bridge = []
        for hook_name in hook_names:
            if hook_name in bridge.hook_dict:
                hook_point = bridge.hook_dict[hook_name]
                handle = hook_point.add_hook(make_bridge_hook(hook_name))  # type: ignore[func-returns-value]
                bridge_handles.append((hook_name, handle))
            else:
                missing_from_bridge.append(hook_name)

        # Run bridge forward pass
        with torch.no_grad():
            if prepend_bos is not None:
                _ = bridge(test_text, prepend_bos=prepend_bos)
            else:
                _ = bridge(test_text)

        # Clean up bridge hooks
        for hook_name, handle in bridge_handles:
            if handle is not None:
                handle.remove()

        # Check for hooks that didn't fire (registered but no activation captured)
        registered_hooks = {name for name, _ in bridge_handles}
        hooks_that_didnt_fire = registered_hooks - set(bridge_activations.keys())

        if reference_model is None:
            # No reference - just verify activations were captured
            if hooks_that_didnt_fire:
                return BenchmarkResult(
                    name="forward_hooks",
                    severity=BenchmarkSeverity.WARNING,
                    message=f"{len(hooks_that_didnt_fire)}/{len(registered_hooks)} hooks didn't fire during forward pass",
                    details={
                        "captured": len(bridge_activations),
                        "registered": len(registered_hooks),
                        "didnt_fire": list(hooks_that_didnt_fire)[:10],
                    },
                )

            return BenchmarkResult(
                name="forward_hooks",
                severity=BenchmarkSeverity.INFO,
                message=f"Bridge captured {len(bridge_activations)} forward hook activations",
                details={"activation_count": len(bridge_activations)},
            )

        # Register hooks on reference model
        def make_reference_hook(name: str):
            def hook_fn(tensor, hook):
                if isinstance(tensor, torch.Tensor):
                    reference_activations[name] = tensor.detach().clone()
                elif isinstance(tensor, tuple) and len(tensor) > 0:
                    if isinstance(tensor[0], torch.Tensor):
                        reference_activations[name] = tensor[0].detach().clone()
                return tensor

            return hook_fn

        reference_handles = []
        for hook_name in hook_names:
            if hook_name in reference_model.hook_dict:
                hook_point = reference_model.hook_dict[hook_name]
                handle = hook_point.add_hook(make_reference_hook(hook_name))  # type: ignore[func-returns-value]
                reference_handles.append(handle)

        # Run reference forward pass
        with torch.no_grad():
            if prepend_bos is not None:
                _ = reference_model(test_text, prepend_bos=prepend_bos)
            else:
                _ = reference_model(test_text)

        # Clean up reference hooks
        for handle in reference_handles:
            if handle is not None:
                handle.remove()

        # CRITICAL CHECK: Bridge must have all hooks that reference has
        # Filter out hooks that bridge models inherently don't have.
        if missing_from_bridge:
            missing_from_bridge = _filter_expected_missing(missing_from_bridge)

        if missing_from_bridge:
            return BenchmarkResult(
                name="forward_hooks",
                severity=BenchmarkSeverity.DANGER,
                message=f"Bridge is MISSING {len(missing_from_bridge)} hooks that exist in reference model",
                details={
                    "missing_count": len(missing_from_bridge),
                    "missing_hooks": missing_from_bridge[:20],  # Show first 20
                    "total_reference_hooks": len(hook_names),
                },
                passed=False,
            )

        # CRITICAL CHECK: All registered hooks must fire
        # Filter out hooks expected to not fire due to architectural differences.
        if hooks_that_didnt_fire:
            hooks_that_didnt_fire = set(_filter_expected_missing(hooks_that_didnt_fire))

        if hooks_that_didnt_fire:
            return BenchmarkResult(
                name="forward_hooks",
                severity=BenchmarkSeverity.DANGER,
                message=f"{len(hooks_that_didnt_fire)} hooks exist but DIDN'T FIRE during forward pass",
                details={
                    "didnt_fire_count": len(hooks_that_didnt_fire),
                    "didnt_fire_hooks": list(hooks_that_didnt_fire)[:20],
                    "total_registered": len(registered_hooks),
                },
                passed=False,
            )

        # Compare activations
        common_hooks = set(bridge_activations.keys()) & set(reference_activations.keys())
        mismatches = []

        for hook_name in sorted(common_hooks):
            bridge_tensor = bridge_activations[hook_name]
            reference_tensor = reference_activations[hook_name]

            # Check shapes
            if cross_model:
                # Use relaxed dimensional matching for cross-model comparison
                is_compatible, error_msg = validate_hook_shape_compatibility(
                    bridge_tensor.shape, reference_tensor.shape, hook_name, cross_model=True
                )
                if not is_compatible:
                    mismatches.append(f"{hook_name}: {error_msg}")
                    continue
                # Skip value comparison for cross-model (different architectures have different values)
                # We only check that hooks exist, fire, and have compatible structure
                continue
            else:
                # Handle batch dimension differences: some HF models (e.g., OPT)
                # internally reshape to 2D for MLP path, producing [seq, dim] hooks
                # while HT always maintains [batch, seq, dim]
                if bridge_tensor.shape != reference_tensor.shape:
                    if (
                        bridge_tensor.ndim == reference_tensor.ndim - 1
                        and reference_tensor.shape[0] == 1
                        and bridge_tensor.shape == reference_tensor.shape[1:]
                    ):
                        bridge_tensor = bridge_tensor.unsqueeze(0)
                    elif (
                        reference_tensor.ndim == bridge_tensor.ndim - 1
                        and bridge_tensor.shape[0] == 1
                        and reference_tensor.shape == bridge_tensor.shape[1:]
                    ):
                        reference_tensor = reference_tensor.unsqueeze(0)
                    else:
                        mismatches.append(
                            f"{hook_name}: Shape mismatch - Bridge{bridge_tensor.shape} vs Ref{reference_tensor.shape}"
                        )
                        continue

            # Check values (only for same-model comparison)
            if not safe_allclose(bridge_tensor, reference_tensor, atol=tolerance, rtol=0.0):
                b = bridge_tensor.float()
                r = reference_tensor.float()
                max_diff = torch.max(torch.abs(b - r)).item()
                mean_diff = torch.mean(torch.abs(b - r)).item()
                mismatches.append(
                    f"{hook_name}: Value mismatch - max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f}"
                )

        if mismatches:
            # Filter out known architectural differences
            significant_mismatches = [
                m
                for m in mismatches
                if "hook_attn_scores" not in m  # Exclude attn_scores which have inf from masking
            ]

            if significant_mismatches:
                return BenchmarkResult(
                    name="forward_hooks",
                    severity=BenchmarkSeverity.DANGER,
                    message=f"Found {len(significant_mismatches)}/{len(common_hooks)} hooks with mismatches",
                    details={
                        "total_hooks": len(common_hooks),
                        "mismatches": len(significant_mismatches),
                        "sample_mismatches": significant_mismatches[:5],
                    },
                    passed=False,
                )
            else:
                return BenchmarkResult(
                    name="forward_hooks",
                    severity=BenchmarkSeverity.WARNING,
                    message=f"All mismatches due to known architectural differences ({len(mismatches)} hooks)",
                    details={"total_hooks": len(common_hooks)},
                )

        return BenchmarkResult(
            name="forward_hooks",
            severity=BenchmarkSeverity.INFO,
            message=f"All {len(common_hooks)} forward hooks match within tolerance",
            details={"hook_count": len(common_hooks), "tolerance": tolerance},
        )

    except Exception as e:
        return BenchmarkResult(
            name="forward_hooks",
            severity=BenchmarkSeverity.ERROR,
            message=f"Forward hooks check failed: {str(e)}",
            passed=False,
        )


def benchmark_critical_forward_hooks(
    bridge: TransformerBridge,
    test_text: str,
    reference_model: Optional[HookedTransformer] = None,
    tolerance: float = 2e-2,
    cross_model: bool = False,
) -> BenchmarkResult:
    """Benchmark critical forward hooks commonly used in interpretability research.

    Args:
        bridge: TransformerBridge model to test
        test_text: Input text for testing
        reference_model: Optional HookedTransformer reference model
        tolerance: Tolerance for activation comparison
        cross_model: If True, uses relaxed dimensional matching instead of exact shape matching

    Returns:
        BenchmarkResult with critical hook comparison details
    """
    # Critical hooks that are commonly used
    critical_hooks = [
        "hook_embed",
        "hook_pos_embed",
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
        "ln_final.hook_normalized",
    ]

    try:
        bridge_activations: Dict[str, torch.Tensor] = {}

        # Register hooks on bridge
        def make_bridge_hook(name: str):
            def hook_fn(tensor, hook):
                if isinstance(tensor, torch.Tensor):
                    bridge_activations[name] = tensor.detach().clone()
                return tensor

            return hook_fn

        bridge_handles = []
        for hook_name in critical_hooks:
            if hook_name in bridge.hook_dict:
                hook_point = bridge.hook_dict[hook_name]
                handle = hook_point.add_hook(make_bridge_hook(hook_name))  # type: ignore[func-returns-value]
                bridge_handles.append(handle)

        # Run bridge forward pass
        with torch.no_grad():
            _ = bridge(test_text)

        # Clean up hooks
        for handle in bridge_handles:
            if handle is not None:
                handle.remove()

        if reference_model is None:
            # No reference - just verify activations were captured
            captured_count = len(bridge_activations)
            return BenchmarkResult(
                name="critical_forward_hooks",
                severity=BenchmarkSeverity.INFO,
                message=f"Bridge captured {captured_count}/{len(critical_hooks)} critical hooks",
                details={"captured": captured_count, "expected": len(critical_hooks)},
            )

        # Compare with reference model
        reference_activations: Dict[str, torch.Tensor] = {}

        def make_reference_hook(name: str):
            def hook_fn(tensor, hook):
                if isinstance(tensor, torch.Tensor):
                    reference_activations[name] = tensor.detach().clone()
                return tensor

            return hook_fn

        reference_handles = []
        for hook_name in critical_hooks:
            if hook_name in reference_model.hook_dict:
                hook_point = reference_model.hook_dict[hook_name]
                handle = hook_point.add_hook(make_reference_hook(hook_name))  # type: ignore[func-returns-value]
                reference_handles.append(handle)

        # Run reference forward pass
        with torch.no_grad():
            _ = reference_model(test_text)

        # Clean up hooks
        for handle in reference_handles:
            if handle is not None:
                handle.remove()

        # Compare activations
        # Only compare hooks that exist in BOTH models
        # If a hook is missing from reference but exists in bridge, that's fine (bridge has more hooks)
        # If a hook is missing from bridge but exists in reference, that's a problem
        mismatches = []
        bridge_missing = []  # Hooks in reference but not in bridge (BAD)
        reference_missing = []  # Hooks in bridge but not in reference (OK - bridge has extras)

        for hook_name in critical_hooks:
            if hook_name not in bridge_activations and hook_name not in reference_activations:
                # Neither has it - skip
                continue
            if hook_name not in bridge_activations:
                # Bridge is missing a hook that reference has - this is a problem
                bridge_missing.append(f"{hook_name}: Not found in Bridge")
                continue
            if hook_name not in reference_activations:
                # Reference doesn't have a hook that bridge has - this is fine (bridge has more)
                reference_missing.append(
                    f"{hook_name}: Not in Reference (Bridge has additional hooks)"
                )
                continue

            bridge_tensor = bridge_activations[hook_name]
            reference_tensor = reference_activations[hook_name]

            # Check shapes
            if cross_model:
                # Use relaxed dimensional matching for cross-model comparison
                is_compatible, error_msg = validate_hook_shape_compatibility(
                    bridge_tensor.shape, reference_tensor.shape, hook_name, cross_model=True
                )
                if not is_compatible:
                    mismatches.append(f"{hook_name}: {error_msg}")
                    continue
                # Skip value comparison for cross-model (different architectures have different values)
                # We only check that hooks exist, fire, and have compatible structure
            else:
                # Handle batch dimension differences (see forward_hooks)
                if bridge_tensor.shape != reference_tensor.shape:
                    if (
                        bridge_tensor.ndim == reference_tensor.ndim - 1
                        and reference_tensor.shape[0] == 1
                        and bridge_tensor.shape == reference_tensor.shape[1:]
                    ):
                        bridge_tensor = bridge_tensor.unsqueeze(0)
                    elif (
                        reference_tensor.ndim == bridge_tensor.ndim - 1
                        and bridge_tensor.shape[0] == 1
                        and reference_tensor.shape == bridge_tensor.shape[1:]
                    ):
                        reference_tensor = reference_tensor.unsqueeze(0)
                    else:
                        mismatches.append(
                            f"{hook_name}: Shape mismatch - Bridge{bridge_tensor.shape} vs Ref{reference_tensor.shape}"
                        )
                        continue

                # Only compare values for same-model comparison
                if not safe_allclose(bridge_tensor, reference_tensor, atol=tolerance, rtol=0.0):
                    max_diff = torch.max(
                        torch.abs(bridge_tensor.float() - reference_tensor.float())
                    ).item()
                    mismatches.append(f"{hook_name}: max_diff={max_diff:.6f}")

        # Filter out hooks expected to be missing in bridge models.
        if bridge_missing:
            bridge_missing = _filter_expected_missing(bridge_missing)

        if bridge_missing:
            return BenchmarkResult(
                name="critical_forward_hooks",
                severity=BenchmarkSeverity.DANGER,
                message=f"Bridge is missing {len(bridge_missing)} critical hooks that exist in reference",
                details={"missing_from_bridge": bridge_missing},
                passed=False,
            )

        # Report if reference is missing hooks that bridge has (INFO - bridge has extras)
        if reference_missing and not mismatches:
            return BenchmarkResult(
                name="critical_forward_hooks",
                severity=BenchmarkSeverity.INFO,
                message=f"All common hooks match. Bridge has {len(reference_missing)} additional hooks not in reference.",
                details={
                    "bridge_extras": reference_missing,
                    "compared": len(critical_hooks) - len(reference_missing),
                },
            )

        if mismatches:
            # Filter out known architectural differences
            significant_mismatches = [m for m in mismatches if "hook_z" not in m]

            if significant_mismatches:
                return BenchmarkResult(
                    name="critical_forward_hooks",
                    severity=BenchmarkSeverity.DANGER,
                    message=f"Found {len(significant_mismatches)} significant mismatches in critical hooks",
                    details={
                        "mismatches": significant_mismatches[:5],
                        "bridge_extras": reference_missing,
                    },
                    passed=False,
                )
            else:
                return BenchmarkResult(
                    name="critical_forward_hooks",
                    severity=BenchmarkSeverity.WARNING,
                    message="All mismatches due to known architectural differences (hook_z shape)",
                    details={
                        "total_hooks": len(critical_hooks),
                        "bridge_extras": reference_missing,
                    },
                )

        compared_count = len(critical_hooks) - len(reference_missing) - len(bridge_missing)
        return BenchmarkResult(
            name="critical_forward_hooks",
            severity=BenchmarkSeverity.INFO,
            message=f"All {compared_count} common critical hooks match",
            details={
                "matched": compared_count,
                "bridge_extras": len(reference_missing),
                "skipped": len(bridge_missing),
            },
        )

    except Exception as e:
        import traceback

        return BenchmarkResult(
            name="critical_forward_hooks",
            severity=BenchmarkSeverity.ERROR,
            message=f"Critical hooks check failed: {str(e)}",
            details={
                "error_type": type(e).__name__,
                "error_message": str(e),
                "traceback": traceback.format_exc(),
            },
            passed=False,
        )


def benchmark_hook_functionality(
    bridge: TransformerBridge,
    test_text: str,
    reference_model: Optional[HookedTransformer] = None,
    atol: float = 2e-3,
    cross_model: bool = False,
) -> BenchmarkResult:
    """Benchmark hook system functionality through ablation effects.

    Args:
        bridge: TransformerBridge model to test
        test_text: Input text for testing
        reference_model: Optional HookedTransformer reference model
        atol: Absolute tolerance for effect comparison
        cross_model: If True, skips this test as ablation effects require same architecture

    Returns:
        BenchmarkResult with hook functionality comparison details
    """
    # Skip ablation tests for cross-model comparison (requires same architecture)
    if cross_model and reference_model is not None:
        return BenchmarkResult(
            name="hook_functionality",
            severity=BenchmarkSeverity.INFO,
            message="Skipped - ablation tests require same model architecture",
            details={"reason": "cross_model_skip"},
        )

    try:
        # For GQA models, V/K tensors have fewer heads than Q
        # Use head 0 which always exists, or last head if we want to test a later one
        # We need to dynamically determine the number of heads available
        head_to_ablate = 0  # Use first head which always exists

        def ablation_hook(activation, hook):
            # Zero out an attention head in layer 0
            # Clone to avoid in-place modification of autograd views
            activation = activation.clone()
            # For GQA models, the head dimension may be smaller than n_heads
            n_heads = activation.shape[2]
            head_idx = min(head_to_ablate, n_heads - 1)
            activation[:, :, head_idx, :] = 0
            return activation

        # Test bridge
        bridge_original = bridge(test_text, return_type="loss")
        bridge_ablated = bridge.run_with_hooks(
            test_text, return_type="loss", fwd_hooks=[("blocks.0.attn.hook_v", ablation_hook)]
        )
        bridge_effect = bridge_ablated - bridge_original

        if reference_model is None:
            # No reference - just verify ablation had an effect
            effect_magnitude = abs(bridge_effect.item())
            if effect_magnitude < 1e-6:
                return BenchmarkResult(
                    name="hook_functionality",
                    severity=BenchmarkSeverity.WARNING,
                    message=f"Ablation had minimal effect: {effect_magnitude:.6f}",
                    details={"effect": effect_magnitude},
                )

            return BenchmarkResult(
                name="hook_functionality",
                severity=BenchmarkSeverity.INFO,
                message=f"Ablation hook functional with effect: {effect_magnitude:.6f}",
                details={"effect": effect_magnitude},
            )

        # Test reference model
        reference_original = reference_model(test_text, return_type="loss")
        reference_ablated = reference_model.run_with_hooks(
            test_text, return_type="loss", fwd_hooks=[("blocks.0.attn.hook_v", ablation_hook)]
        )
        reference_effect = reference_ablated - reference_original

        return compare_scalars(
            bridge_effect.item(),
            reference_effect.item(),
            atol=atol,
            name="hook_functionality",
        )

    except Exception as e:
        import traceback

        return BenchmarkResult(
            name="hook_functionality",
            severity=BenchmarkSeverity.ERROR,
            message=f"Hook functionality check failed: {str(e)}",
            details={
                "error_type": type(e).__name__,
                "error_message": str(e),
                "traceback": traceback.format_exc(),
            },
            passed=False,
        )
