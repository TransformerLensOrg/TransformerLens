"""Hook registration and behavior benchmarks for TransformerBridge."""

from typing import Dict, Optional

import torch

from transformer_lens import HookedTransformer
from transformer_lens.benchmarks.utils import (
    BenchmarkResult,
    BenchmarkSeverity,
    compare_scalars,
)
from transformer_lens.model_bridge import TransformerBridge


def validate_hook_shape_compatibility(
    target_shape: tuple,
    reference_shape: tuple,
    hook_name: str,
) -> tuple[bool, Optional[str]]:
    """Validate that hook shapes have compatible structure across different models.

    This allows comparing hooks from different models (e.g., Llama vs GPT-2) by checking
    structural compatibility rather than exact shape matching.

    Args:
        target_shape: Shape of the tensor from the target model
        reference_shape: Shape of the tensor from the reference model
        hook_name: Name of the hook (for error messages)

    Returns:
        Tuple of (is_compatible, error_message)
        - is_compatible: True if shapes are structurally compatible
        - error_message: None if compatible, otherwise description of incompatibility
    """
    # Same rank (number of dimensions) is required
    if len(target_shape) != len(reference_shape):
        return False, f"Rank mismatch: {len(target_shape)} vs {len(reference_shape)}"

    # For each dimension, check compatibility
    for i, (target_dim, ref_dim) in enumerate(zip(target_shape, reference_shape)):
        if i == 0:  # Batch dimension
            # Should be same (both use same test input)
            if target_dim != ref_dim:
                return False, f"Batch dimension mismatch: {target_dim} vs {ref_dim}"
        elif i == 1:  # Sequence dimension
            # Should be same (both use same test input)
            if target_dim != ref_dim:
                return False, f"Sequence dimension mismatch: {target_dim} vs {ref_dim}"
        else:  # Model-specific dimensions (d_model, n_heads, d_head, etc.)
            # Can differ between models - just verify it's valid
            if target_dim <= 0:
                return False, f"Invalid dimension {i}: {target_dim} <= 0"
            if ref_dim <= 0:
                return False, f"Invalid reference dimension {i}: {ref_dim} <= 0"

    return True, None


def benchmark_forward_hooks_structure(
    bridge: TransformerBridge,
    test_text: str,
    reference_model: Optional[HookedTransformer] = None,
    prepend_bos: Optional[bool] = None,
    cross_model: bool = False,
) -> BenchmarkResult:
    """Benchmark forward hooks for structural correctness (existence, firing, shapes).

    This checks:
    - All reference hooks exist in bridge
    - Hooks can be registered
    - Hooks fire during forward pass
    - Hook tensor shapes are compatible (allows cross-model comparison)

    Args:
        bridge: TransformerBridge model to test
        test_text: Input text for testing
        reference_model: Optional HookedTransformer for comparison
        prepend_bos: Whether to prepend BOS token. If None, uses model default.
        cross_model: If True, uses relaxed shape matching for cross-model comparison

    Returns:
        BenchmarkResult with structural validation details
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

        # Check for hooks that didn't fire
        registered_hooks = {name for name, _ in bridge_handles}
        hooks_that_didnt_fire = registered_hooks - set(bridge_activations.keys())

        if reference_model is None:
            # No reference - just verify hooks were captured
            if hooks_that_didnt_fire:
                return BenchmarkResult(
                    name="forward_hooks_structure",
                    severity=BenchmarkSeverity.WARNING,
                    message=f"{len(hooks_that_didnt_fire)}/{len(registered_hooks)} hooks didn't fire",
                    details={
                        "captured": len(bridge_activations),
                        "registered": len(registered_hooks),
                        "didnt_fire": list(hooks_that_didnt_fire)[:10],
                    },
                )

            return BenchmarkResult(
                name="forward_hooks_structure",
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
        if missing_from_bridge:
            return BenchmarkResult(
                name="forward_hooks_structure",
                severity=BenchmarkSeverity.DANGER,
                message=f"Bridge MISSING {len(missing_from_bridge)} hooks from reference",
                details={
                    "missing_count": len(missing_from_bridge),
                    "missing_hooks": missing_from_bridge[:20],
                    "total_reference_hooks": len(hook_names),
                },
                passed=False,
            )

        # CRITICAL CHECK: All registered hooks must fire
        if hooks_that_didnt_fire:
            return BenchmarkResult(
                name="forward_hooks_structure",
                severity=BenchmarkSeverity.DANGER,
                message=f"{len(hooks_that_didnt_fire)} hooks DIDN'T FIRE during forward pass",
                details={
                    "didnt_fire_count": len(hooks_that_didnt_fire),
                    "didnt_fire_hooks": list(hooks_that_didnt_fire)[:20],
                    "total_registered": len(registered_hooks),
                },
                passed=False,
            )

        # Check shapes
        common_hooks = set(bridge_activations.keys()) & set(reference_activations.keys())
        shape_mismatches = []

        for hook_name in sorted(common_hooks):
            bridge_tensor = bridge_activations[hook_name]
            reference_tensor = reference_activations[hook_name]

            if cross_model:
                # Use relaxed shape matching for cross-model comparison
                is_compatible, error_msg = validate_hook_shape_compatibility(
                    bridge_tensor.shape, reference_tensor.shape, hook_name
                )
                if not is_compatible:
                    shape_mismatches.append(f"{hook_name}: {error_msg}")
            else:
                # Exact shape matching for same-model comparison
                if bridge_tensor.shape != reference_tensor.shape:
                    shape_mismatches.append(
                        f"{hook_name}: Shape {bridge_tensor.shape} vs {reference_tensor.shape}"
                    )

        if shape_mismatches:
            return BenchmarkResult(
                name="forward_hooks_structure",
                severity=BenchmarkSeverity.DANGER,
                message=f"Found {len(shape_mismatches)}/{len(common_hooks)} hooks with shape incompatibilities",
                details={
                    "total_hooks": len(common_hooks),
                    "shape_mismatches": len(shape_mismatches),
                    "sample_mismatches": shape_mismatches[:5],
                    "cross_model": cross_model,
                },
                passed=False,
            )

        ref_type = "cross-model reference" if cross_model else "same-model reference"
        return BenchmarkResult(
            name="forward_hooks_structure",
            severity=BenchmarkSeverity.INFO,
            message=f"All {len(common_hooks)} forward hooks structurally compatible ({ref_type})",
            details={"hook_count": len(common_hooks), "cross_model": cross_model},
        )

    except Exception as e:
        return BenchmarkResult(
            name="forward_hooks_structure",
            severity=BenchmarkSeverity.ERROR,
            message=f"Forward hooks structure check failed: {str(e)}",
            passed=False,
        )


def benchmark_forward_hooks_values(
    bridge: TransformerBridge,
    test_text: str,
    reference_model: HookedTransformer,
    tolerance: float = 0.5,
    prepend_bos: Optional[bool] = None,
) -> BenchmarkResult:
    """Benchmark forward hooks for value equivalence (requires same-model reference).

    This checks that hook activation values match between bridge and reference.
    Should only be called when reference_model is the same architecture as bridge.

    Args:
        bridge: TransformerBridge model to test
        test_text: Input text for testing
        reference_model: HookedTransformer reference (must be same architecture)
        tolerance: Tolerance for activation matching
        prepend_bos: Whether to prepend BOS token. If None, uses model default.

    Returns:
        BenchmarkResult with value comparison details
    """
    try:
        bridge_activations: Dict[str, torch.Tensor] = {}
        reference_activations: Dict[str, torch.Tensor] = {}

        hook_names = list(reference_model.hook_dict.keys())

        # Register hooks on bridge
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
        for hook_name in hook_names:
            if hook_name in bridge.hook_dict:
                hook_point = bridge.hook_dict[hook_name]
                handle = hook_point.add_hook(make_bridge_hook(hook_name))  # type: ignore[func-returns-value]
                bridge_handles.append((hook_name, handle))

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

        # Register hooks on reference
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

        # Compare activation values
        common_hooks = set(bridge_activations.keys()) & set(reference_activations.keys())
        value_mismatches = []

        for hook_name in sorted(common_hooks):
            bridge_tensor = bridge_activations[hook_name]
            reference_tensor = reference_activations[hook_name]

            # Check values
            if not torch.allclose(bridge_tensor, reference_tensor, atol=tolerance, rtol=0):
                max_diff = torch.max(torch.abs(bridge_tensor - reference_tensor)).item()
                mean_diff = torch.mean(torch.abs(bridge_tensor - reference_tensor)).item()
                value_mismatches.append(
                    f"{hook_name}: max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f}"
                )

        if value_mismatches:
            # Filter out known architectural differences
            significant_mismatches = [
                m
                for m in value_mismatches
                if "hook_attn_scores" not in m  # Exclude attn_scores (has inf from masking)
            ]

            if significant_mismatches:
                return BenchmarkResult(
                    name="forward_hooks_values",
                    severity=BenchmarkSeverity.DANGER,
                    message=f"Found {len(significant_mismatches)}/{len(common_hooks)} hooks with value mismatches",
                    details={
                        "total_hooks": len(common_hooks),
                        "mismatches": len(significant_mismatches),
                        "sample_mismatches": significant_mismatches[:5],
                        "tolerance": tolerance,
                    },
                    passed=False,
                )
            else:
                return BenchmarkResult(
                    name="forward_hooks_values",
                    severity=BenchmarkSeverity.WARNING,
                    message=f"All mismatches due to known differences ({len(value_mismatches)} hooks)",
                    details={"total_hooks": len(common_hooks), "tolerance": tolerance},
                )

        return BenchmarkResult(
            name="forward_hooks_values",
            severity=BenchmarkSeverity.INFO,
            message=f"All {len(common_hooks)} forward hooks match within tolerance",
            details={"hook_count": len(common_hooks), "tolerance": tolerance},
        )

    except Exception as e:
        return BenchmarkResult(
            name="forward_hooks_values",
            severity=BenchmarkSeverity.ERROR,
            message=f"Forward hooks values check failed: {str(e)}",
            passed=False,
        )


def benchmark_hook_registry(
    bridge: TransformerBridge,
    reference_model: Optional[HookedTransformer] = None,
) -> BenchmarkResult:
    """Benchmark hook registry completeness.

    Args:
        bridge: TransformerBridge model to test
        reference_model: Optional HookedTransformer reference model

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
) -> BenchmarkResult:
    """Benchmark all forward hooks for activation matching.

    Args:
        bridge: TransformerBridge model to test
        test_text: Input text for testing
        reference_model: Optional HookedTransformer for comparison
        tolerance: Tolerance for activation matching (fraction of mismatches allowed)
        prepend_bos: Whether to prepend BOS token. If None, uses model default.

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
            if bridge_tensor.shape != reference_tensor.shape:
                mismatches.append(
                    f"{hook_name}: Shape mismatch - Bridge{bridge_tensor.shape} vs Ref{reference_tensor.shape}"
                )
                continue

            # Check values
            if not torch.allclose(bridge_tensor, reference_tensor, atol=tolerance, rtol=0):
                max_diff = torch.max(torch.abs(bridge_tensor - reference_tensor)).item()
                mean_diff = torch.mean(torch.abs(bridge_tensor - reference_tensor)).item()
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
) -> BenchmarkResult:
    """Benchmark critical forward hooks commonly used in interpretability research.

    Args:
        bridge: TransformerBridge model to test
        test_text: Input text for testing
        reference_model: Optional HookedTransformer reference model
        tolerance: Tolerance for activation comparison

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

            if bridge_tensor.shape != reference_tensor.shape:
                mismatches.append(
                    f"{hook_name}: Shape mismatch - Bridge{bridge_tensor.shape} vs Ref{reference_tensor.shape}"
                )
                continue

            if not torch.allclose(bridge_tensor, reference_tensor, atol=tolerance, rtol=0):
                max_diff = torch.max(torch.abs(bridge_tensor - reference_tensor)).item()
                mismatches.append(f"{hook_name}: max_diff={max_diff:.6f}")

        # Check if bridge is missing critical hooks (BAD)
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
        return BenchmarkResult(
            name="critical_forward_hooks",
            severity=BenchmarkSeverity.ERROR,
            message=f"Critical hooks check failed: {str(e)}",
            passed=False,
        )


def benchmark_hook_functionality(
    bridge: TransformerBridge,
    test_text: str,
    reference_model: Optional[HookedTransformer] = None,
    atol: float = 2e-3,
) -> BenchmarkResult:
    """Benchmark hook system functionality through ablation effects.

    Args:
        bridge: TransformerBridge model to test
        test_text: Input text for testing
        reference_model: Optional HookedTransformer reference model
        atol: Absolute tolerance for effect comparison

    Returns:
        BenchmarkResult with hook functionality comparison details
    """
    try:
        # For GQA models, V/K tensors have fewer heads than Q
        # Use head 0 which always exists, or last head if we want to test a later one
        # We need to dynamically determine the number of heads available
        head_to_ablate = 0  # Use first head which always exists

        def ablation_hook(activation, hook):
            # Zero out an attention head in layer 0
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
        return BenchmarkResult(
            name="hook_functionality",
            severity=BenchmarkSeverity.ERROR,
            message=f"Hook functionality check failed: {str(e)}",
            passed=False,
        )
