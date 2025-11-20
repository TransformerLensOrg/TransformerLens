"""Hook structure validation benchmarks - cross-model compatible.

This module provides structure-only validation of hooks that can work across
different model architectures. It checks hook existence, registration, firing,
and shape compatibility without comparing activation values.
"""

from typing import Dict, Optional

import torch

from transformer_lens import HookedTransformer
from transformer_lens.benchmarks.utils import BenchmarkResult, BenchmarkSeverity
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
    # For GQA (Grouped Query Attention) models, k/v hooks may have different ranks
    # GPT-2: (batch, seq, n_heads, d_head) = 4D
    # Gemma/Llama with GQA: (batch, seq, d_head) = 3D (heads are already collapsed)
    # This is expected and fine - both are valid attention representations
    gqa_attention_hooks = ["hook_q", "hook_k", "hook_v", "hook_z"]
    is_gqa_hook = any(pattern in hook_name for pattern in gqa_attention_hooks)

    # Attention pattern hooks have shape [batch, n_heads, seq_q, seq_k]
    # Different models can have different numbers of heads
    is_attention_pattern_hook = "hook_pattern" in hook_name or "hook_attn_scores" in hook_name

    # Same rank (number of dimensions) is required, except for GQA attention hooks
    if len(target_shape) != len(reference_shape):
        if is_gqa_hook:
            # For GQA hooks, different ranks are okay - just verify batch and sequence dims match
            if len(target_shape) >= 2 and len(reference_shape) >= 2:
                if target_shape[0] != reference_shape[0]:
                    return (
                        False,
                        f"Batch dimension mismatch: {target_shape[0]} vs {reference_shape[0]}",
                    )
                if target_shape[1] != reference_shape[1]:
                    return (
                        False,
                        f"Sequence dimension mismatch: {target_shape[1]} vs {reference_shape[1]}",
                    )
                # Rank mismatch is fine for GQA - different attention implementations
                return True, None
            else:
                return False, f"Invalid tensor rank: {len(target_shape)} or {len(reference_shape)}"
        return False, f"Rank mismatch: {len(target_shape)} vs {len(reference_shape)}"

    # For each dimension, check compatibility
    for i, (target_dim, ref_dim) in enumerate(zip(target_shape, reference_shape)):
        if i == 0:  # Batch dimension
            # Should be same (both use same test input)
            if target_dim != ref_dim:
                return False, f"Batch dimension mismatch: {target_dim} vs {ref_dim}"
        elif i == 1:  # Usually sequence dimension, but n_heads for attention patterns
            if is_attention_pattern_hook:
                # For attention patterns: [batch, n_heads, seq_q, seq_k]
                # Dimension 1 is n_heads, which can differ between models
                # Just verify it's valid
                if target_dim <= 0 or ref_dim <= 0:
                    return False, f"Invalid n_heads dimension: {target_dim} vs {ref_dim}"
            else:
                # For other hooks, dimension 1 is sequence - should be same
                if target_dim != ref_dim:
                    return False, f"Sequence dimension mismatch: {target_dim} vs {ref_dim}"
        elif i >= 2 and is_attention_pattern_hook:
            # For attention patterns, dimensions 2 and 3 are seq_q and seq_k
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


def benchmark_backward_hooks_structure(
    bridge: TransformerBridge,
    test_text: str,
    reference_model: Optional[HookedTransformer] = None,
    prepend_bos: Optional[bool] = None,
    cross_model: bool = False,
) -> BenchmarkResult:
    """Benchmark backward hooks for structural correctness (existence, firing, shapes).

    This checks:
    - All reference backward hooks exist in bridge
    - Hooks can be registered
    - Hooks fire during backward pass
    - Gradient tensor shapes are compatible (allows cross-model comparison)

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
        bridge_grads: Dict[str, torch.Tensor] = {}
        reference_grads: Dict[str, torch.Tensor] = {}

        # Get all hook names that support gradients
        if reference_model is not None:
            hook_names = list(reference_model.hook_dict.keys())
        else:
            hook_names = list(bridge.hook_dict.keys())

        # Filter to hooks that typically have gradients
        grad_hook_names = [
            name
            for name in hook_names
            if any(
                keyword in name
                for keyword in [
                    "hook_embed",
                    "hook_pos_embed",
                    "hook_resid",
                    "hook_q",
                    "hook_k",
                    "hook_v",
                    "hook_z",
                    "hook_result",
                    "hook_mlp_out",
                    "hook_pre",
                    "hook_post",
                ]
            )
        ]

        # Register backward hooks on bridge
        def make_bridge_backward_hook(name: str):
            def hook_fn(grad):
                if grad is not None and isinstance(grad, torch.Tensor):
                    bridge_grads[name] = grad.detach().clone()
                elif isinstance(grad, tuple) and len(grad) > 0:
                    if grad[0] is not None and isinstance(grad[0], torch.Tensor):
                        bridge_grads[name] = grad[0].detach().clone()
                return grad

            return hook_fn

        bridge_handles = []
        missing_from_bridge = []
        for hook_name in grad_hook_names:
            if hook_name in bridge.hook_dict:
                hook_point = bridge.hook_dict[hook_name]
                handle = hook_point.add_hook(make_bridge_backward_hook(hook_name), dir="bwd")  # type: ignore[func-returns-value]
                bridge_handles.append((hook_name, handle))
            else:
                missing_from_bridge.append(hook_name)

        # Run bridge forward + backward pass
        if prepend_bos is not None:
            logits = bridge(test_text, prepend_bos=prepend_bos)
        else:
            logits = bridge(test_text)

        loss = logits[:, -1, :].sum()
        loss.backward()

        # Clean up bridge hooks
        for hook_name, handle in bridge_handles:
            if handle is not None:
                handle.remove()

        # Check for hooks that didn't fire
        registered_hooks = {name for name, _ in bridge_handles}
        hooks_that_didnt_fire = registered_hooks - set(bridge_grads.keys())

        if reference_model is None:
            # No reference - just verify gradients were captured
            if hooks_that_didnt_fire:
                return BenchmarkResult(
                    name="backward_hooks_structure",
                    severity=BenchmarkSeverity.WARNING,
                    message=f"{len(hooks_that_didnt_fire)}/{len(registered_hooks)} backward hooks didn't fire",
                    details={
                        "captured": len(bridge_grads),
                        "registered": len(registered_hooks),
                        "didnt_fire": list(hooks_that_didnt_fire)[:10],
                    },
                )

            return BenchmarkResult(
                name="backward_hooks_structure",
                severity=BenchmarkSeverity.INFO,
                message=f"Bridge captured {len(bridge_grads)} backward hook gradients",
                details={"gradient_count": len(bridge_grads)},
            )

        # Register backward hooks on reference
        def make_reference_backward_hook(name: str):
            def hook_fn(grad):
                if grad is not None and isinstance(grad, torch.Tensor):
                    reference_grads[name] = grad.detach().clone()
                elif isinstance(grad, tuple) and len(grad) > 0:
                    if grad[0] is not None and isinstance(grad[0], torch.Tensor):
                        reference_grads[name] = grad[0].detach().clone()
                return grad

            return hook_fn

        reference_handles = []
        for hook_name in grad_hook_names:
            if hook_name in reference_model.hook_dict:
                hook_point = reference_model.hook_dict[hook_name]
                handle = hook_point.add_hook(make_reference_backward_hook(hook_name), dir="bwd")  # type: ignore[func-returns-value]
                reference_handles.append(handle)

        # Run reference forward + backward pass
        if prepend_bos is not None:
            ref_logits = reference_model(test_text, prepend_bos=prepend_bos)
        else:
            ref_logits = reference_model(test_text)

        ref_loss = ref_logits[:, -1, :].sum()
        ref_loss.backward()

        # Clean up reference hooks
        for handle in reference_handles:
            if handle is not None:
                handle.remove()

        # CRITICAL CHECK: Bridge must have all backward hooks that reference has
        if missing_from_bridge:
            return BenchmarkResult(
                name="backward_hooks_structure",
                severity=BenchmarkSeverity.DANGER,
                message=f"Bridge MISSING {len(missing_from_bridge)} backward hooks from reference",
                details={
                    "missing_count": len(missing_from_bridge),
                    "missing_hooks": missing_from_bridge[:20],
                    "total_reference_hooks": len(grad_hook_names),
                },
                passed=False,
            )

        # CRITICAL CHECK: All registered hooks must fire
        if hooks_that_didnt_fire:
            return BenchmarkResult(
                name="backward_hooks_structure",
                severity=BenchmarkSeverity.DANGER,
                message=f"{len(hooks_that_didnt_fire)} backward hooks DIDN'T FIRE",
                details={
                    "didnt_fire_count": len(hooks_that_didnt_fire),
                    "didnt_fire_hooks": list(hooks_that_didnt_fire)[:20],
                    "total_registered": len(registered_hooks),
                },
                passed=False,
            )

        # Check gradient shapes
        common_hooks = set(bridge_grads.keys()) & set(reference_grads.keys())
        shape_mismatches = []

        for hook_name in sorted(common_hooks):
            bridge_grad = bridge_grads[hook_name]
            reference_grad = reference_grads[hook_name]

            if cross_model:
                # Use relaxed shape matching for cross-model comparison
                is_compatible, error_msg = validate_hook_shape_compatibility(
                    bridge_grad.shape, reference_grad.shape, hook_name
                )
                if not is_compatible:
                    shape_mismatches.append(f"{hook_name}: {error_msg}")
            else:
                # Exact shape matching for same-model comparison
                if bridge_grad.shape != reference_grad.shape:
                    shape_mismatches.append(
                        f"{hook_name}: Shape {bridge_grad.shape} vs {reference_grad.shape}"
                    )

        if shape_mismatches:
            return BenchmarkResult(
                name="backward_hooks_structure",
                severity=BenchmarkSeverity.DANGER,
                message=f"Found {len(shape_mismatches)}/{len(common_hooks)} hooks with gradient shape incompatibilities",
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
            name="backward_hooks_structure",
            severity=BenchmarkSeverity.INFO,
            message=f"All {len(common_hooks)} backward hooks structurally compatible ({ref_type})",
            details={"hook_count": len(common_hooks), "cross_model": cross_model},
        )

    except Exception as e:
        return BenchmarkResult(
            name="backward_hooks_structure",
            severity=BenchmarkSeverity.ERROR,
            message=f"Backward hooks structure check failed: {str(e)}",
            passed=False,
        )


def benchmark_activation_cache_structure(
    bridge: TransformerBridge,
    test_text: str,
    reference_model: Optional[HookedTransformer] = None,
    prepend_bos: Optional[bool] = None,
    cross_model: bool = False,
) -> BenchmarkResult:
    """Benchmark activation cache for structural correctness (keys, shapes).

    This checks:
    - Cache returns expected keys
    - Cache tensor shapes are compatible
    - run_with_cache works correctly

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
        # Run bridge with cache
        if prepend_bos is not None:
            _, bridge_cache = bridge.run_with_cache(test_text, prepend_bos=prepend_bos)
        else:
            _, bridge_cache = bridge.run_with_cache(test_text)

        bridge_keys = set(bridge_cache.keys())

        if reference_model is None:
            # No reference - just verify cache works
            if len(bridge_keys) == 0:
                return BenchmarkResult(
                    name="activation_cache_structure",
                    severity=BenchmarkSeverity.DANGER,
                    message="Cache is empty",
                    passed=False,
                )

            return BenchmarkResult(
                name="activation_cache_structure",
                severity=BenchmarkSeverity.INFO,
                message=f"Cache captured {len(bridge_keys)} activations",
                details={"cache_size": len(bridge_keys)},
            )

        # Run reference with cache
        if prepend_bos is not None:
            _, ref_cache = reference_model.run_with_cache(test_text, prepend_bos=prepend_bos)
        else:
            _, ref_cache = reference_model.run_with_cache(test_text)

        ref_keys = set(ref_cache.keys())

        # Check for missing keys
        missing_keys = ref_keys - bridge_keys

        # Filter out expected missing hooks in cross-model mode
        if cross_model and missing_keys:
            # In cross-model mode, some hooks are expected to be missing due to architectural differences
            # For example, rotary embedding models (Gemma, LLaMA) don't have hook_pos_embed
            expected_missing_patterns = ["hook_pos_embed"]
            actual_missing = [
                k
                for k in missing_keys
                if not any(pattern in k for pattern in expected_missing_patterns)
            ]
            missing_keys = set(actual_missing)

        if missing_keys:
            return BenchmarkResult(
                name="activation_cache_structure",
                severity=BenchmarkSeverity.DANGER,
                message=f"Cache MISSING {len(missing_keys)} keys from reference",
                details={
                    "missing_count": len(missing_keys),
                    "missing_keys": list(missing_keys)[:20],
                    "total_reference_keys": len(ref_keys),
                },
                passed=False,
            )

        # Check shapes of common keys
        common_keys = bridge_keys & ref_keys
        shape_mismatches = []

        for key in sorted(common_keys):
            bridge_tensor = bridge_cache[key]
            ref_tensor = ref_cache[key]

            if cross_model:
                # Use relaxed shape matching for cross-model comparison
                is_compatible, error_msg = validate_hook_shape_compatibility(
                    bridge_tensor.shape, ref_tensor.shape, key
                )
                if not is_compatible:
                    shape_mismatches.append(f"{key}: {error_msg}")
            else:
                # Exact shape matching for same-model comparison
                if bridge_tensor.shape != ref_tensor.shape:
                    shape_mismatches.append(
                        f"{key}: Shape {bridge_tensor.shape} vs {ref_tensor.shape}"
                    )

        if shape_mismatches:
            return BenchmarkResult(
                name="activation_cache_structure",
                severity=BenchmarkSeverity.DANGER,
                message=f"Found {len(shape_mismatches)}/{len(common_keys)} cache entries with shape incompatibilities",
                details={
                    "total_keys": len(common_keys),
                    "shape_mismatches": len(shape_mismatches),
                    "sample_mismatches": shape_mismatches[:5],
                    "cross_model": cross_model,
                },
                passed=False,
            )

        ref_type = "cross-model reference" if cross_model else "same-model reference"
        return BenchmarkResult(
            name="activation_cache_structure",
            severity=BenchmarkSeverity.INFO,
            message=f"All {len(common_keys)} cache entries structurally compatible ({ref_type})",
            details={"cache_size": len(common_keys), "cross_model": cross_model},
        )

    except Exception as e:
        import traceback

        return BenchmarkResult(
            name="activation_cache_structure",
            severity=BenchmarkSeverity.ERROR,
            message=f"Activation cache structure check failed: {str(e)}",
            details={
                "error_type": type(e).__name__,
                "error_message": str(e),
                "traceback": traceback.format_exc(),
            },
            passed=False,
        )
