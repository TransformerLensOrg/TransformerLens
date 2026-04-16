"""Hook registration and behavior benchmarks for TransformerBridge."""

from typing import Dict, Optional

import torch

from transformer_lens import HookedTransformer
from transformer_lens.benchmarks.utils import (
    BenchmarkResult,
    BenchmarkSeverity,
    compare_activation_dicts,
    compare_scalars,
    filter_expected_missing_hooks,
    make_capture_hook,
)
from transformer_lens.model_bridge import TransformerBridge


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

        # Filter out hooks that are expected to differ due to architectural differences.
        if missing_hooks:
            missing_hooks = set(filter_expected_missing_hooks(missing_hooks))

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
        bridge_handles = []
        missing_from_bridge = []
        for hook_name in hook_names:
            if hook_name in bridge.hook_dict:
                hook_point = bridge.hook_dict[hook_name]
                handle = hook_point.add_hook(make_capture_hook(bridge_activations, hook_name))  # type: ignore[func-returns-value]
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
        reference_handles = []
        for hook_name in hook_names:
            if hook_name in reference_model.hook_dict:
                hook_point = reference_model.hook_dict[hook_name]
                handle = hook_point.add_hook(make_capture_hook(reference_activations, hook_name))  # type: ignore[func-returns-value]
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

        # CRITICAL CHECK: Bridge must have all hooks that reference has.
        # Filter out hooks that bridge models inherently don't have.
        if missing_from_bridge:
            missing_from_bridge = filter_expected_missing_hooks(missing_from_bridge)

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
            hooks_that_didnt_fire = set(filter_expected_missing_hooks(hooks_that_didnt_fire))

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
        mismatches = compare_activation_dicts(
            bridge_activations, reference_activations, atol=tolerance
        )

        if mismatches:
            # Detect Bloom-style residual-merged hooks: Bloom adds residual inside
            # attn/MLP modules (dropout_add), so hook_attn_out and hook_mlp_out capture
            # attn+residual instead of just attn. This is a known HF architectural difference.
            has_bloom_blocks = any(type(m).__name__ == "BloomBlockBridge" for m in bridge.modules())
            # Filter out known architectural differences
            significant_mismatches = [
                m
                for m in mismatches
                if "hook_attn_scores" not in m  # Exclude attn_scores which have inf from masking
                and not (has_bloom_blocks and ("hook_attn_out" in m or "hook_mlp_out" in m))
                # QK norm hooks: Bridge preserves HF's 4D [batch, heads, seq, d_head]
                # while HT flattens to [batch*seq*heads, d_head]. This is an intentional
                # shape convention difference, not a computation error.
                and "q_norm" not in m and "k_norm" not in m
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


# Configuration for cfg-gated attention hooks. Each entry names a config flag
# and the hook-name stems that should fire on supporting layers when that flag
# is on. Stems are matched against `hook_dict` keys via substring; both
# block-level aliases (blocks.N.hook_X) and attn-level primaries
# (blocks.N.attn.hook_X) are accepted.
_GATED_HOOK_CONFIGS: list[tuple[str, tuple[str, ...]]] = [
    ("use_attn_result", ("hook_result",)),
    ("use_split_qkv_input", ("hook_q_input", "hook_k_input", "hook_v_input")),
    ("use_attn_in", ("hook_attn_in",)),
]


def benchmark_gated_hooks_fire(
    bridge: TransformerBridge,
    test_text: str = "The quick brown fox",
    prepend_bos: Optional[bool] = None,
) -> BenchmarkResult:
    """Verify each cfg-gated attention hook fires when its flag is enabled.

    Hooks like `hook_result`, `hook_q_input`, `hook_attn_in` exist
    unconditionally on the attention bridge but are only populated when the
    corresponding config flag is set (keeping default-path cost at zero).
    This benchmark toggles each flag in turn, runs a short forward, and asserts
    at least one layer's matching hook actually captured an activation.

    `use_attn_in` and `use_split_qkv_input` are mutually exclusive, so each
    flag runs in its own forward pass. Plain `AttentionBridge` (non-PEA/JPEA)
    adapters raise `NotImplementedError` from the setter — recorded as skipped
    rather than failed, since the applicability gate is intentional.
    """
    try:
        if not hasattr(bridge, "blocks") or not len(bridge.blocks):
            return BenchmarkResult(
                name="gated_hooks_fire",
                severity=BenchmarkSeverity.INFO,
                message="Bridge has no blocks attribute; gated-hook check skipped",
            )

        fired: dict[str, int] = {}
        skipped: list[tuple[str, str]] = []
        failed: list[tuple[str, str]] = []
        tested_flags: list[str] = []

        for flag_name, hook_stems in _GATED_HOOK_CONFIGS:
            # Force a clean baseline: all three flags off before toggling.
            for reset_flag in ("use_attn_result", "use_split_qkv_input", "use_attn_in"):
                setattr(bridge.cfg, reset_flag, False)

            setter = getattr(bridge, f"set_{flag_name}", None)
            if setter is None:
                skipped.append((flag_name, "setter missing on bridge"))
                continue
            try:
                setter(True)
            except NotImplementedError as e:
                skipped.append((flag_name, str(e).split("\n", 1)[0][:120]))
                continue
            except ValueError as e:
                # Defensive: mutual-exclusivity shouldn't trigger because we
                # reset all flags first, but record if something upstream
                # left the cfg dirty.
                skipped.append((flag_name, f"setter refused: {e}"))
                continue

            tested_flags.append(flag_name)
            try:
                activations: dict[str, torch.Tensor] = {}
                handles: list[tuple[str, object]] = []
                target_hook_names = [
                    name
                    for name in bridge.hook_dict
                    if any(f".{stem}" in name or name.endswith(stem) for stem in hook_stems)
                    # Exclude the cross-stem substring collisions, e.g. a hook
                    # named "...hook_q_input_foo" — not expected today but be
                    # defensive.
                    and any(name.rsplit(".", 1)[-1] == stem for stem in hook_stems)
                ]
                for hname in target_hook_names:
                    hp = bridge.hook_dict[hname]
                    h = hp.add_hook(make_capture_hook(activations, hname))  # type: ignore[func-returns-value]
                    handles.append((hname, h))

                with torch.no_grad():
                    if prepend_bos is not None:
                        _ = bridge(test_text, prepend_bos=prepend_bos)
                    else:
                        _ = bridge(test_text)

                for _, h in handles:
                    if h is not None and hasattr(h, "remove"):
                        h.remove()

                # Bucket fired counts per stem.
                for stem in hook_stems:
                    fired_count = sum(1 for name in activations if name.rsplit(".", 1)[-1] == stem)
                    fired[stem] = fired_count
                    if fired_count == 0 and any(
                        name.rsplit(".", 1)[-1] == stem for name in target_hook_names
                    ):
                        failed.append((flag_name, stem))
            finally:
                setter(False)

        for reset_flag in ("use_attn_result", "use_split_qkv_input", "use_attn_in"):
            setattr(bridge.cfg, reset_flag, False)

        if failed:
            return BenchmarkResult(
                name="gated_hooks_fire",
                severity=BenchmarkSeverity.DANGER,
                message=f"{len(failed)} gated hooks did not fire when their flag was enabled",
                details={
                    "failed": failed,
                    "fired_counts": fired,
                    "tested_flags": tested_flags,
                    "skipped": skipped,
                },
                passed=False,
            )

        if not tested_flags:
            return BenchmarkResult(
                name="gated_hooks_fire",
                severity=BenchmarkSeverity.INFO,
                message=(
                    "Architecture does not support any gated attention hooks "
                    f"({len(skipped)} flags skipped)"
                ),
                details={"skipped": skipped},
            )

        msg = (
            f"All gated hooks fired on their supporting layers "
            f"({sum(fired.values())} activations across {len(fired)} hook stems"
            f", {len(tested_flags)} flags tested)"
        )
        if skipped:
            msg += f"; {len(skipped)} flags not applicable to this architecture"
        return BenchmarkResult(
            name="gated_hooks_fire",
            severity=BenchmarkSeverity.INFO,
            message=msg,
            details={"fired_counts": fired, "tested_flags": tested_flags, "skipped": skipped},
        )

    except Exception as e:
        return BenchmarkResult(
            name="gated_hooks_fire",
            severity=BenchmarkSeverity.ERROR,
            message=f"Gated-hook check failed: {str(e)}",
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
    # Scale tolerance for deep models — numerical precision differences
    # accumulate through layers, especially for ln_final.hook_normalized
    # which passes through the entire model. Cap at 3x base to avoid
    # overly permissive tolerance for very deep models (70B+).
    n_layers = getattr(bridge.cfg, "n_layers", 1)
    if n_layers > 12:
        tolerance = min(tolerance * (1 + 0.05 * (n_layers - 12)), tolerance * 3.0)

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
        bridge_handles = []
        for hook_name in critical_hooks:
            if hook_name in bridge.hook_dict:
                hook_point = bridge.hook_dict[hook_name]
                handle = hook_point.add_hook(make_capture_hook(bridge_activations, hook_name))  # type: ignore[func-returns-value]
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

        reference_handles = []
        for hook_name in critical_hooks:
            if hook_name in reference_model.hook_dict:
                hook_point = reference_model.hook_dict[hook_name]
                handle = hook_point.add_hook(make_capture_hook(reference_activations, hook_name))  # type: ignore[func-returns-value]
                reference_handles.append(handle)

        # Run reference forward pass
        with torch.no_grad():
            _ = reference_model(test_text)

        # Clean up hooks
        for handle in reference_handles:
            if handle is not None:
                handle.remove()

        # Compare activations — categorize by presence
        bridge_missing = []  # Hooks in reference but not in bridge (BAD)
        reference_missing = []  # Hooks in bridge but not in reference (OK)

        for hook_name in critical_hooks:
            if hook_name not in bridge_activations and hook_name not in reference_activations:
                continue
            if hook_name not in bridge_activations:
                bridge_missing.append(f"{hook_name}: Not found in Bridge")
                continue
            if hook_name not in reference_activations:
                reference_missing.append(
                    f"{hook_name}: Not in Reference (Bridge has additional hooks)"
                )

        mismatches = compare_activation_dicts(
            bridge_activations, reference_activations, atol=tolerance
        )

        # Filter out hooks expected to be missing in bridge models.
        if bridge_missing:
            bridge_missing = filter_expected_missing_hooks(bridge_missing)

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
            # Detect Bloom-style residual-merged hooks
            has_bloom_blocks = any(type(m).__name__ == "BloomBlockBridge" for m in bridge.modules())
            # Filter out known architectural differences
            significant_mismatches = [
                m
                for m in mismatches
                if "hook_z" not in m
                and not (has_bloom_blocks and ("hook_mlp_out" in m or "hook_attn_out" in m))
            ]

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
            # Clone to avoid in-place modification of autograd views
            activation = activation.clone()
            if activation.ndim == 4:
                # Standard: [batch, seq, n_heads, d_head]
                # For GQA models, the head dimension may be smaller than n_heads
                n_heads = activation.shape[2]
                head_idx = min(head_to_ablate, n_heads - 1)
                activation[:, :, head_idx, :] = 0
            elif activation.ndim == 3:
                # Bridge with joint QKV projection (e.g., Phi-3): [batch, seq, d_model]
                # hook_conversion may not reshape when the underlying linear is a
                # combined qkv_proj. Zero out a head-sized slice instead.
                d_model = activation.shape[-1]
                n_heads = bridge.cfg.n_heads
                d_head = d_model // n_heads
                head_idx = min(head_to_ablate, n_heads - 1)
                start = head_idx * d_head
                end = start + d_head
                activation[:, :, start:end] = 0
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
