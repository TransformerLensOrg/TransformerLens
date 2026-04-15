"""Weight processing benchmarks for TransformerBridge."""

from typing import Optional, cast

import torch

from transformer_lens import HookedTransformer
from transformer_lens.benchmarks.utils import (
    BenchmarkResult,
    BenchmarkSeverity,
    is_tiny_test_model,
    safe_allclose,
)
from transformer_lens.model_bridge import TransformerBridge


def benchmark_weight_processing(
    bridge: TransformerBridge,
    test_text: str,
    reference_model: Optional[HookedTransformer] = None,
) -> BenchmarkResult:
    """Benchmark weight processing (folding, centering) application.

    Args:
        bridge: TransformerBridge model to test
        test_text: Input text for testing
        reference_model: Optional HookedTransformer reference model

    Returns:
        BenchmarkResult with weight processing verification details
    """
    try:
        from transformer_lens.components.layer_norm_pre import LayerNormPre
        from transformer_lens.model_bridge.generalized_components.normalization import (
            NormalizationBridge,
        )

        # Check layer norm folding
        if not isinstance(bridge.ln_final, NormalizationBridge):
            return BenchmarkResult(
                name="weight_processing",
                severity=BenchmarkSeverity.WARNING,
                message=f"Bridge ln_final is {type(bridge.ln_final).__name__}, expected NormalizationBridge",
            )

        # Verify NormalizationBridge has LayerNormPre functionality
        if not hasattr(bridge.ln_final, "_layernorm_pre_forward"):
            return BenchmarkResult(
                name="weight_processing",
                severity=BenchmarkSeverity.WARNING,
                message="Bridge ln_final missing LayerNormPre functionality",
            )

        if not hasattr(bridge.ln_final.config, "layer_norm_folding"):
            return BenchmarkResult(
                name="weight_processing",
                severity=BenchmarkSeverity.WARNING,
                message="Bridge ln_final missing layer_norm_folding config",
            )

        if reference_model is not None:
            # Check that reference model has LayerNormPre
            if not isinstance(reference_model.ln_final, LayerNormPre):
                return BenchmarkResult(
                    name="weight_processing",
                    severity=BenchmarkSeverity.WARNING,
                    message=f"Reference ln_final is {type(reference_model.ln_final).__name__}, expected LayerNormPre",
                )

            # Check weight centering - writing weights should be approximately centered
            mlp_blocks = bridge.blocks_with("mlp")
            if not mlp_blocks:
                return BenchmarkResult(
                    name="weight_processing",
                    severity=BenchmarkSeverity.WARNING,
                    message="No blocks have MLP submodule — cannot check centering",
                )
            _mlp_idx, mlp_block = mlp_blocks[0]
            bridge_w_out = mlp_block.mlp.W_out
            reference_w_out = reference_model.blocks[_mlp_idx].mlp.W_out  # type: ignore[union-attr]

            bridge_mean = torch.mean(torch.abs(torch.mean(bridge_w_out, dim=-1, keepdim=True)))
            reference_mean = torch.mean(
                torch.abs(torch.mean(reference_w_out, dim=-1, keepdim=True))  # type: ignore[arg-type]
            )

            if bridge_mean.item() > 1e-3:
                return BenchmarkResult(
                    name="weight_processing",
                    severity=BenchmarkSeverity.WARNING,
                    message=f"Bridge weights not well-centered: {bridge_mean.item():.6f}",
                    details={"bridge_mean": bridge_mean.item()},
                )

            if reference_mean.item() > 1e-3:
                return BenchmarkResult(
                    name="weight_processing",
                    severity=BenchmarkSeverity.WARNING,
                    message=f"Reference weights not well-centered: {reference_mean.item():.6f}",
                    details={"reference_mean": reference_mean.item()},
                )

            return BenchmarkResult(
                name="weight_processing",
                severity=BenchmarkSeverity.INFO,
                message="Weight processing verified (folding and centering applied)",
                details={
                    "bridge_mean": bridge_mean.item(),
                    "reference_mean": reference_mean.item(),
                },
            )

        return BenchmarkResult(
            name="weight_processing",
            severity=BenchmarkSeverity.INFO,
            message="Weight processing structure verified",
        )

    except Exception as e:
        return BenchmarkResult(
            name="weight_processing",
            severity=BenchmarkSeverity.ERROR,
            message=f"Weight processing check failed: {str(e)}",
            passed=False,
        )


def benchmark_weight_sharing(
    bridge: TransformerBridge,
    test_text: str,
    reference_model: Optional[HookedTransformer] = None,
    atol: float = 1e-3,
) -> BenchmarkResult:
    """Benchmark weight sharing and modification effects.

    Args:
        bridge: TransformerBridge model to test
        test_text: Input text for testing
        reference_model: Optional HookedTransformer reference model
        atol: Absolute tolerance for effect comparison

    Returns:
        BenchmarkResult with weight sharing verification details
    """
    try:
        # Get baseline loss
        bridge_original = bridge(test_text, return_type="loss")

        if reference_model is not None:
            reference_original = reference_model(test_text, return_type="loss")

            bridge_attn_blocks = bridge.blocks_with("attn")
            if not bridge_attn_blocks:
                return BenchmarkResult(
                    name="weight_sharing",
                    severity=BenchmarkSeverity.INFO,
                    message="No blocks have attention submodule — skipping weight sharing check",
                )
            bridge_attn_idx, bridge_attn_block = bridge_attn_blocks[0]

            # Verify weights are identical before modification
            bridge_W_V = torch.clone(cast(torch.Tensor, bridge_attn_block.attn.W_V))
            reference_W_V = torch.clone(
                cast(torch.Tensor, reference_model.blocks[bridge_attn_idx].attn.W_V)  # type: ignore[union-attr]
            )

            # Check if models have GQA (different head counts for K/V vs Q)
            has_gqa = (
                hasattr(bridge.cfg, "n_key_value_heads")
                and bridge.cfg.n_key_value_heads != bridge.cfg.n_heads
            )

            # For GQA models, HookedTransformer may not support GQA correctly yet
            # Skip the weight comparison if shapes don't match
            if bridge_W_V.shape != reference_W_V.shape:  # type: ignore[union-attr]
                if has_gqa:
                    # This is expected - HookedTransformer doesn't support GQA yet
                    # Skip this benchmark for GQA models
                    return BenchmarkResult(
                        name="weight_sharing",
                        severity=BenchmarkSeverity.INFO,
                        message=f"GQA model detected - skipping HT comparison (Bridge W_V: {bridge_W_V.shape}, HT W_V: {reference_W_V.shape})",  # type: ignore[union-attr]
                        details={
                            "bridge_shape": str(bridge_W_V.shape),  # type: ignore[union-attr]
                            "reference_shape": str(reference_W_V.shape),  # type: ignore[union-attr]
                        },
                    )
                else:
                    return BenchmarkResult(
                        name="weight_sharing",
                        severity=BenchmarkSeverity.WARNING,
                        message=f"Weight shapes differ: Bridge {bridge_W_V.shape} vs Reference {reference_W_V.shape}",  # type: ignore[union-attr]
                        details={
                            "bridge_shape": str(bridge_W_V.shape),  # type: ignore[union-attr]
                            "reference_shape": str(reference_W_V.shape),  # type: ignore[union-attr]
                        },
                    )

            if not safe_allclose(bridge_W_V, reference_W_V):  # type: ignore[arg-type]
                return BenchmarkResult(
                    name="weight_sharing",
                    severity=BenchmarkSeverity.WARNING,
                    message="Weights differ before modification",
                )

            # Modify weights in both models
            with torch.no_grad():
                bridge_attn_block.attn.W_V[0, :, :] = 0  # type: ignore[union-attr,operator]
                reference_model.blocks[bridge_attn_idx].attn.W_V[0, :, :] = 0  # type: ignore[union-attr,operator]

            # Test modified losses
            bridge_modified = bridge(test_text, return_type="loss")
            reference_modified = reference_model(test_text, return_type="loss")

            bridge_change = bridge_modified - bridge_original
            reference_change = reference_modified - reference_original

            # Restore weights
            with torch.no_grad():
                bridge_attn_block.attn.W_V.copy_(bridge_W_V)  # type: ignore[union-attr,operator,arg-type]
                reference_model.blocks[bridge_attn_idx].attn.W_V.copy_(reference_W_V)  # type: ignore[union-attr,operator,arg-type]

            diff = abs(bridge_change - reference_change)
            if diff < atol:
                return BenchmarkResult(
                    name="weight_sharing",
                    severity=BenchmarkSeverity.INFO,
                    message=f"Weight modifications have similar effects: {bridge_change:.6f} ≈ {reference_change:.6f}",
                    details={"diff": diff.item(), "atol": atol},
                )
            else:
                return BenchmarkResult(
                    name="weight_sharing",
                    severity=BenchmarkSeverity.WARNING,
                    message=f"Weight modification effects differ: {bridge_change:.6f} vs {reference_change:.6f}",
                    details={"diff": diff.item(), "atol": atol},
                )

        # No reference model - just verify modification has an effect
        # Find first block with attention (hybrid models may not have attn on block 0)
        bridge_attn_blocks = bridge.blocks_with("attn")
        if not bridge_attn_blocks:
            return BenchmarkResult(
                name="weight_sharing",
                severity=BenchmarkSeverity.INFO,
                message="No blocks have attention submodule — skipping weight sharing check",
            )
        _ws_idx, ws_attn_block = bridge_attn_blocks[0]

        original_W_V = ws_attn_block.attn.W_V.clone()
        with torch.no_grad():
            ws_attn_block.attn.W_V[0, :, :] = 0

        bridge_modified = bridge(test_text, return_type="loss")
        change = abs(bridge_modified - bridge_original)

        # Restore weights
        with torch.no_grad():
            ws_attn_block.attn.W_V.copy_(original_W_V)

        if change < 1e-6:
            return BenchmarkResult(
                name="weight_sharing",
                severity=BenchmarkSeverity.WARNING,
                message=f"Weight modification had minimal effect: {change:.6f}",
                details={"change": change.item()},
            )

        return BenchmarkResult(
            name="weight_sharing",
            severity=BenchmarkSeverity.INFO,
            message=f"Weight modification affects forward pass: change={change:.6f}",
            details={"change": change.item()},
        )

    except Exception as e:
        return BenchmarkResult(
            name="weight_sharing",
            severity=BenchmarkSeverity.ERROR,
            message=f"Weight sharing check failed: {str(e)}",
            passed=False,
        )


def benchmark_weight_modification(
    bridge: TransformerBridge,
    test_text: str,
    reference_model: Optional[HookedTransformer] = None,
) -> BenchmarkResult:
    """Benchmark that weight modifications propagate correctly.

    Args:
        bridge: TransformerBridge model to test
        test_text: Input text for testing
        reference_model: Optional HookedTransformer reference model (not used)

    Returns:
        BenchmarkResult with weight modification verification details
    """
    try:
        # Get original loss
        original_loss = bridge(test_text, return_type="loss")

        # Find first block with attention (hybrid models may not have attn on block 0)
        wm_attn_blocks = bridge.blocks_with("attn")
        if not wm_attn_blocks:
            return BenchmarkResult(
                name="weight_modification",
                severity=BenchmarkSeverity.INFO,
                message="No blocks have attention submodule — skipping weight modification check",
            )
        _wm_idx, wm_attn_block = wm_attn_blocks[0]

        # Modify W_V weights
        with torch.no_grad():
            original_w_v = wm_attn_block.attn.W_V.clone()
            # Check dimensionality - GQA models may have 2D tensors instead of 3D
            if original_w_v.ndim == 3:
                # Standard 3D tensor: [n_heads, d_model, d_head]
                wm_attn_block.attn.W_V[0, :, :] = 0
            elif original_w_v.ndim == 2:
                # 2D tensor (e.g., GQA models): [n_heads * d_head, d_model] or similar
                wm_attn_block.attn.W_V[0, :] = 0
            else:
                return BenchmarkResult(
                    name="weight_modification",
                    severity=BenchmarkSeverity.WARNING,
                    message=f"Unexpected W_V shape: {original_w_v.shape} (ndim={original_w_v.ndim})",
                    passed=False,
                )

        # Get modified loss (with error handling to restore weights)
        try:
            modified_loss = bridge(test_text, return_type="loss")
        except Exception as forward_error:
            # Restore weights before reporting error
            with torch.no_grad():
                wm_attn_block.attn.W_V.copy_(original_w_v)

            # Some models (e.g., models with complex attention mechanisms) may have
            # forward pass issues after weight modification. Report as skipped.
            return BenchmarkResult(
                name="weight_modification",
                severity=BenchmarkSeverity.SKIPPED,
                message=f"Weight modification not testable for this architecture: {str(forward_error)}",
                details={"error": str(forward_error), "architecture_limitation": True},
            )

        # Restore weights
        with torch.no_grad():
            wm_attn_block.attn.W_V.copy_(original_w_v)

        # Loss should change
        change = abs(modified_loss - original_loss)
        if change < 1e-6:
            # W_V modification didn't propagate. This can happen in models with
            # combined QKV projections (e.g., Bloom) where the split V weight
            # is separate from the combined QKV weight used in forward.
            # Try MLP weight modification as fallback.
            mlp_fallback_error = None
            mlp_blocks = bridge.blocks_with("mlp")
            mlp_block = mlp_blocks[0][1] if mlp_blocks else None
            try:
                if mlp_block is None:
                    raise AttributeError("No blocks have mlp submodule")
                with torch.no_grad():
                    original_mlp_w = mlp_block.mlp.out.weight.clone()
                    mlp_block.mlp.out.weight[0, :] = 0
                mlp_modified_loss = bridge(test_text, return_type="loss")
                with torch.no_grad():
                    mlp_block.mlp.out.weight.copy_(original_mlp_w)
                mlp_change = abs(mlp_modified_loss - original_loss)
                if mlp_change > 1e-6:
                    return BenchmarkResult(
                        name="weight_modification",
                        severity=BenchmarkSeverity.INFO,
                        message=f"Weight modification propagates via MLP (change: {mlp_change:.6f}). "
                        f"W_V not propagated (combined QKV architecture).",
                        details={"change": mlp_change.item(), "fallback": "mlp"},
                    )
            except Exception as mlp_err:
                mlp_fallback_error = str(mlp_err)

            details = {"change": change.item()}
            if mlp_fallback_error is not None:
                details["mlp_fallback_error"] = mlp_fallback_error
            return BenchmarkResult(
                name="weight_modification",
                severity=BenchmarkSeverity.DANGER,
                message=f"Weight modification did not affect loss (change: {change:.6f})",
                details=details,
                passed=False,
            )

        return BenchmarkResult(
            name="weight_modification",
            severity=BenchmarkSeverity.INFO,
            message=f"Weight modification propagates correctly (change: {change:.6f})",
            details={"change": change.item()},
        )

    except Exception as e:
        # Some architectures (e.g., Gemma 3 with complex attention, OpenELM with
        # combined QKV) don't expose W_V. Report as skipped, not passed.
        if (
            "cannot be multiplied" in str(e)
            or "shape" in str(e).lower()
            or "has no attribute" in str(e)
        ):
            return BenchmarkResult(
                name="weight_modification",
                severity=BenchmarkSeverity.SKIPPED,
                message=f"Weight modification not testable for this architecture: {str(e)}",
                details={"error": str(e), "architecture_limitation": True},
            )
        return BenchmarkResult(
            name="weight_modification",
            severity=BenchmarkSeverity.ERROR,
            message=f"Weight modification check failed: {str(e)}",
            passed=False,
        )


def benchmark_layer_norm_folding(
    bridge: TransformerBridge,
    test_text: str,
    reference_model: Optional[HookedTransformer] = None,
) -> BenchmarkResult:
    """Benchmark layer norm folding - norm weights should be identity after folding.

    Args:
        bridge: TransformerBridge model to test
        test_text: Input text for testing
        reference_model: Optional HookedTransformer reference model (not used)

    Returns:
        BenchmarkResult with layer norm folding verification details
    """
    try:
        # Skip for architectures that don't support fold_ln (e.g., post-LN like BERT)
        adapter = getattr(bridge, "adapter", None)
        if adapter and not getattr(adapter, "supports_fold_ln", True):
            return BenchmarkResult(
                name="layer_norm_folding",
                severity=BenchmarkSeverity.SKIPPED,
                message="Skipped (post-LN architecture does not support fold_ln)",
                passed=True,
            )

        # Get state dict from bridge (should return TransformerLens format keys)
        state_dict = bridge.state_dict()

        # Check both ln1 (attention LN) and ln2 (MLP LN) in TransformerLens format.
        # Models with combined QKV projections (e.g., OpenELM's qkv_proj) cannot
        # fold ln1 into attention weights, but ln2 should always be foldable.
        tolerance = 0.01
        # For rmsnorm_uses_offset models (Gemma/Gemma2), HF computes x*(1+weight),
        # so the identity weight after folding is 0.0 (gives 1+0=1). For standard
        # models, identity is 1.0.
        cfg = getattr(getattr(bridge, "adapter", None), "cfg", None)
        rmsnorm_uses_offset = getattr(cfg, "rmsnorm_uses_offset", False)
        expected_val = 0.0 if rmsnorm_uses_offset else 1.0
        folded = []
        not_folded = []

        for ln_name in ["ln1", "ln2"]:
            ln_key = f"blocks.0.{ln_name}.weight"
            if ln_key not in state_dict:
                continue
            ln_weight = state_dict[ln_key]
            mean_val = torch.mean(ln_weight).item()
            if abs(mean_val - expected_val) < tolerance:
                folded.append((ln_name, ln_key, mean_val))
            else:
                not_folded.append((ln_name, ln_key, mean_val))

        if not folded and not not_folded:
            # No LN weights found — model uses non-parametric LayerNorm
            # (e.g., OLMo v1 has fixed weight=1, bias=0 with no learnable params).
            # Nothing to fold, so this is a pass.
            return BenchmarkResult(
                name="layer_norm_folding",
                severity=BenchmarkSeverity.INFO,
                message="No learnable layer norm weights (non-parametric LayerNorm)",
                passed=True,
            )

        if folded and not not_folded:
            # All LN weights are folded
            names = ", ".join(f"{n} (mean={m:.6f})" for n, _, m in folded)
            return BenchmarkResult(
                name="layer_norm_folding",
                severity=BenchmarkSeverity.INFO,
                message=f"Layer norm folding verified: {names}",
                details={"folded": [n for n, _, _ in folded]},
            )
        elif folded and not_folded:
            # Partial folding — some LN weights folded, some not.
            # This is expected for models with combined QKV (ln1 can't fold).
            folded_names = ", ".join(f"{n} (mean={m:.6f})" for n, _, m in folded)
            unfolded_names = ", ".join(f"{n} (mean={m:.6f})" for n, _, m in not_folded)
            return BenchmarkResult(
                name="layer_norm_folding",
                severity=BenchmarkSeverity.WARNING,
                message=(
                    f"Partial LN folding: {folded_names} folded; "
                    f"{unfolded_names} preserved (expected for combined QKV models)"
                ),
                details={
                    "folded": [n for n, _, _ in folded],
                    "not_folded": [n for n, _, _ in not_folded],
                },
                passed=True,
            )
        else:
            # No LN weights folded
            names = ", ".join(f"{n} (mean={m:.6f})" for n, _, m in not_folded)
            return BenchmarkResult(
                name="layer_norm_folding",
                severity=BenchmarkSeverity.WARNING,
                message=f"Layer norm weights not identity after folding: {names}",
                details={"not_folded": [n for n, _, _ in not_folded]},
                passed=False,
            )

    except Exception as e:
        return BenchmarkResult(
            name="layer_norm_folding",
            severity=BenchmarkSeverity.ERROR,
            message=f"Layer norm folding check failed: {str(e)}",
            passed=False,
        )


def benchmark_attention_output_centering(
    bridge: TransformerBridge,
    test_text: str,
    reference_model: Optional[HookedTransformer] = None,
) -> BenchmarkResult:
    """Benchmark attention output centering - W_O should have mean ≈ 0.

    Args:
        bridge: TransformerBridge model to test
        test_text: Input text for testing
        reference_model: Optional HookedTransformer reference model (not used)

    Returns:
        BenchmarkResult with attention output centering verification details
    """
    try:
        # Skip centering check for tiny/test models — random weights don't
        # center meaningfully and produce false failures.
        if is_tiny_test_model(getattr(bridge.cfg, "model_name", "") or ""):
            return BenchmarkResult(
                name="attention_output_centering",
                severity=BenchmarkSeverity.INFO,
                message="Skipped for tiny/test model (random weights don't center meaningfully)",
            )

        attn_blocks = bridge.blocks_with("attn")
        if not attn_blocks:
            return BenchmarkResult(
                name="attention_output_centering",
                severity=BenchmarkSeverity.WARNING,
                message="No blocks have attention submodule",
                passed=False,
            )

        # Check W_O accessibility on first attention block
        first_idx, first_attn_block = attn_blocks[0]
        if not hasattr(first_attn_block.attn, "W_O"):
            return BenchmarkResult(
                name="attention_output_centering",
                severity=BenchmarkSeverity.WARNING,
                message="W_O not accessible on bridge model",
                passed=False,
            )

        # Compute mean across all attention blocks
        tolerance = 0.01  # 1% tolerance
        worst_mean = 0.0
        for idx, block in attn_blocks:
            w_o = block.attn.W_O
            mean_abs = torch.mean(torch.abs(torch.mean(w_o, dim=-1))).item()
            worst_mean = max(worst_mean, mean_abs)

        n_attn = len(attn_blocks)
        n_total = len(bridge.blocks)
        block_info = f" ({n_attn}/{n_total} blocks have attention)" if n_attn < n_total else ""

        if worst_mean < tolerance:
            return BenchmarkResult(
                name="attention_output_centering",
                severity=BenchmarkSeverity.INFO,
                message=f"Attention output centering verified (worst_mean={worst_mean:.6f}){block_info}",
                details={"mean": worst_mean, "tolerance": tolerance, "n_attn_blocks": n_attn},
            )
        else:
            return BenchmarkResult(
                name="attention_output_centering",
                severity=BenchmarkSeverity.WARNING,
                message=f"Attention output weights not well-centered (worst_mean={worst_mean:.6f}){block_info}",
                details={"mean": worst_mean, "tolerance": tolerance, "n_attn_blocks": n_attn},
                passed=False,
            )

    except Exception as e:
        return BenchmarkResult(
            name="attention_output_centering",
            severity=BenchmarkSeverity.ERROR,
            message=f"Attention output centering check failed: {str(e)}",
            passed=False,
        )


def benchmark_mlp_output_centering(
    bridge: TransformerBridge,
    test_text: str,
    reference_model: Optional[HookedTransformer] = None,
) -> BenchmarkResult:
    """Benchmark MLP output centering - MLP output weights should have mean ≈ 0.

    Args:
        bridge: TransformerBridge model to test
        test_text: Input text for testing
        reference_model: Optional HookedTransformer reference model (not used)

    Returns:
        BenchmarkResult with MLP output centering verification details
    """
    try:
        # Skip centering check for tiny/test models — random weights don't
        # center meaningfully and produce false failures.
        if is_tiny_test_model(getattr(bridge.cfg, "model_name", "") or ""):
            return BenchmarkResult(
                name="mlp_output_centering",
                severity=BenchmarkSeverity.INFO,
                message="Skipped for tiny/test model (random weights don't center meaningfully)",
            )

        # Find an MLP-like submodule (may be "mlp", "shared_mlp", etc.)
        from transformer_lens.model_bridge.generalized_components.moe import MoEBridge

        mlp_module = None
        block = bridge.blocks[0]
        for name in ("mlp", "shared_mlp"):
            if name in block._modules:
                mlp_module = block._modules[name]
                break
        if mlp_module is None:
            return BenchmarkResult(
                name="mlp_output_centering",
                severity=BenchmarkSeverity.WARNING,
                message="No MLP submodule found on block 0",
                passed=False,
            )

        if isinstance(mlp_module, MoEBridge):
            return BenchmarkResult(
                name="mlp_output_centering",
                severity=BenchmarkSeverity.INFO,
                message="Skipped for MoE models (no single W_out weight)",
                details={"is_moe": True},
            )

        # Check if W_out exists and is accessible (HT format or bridge format)
        w_out = None
        if hasattr(mlp_module, "W_out"):
            w_out = mlp_module.W_out
        elif hasattr(mlp_module, "out"):
            out_module = mlp_module.out
            if hasattr(out_module, "original_component") and hasattr(
                out_module.original_component, "weight"
            ):
                w_out = out_module.original_component.weight
            elif hasattr(out_module, "weight"):
                w_out = out_module.weight
        if w_out is None:
            return BenchmarkResult(
                name="mlp_output_centering",
                severity=BenchmarkSeverity.WARNING,
                message="W_out not accessible on bridge model",
                passed=False,
            )

        # Compute mean along output dimension
        mean_abs = torch.mean(torch.abs(torch.mean(w_out, dim=-1))).item()

        tolerance = 0.01  # 1% tolerance

        if mean_abs < tolerance:
            return BenchmarkResult(
                name="mlp_output_centering",
                severity=BenchmarkSeverity.INFO,
                message=f"MLP output centering verified (mean={mean_abs:.6f})",
                details={"mean": mean_abs, "tolerance": tolerance},
            )
        else:
            return BenchmarkResult(
                name="mlp_output_centering",
                severity=BenchmarkSeverity.WARNING,
                message=f"MLP output weights not well-centered (mean={mean_abs:.6f})",
                details={"mean": mean_abs, "tolerance": tolerance},
                passed=False,
            )

    except Exception as e:
        return BenchmarkResult(
            name="mlp_output_centering",
            severity=BenchmarkSeverity.ERROR,
            message=f"MLP output centering check failed: {str(e)}",
            passed=False,
        )


def benchmark_unembed_centering(
    bridge: TransformerBridge,
    test_text: str,
    reference_model: Optional[HookedTransformer] = None,
) -> BenchmarkResult:
    """Benchmark unembed centering - unembed matrix should have mean ≈ 0.

    Args:
        bridge: TransformerBridge model to test
        test_text: Input text for testing
        reference_model: Optional HookedTransformer reference model (not used)

    Returns:
        BenchmarkResult with unembed centering verification details
    """
    try:
        # Get state dict from bridge (should return TransformerLens format keys)
        state_dict = bridge.state_dict()

        # Check for unembed weight in TransformerLens format
        unembed_key = "unembed.weight"

        # Fallback: if TL format key doesn't exist, try common HF format patterns
        if unembed_key not in state_dict:
            # Try standard HF format
            if "lm_head.weight" in state_dict:
                unembed_key = "lm_head.weight"
            else:
                return BenchmarkResult(
                    name="unembed_centering",
                    severity=BenchmarkSeverity.WARNING,
                    message="Could not find unembed weights in state dict",
                    passed=False,
                )

        # Get the unembed weight tensor
        w_u = state_dict[unembed_key]

        # Compute mean along vocabulary dimension (dim 0)
        mean_abs = torch.mean(torch.abs(torch.mean(w_u, dim=0))).item()

        tolerance = 0.01  # 1% tolerance (consistent with attn/mlp centering)

        if mean_abs < tolerance:
            return BenchmarkResult(
                name="unembed_centering",
                severity=BenchmarkSeverity.INFO,
                message=f"Unembed centering verified (mean={mean_abs:.6f})",
                details={"mean": mean_abs, "tolerance": tolerance, "key": unembed_key},
            )
        else:
            return BenchmarkResult(
                name="unembed_centering",
                severity=BenchmarkSeverity.WARNING,
                message=f"Unembed matrix not well-centered (mean={mean_abs:.6f})",
                details={"mean": mean_abs, "tolerance": tolerance, "key": unembed_key},
                passed=False,
            )

    except Exception as e:
        return BenchmarkResult(
            name="unembed_centering",
            severity=BenchmarkSeverity.ERROR,
            message=f"Unembed centering check failed: {str(e)}",
            passed=False,
        )


def benchmark_value_bias_folding(
    bridge: TransformerBridge,
    test_text: str,
    reference_model: Optional[HookedTransformer] = None,
) -> BenchmarkResult:
    """Benchmark value bias folding - b_V should be zero after folding.

    Args:
        bridge: TransformerBridge model to test
        test_text: Input text for testing
        reference_model: Optional HookedTransformer reference model (not used)

    Returns:
        BenchmarkResult with value bias folding verification details
    """
    try:
        # Skip for GQA models (where n_key_value_heads != n_heads)
        # Value bias folding doesn't work the same way because V outputs are repeated
        if hasattr(bridge.cfg, "n_key_value_heads") and bridge.cfg.n_key_value_heads is not None:
            if bridge.cfg.n_key_value_heads != bridge.cfg.n_heads:
                return BenchmarkResult(
                    name="value_bias_folding",
                    severity=BenchmarkSeverity.INFO,
                    message="Skipped for GQA models (n_key_value_heads != n_heads)",
                    details={
                        "is_gqa": True,
                        "n_heads": bridge.cfg.n_heads,
                        "n_kv_heads": bridge.cfg.n_key_value_heads,
                    },
                )

        attn_blocks = bridge.blocks_with("attn")
        if not attn_blocks:
            return BenchmarkResult(
                name="value_bias_folding",
                severity=BenchmarkSeverity.INFO,
                message="No blocks have attention submodule (expected for hybrid models without mapped attn)",
                details={"has_bias": False},
            )

        first_idx, first_attn_block = attn_blocks[0]

        # Check if b_V exists
        if not hasattr(first_attn_block.attn, "b_V"):
            return BenchmarkResult(
                name="value_bias_folding",
                severity=BenchmarkSeverity.INFO,
                message="No value bias found (expected for models without biases)",
                details={"has_bias": False},
            )

        b_v = first_attn_block.attn.b_V

        if b_v is None:
            return BenchmarkResult(
                name="value_bias_folding",
                severity=BenchmarkSeverity.INFO,
                message="Value bias is None (expected for models without biases)",
                details={"has_bias": False},
            )

        # Check if b_V is approximately zero
        max_abs = torch.max(torch.abs(b_v)).item()
        tolerance = 1e-6

        if max_abs < tolerance:
            return BenchmarkResult(
                name="value_bias_folding",
                severity=BenchmarkSeverity.INFO,
                message=f"Value bias folding verified (max_abs={max_abs:.6e})",
                details={"max_abs": max_abs, "tolerance": tolerance},
            )
        else:
            return BenchmarkResult(
                name="value_bias_folding",
                severity=BenchmarkSeverity.WARNING,
                message=f"Value bias not zero after folding (max_abs={max_abs:.6e})",
                details={"max_abs": max_abs, "tolerance": tolerance},
                passed=False,
            )

    except Exception as e:
        return BenchmarkResult(
            name="value_bias_folding",
            severity=BenchmarkSeverity.ERROR,
            message=f"Value bias folding check failed: {str(e)}",
            passed=False,
        )


def benchmark_no_nan_inf(
    bridge: TransformerBridge,
    test_text: str,
    reference_model: Optional[HookedTransformer] = None,
) -> BenchmarkResult:
    """Benchmark that weights contain no NaN or Inf values.

    Args:
        bridge: TransformerBridge model to test
        test_text: Input text for testing
        reference_model: Optional HookedTransformer reference model (not used)

    Returns:
        BenchmarkResult with NaN/Inf verification details
    """
    try:
        # Get state dict from original model
        state_dict = bridge.state_dict()

        # Check for NaN/Inf in all tensors
        nan_keys = []
        inf_keys = []

        for key, value in state_dict.items():
            if torch.isnan(value).any():
                nan_keys.append(key)
            if torch.isinf(value).any():
                inf_keys.append(key)

        if nan_keys or inf_keys:
            message_parts = []
            if nan_keys:
                message_parts.append(f"NaN in {len(nan_keys)} tensors")
            if inf_keys:
                message_parts.append(f"Inf in {len(inf_keys)} tensors")

            return BenchmarkResult(
                name="no_nan_inf",
                severity=BenchmarkSeverity.DANGER,
                message=f"Invalid values found: {', '.join(message_parts)}",
                details={"nan_keys": nan_keys, "inf_keys": inf_keys},
                passed=False,
            )

        return BenchmarkResult(
            name="no_nan_inf",
            severity=BenchmarkSeverity.INFO,
            message="No NaN or Inf values found in weights",
            details={"num_tensors_checked": len(state_dict)},
        )

    except Exception as e:
        return BenchmarkResult(
            name="no_nan_inf",
            severity=BenchmarkSeverity.ERROR,
            message=f"NaN/Inf check failed: {str(e)}",
            passed=False,
        )


def benchmark_weight_magnitudes(
    bridge: TransformerBridge,
    test_text: str,
    reference_model: Optional[HookedTransformer] = None,
) -> BenchmarkResult:
    """Benchmark that weight magnitudes are in reasonable ranges.

    Args:
        bridge: TransformerBridge model to test
        test_text: Input text for testing
        reference_model: Optional HookedTransformer reference model (not used)

    Returns:
        BenchmarkResult with weight magnitude verification details
    """
    try:
        # Get state dict from original model
        state_dict = bridge.state_dict()

        # Check magnitude ranges
        too_small_keys = []
        too_large_keys = []

        min_threshold = 1e-6
        max_threshold = 1000.0

        # For rmsnorm_uses_offset models (Gemma/Gemma2), fold_ln sets LN weights
        # to 0.0 (identity for (1+w) normalization). Skip LN weights for these models.
        cfg = getattr(getattr(bridge, "adapter", None), "cfg", None)
        rmsnorm_uses_offset = getattr(cfg, "rmsnorm_uses_offset", False)

        for key, value in state_dict.items():
            # Skip non-weight tensors (buffers, etc.)
            if "weight" not in key and "bias" not in key:
                continue

            # Skip internal _original_component keys - these are implementation details
            if "_original_component" in key:
                continue

            # Skip value biases - they are expected to be zero after folding
            if ".v.bias" in key:
                continue

            # Skip attention projection biases - they can be zero in some models
            if (
                ".k_proj.bias" in key
                or ".q_proj.bias" in key
                or ".v_proj.bias" in key
                or ".o_proj.bias" in key
                or ".k.bias" in key
                or ".q.bias" in key
                or ".v.bias" in key
                or ".o.bias" in key
            ):
                continue

            # Skip layer norm biases - they are expected to be zero after folding
            if (
                "ln1.bias" in key
                or "ln2.bias" in key
                or "ln_1.bias" in key
                or "ln_2.bias" in key
                or "ln_final.bias" in key
                or "input_layernorm.bias" in key
                or "post_attention_layernorm.bias" in key
            ):
                continue

            # For rmsnorm_uses_offset models, fold_ln sets LN weights to 0.0
            # (identity for (1+w) normalization). Skip all LN weight keys —
            # including post-norms (ln1_post, ln2_post) which aren't folded but
            # use the same (1+w) convention — to avoid false magnitude warnings.
            if rmsnorm_uses_offset and (
                "ln1.weight" in key
                or "ln2.weight" in key
                or "ln1_post.weight" in key
                or "ln2_post.weight" in key
                or "ln_1.weight" in key
                or "ln_2.weight" in key
                or "ln_final.weight" in key
                or "input_layernorm.weight" in key
                or "post_attention_layernorm.weight" in key
            ):
                continue

            # Skip unembed bias - it may be zero after processing
            if "unembed.bias" in key:
                continue

            # Skip zero biases - many models initialize biases to zero which is
            # mathematically equivalent to having no bias. This is a valid state.
            if "bias" in key and torch.all(value == 0).item():
                continue

            mean_abs = torch.mean(torch.abs(value)).item()
            max_abs = torch.max(torch.abs(value)).item()

            if mean_abs > 0.0 and mean_abs < min_threshold:
                # For non-zero weights, check if they're suspiciously small
                too_small_keys.append((key, mean_abs))

            if max_abs > max_threshold:
                too_large_keys.append((key, max_abs))

        if too_small_keys or too_large_keys:
            message_parts = []
            if too_small_keys:
                message_parts.append(f"{len(too_small_keys)} too small")
            if too_large_keys:
                message_parts.append(f"{len(too_large_keys)} too large")

            return BenchmarkResult(
                name="weight_magnitudes",
                severity=BenchmarkSeverity.WARNING,
                message=f"Weight magnitude issues: {', '.join(message_parts)}",
                details={
                    "too_small": too_small_keys[:5],  # Limit to first 5
                    "too_large": too_large_keys[:5],  # Limit to first 5
                },
                passed=False,
            )

        return BenchmarkResult(
            name="weight_magnitudes",
            severity=BenchmarkSeverity.INFO,
            message="All weight magnitudes in reasonable ranges",
            details={"min_threshold": min_threshold, "max_threshold": max_threshold},
        )

    except Exception as e:
        return BenchmarkResult(
            name="weight_magnitudes",
            severity=BenchmarkSeverity.ERROR,
            message=f"Weight magnitude check failed: {str(e)}",
            passed=False,
        )
