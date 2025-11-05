"""Weight processing benchmarks for TransformerBridge."""

from typing import Optional

import torch

from transformer_lens import HookedTransformer
from transformer_lens.benchmarks.utils import BenchmarkResult, BenchmarkSeverity
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
            bridge_w_out = bridge.blocks[0].mlp.W_out
            reference_w_out = reference_model.blocks[0].mlp.W_out

            bridge_mean = torch.mean(torch.abs(torch.mean(bridge_w_out, dim=-1, keepdim=True)))
            reference_mean = torch.mean(
                torch.abs(torch.mean(reference_w_out, dim=-1, keepdim=True))
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

            # Verify weights are identical before modification
            bridge_W_V = bridge.blocks[0].attn.W_V
            reference_W_V = reference_model.blocks[0].attn.W_V  # type: ignore[union-attr]

            if not torch.allclose(bridge_W_V, reference_W_V):  # type: ignore[arg-type]
                return BenchmarkResult(
                    name="weight_sharing",
                    severity=BenchmarkSeverity.WARNING,
                    message="Weights differ before modification",
                )

            # Modify weights in both models
            with torch.no_grad():
                bridge.blocks[0].attn.W_V[0, :, :] = 0  # type: ignore[union-attr,operator]
                reference_model.blocks[0].attn.W_V[0, :, :] = 0  # type: ignore[union-attr,operator]

            # Test modified losses
            bridge_modified = bridge(test_text, return_type="loss")
            reference_modified = reference_model(test_text, return_type="loss")

            bridge_change = bridge_modified - bridge_original
            reference_change = reference_modified - reference_original

            # Restore weights
            with torch.no_grad():
                bridge.blocks[0].attn.W_V.copy_(bridge_W_V)  # type: ignore[union-attr,operator,arg-type]
                reference_model.blocks[0].attn.W_V.copy_(reference_W_V)  # type: ignore[union-attr,operator,arg-type]

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
        original_W_V = bridge.blocks[0].attn.W_V.clone()
        with torch.no_grad():
            bridge.blocks[0].attn.W_V[0, :, :] = 0

        bridge_modified = bridge(test_text, return_type="loss")
        change = abs(bridge_modified - bridge_original)

        # Restore weights
        with torch.no_grad():
            bridge.blocks[0].attn.W_V.copy_(original_W_V)

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

        # Modify W_V weights
        with torch.no_grad():
            original_w_v = bridge.blocks[0].attn.W_V.clone()
            bridge.blocks[0].attn.W_V[0, :, :] = 0  # Zero out first head

        # Get modified loss
        modified_loss = bridge(test_text, return_type="loss")

        # Restore weights
        with torch.no_grad():
            bridge.blocks[0].attn.W_V.copy_(original_w_v)

        # Loss should change
        change = abs(modified_loss - original_loss)
        if change < 1e-6:
            return BenchmarkResult(
                name="weight_modification",
                severity=BenchmarkSeverity.DANGER,
                message=f"Weight modification did not affect loss (change: {change:.6f})",
                details={"change": change.item()},
                passed=False,
            )

        return BenchmarkResult(
            name="weight_modification",
            severity=BenchmarkSeverity.INFO,
            message=f"Weight modification propagates correctly (change: {change:.6f})",
            details={"change": change.item()},
        )

    except Exception as e:
        return BenchmarkResult(
            name="weight_modification",
            severity=BenchmarkSeverity.ERROR,
            message=f"Weight modification check failed: {str(e)}",
            passed=False,
        )
