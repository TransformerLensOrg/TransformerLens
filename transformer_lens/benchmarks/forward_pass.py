"""Forward pass benchmarks for TransformerBridge."""

from typing import Optional, Union

import torch

from transformer_lens import HookedTransformer
from transformer_lens.benchmarks.utils import (
    BenchmarkResult,
    BenchmarkSeverity,
    compare_scalars,
    compare_tensors,
)
from transformer_lens.model_bridge import TransformerBridge


def benchmark_forward_pass(
    bridge: TransformerBridge,
    test_text: str,
    reference_model: Optional[Union[HookedTransformer, torch.nn.Module]] = None,
    atol: float = 1e-3,
    rtol: float = 3e-2,
) -> BenchmarkResult:
    """Benchmark forward pass between TransformerBridge and reference model.

    Args:
        bridge: TransformerBridge model to test
        test_text: Input text for testing
        reference_model: Optional reference model (HookedTransformer or HF model)
        atol: Absolute tolerance for comparison
        rtol: Relative tolerance for comparison

    Returns:
        BenchmarkResult with comparison details
    """
    try:
        # Run bridge forward pass
        bridge_output = bridge(test_text, return_type="logits")

        if reference_model is None:
            # No reference model - just verify output shape and validity
            if not isinstance(bridge_output, torch.Tensor):
                return BenchmarkResult(
                    name="forward_pass",
                    severity=BenchmarkSeverity.DANGER,
                    message="Bridge output is not a tensor",
                    passed=False,
                )

            if bridge_output.numel() == 0:
                return BenchmarkResult(
                    name="forward_pass",
                    severity=BenchmarkSeverity.DANGER,
                    message="Bridge output is empty",
                    passed=False,
                )

            return BenchmarkResult(
                name="forward_pass",
                severity=BenchmarkSeverity.INFO,
                message=f"Bridge forward pass successful (shape: {bridge_output.shape})",
                details={"output_shape": str(bridge_output.shape)},
            )

        # Compare with reference model
        if isinstance(reference_model, HookedTransformer):
            reference_output = reference_model(test_text, return_type="logits")
        else:
            # HuggingFace model
            tokens = bridge.to_tokens(test_text)
            with torch.no_grad():
                hf_output = reference_model(tokens)
                reference_output = hf_output.logits

        return compare_tensors(
            bridge_output,
            reference_output,
            atol=atol,
            rtol=rtol,
            name="forward_pass_logits",
        )

    except Exception as e:
        return BenchmarkResult(
            name="forward_pass",
            severity=BenchmarkSeverity.ERROR,
            message=f"Forward pass failed: {str(e)}",
            passed=False,
        )


def benchmark_loss_equivalence(
    bridge: TransformerBridge,
    test_text: str,
    reference_model: Optional[HookedTransformer] = None,
    atol: float = 1e-3,
) -> BenchmarkResult:
    """Benchmark loss computation between TransformerBridge and HookedTransformer.

    Args:
        bridge: TransformerBridge model to test
        test_text: Input text for testing
        reference_model: Optional HookedTransformer reference model
        atol: Absolute tolerance for comparison

    Returns:
        BenchmarkResult with comparison details
    """
    try:
        # Run bridge loss computation
        bridge_loss = bridge(test_text, return_type="loss")

        if reference_model is None:
            # No reference - just verify loss is valid
            if not isinstance(bridge_loss, torch.Tensor):
                return BenchmarkResult(
                    name="loss_equivalence",
                    severity=BenchmarkSeverity.DANGER,
                    message="Bridge loss is not a tensor",
                    passed=False,
                )

            loss_value = bridge_loss.item()
            if torch.isnan(bridge_loss) or torch.isinf(bridge_loss):
                return BenchmarkResult(
                    name="loss_equivalence",
                    severity=BenchmarkSeverity.DANGER,
                    message=f"Bridge loss is invalid: {loss_value}",
                    passed=False,
                )

            return BenchmarkResult(
                name="loss_equivalence",
                severity=BenchmarkSeverity.INFO,
                message=f"Bridge loss computed successfully: {loss_value:.6f}",
                details={"loss": loss_value},
            )

        # Compare with reference model
        reference_loss = reference_model(test_text, return_type="loss")

        return compare_scalars(
            bridge_loss.item(),
            reference_loss.item(),
            atol=atol,
            name="loss_equivalence",
        )

    except Exception as e:
        return BenchmarkResult(
            name="loss_equivalence",
            severity=BenchmarkSeverity.ERROR,
            message=f"Loss computation failed: {str(e)}",
            passed=False,
        )


def benchmark_logits_equivalence(
    bridge: TransformerBridge,
    test_text: str,
    reference_model: Optional[HookedTransformer] = None,
    atol: float = 3e-2,
    rtol: float = 3e-2,
) -> BenchmarkResult:
    """Benchmark logits output between TransformerBridge and HookedTransformer.

    Note: Uses relaxed tolerance (3e-2) as forward pass implementations differ
    slightly, leading to accumulated numerical precision differences.

    Args:
        bridge: TransformerBridge model to test
        test_text: Input text for testing
        reference_model: Optional HookedTransformer reference model
        atol: Absolute tolerance for comparison
        rtol: Relative tolerance for comparison

    Returns:
        BenchmarkResult with comparison details
    """
    try:
        # Run bridge forward pass
        bridge_logits = bridge(test_text, return_type="logits")

        if reference_model is None:
            # No reference - just verify logits shape and validity
            if not isinstance(bridge_logits, torch.Tensor):
                return BenchmarkResult(
                    name="logits_equivalence",
                    severity=BenchmarkSeverity.DANGER,
                    message="Bridge logits is not a tensor",
                    passed=False,
                )

            if bridge_logits.numel() == 0:
                return BenchmarkResult(
                    name="logits_equivalence",
                    severity=BenchmarkSeverity.DANGER,
                    message="Bridge logits is empty",
                    passed=False,
                )

            return BenchmarkResult(
                name="logits_equivalence",
                severity=BenchmarkSeverity.INFO,
                message=f"Bridge logits computed successfully (shape: {bridge_logits.shape})",
                details={"output_shape": str(bridge_logits.shape)},
            )

        # Compare with reference model
        reference_logits = reference_model(test_text, return_type="logits")

        return compare_tensors(
            bridge_logits,
            reference_logits,
            atol=atol,
            rtol=rtol,
            name="logits_equivalence",
        )

    except Exception as e:
        return BenchmarkResult(
            name="logits_equivalence",
            severity=BenchmarkSeverity.ERROR,
            message=f"Logits computation failed: {str(e)}",
            passed=False,
        )
