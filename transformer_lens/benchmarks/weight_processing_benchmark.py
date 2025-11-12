#!/usr/bin/env python3
"""Benchmark suite for validating weight processing in TransformerBridge models.

This suite verifies that each weight processing step (layer norm folding, centering,
value bias folding, etc.) has been correctly applied to the model weights.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import torch


@dataclass
class WeightProcessingCheck:
    """Result of a single weight processing verification check."""

    name: str
    passed: bool
    details: Dict[str, Any]
    message: str


class WeightProcessingBenchmark:
    """Benchmark suite for validating weight processing steps."""

    def __init__(self, bridge_model: Any, verbose: bool = True):
        """Initialize the benchmark suite.

        Args:
            bridge_model: TransformerBridge model instance
            verbose: Whether to print detailed output
        """
        self.bridge = bridge_model
        self.cfg = bridge_model.cfg
        self.verbose = verbose
        self.results: List[WeightProcessingCheck] = []

    def run_all_checks(self) -> Tuple[int, int]:
        """Run all weight processing validation checks.

        Returns:
            Tuple of (passed_count, total_count)
        """
        if self.verbose:
            print("=" * 80)
            print(f"Weight Processing Benchmark: {self.cfg.model_name}")
            print("=" * 80)

        # Get state dict from the processed model
        state_dict = self.bridge.original_model.state_dict()

        # Clean keys
        cleaned_state_dict = {}
        for key, value in state_dict.items():
            clean_key = key.replace("._original_component", "")
            cleaned_state_dict[clean_key] = value

        # Run checks
        self._check_layer_norm_folding(cleaned_state_dict)
        self._check_weight_centering(cleaned_state_dict)
        self._check_unembed_centering(cleaned_state_dict)
        self._check_value_bias_folding(cleaned_state_dict)
        self._check_no_nan_inf(cleaned_state_dict)
        self._check_weight_magnitudes(cleaned_state_dict)

        # Print summary
        passed = sum(1 for r in self.results if r.passed)
        total = len(self.results)

        if self.verbose:
            print("\n" + "=" * 80)
            print("RESULTS")
            print("=" * 80)
            for result in self.results:
                status = "✅" if result.passed else "❌"
                print(f"\n{status} {result.name}")
                print(f"   {result.message}")
                if not result.passed or self.verbose:
                    for key, value in result.details.items():
                        print(f"   {key}: {value}")

            print("\n" + "=" * 80)
            print(f"SUMMARY: {passed}/{total} checks passed ({100*passed//total}%)")
            print("=" * 80)

        return passed, total

    def _check_layer_norm_folding(self, state_dict: Dict[str, torch.Tensor]) -> None:
        """Check that layer norm weights have been folded into subsequent layers."""
        # For models with LayerNorm/RMSNorm, check if normalization weights still exist
        # After folding, ln weights should be removed or set to identity
        uses_rms_norm = getattr(self.cfg, "uses_rms_norm", False)

        # Check if ln1 weights exist for first block
        ln1_key_patterns = [
            f"blocks.0.ln1.weight",  # GPT-2 (TransformerLens format)
            f"model.layers.0.input_layernorm.weight",  # Gemma
        ]

        ln1_exists = False
        ln1_key = None
        for pattern in ln1_key_patterns:
            if pattern in state_dict:
                ln1_exists = True
                ln1_key = pattern
                break

        if ln1_exists and ln1_key:
            ln1_weight = state_dict[ln1_key]

            if uses_rms_norm:
                # RMS norm weights should be folded (multiplied into downstream weights)
                # After folding, they might still exist but should be identity-like
                # For RMS norm, "identity" means all ones
                expected_val = 1.0
                is_identity = torch.allclose(ln1_weight, torch.ones_like(ln1_weight), atol=1e-4)

                self.results.append(
                    WeightProcessingCheck(
                        name="layer_norm_folding",
                        passed=is_identity,
                        details={
                            "norm_type": "RMSNorm",
                            "ln1_mean": ln1_weight.mean().item(),
                            "ln1_std": ln1_weight.std().item(),
                            "expected_mean": expected_val,
                            "is_identity": is_identity,
                        },
                        message=f"RMSNorm weights {'are' if is_identity else 'are NOT'} identity after folding",
                    )
                )
            else:
                # LayerNorm weights should be folded
                # After folding, they should be identity (all ones)
                expected_val = 1.0
                is_identity = torch.allclose(ln1_weight, torch.ones_like(ln1_weight), atol=1e-4)

                self.results.append(
                    WeightProcessingCheck(
                        name="layer_norm_folding",
                        passed=is_identity,
                        details={
                            "norm_type": "LayerNorm",
                            "ln1_mean": ln1_weight.mean().item(),
                            "ln1_std": ln1_weight.std().item(),
                            "expected_mean": expected_val,
                            "is_identity": is_identity,
                        },
                        message=f"LayerNorm weights {'are' if is_identity else 'are NOT'} identity after folding",
                    )
                )
        else:
            # No ln1 found - might be architecture without it
            self.results.append(
                WeightProcessingCheck(
                    name="layer_norm_folding",
                    passed=True,
                    details={"norm_type": "None", "reason": "No ln1 layer found"},
                    message="No normalization layer found (expected for some architectures)",
                )
            )

    def _check_weight_centering(self, state_dict: Dict[str, torch.Tensor]) -> None:
        """Check that writing weights have been centered."""
        # Writing weights are those that write to the residual stream:
        # - Attention output projection (W_O)
        # - MLP output projection
        # These should have mean ≈ 0 along the output dimension after centering

        # Check attention output (W_O / o_proj)
        wo_key_patterns = [
            "blocks.0.attn.o.weight",  # GPT-2 (TransformerLens format)
            "model.layers.0.self_attn.o_proj.weight",  # Gemma
        ]

        wo_key = None
        for pattern in wo_key_patterns:
            if pattern in state_dict:
                wo_key = pattern
                break

        if wo_key:
            wo = state_dict[wo_key]
            # W_O shape is typically (d_model, d_model) or similar
            # After centering, mean along output dimension (dim=0) should be ~0
            output_means = wo.mean(dim=-1)  # Mean along input dimension
            mean_magnitude = output_means.abs().mean().item()

            # Threshold for "centered" - mean should be small
            is_centered = mean_magnitude < 0.01

            self.results.append(
                WeightProcessingCheck(
                    name="attention_output_centering",
                    passed=is_centered,
                    details={
                        "weight": "W_O (attention output)",
                        "mean_magnitude": mean_magnitude,
                        "threshold": 0.01,
                        "shape": list(wo.shape),
                    },
                    message=f"Attention output weights {'are' if is_centered else 'are NOT'} centered (mean={mean_magnitude:.6f})",
                )
            )
        else:
            self.results.append(
                WeightProcessingCheck(
                    name="attention_output_centering",
                    passed=True,
                    details={"reason": "No W_O found"},
                    message="No attention output weight found",
                )
            )

        # Check MLP output
        mlp_out_patterns = [
            "blocks.0.mlp.out",  # GPT-2 (TransformerLens format)
            "model.layers.0.mlp.down_proj.weight",  # Gemma
        ]

        mlp_out_key = None
        for pattern in mlp_out_patterns:
            if pattern in state_dict:
                mlp_out_key = pattern
                break

        if mlp_out_key:
            mlp_out = state_dict[mlp_out_key]
            output_means = mlp_out.mean(dim=-1)
            mean_magnitude = output_means.abs().mean().item()

            is_centered = mean_magnitude < 0.01

            self.results.append(
                WeightProcessingCheck(
                    name="mlp_output_centering",
                    passed=is_centered,
                    details={
                        "weight": "MLP output",
                        "mean_magnitude": mean_magnitude,
                        "threshold": 0.01,
                        "shape": list(mlp_out.shape),
                    },
                    message=f"MLP output weights {'are' if is_centered else 'are NOT'} centered (mean={mean_magnitude:.6f})",
                )
            )
        else:
            self.results.append(
                WeightProcessingCheck(
                    name="mlp_output_centering",
                    passed=True,
                    details={"reason": "No MLP output found"},
                    message="No MLP output weight found",
                )
            )

    def _check_unembed_centering(self, state_dict: Dict[str, torch.Tensor]) -> None:
        """Check that unembedding matrix has been centered."""
        unembed_patterns = [
            "lm_head.weight",  # Most models
            "output.weight",  # Some models
        ]

        unembed_key = None
        for pattern in unembed_patterns:
            if pattern in state_dict:
                unembed_key = pattern
                break

        if unembed_key:
            unembed = state_dict[unembed_key]
            # Unembed should have mean ≈ 0 along vocabulary dimension
            vocab_means = unembed.mean(dim=0)  # Mean across vocabulary
            mean_magnitude = vocab_means.abs().mean().item()

            is_centered = mean_magnitude < 0.1  # Slightly higher tolerance

            self.results.append(
                WeightProcessingCheck(
                    name="unembed_centering",
                    passed=is_centered,
                    details={
                        "mean_magnitude": mean_magnitude,
                        "threshold": 0.1,
                        "shape": list(unembed.shape),
                    },
                    message=f"Unembedding matrix {'is' if is_centered else 'is NOT'} centered (mean={mean_magnitude:.6f})",
                )
            )
        else:
            self.results.append(
                WeightProcessingCheck(
                    name="unembed_centering",
                    passed=True,
                    details={"reason": "No unembed found"},
                    message="No unembedding matrix found",
                )
            )

    def _check_value_bias_folding(self, state_dict: Dict[str, torch.Tensor]) -> None:
        """Check that value biases have been folded into output bias."""
        # After value bias folding, b_V should be zero and b_O should be modified

        # Check if b_V exists and is zero
        bv_patterns = [
            "blocks.0.attn.v.bias",  # GPT-2 (TransformerLens format)
            "model.layers.0.self_attn.v_proj.bias",  # Gemma
        ]

        bv_key = None
        for pattern in bv_patterns:
            if pattern in state_dict:
                bv_key = pattern
                break

        # Check value bias (already split in TransformerLens format)
        if bv_key:
            bv = state_dict[bv_key]
            bv_is_zero = torch.allclose(bv, torch.zeros_like(bv), atol=1e-6)
            bv_mean = bv.abs().mean().item()

            self.results.append(
                WeightProcessingCheck(
                    name="value_bias_folding",
                    passed=bv_is_zero,
                    details={
                        "bv_mean_abs": bv_mean,
                        "threshold": 1e-6,
                    },
                    message=f"Value bias {'is' if bv_is_zero else 'is NOT'} zero after folding (mean={bv_mean:.8f})",
                )
            )
        else:
            # No value bias found (some models don't have biases)
            self.results.append(
                WeightProcessingCheck(
                    name="value_bias_folding",
                    passed=True,
                    details={"reason": "No value bias found"},
                    message="No value bias found (expected for some architectures)",
                )
            )

    def _check_no_nan_inf(self, state_dict: Dict[str, torch.Tensor]) -> None:
        """Check that no weights contain NaN or Inf values."""
        nan_keys = []
        inf_keys = []

        for key, tensor in state_dict.items():
            if torch.isnan(tensor).any():
                nan_keys.append(key)
            if torch.isinf(tensor).any():
                inf_keys.append(key)

        has_issues = len(nan_keys) > 0 or len(inf_keys) > 0

        self.results.append(
            WeightProcessingCheck(
                name="no_nan_inf",
                passed=not has_issues,
                details={
                    "nan_count": len(nan_keys),
                    "inf_count": len(inf_keys),
                    "nan_keys": nan_keys[:5] if nan_keys else [],
                    "inf_keys": inf_keys[:5] if inf_keys else [],
                },
                message=f"Weights {'contain' if has_issues else 'do not contain'} NaN/Inf values",
            )
        )

    def _check_weight_magnitudes(self, state_dict: Dict[str, torch.Tensor]) -> None:
        """Check that weight magnitudes are reasonable (no explosion/vanishing)."""
        issues = []

        for key, tensor in state_dict.items():
            if "weight" not in key.lower():
                continue

            mean_abs = tensor.abs().mean().item()
            max_abs = tensor.abs().max().item()

            # Check for suspiciously large or small weights
            if mean_abs > 100:
                issues.append(f"{key}: mean_abs={mean_abs:.2f} (too large)")
            elif mean_abs < 1e-6 and "norm" not in key.lower():
                issues.append(f"{key}: mean_abs={mean_abs:.2e} (too small)")

            if max_abs > 1000:
                issues.append(f"{key}: max_abs={max_abs:.2f} (too large)")

        has_issues = len(issues) > 0

        self.results.append(
            WeightProcessingCheck(
                name="weight_magnitudes",
                passed=not has_issues,
                details={
                    "issue_count": len(issues),
                    "issues": issues[:10],  # First 10 issues
                },
                message=f"Weight magnitudes {'are suspicious' if has_issues else 'are reasonable'}",
            )
        )


def benchmark_weight_processing(
    model_name: str, device: str = "cpu", verbose: bool = True
) -> Tuple[int, int]:
    """Run weight processing benchmark on a model.

    Args:
        model_name: HuggingFace model name
        device: Device to load model on
        verbose: Whether to print detailed output

    Returns:
        Tuple of (passed_count, total_count)
    """
    import torch

    from transformer_lens.model_bridge import TransformerBridge

    if verbose:
        print(f"\nLoading {model_name}...")

    # Load model with weight processing
    bridge = TransformerBridge.boot_transformers(model_name, device=device, dtype=torch.float32)  # type: ignore[attr-defined]

    if verbose:
        print(f"Processing weights...")

    bridge.process_compatibility_weights(verbose=False)

    # Run benchmark
    benchmark = WeightProcessingBenchmark(bridge, verbose=verbose)
    return benchmark.run_all_checks()


if __name__ == "__main__":
    import sys

    # Test on multiple models
    models = [
        "gpt2",
        "google/gemma-2-2b-it",
    ]

    if len(sys.argv) > 1:
        models = sys.argv[1:]

    total_passed = 0
    total_checks = 0

    for model_name in models:
        try:
            passed, total = benchmark_weight_processing(model_name, verbose=True)
            total_passed += passed
            total_checks += total
            print()
        except Exception as e:
            print(f"\n❌ Error benchmarking {model_name}: {e}")
            import traceback

            traceback.print_exc()

    print("\n" + "=" * 80)
    print("OVERALL SUMMARY")
    print("=" * 80)
    print(
        f"Total: {total_passed}/{total_checks} checks passed ({100*total_passed//total_checks if total_checks > 0 else 0}%)"
    )
    print("=" * 80)
