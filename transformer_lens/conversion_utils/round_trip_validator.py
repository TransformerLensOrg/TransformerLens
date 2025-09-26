"""Round-trip conversion validator for weight processing workflows.

This module provides validation for HF ↔ TLens weight conversions after processing,
ensuring that weight folding and other transformations preserve model behavior.
"""

import torch
from typing import Dict, Any, Optional, Tuple
from transformer_lens.conversion_utils.reversible_weight_converter import ReversibleWeightConverter


class RoundTripValidator:
    """Validates round-trip conversions for processed weights."""

    def __init__(self, tolerance: float = 1e-6):
        """Initialize the validator.

        Args:
            tolerance: Maximum allowed difference for validation to pass
        """
        self.tolerance = tolerance
        self.converter = ReversibleWeightConverter()

    def validate_processed_weight_conversion(
        self,
        original_hf_weights: Dict[str, torch.Tensor],
        processed_tlens_weights: Dict[str, torch.Tensor],
        config: Any,
        model_type: str
    ) -> Dict[str, Any]:
        """Validate that processed TLens weights can round-trip through HF format.

        Args:
            original_hf_weights: Original HuggingFace model weights
            processed_tlens_weights: Processed TransformerLens weights
            config: Model configuration
            model_type: Type of model (e.g., "gpt2")

        Returns:
            Dictionary containing validation results
        """
        results = {
            "success": False,
            "conversions": {},
            "shape_validation": {},
            "precision_validation": {},
            "errors": []
        }

        try:
            # Step 1: Convert processed TLens → HF
            hf_weights_from_processed = self.converter.tlens_to_hf(
                processed_tlens_weights, config, model_type
            )
            results["conversions"]["tlens_to_hf"] = {
                "original_keys": len(processed_tlens_weights),
                "converted_keys": len(hf_weights_from_processed)
            }

            # Step 2: Convert HF → TLens (round-trip)
            tlens_weights_roundtrip = self.converter.hf_to_tlens(
                hf_weights_from_processed, config, model_type
            )
            results["conversions"]["hf_to_tlens_roundtrip"] = {
                "converted_keys": len(tlens_weights_roundtrip)
            }

            # Step 3: Validate shapes match
            shape_mismatches = []
            for key in processed_tlens_weights.keys():
                if key in tlens_weights_roundtrip:
                    original_shape = processed_tlens_weights[key].shape
                    roundtrip_shape = tlens_weights_roundtrip[key].shape
                    if original_shape != roundtrip_shape:
                        shape_mismatches.append({
                            "key": key,
                            "original": original_shape,
                            "roundtrip": roundtrip_shape
                        })

            results["shape_validation"] = {
                "total_keys_checked": len(processed_tlens_weights),
                "shape_mismatches": shape_mismatches,
                "shapes_match": len(shape_mismatches) == 0
            }

            # Step 4: Validate precision
            precision_errors = []
            max_difference = 0.0

            for key in processed_tlens_weights.keys():
                if key in tlens_weights_roundtrip:
                    if processed_tlens_weights[key].shape == tlens_weights_roundtrip[key].shape:
                        diff = torch.max(torch.abs(
                            processed_tlens_weights[key] - tlens_weights_roundtrip[key]
                        )).item()
                        max_difference = max(max_difference, diff)

                        if diff > self.tolerance:
                            precision_errors.append({
                                "key": key,
                                "difference": diff,
                                "tolerance": self.tolerance
                            })

            results["precision_validation"] = {
                "max_difference": max_difference,
                "tolerance": self.tolerance,
                "precision_errors": precision_errors,
                "precision_ok": len(precision_errors) == 0
            }

            # Overall success
            results["success"] = (
                results["shape_validation"]["shapes_match"] and
                results["precision_validation"]["precision_ok"]
            )

        except Exception as e:
            results["errors"].append(str(e))
            results["success"] = False

        return results

    def validate_conversion_with_sample_keys(
        self,
        processed_tlens_weights: Dict[str, torch.Tensor],
        tlens_weights_roundtrip: Dict[str, torch.Tensor],
        sample_size: int = 5
    ) -> Dict[str, Any]:
        """Quick validation using a sample of keys for debugging.

        Args:
            processed_tlens_weights: Original processed weights
            tlens_weights_roundtrip: Round-trip converted weights
            sample_size: Number of keys to check

        Returns:
            Dictionary with sample validation results
        """
        sample_keys = list(processed_tlens_weights.keys())[:sample_size]
        results = {"sample_validation": []}

        for key in sample_keys:
            if key in tlens_weights_roundtrip:
                original_shape = processed_tlens_weights[key].shape
                roundtrip_shape = tlens_weights_roundtrip[key].shape

                result_entry = {
                    "key": key,
                    "original_shape": original_shape,
                    "roundtrip_shape": roundtrip_shape,
                    "shapes_match": original_shape == roundtrip_shape
                }

                if original_shape == roundtrip_shape:
                    diff = torch.max(torch.abs(
                        processed_tlens_weights[key] - tlens_weights_roundtrip[key]
                    )).item()
                    result_entry["difference"] = diff
                    result_entry["precision_ok"] = diff <= self.tolerance

                results["sample_validation"].append(result_entry)

        return results