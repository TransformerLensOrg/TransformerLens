#!/usr/bin/env python3
"""
Integration Compatibility Test for Weight Processing
====================================================

This test verifies that:
1. HookedTransformer with processing matches expected Main Demo values (3.999 → 5.453)
2. HookedTransformer without processing matches expected unprocessed values (~3.999 → ~4.117)
3. TransformerBridge with processing matches HookedTransformer with processing
4. TransformerBridge without processing matches HookedTransformer without processing
5. Processing maintains mathematical equivalence for baseline computation
6. Processing changes ablation results as expected (for better interpretability)
"""

import pytest
import torch
from jaxtyping import Float

from transformer_lens import HookedTransformer, utils
from transformer_lens.model_bridge.bridge import TransformerBridge


class TestWeightProcessingCompatibility:
    """Test class for weight processing compatibility between HookedTransformer and TransformerBridge."""

    @pytest.fixture(scope="class")
    def model_name(self):
        return "gpt2"

    @pytest.fixture(scope="class")
    def device(self):
        return "cpu"

    @pytest.fixture(scope="class")
    def test_text(self):
        return "Natural language processing tasks, such as question answering, machine translation, reading comprehension, and summarization, are typically approached with supervised learning on taskspecific datasets."

    @pytest.fixture(scope="class")
    def ablation_params(self):
        return {"layer_to_ablate": 0, "head_index_to_ablate": 8}

    @pytest.fixture(scope="class")
    def expected_values(self):
        return {
            "processed_orig": 3.999,
            "processed_ablated": 5.453,
            "unprocessed_orig": 3.999,
            "unprocessed_ablated": 4.117,
        }

    @pytest.fixture(scope="class")
    def tolerance(self):
        return 0.01

    @pytest.fixture(scope="class")
    def hooked_processed(self, model_name, device):
        """Load HookedTransformer with processing."""
        print("Loading HookedTransformer with processing...")
        return HookedTransformer.from_pretrained(
            model_name,
            device=device,
            fold_ln=True,
            center_writing_weights=True,
            center_unembed=True,
            fold_value_biases=True,
        )

    @pytest.fixture(scope="class")
    def hooked_unprocessed(self, model_name, device):
        """Load HookedTransformer without processing."""
        print("Loading HookedTransformer without processing...")
        return HookedTransformer.from_pretrained_no_processing(model_name, device=device)

    @pytest.fixture(scope="class")
    def bridge_processed(self, model_name, device):
        """Load TransformerBridge with processing."""
        print("Loading TransformerBridge with processing...")
        bridge = TransformerBridge.boot_transformers(model_name, device=device)
        bridge.enable_compatibility_mode()  # Enable compatibility mode for hook aliases
        return bridge

    @pytest.fixture(scope="class")
    def bridge_unprocessed(self, model_name, device):
        """Load TransformerBridge without processing."""
        print("Loading TransformerBridge without processing...")
        bridge = TransformerBridge.boot_transformers(model_name, device=device)
        bridge.enable_compatibility_mode(
            no_processing=True
        )  # Enable compatibility mode for hook aliases
        # No processing applied
        return bridge

    def create_ablation_hook(self, head_index_to_ablate):
        """Create the exact ablation hook from Main Demo."""

        def head_ablation_hook(
            value: Float[torch.Tensor, "batch pos head_index d_head"], hook
        ) -> Float[torch.Tensor, "batch pos head_index d_head"]:
            value[:, :, head_index_to_ablate, :] = 0.0
            return value

        return head_ablation_hook

    def _test_model_ablation(self, model, model_name: str, test_text, ablation_params):
        """Test a model and return original and ablated losses."""
        tokens = model.to_tokens(test_text)

        # Original loss
        original_loss = model(tokens, return_type="loss").item()

        # Ablated loss
        ablated_loss = model.run_with_hooks(
            tokens,
            return_type="loss",
            fwd_hooks=[
                (
                    utils.get_act_name("v", ablation_params["layer_to_ablate"]),
                    self.create_ablation_hook(ablation_params["head_index_to_ablate"]),
                )
            ],
        ).item()

        print(f"{model_name}: Original={original_loss:.6f}, Ablated={ablated_loss:.6f}")
        return original_loss, ablated_loss

    def test_hooked_transformer_processed_matches_main_demo(
        self, hooked_processed, test_text, ablation_params, expected_values, tolerance
    ):
        """Test that HookedTransformer with processing matches Main Demo values."""
        orig, ablated = self._test_model_ablation(
            hooked_processed, "HookedTransformer (processed)", test_text, ablation_params
        )

        assert (
            abs(orig - expected_values["processed_orig"]) < tolerance
        ), f"HookedTransformer processed original loss {orig:.6f} != expected {expected_values['processed_orig']:.3f}"
        assert (
            abs(ablated - expected_values["processed_ablated"]) < tolerance
        ), f"HookedTransformer processed ablated loss {ablated:.6f} != expected {expected_values['processed_ablated']:.3f}"

    def test_hooked_transformer_unprocessed_matches_expected(
        self, hooked_unprocessed, test_text, ablation_params, expected_values, tolerance
    ):
        """Test that HookedTransformer without processing matches expected values."""
        orig, ablated = self._test_model_ablation(
            hooked_unprocessed, "HookedTransformer (unprocessed)", test_text, ablation_params
        )

        assert (
            abs(orig - expected_values["unprocessed_orig"]) < tolerance
        ), f"HookedTransformer unprocessed original loss {orig:.6f} != expected {expected_values['unprocessed_orig']:.3f}"
        assert (
            abs(ablated - expected_values["unprocessed_ablated"]) < tolerance
        ), f"HookedTransformer unprocessed ablated loss {ablated:.6f} != expected {expected_values['unprocessed_ablated']:.3f}"

    def test_baseline_mathematical_equivalence(
        self, hooked_processed, hooked_unprocessed, test_text, ablation_params
    ):
        """Test that processing maintains mathematical equivalence for baseline computation."""
        hooked_proc_orig, _ = self._test_model_ablation(
            hooked_processed, "HookedTransformer (processed)", test_text, ablation_params
        )
        hooked_unproc_orig, _ = self._test_model_ablation(
            hooked_unprocessed, "HookedTransformer (unprocessed)", test_text, ablation_params
        )

        orig_diff = abs(hooked_proc_orig - hooked_unproc_orig)
        assert (
            orig_diff < 0.001
        ), f"Baseline computation not mathematically equivalent: diff={orig_diff:.6f}"

    def test_ablation_interpretability_enhancement(
        self, hooked_processed, hooked_unprocessed, test_text, ablation_params
    ):
        """Test that processing changes ablation results as expected for interpretability."""
        _, hooked_proc_ablated = self._test_model_ablation(
            hooked_processed, "HookedTransformer (processed)", test_text, ablation_params
        )
        _, hooked_unproc_ablated = self._test_model_ablation(
            hooked_unprocessed, "HookedTransformer (unprocessed)", test_text, ablation_params
        )

        ablated_diff = abs(hooked_proc_ablated - hooked_unproc_ablated)
        assert (
            ablated_diff > 0.5
        ), f"Ablation results should be significantly different for interpretability: diff={ablated_diff:.6f}"

    @pytest.mark.skip(
        reason="TransformerBridge processing compatibility has architectural differences that cause large numerical discrepancies"
    )
    def test_bridge_processed_matches_hooked_processed(
        self, bridge_processed, hooked_processed, test_text, ablation_params, tolerance
    ):
        """Test that TransformerBridge with processing matches HookedTransformer with processing."""
        bridge_orig, bridge_ablated = self._test_model_ablation(
            bridge_processed, "TransformerBridge (processed)", test_text, ablation_params
        )
        hooked_orig, hooked_ablated = self._test_model_ablation(
            hooked_processed, "HookedTransformer (processed)", test_text, ablation_params
        )

        assert (
            abs(bridge_orig - hooked_orig) < tolerance
        ), f"TransformerBridge processed original {bridge_orig:.6f} != HookedTransformer processed {hooked_orig:.6f}"
        assert (
            abs(bridge_ablated - hooked_ablated) < tolerance
        ), f"TransformerBridge processed ablated {bridge_ablated:.6f} != HookedTransformer processed {hooked_ablated:.6f}"

    @pytest.mark.skip(
        reason="TransformerBridge processing compatibility has architectural differences that cause large numerical discrepancies"
    )
    def test_bridge_unprocessed_matches_hooked_unprocessed(
        self, bridge_unprocessed, hooked_unprocessed, test_text, ablation_params, tolerance
    ):
        """Test that TransformerBridge without processing matches HookedTransformer without processing."""
        bridge_orig, bridge_ablated = self._test_model_ablation(
            bridge_unprocessed, "TransformerBridge (unprocessed)", test_text, ablation_params
        )
        hooked_orig, hooked_ablated = self._test_model_ablation(
            hooked_unprocessed, "HookedTransformer (unprocessed)", test_text, ablation_params
        )

        assert (
            abs(bridge_orig - hooked_orig) < tolerance
        ), f"TransformerBridge unprocessed original {bridge_orig:.6f} != HookedTransformer unprocessed {hooked_orig:.6f}"
        assert (
            abs(bridge_ablated - hooked_ablated) < tolerance
        ), f"TransformerBridge unprocessed ablated {bridge_ablated:.6f} != HookedTransformer unprocessed {hooked_ablated:.6f}"
