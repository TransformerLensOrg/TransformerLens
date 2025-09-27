#!/usr/bin/env python3
"""Comprehensive test suite for TransformerBridge compatibility with HookedTransformer.

This test suite ensures that TransformerBridge maintains perfect compatibility
with HookedTransformer while using direct weight processing instead of delegation.
"""

import pytest
import torch

from transformer_lens import HookedTransformer
from transformer_lens.model_bridge import TransformerBridge


class TestTransformerBridgeCompatibility:
    """Test TransformerBridge compatibility and behavioral equivalence with HookedTransformer."""

    @pytest.fixture
    def models(self):
        """Create HookedTransformer and TransformerBridge for comparison."""
        device = "cpu"
        model_name = "gpt2"

        # HookedTransformer with processing
        ht = HookedTransformer.from_pretrained(
            model_name,
            device=device,
            fold_ln=True,
            center_writing_weights=True,
            center_unembed=True,
            fold_value_biases=True,
            refactor_factored_attn_matrices=False,
        )

        # TransformerBridge with compatibility mode
        bridge = TransformerBridge.boot_transformers(model_name, device=device)
        bridge.enable_compatibility_mode()

        return {"ht": ht, "bridge": bridge}

    @pytest.fixture
    def test_text(self):
        """Test text for forward pass verification."""
        return "Natural language processing tasks, such as question answering, machine translation, reading comprehension, and summarization, are typically approached with supervised learning on taskspecific datasets."

    def test_forward_pass_equivalence(self, models, test_text):
        """Test that forward passes produce identical results."""
        ht = models["ht"]
        bridge = models["bridge"]

        ht_loss = ht(test_text, return_type="loss")
        bridge_loss = bridge(test_text, return_type="loss")

        # Results should be nearly identical (allow for floating point precision)
        assert (
            abs(ht_loss - bridge_loss) < 1e-5
        ), f"Loss mismatch: HT={ht_loss:.6f}, Bridge={bridge_loss:.6f}"

    def test_logits_equivalence(self, models, test_text):
        """Test that logits outputs are identical."""
        ht = models["ht"]
        bridge = models["bridge"]

        ht_logits = ht(test_text, return_type="logits")
        bridge_logits = bridge(test_text, return_type="logits")

        assert torch.allclose(
            ht_logits, bridge_logits, rtol=1e-4, atol=1e-5
        ), "Logits should be nearly identical"

    def test_hook_functionality_equivalence(self, models, test_text):
        """Test that hook system produces identical ablation effects."""
        ht = models["ht"]
        bridge = models["bridge"]

        def ablation_hook(activation, hook):
            # Zero out attention head 8 in layer 0
            activation[:, :, 8, :] = 0
            return activation

        # Test with HookedTransformer using standard hook names
        ht_original = ht(test_text, return_type="loss")
        ht_ablated = ht.run_with_hooks(
            test_text, return_type="loss", fwd_hooks=[("blocks.0.attn.hook_v", ablation_hook)]
        )

        # Test with TransformerBridge using same hook names (should work due to extracted hooks)
        bridge_original = bridge(test_text, return_type="loss")
        bridge_ablated = bridge.run_with_hooks(
            test_text, return_type="loss", fwd_hooks=[("blocks.0.attn.hook_v", ablation_hook)]
        )

        ht_effect = ht_ablated - ht_original
        bridge_effect = bridge_ablated - bridge_original

        # Both should have similar ablation effects
        assert (
            abs(ht_effect - bridge_effect) < 1e-5
        ), f"Ablation effects should match: HT={ht_effect:.6f}, Bridge={bridge_effect:.6f}"

    def test_weight_sharing_verification(self, models, test_text):
        """Test that modifying weights affects both models similarly."""
        ht = models["ht"]
        bridge = models["bridge"]

        # Get baseline losses
        ht_original = ht(test_text, return_type="loss")
        bridge_original = bridge(test_text, return_type="loss")

        # Verify weights are identical before modification
        ht_W_V = ht.blocks[0].attn.W_V
        bridge_W_V = bridge.blocks[0].attn.W_V
        assert torch.allclose(ht_W_V, bridge_W_V), "Weights should be identical"

        # Modify weights in both models
        with torch.no_grad():
            ht.blocks[0].attn.W_V[0, :, :] = 0
            bridge.blocks[0].attn.W_V[0, :, :] = 0

        # Test modified losses
        ht_modified = ht(test_text, return_type="loss")
        bridge_modified = bridge(test_text, return_type="loss")

        ht_change = ht_modified - ht_original
        bridge_change = bridge_modified - bridge_original

        # Both models should respond similarly to weight changes
        assert (
            abs(ht_change - bridge_change) < 1e-4
        ), "Models should respond similarly to weight changes"

    def test_component_structure_equivalence(self, models):
        """Test that component structures are equivalent."""
        ht = models["ht"]
        bridge = models["bridge"]

        # Test that core components exist and have correct types
        assert hasattr(bridge, "embed"), "Bridge should have embed component"
        assert hasattr(bridge, "blocks"), "Bridge should have blocks component"
        assert hasattr(bridge, "ln_final"), "Bridge should have ln_final component"
        assert hasattr(bridge, "unembed"), "Bridge should have unembed component"

        # Test component structure matches
        assert len(bridge.blocks) == len(ht.blocks), "Should have same number of blocks"
        assert type(bridge.ln_final) == type(ht.ln_final), "Should have same ln_final type"

        # Test weight shapes match
        assert (
            ht.embed.W_E.shape == bridge.embed.W_E.shape
        ), "Embedding weights should have same shape"
        assert (
            ht.blocks[0].attn.W_V.shape == bridge.blocks[0].attn.W_V.shape
        ), "Attention weights should have same shape"

    def test_weight_processing_verification(self, models):
        """Test that weight processing (folding, centering) was applied correctly."""
        ht = models["ht"]
        bridge = models["bridge"]

        # Check layer norm folding - HT should have LayerNormPre, Bridge should have NormalizationBridge
        from transformer_lens.components.layer_norm_pre import LayerNormPre
        from transformer_lens.model_bridge.generalized_components.normalization import (
            NormalizationBridge,
        )

        assert isinstance(ht.ln_final, LayerNormPre), "HT should have LayerNormPre (folded)"
        assert isinstance(
            bridge.ln_final, NormalizationBridge
        ), "Bridge should have NormalizationBridge (integrated folding)"

        # Verify that the NormalizationBridge has LayerNormPre functionality
        assert hasattr(
            bridge.ln_final, "_layernorm_pre_forward"
        ), "Bridge ln_final should have LayerNormPre functionality"
        assert hasattr(
            bridge.ln_final.config, "layer_norm_folding"
        ), "Bridge ln_final should have layer_norm_folding config"
        assert (
            bridge.ln_final.config.layer_norm_folding
        ), "Bridge ln_final should be in folding mode"

        # Check weight centering - writing weights should be approximately centered
        ht_w_out = ht.blocks[0].mlp.W_out
        bridge_w_out = bridge.blocks[0].mlp.W_out

        ht_mean = torch.mean(ht_w_out, dim=-1, keepdim=True)
        bridge_mean = torch.mean(bridge_w_out, dim=-1, keepdim=True)

        # Both should be centered (mean ~0)
        assert torch.mean(torch.abs(ht_mean)).item() < 1e-4, "HT weights should be centered"
        assert torch.mean(torch.abs(bridge_mean)).item() < 1e-4, "Bridge weights should be centered"

    def test_hook_registry_completeness(self, models):
        """Test that TransformerBridge has complete hook registry from HookedTransformer."""
        ht = models["ht"]
        bridge = models["bridge"]

        # Key hooks that should be available
        important_hooks = [
            "hook_embed",
            "blocks.0.attn.hook_q",
            "blocks.0.attn.hook_k",
            "blocks.0.attn.hook_v",
            "blocks.0.attn.hook_z",
            "blocks.0.mlp.hook_pre",
            "blocks.0.mlp.hook_post",
        ]

        for hook_name in important_hooks:
            assert hook_name in ht.hook_dict, f"HT should have {hook_name}"
            assert hook_name in bridge._hook_registry, f"Bridge should have {hook_name} in registry"

    def test_no_persistent_hookedtransformer_reference(self, models):
        """Test that TransformerBridge has no persistent HookedTransformer references."""
        bridge = models["bridge"]

        # Should not have any persistent HookedTransformer references
        assert not hasattr(
            bridge, "_processed_hooked_transformer"
        ), "Should not have persistent HookedTransformer reference"

        # Should have extracted components
        assert hasattr(bridge, "blocks"), "Should have extracted blocks"
        assert hasattr(bridge, "_hook_registry"), "Should have hook registry"
        assert len(bridge._hook_registry) > 0, "Hook registry should not be empty"


class TestTransformerBridgeWeightModification:
    """Test that TransformerBridge properly responds to weight modifications."""

    @pytest.fixture
    def bridge_model(self):
        """Create TransformerBridge for testing."""
        device = "cpu"
        model_name = "gpt2"

        bridge = TransformerBridge.boot_transformers(model_name, device=device)
        bridge.enable_compatibility_mode()
        return bridge

    def test_weight_modification_propagates(self, bridge_model):
        """Test that weight modifications affect forward pass."""
        test_text = "Natural language processing"

        # Get original loss
        original_loss = bridge_model(test_text, return_type="loss")

        # Modify W_V weights
        with torch.no_grad():
            original_w_v = bridge_model.blocks[0].attn.W_V.clone()
            bridge_model.blocks[0].attn.W_V[0, :, :] = 0  # Zero out first head

        # Get modified loss
        modified_loss = bridge_model(test_text, return_type="loss")

        # Loss should change
        change = abs(modified_loss - original_loss)
        assert change > 1e-6, f"Weight modification should affect loss (change: {change:.6f})"

        # Restore weights
        with torch.no_grad():
            bridge_model.blocks[0].attn.W_V.copy_(original_w_v)


# Test cases that can be run individually for debugging
def test_simple_forward_equivalence():
    """Simple standalone test for forward pass equivalence."""
    device = "cpu"
    model_name = "gpt2"
    test_text = "The quick brown fox"

    # Create models
    ht = HookedTransformer.from_pretrained(
        model_name,
        device=device,
        fold_ln=True,
        center_writing_weights=True,
        center_unembed=True,
        fold_value_biases=True,
        refactor_factored_attn_matrices=False,
    )

    bridge = TransformerBridge.boot_transformers(model_name, device=device)
    bridge.enable_compatibility_mode()

    # Test forward pass
    ht_loss = ht(test_text, return_type="loss")
    bridge_loss = bridge(test_text, return_type="loss")

    print(f"HT loss: {ht_loss:.6f}")
    print(f"Bridge loss: {bridge_loss:.6f}")
    print(f"Difference: {abs(ht_loss - bridge_loss):.6f}")

    assert abs(ht_loss - bridge_loss) < 1e-5


if __name__ == "__main__":
    # Run simple test when executed directly
    test_simple_forward_equivalence()
    print("✅ Simple forward equivalence test passed!")
