#!/usr/bin/env python3
"""Test suite for TransformerBridge hook system functionality.

This test suite ensures that the TransformerBridge hook system works correctly
and maintains compatibility with HookedTransformer hook behavior.
"""

import torch
import pytest
from transformer_lens import HookedTransformer
from transformer_lens.model_bridge import TransformerBridge


class TestTransformerBridgeHooks:
    """Test TransformerBridge hook system functionality."""

    @pytest.fixture
    def bridge_model(self):
        """Create TransformerBridge with compatibility mode enabled."""
        device = "cpu"
        model_name = "gpt2"

        bridge = TransformerBridge.boot_transformers(model_name, device=device)
        bridge.enable_compatibility_mode()
        return bridge

    @pytest.fixture
    def reference_ht(self):
        """Create reference HookedTransformer for comparison."""
        device = "cpu"
        model_name = "gpt2"

        return HookedTransformer.from_pretrained(
            model_name, device=device,
            fold_ln=True, center_writing_weights=True, center_unembed=True,
            fold_value_biases=True, refactor_factored_attn_matrices=False,
        )

    def test_hook_registry_completeness(self, bridge_model, reference_ht):
        """Test that bridge has complete hook registry."""
        # Check that important hooks are available
        key_hooks = [
            "hook_embed",
            "hook_pos_embed",
            "blocks.0.attn.hook_q",
            "blocks.0.attn.hook_k",
            "blocks.0.attn.hook_v",
            "blocks.0.attn.hook_z",
        ]

        for hook_name in key_hooks:
            assert hook_name in reference_ht.hook_dict, f"Reference HT missing {hook_name}"
            assert hook_name in bridge_model._hook_registry, f"Bridge missing {hook_name}"

        # Bridge should have substantial number of hooks
        assert len(bridge_model._hook_registry) > 100, "Bridge should have substantial hook registry"

    def test_basic_hook_functionality(self, bridge_model):
        """Test that hooks fire and can modify activations."""
        test_text = "Natural language processing"
        hook_fired = False

        def test_hook(activation, hook):
            nonlocal hook_fired
            hook_fired = True
            assert isinstance(activation, torch.Tensor), "Hook should receive tensor"
            assert activation.shape[-1] > 0, "Activation should have meaningful shape"
            return activation

        # Run with hook
        result = bridge_model.run_with_hooks(
            test_text,
            return_type="logits",
            fwd_hooks=[("hook_embed", test_hook)]
        )

        assert hook_fired, "Hook should have fired"
        assert isinstance(result, torch.Tensor), "Should return tensor result"

    def test_ablation_hook_effect(self, bridge_model):
        """Test that ablation hooks actually affect output."""
        test_text = "Natural language processing"

        # Get baseline
        baseline_loss = bridge_model(test_text, return_type="loss")

        def ablation_hook(activation, hook):
            # Zero out first attention head
            activation[:, :, 0, :] = 0
            return activation

        # Run with ablation
        ablated_loss = bridge_model.run_with_hooks(
            test_text,
            return_type="loss",
            fwd_hooks=[("blocks.0.attn.hook_v", ablation_hook)]
        )

        # Should see meaningful change
        effect = abs(ablated_loss - baseline_loss)
        assert effect > 1e-6, f"Ablation should have meaningful effect (got {effect:.6f})"

    def test_hook_equivalence_with_reference(self, bridge_model, reference_ht):
        """Test that hooks produce equivalent effects to reference HookedTransformer."""
        test_text = "Natural language processing"

        def ablation_hook(activation, hook):
            # Zero out attention head 5 in layer 0
            activation[:, :, 5, :] = 0
            return activation

        # Test reference HookedTransformer
        ht_baseline = reference_ht(test_text, return_type="loss")
        ht_ablated = reference_ht.run_with_hooks(
            test_text,
            return_type="loss",
            fwd_hooks=[("blocks.0.attn.hook_v", ablation_hook)]
        )

        # Test TransformerBridge
        bridge_baseline = bridge_model(test_text, return_type="loss")
        bridge_ablated = bridge_model.run_with_hooks(
            test_text,
            return_type="loss",
            fwd_hooks=[("blocks.0.attn.hook_v", ablation_hook)]
        )

        # Effects should be similar
        ht_effect = ht_ablated - ht_baseline
        bridge_effect = bridge_ablated - bridge_baseline

        effect_diff = abs(ht_effect - bridge_effect)
        assert effect_diff < 1e-5, f"Hook effects should match between models (diff: {effect_diff:.6f})"

    def test_multiple_hooks(self, bridge_model):
        """Test that multiple hooks can be applied simultaneously."""
        test_text = "Natural language processing"
        hooks_fired = set()

        def make_hook(hook_id):
            def hook_fn(activation, hook):
                hooks_fired.add(hook_id)
                return activation
            return hook_fn

        # Apply multiple hooks
        result = bridge_model.run_with_hooks(
            test_text,
            return_type="logits",
            fwd_hooks=[
                ("hook_embed", make_hook("embed")),
                ("blocks.0.attn.hook_q", make_hook("q")),
                ("blocks.0.attn.hook_v", make_hook("v")),
            ]
        )

        # All hooks should have fired
        expected_hooks = {"embed", "q", "v"}
        assert hooks_fired == expected_hooks, f"Expected {expected_hooks}, got {hooks_fired}"

    def test_hook_activation_shapes(self, bridge_model):
        """Test that hook activations have expected shapes."""
        test_text = "The quick brown fox"
        captured_shapes = {}

        def capture_shape_hook(hook_name):
            def hook_fn(activation, hook):
                captured_shapes[hook_name] = activation.shape
                return activation
            return hook_fn

        # Test various hook points
        bridge_model.run_with_hooks(
            test_text,
            return_type="logits",
            fwd_hooks=[
                ("hook_embed", capture_shape_hook("embed")),
                ("blocks.0.attn.hook_v", capture_shape_hook("v")),
                ("blocks.0.mlp.hook_pre", capture_shape_hook("mlp_pre")),
            ]
        )

        # Verify shapes make sense
        assert len(captured_shapes) == 3, "Should have captured 3 activations"

        # Embedding should be [batch, seq, d_model]
        embed_shape = captured_shapes["embed"]
        assert len(embed_shape) == 3, "Embedding should be 3D"
        assert embed_shape[-1] == 768, "Should have d_model=768 for GPT2"

        # Attention values should be [batch, seq, n_heads, d_head]
        v_shape = captured_shapes["v"]
        assert len(v_shape) == 4, "Attention values should be 4D"
        assert v_shape[2] == 12, "Should have 12 heads for GPT2"

    def test_hook_context_manager(self, bridge_model):
        """Test hook context manager functionality."""
        test_text = "Natural language processing"
        hook_fired = False

        def test_hook(activation, hook):
            nonlocal hook_fired
            hook_fired = True
            return activation

        # Use context manager
        with bridge_model.hooks(fwd_hooks=[("hook_embed", test_hook)]):
            result = bridge_model(test_text, return_type="logits")

        assert hook_fired, "Hook should have fired in context"

        # Hook should be removed after context
        hook_fired = False
        bridge_model(test_text, return_type="logits")
        assert not hook_fired, "Hook should be removed after context"


def test_standalone_hook_functionality():
    """Standalone test for basic hook functionality."""
    device = "cpu"
    model_name = "gpt2"

    # Create bridge
    bridge = TransformerBridge.boot_transformers(model_name, device=device)
    bridge.enable_compatibility_mode()

    test_text = "The quick brown fox"

    # Test basic hook
    hook_called = False
    def test_hook(activation, hook):
        nonlocal hook_called
        hook_called = True
        print(f"Hook fired: {hook.name}, shape: {activation.shape}")
        return activation

    result = bridge.run_with_hooks(
        test_text,
        return_type="loss",
        fwd_hooks=[("blocks.0.attn.hook_v", test_hook)]
    )

    assert hook_called, "Hook should have been called"
    assert isinstance(result, torch.Tensor), "Should return tensor result"
    print(f"✅ Hook test passed! Loss: {result:.6f}")


if __name__ == "__main__":
    # Run standalone test when executed directly
    test_standalone_hook_functionality()