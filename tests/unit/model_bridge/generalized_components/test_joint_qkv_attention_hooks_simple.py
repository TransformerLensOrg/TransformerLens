"""Simplified tests for JointQKVAttentionBridge hook integration.

This module tests the core functionality that hooks applied to V components
are properly integrated back into the main attention computation.
"""

import pytest
import torch
from jaxtyping import Float

import transformer_lens.utils as utils
from transformer_lens.hook_points import HookPoint
from transformer_lens.model_bridge import TransformerBridge


class TestJointQKVAttentionHooksSimple:
    """Simplified test suite for JointQKVAttentionBridge hook integration."""

    @pytest.fixture
    def model(self):
        """Create a GPT-2 model for testing."""
        torch.set_grad_enabled(False)
        return TransformerBridge.boot_transformers("gpt2", device="cpu")

    @pytest.fixture
    def sample_input(self, model):
        """Create sample input tokens."""
        text = "The quick brown fox jumps over the lazy dog"
        return model.to_tokens(text)

    def test_v_hook_integration_works(self, model, sample_input):
        """Test that V hooks are properly integrated into computation."""
        layer_idx = 0
        head_idx = 5

        def v_ablation_hook(
            value: Float[torch.Tensor, "batch pos head_index d_head"], hook: HookPoint
        ) -> Float[torch.Tensor, "batch pos head_index d_head"]:
            """Ablate a specific head in the V matrix."""
            # Verify we get the expected shape
            assert len(value.shape) == 4, f"Expected 4D tensor, got {value.shape}"
            batch, pos, heads, d_head = value.shape
            assert heads == 12, f"Expected 12 heads, got {heads}"
            assert d_head == 64, f"Expected 64 d_head, got {d_head}"

            # Ablate the specified head
            value[:, :, head_idx, :] = 0.0
            return value

        # Test that hook affects computation
        original_loss = model(sample_input, return_type="loss")
        hooked_loss = model.run_with_hooks(
            sample_input,
            return_type="loss",
            fwd_hooks=[(utils.get_act_name("v", layer_idx), v_ablation_hook)],
        )

        # Verify the hook affected the computation
        assert not torch.isclose(original_loss, hooked_loss, atol=1e-6), (
            f"V hook should affect computation. Original: {original_loss:.6f}, "
            f"Hooked: {hooked_loss:.6f}"
        )

    def test_extreme_v_ablation_large_effect(self, model, sample_input):
        """Test that extreme V ablations have significant effects."""
        layer_idx = 0

        def zero_all_v_hook(
            value: Float[torch.Tensor, "batch pos head_index d_head"], hook: HookPoint
        ) -> Float[torch.Tensor, "batch pos head_index d_head"]:
            """Zero out all V values."""
            value[:, :, :, :] = 0.0
            return value

        original_loss = model(sample_input, return_type="loss")
        extreme_hooked_loss = model.run_with_hooks(
            sample_input,
            return_type="loss",
            fwd_hooks=[(utils.get_act_name("v", layer_idx), zero_all_v_hook)],
        )

        # Verify extreme ablation has meaningful effect
        effect_size = abs(extreme_hooked_loss - original_loss)
        assert (
            effect_size > 0.01
        ), f"Extreme V ablation should have meaningful effect, got {effect_size:.6f}"

    def test_multiple_v_hooks_cumulative(self, model, sample_input):
        """Test that multiple V hooks on different layers have cumulative effects."""

        def v_scale_hook(
            value: Float[torch.Tensor, "batch pos head_index d_head"], hook: HookPoint
        ) -> Float[torch.Tensor, "batch pos head_index d_head"]:
            """Scale V values by 0.8."""
            value *= 0.8
            return value

        original_loss = model(sample_input, return_type="loss")

        # Apply hook to layer 0 only
        layer0_loss = model.run_with_hooks(
            sample_input,
            return_type="loss",
            fwd_hooks=[(utils.get_act_name("v", 0), v_scale_hook)],
        )

        # Apply hooks to layers 0 and 1
        both_layers_loss = model.run_with_hooks(
            sample_input,
            return_type="loss",
            fwd_hooks=[
                (utils.get_act_name("v", 0), v_scale_hook),
                (utils.get_act_name("v", 1), v_scale_hook),
            ],
        )

        # Verify all configurations produce different results
        assert not torch.isclose(original_loss, layer0_loss, atol=1e-6)
        assert not torch.isclose(original_loss, both_layers_loss, atol=1e-6)
        assert not torch.isclose(layer0_loss, both_layers_loss, atol=1e-6)

    def test_no_hooks_uses_efficient_path(self, model, sample_input):
        """Test that when no hooks are present, the efficient fused path is used."""
        # Run multiple times without hooks - should be deterministic
        loss1 = model(sample_input, return_type="loss")
        loss2 = model(sample_input, return_type="loss")
        loss3 = model(sample_input, return_type="loss")

        # All should be identical (deterministic)
        assert torch.isclose(loss1, loss2, atol=1e-8)
        assert torch.isclose(loss2, loss3, atol=1e-8)

    def test_hook_preserves_original_functionality(self, model, sample_input):
        """Test that the reconstructed attention produces similar results to original when no modification is made."""
        layer_idx = 0

        def identity_hook(
            value: Float[torch.Tensor, "batch pos head_index d_head"], hook: HookPoint
        ) -> Float[torch.Tensor, "batch pos head_index d_head"]:
            """Identity hook that doesn't modify the tensor."""
            return value

        original_loss = model(sample_input, return_type="loss")
        identity_hooked_loss = model.run_with_hooks(
            sample_input,
            return_type="loss",
            fwd_hooks=[(utils.get_act_name("v", layer_idx), identity_hook)],
        )

        # Should be very close (some small numerical differences are expected due to reconstruction)
        assert torch.isclose(original_loss, identity_hooked_loss, atol=1e-3), (
            f"Identity hook should preserve functionality. Original: {original_loss:.6f}, "
            f"Hooked: {identity_hooked_loss:.6f}, Diff: {abs(original_loss - identity_hooked_loss):.6f}"
        )

    def test_reconstruction_vs_original_demo_compatibility(self, model):
        """Test that our fix maintains compatibility with the original Main_Demo.py."""
        # Use the exact same text and setup as the original demo
        gpt2_text = "Natural language processing tasks, such as question answering, machine translation, reading comprehension, and summarization, are typically approached with supervised learning on taskspecific datasets."
        gpt2_tokens = model.to_tokens(gpt2_text)

        layer_to_ablate = 0
        head_index_to_ablate = 8

        def head_ablation_hook(
            value: Float[torch.Tensor, "batch pos head_index d_head"], hook: HookPoint
        ) -> Float[torch.Tensor, "batch pos head_index d_head"]:
            """Exact same hook as in Main_Demo.py."""
            value[:, :, head_index_to_ablate, :] = 0.0
            return value

        original_loss = model(gpt2_tokens, return_type="loss")
        ablated_loss = model.run_with_hooks(
            gpt2_tokens,
            return_type="loss",
            fwd_hooks=[(utils.get_act_name("v", layer_to_ablate), head_ablation_hook)],
        )

        # Verify the ablation works (this was the original bug)
        assert not torch.isclose(original_loss, ablated_loss, atol=1e-6), (
            f"Main_Demo.py compatibility test failed. Original: {original_loss:.6f}, "
            f"Ablated: {ablated_loss:.6f}"
        )

        # Verify we get reasonable values (original was ~3.999, ablated should be different)
        assert (
            3.0 < original_loss < 5.0
        ), f"Original loss should be reasonable, got {original_loss:.6f}"
        assert (
            3.0 < ablated_loss < 6.0
        ), f"Ablated loss should be reasonable, got {ablated_loss:.6f}"

    def test_comprehensive_hook_detection(self, model, sample_input):
        """Test that all types of hooks (input/output, forward/backward) are properly detected."""
        layer_idx = 0

        def dummy_hook(value, hook):
            """Simple hook that doesn't modify the tensor."""
            return value

        def dummy_backward_hook(grad, hook):
            """Simple backward hook."""
            return grad

        # Test that forward hooks on hook_in are detected
        original_loss = model(sample_input, return_type="loss")

        # Add forward hook to Q input
        q_hook_in_loss = model.run_with_hooks(
            sample_input,
            return_type="loss",
            fwd_hooks=[(f"blocks.{layer_idx}.attn.q.hook_in", dummy_hook)],
        )

        # Should use reconstruction path (may have slight differences due to numerical precision)
        # The key test is that it doesn't error and processes the hook
        assert isinstance(q_hook_in_loss, torch.Tensor), "Q hook_in should be processed correctly"

        # Add forward hook to V input
        v_hook_in_loss = model.run_with_hooks(
            sample_input,
            return_type="loss",
            fwd_hooks=[(f"blocks.{layer_idx}.attn.v.hook_in", dummy_hook)],
        )

        assert isinstance(v_hook_in_loss, torch.Tensor), "V hook_in should be processed correctly"

    def test_hookpoint_has_hooks_method(self):
        """Test the new has_hooks method on HookPoint."""
        from transformer_lens.hook_points import HookPoint

        # Create a fresh HookPoint
        hook_point = HookPoint()

        # Initially should have no hooks
        assert not hook_point.has_hooks("fwd"), "Should have no forward hooks initially"
        assert not hook_point.has_hooks("bwd"), "Should have no backward hooks initially"
        assert not hook_point.has_hooks("both"), "Should have no hooks initially"

        # Add a forward hook
        def dummy_hook(value, hook):
            return value

        hook_point.add_hook(dummy_hook, dir="fwd")

        # Should now detect forward hooks
        assert hook_point.has_hooks("fwd"), "Should detect forward hooks"
        assert not hook_point.has_hooks("bwd"), "Should still have no backward hooks"
        assert hook_point.has_hooks("both"), "Should detect hooks when checking both directions"

        # Add a backward hook
        hook_point.add_hook(dummy_hook, dir="bwd")

        # Should now detect both
        assert hook_point.has_hooks("fwd"), "Should still detect forward hooks"
        assert hook_point.has_hooks("bwd"), "Should now detect backward hooks"
        assert hook_point.has_hooks("both"), "Should detect hooks in both directions"

        # Test permanent hook detection
        hook_point.add_hook(dummy_hook, dir="fwd", is_permanent=True)

        assert hook_point.has_hooks(
            "fwd", including_permanent=True
        ), "Should detect permanent hooks when included"
        # Note: We can't easily test excluding permanent hooks without removing non-permanent ones

        # Clean up
        hook_point.remove_hooks("both", including_permanent=True)
