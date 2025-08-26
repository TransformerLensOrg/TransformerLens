"""Integration tests for QKV hook compatibility in TransformerBridge."""

import torch

from transformer_lens.model_bridge import TransformerBridge


class TestQKVHookCompatibility:
    """Test that QKV bridge hooks are compatible with overall model hook access."""

    def test_v_hook_out_equals_blocks_attn_hook_v(self):
        """Test that v_hook_out in QKV bridge equals blocks.0.attn.hook_v on the overall model."""
        # Load GPT-2 in TransformerBridge
        bridge = TransformerBridge.boot_transformers("gpt2", device="cpu")

        # Turn on compatibility mode
        bridge.enable_compatibility_mode(disable_warnings=True)

        # Create test input
        test_input = torch.tensor([[1, 2, 3, 4, 5]])  # Simple test sequence

        # Get the QKV bridge from the first attention layer
        qkv_bridge = bridge.blocks[0].attn.qkv

        # Verify that qkv_bridge is indeed a QKVBridge
        from transformer_lens.model_bridge.generalized_components.qkv_bridge import (
            QKVBridge,
        )

        assert isinstance(qkv_bridge, QKVBridge), "First attention layer should have a QKVBridge"

        # Run a forward pass to populate the hooks
        with torch.no_grad():
            _ = bridge(test_input)

        # Assert that v_hook_out in the QKV bridge is the same object as
        # blocks.0.attn.hook_v on the overall model
        assert (
            qkv_bridge.v_hook_out is bridge.blocks[0].attn.hook_v
        ), "v_hook_out in QKV bridge should be the same object as blocks.0.attn.hook_v"

        # Also test that the hook points have the same properties
        assert (
            qkv_bridge.v_hook_out.has_hooks() == bridge.blocks[0].attn.hook_v.has_hooks()
        ), "Hook points should have the same hook status"

    def test_q_hook_out_equals_blocks_attn_hook_q(self):
        """Test that q_hook_out in QKV bridge equals blocks.0.attn.hook_q on the overall model."""
        # Load GPT-2 in TransformerBridge
        bridge = TransformerBridge.boot_transformers("gpt2", device="cpu")

        # Turn on compatibility mode
        bridge.enable_compatibility_mode(disable_warnings=True)

        # Create test input
        test_input = torch.tensor([[1, 2, 3, 4, 5]])  # Simple test sequence

        # Get the QKV bridge from the first attention layer
        qkv_bridge = bridge.blocks[0].attn.qkv

        # Run a forward pass to populate the hooks
        with torch.no_grad():
            _ = bridge(test_input)

        # Assert that q_hook_out in the QKV bridge is the same object as
        # blocks.0.attn.hook_q on the overall model
        assert (
            qkv_bridge.q_hook_out is bridge.blocks[0].attn.hook_q
        ), "q_hook_out in QKV bridge should be the same object as blocks.0.attn.hook_q"

    def test_k_hook_out_equals_blocks_attn_hook_k(self):
        """Test that k_hook_out in QKV bridge equals blocks.0.attn.hook_k on the overall model."""
        # Load GPT-2 in TransformerBridge
        bridge = TransformerBridge.boot_transformers("gpt2", device="cpu")

        # Turn on compatibility mode
        bridge.enable_compatibility_mode(disable_warnings=True)

        # Create test input
        test_input = torch.tensor([[1, 2, 3, 4, 5]])  # Simple test sequence

        # Get the QKV bridge from the first attention layer
        qkv_bridge = bridge.blocks[0].attn.qkv

        # Run a forward pass to populate the hooks
        with torch.no_grad():
            _ = bridge(test_input)

        # Assert that k_hook_out in the QKV bridge is the same object as
        # blocks.0.attn.hook_k on the overall model
        assert (
            qkv_bridge.k_hook_out is bridge.blocks[0].attn.hook_k
        ), "k_hook_out in QKV bridge should be the same object as blocks.0.attn.hook_k"

    def test_hook_aliases_work_correctly(self):
        """Test that hook aliases work correctly in compatibility mode."""
        # Load GPT-2 in TransformerBridge
        bridge = TransformerBridge.boot_transformers("gpt2", device="cpu")

        # Turn on compatibility mode
        bridge.enable_compatibility_mode(disable_warnings=True)

        # Create test input
        test_input = torch.tensor([[1, 2, 3, 4, 5]])  # Simple test sequence

        # Get the QKV bridge from the first attention layer
        qkv_bridge = bridge.blocks[0].attn.qkv

        # Run a forward pass to populate the hooks
        with torch.no_grad():
            _ = bridge(test_input)

        # Test that hook aliases work correctly
        # These should all reference the same hook points
        assert qkv_bridge.q_hook_out is bridge.blocks[0].attn.hook_q, "Q hook alias should work"
        assert qkv_bridge.k_hook_out is bridge.blocks[0].attn.hook_k, "K hook alias should work"
        assert qkv_bridge.v_hook_out is bridge.blocks[0].attn.hook_v, "V hook alias should work"

        # Test that the hook points are accessible through the attention bridge properties
        assert qkv_bridge.q_hook_out is bridge.blocks[0].attn.q.hook_out, "Q property should work"
        assert qkv_bridge.k_hook_out is bridge.blocks[0].attn.k.hook_out, "K property should work"
        assert qkv_bridge.v_hook_out is bridge.blocks[0].attn.v.hook_out, "V property should work"

    def test_head_ablation_hook_works_correctly(self):
        """Test that head ablation hook works correctly with TransformerBridge."""
        # Load GPT-2 in TransformerBridge
        bridge = TransformerBridge.boot_transformers("gpt2", device="cpu")

        # Turn on compatibility mode
        bridge.enable_compatibility_mode(disable_warnings=True)

        # Create test tokens (same as in the demo)
        gpt2_tokens = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])

        layer_to_ablate = 0
        head_index_to_ablate = 8

        # Test both hook names
        hook_names_to_test = [
            "blocks.0.attn.hook_v",  # Compatibility mode alias
            "blocks.0.attn.v.hook_out",  # Direct property access
        ]

        for hook_name in hook_names_to_test:
            print(f"\nTesting hook name: {hook_name}")

            # Track if the hook was called
            hook_called = False
            mutation_applied = False

            # We define a head ablation hook
            def head_ablation_hook(value, hook):
                nonlocal hook_called, mutation_applied
                hook_called = True
                print(f"Shape of the value tensor: {value.shape}")

                # Apply the ablation (out-of-place to avoid view modification error)
                result = value.clone()
                result[:, :, head_index_to_ablate, :] = 0.0

                # Check if the mutation was applied (the result should be zero for the ablated head)
                if torch.all(result[:, :, head_index_to_ablate, :] == 0.0):
                    mutation_applied = True

                return result

            # Get original loss
            original_loss = bridge(gpt2_tokens, return_type="loss")

            # Run with head ablation hook
            ablated_loss = bridge.run_with_hooks(
                gpt2_tokens, return_type="loss", fwd_hooks=[(hook_name, head_ablation_hook)]
            )

            print(f"Original Loss: {original_loss.item():.3f}")
            print(f"Ablated Loss: {ablated_loss.item():.3f}")

            # Assert that the hook was called
            assert hook_called, f"Head ablation hook should have been called for {hook_name}"

            # Assert that the mutation was applied
            assert (
                mutation_applied
            ), f"Mutation should have been applied to the value tensor for {hook_name}"

            # Assert that ablated loss is higher than original loss (ablation should hurt performance)
            assert (
                ablated_loss.item() > original_loss.item()
            ), f"Ablated loss should be higher than original loss for {hook_name}"

            print(f"✅ Hook {hook_name} works correctly!")
