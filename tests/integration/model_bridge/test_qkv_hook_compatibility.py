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
