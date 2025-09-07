"""Integration tests for Joint QKV Attention bridge hook compatibility in TransformerBridge."""

import torch

from transformer_lens.model_bridge import TransformerBridge


class TestJointQKVAttentionHookCompatibility:
    """Test that Joint QKV Attention bridge hooks are compatible with overall model hook access."""

    def test_v_hook_out_equals_blocks_attn_hook_v(self):
        """Test that v_hook_out in Joint QKV Attention bridge equals blocks.0.attn.hook_v on the overall model."""
        # Load GPT-2 in TransformerBridge
        bridge = TransformerBridge.boot_transformers("gpt2", device="cpu")

        # Turn on compatibility mode
        bridge.enable_compatibility_mode(disable_warnings=True)

        # Create test input
        test_input = torch.tensor([[1, 2, 3, 4, 5]])  # Simple test sequence

        # Get the QKV bridge from the first attention layer
        joint_qkv_attention_bridge = bridge.blocks[0].attn

        # Verify that joint_qkv_attention_bridge is indeed a JointQKVAttentionBridge
        from transformer_lens.model_bridge.generalized_components.joint_qkv_attention import (
            JointQKVAttentionBridge,
        )

        assert isinstance(
            joint_qkv_attention_bridge, JointQKVAttentionBridge
        ), "First attention layer should be a JointQKVAttentionBridge"

        # Run a forward pass to populate the hooks
        with torch.no_grad():
            _ = bridge(test_input)

        # Assert that v.hook_out in the Joint QKV Attention bridge is the same object as
        # blocks.0.attn.hook_v on the overall model
        assert (
            joint_qkv_attention_bridge.v.hook_out is bridge.blocks[0].attn.hook_v
        ), "v.hook_out in Joint QKV Attention bridge should be the same object as blocks.0.attn.hook_v"

        # Also test that the hook points have the same properties
        assert (
            joint_qkv_attention_bridge.v.hook_out.has_hooks()
            == bridge.blocks[0].attn.hook_v.has_hooks()
        ), "Hook points should have the same hook status"

    def test_q_hook_out_equals_blocks_attn_hook_q(self):
        """Test that q.hook_out in Joint QKV Attention bridge equals blocks.0.attn.hook_q on the overall model."""
        # Load GPT-2 in TransformerBridge
        bridge = TransformerBridge.boot_transformers("gpt2", device="cpu")

        # Turn on compatibility mode
        bridge.enable_compatibility_mode(disable_warnings=True)

        # Create test input
        test_input = torch.tensor([[1, 2, 3, 4, 5]])  # Simple test sequence

        # Get the QKV bridge from the first attention layer
        joint_qkv_attention_bridge = bridge.blocks[0].attn

        # Run a forward pass to populate the hooks
        with torch.no_grad():
            _ = bridge(test_input)

        # Assert that q.hook_out in the Joint QKV Attention bridge is the same object as
        # blocks.0.attn.hook_q on the overall model
        assert (
            joint_qkv_attention_bridge.q.hook_out is bridge.blocks[0].attn.hook_q
        ), "q.hook_out in Joint QKV Attention bridge should be the same object as blocks.0.attn.hook_q"

    def test_k_hook_out_equals_blocks_attn_hook_k(self):
        """Test that k.hook_out in Joint QKV Attention bridge equals blocks.0.attn.hook_k on the overall model."""
        # Load GPT-2 in TransformerBridge
        bridge = TransformerBridge.boot_transformers("gpt2", device="cpu")

        # Turn on compatibility mode
        bridge.enable_compatibility_mode(disable_warnings=True)

        # Create test input
        test_input = torch.tensor([[1, 2, 3, 4, 5]])  # Simple test sequence

        # Get the QKV bridge from the first attention layer
        joint_qkv_attention_bridge = bridge.blocks[0].attn

        # Run a forward pass to populate the hooks
        with torch.no_grad():
            _ = bridge(test_input)

        # Assert that k.hook_out in the Joint QKV Attention bridge is the same object as
        # blocks.0.attn.hook_k on the overall model
        assert (
            joint_qkv_attention_bridge.k.hook_out is bridge.blocks[0].attn.hook_k
        ), "k.hook_out in Joint QKV Attention bridge should be the same object as blocks.0.attn.hook_k"

    def test_hook_aliases_work_correctly(self):
        """Test that hook aliases work correctly in compatibility mode."""
        # Load GPT-2 in TransformerBridge
        bridge = TransformerBridge.boot_transformers("gpt2", device="cpu")

        # Turn on compatibility mode
        bridge.enable_compatibility_mode(disable_warnings=True)

        # Create test input
        test_input = torch.tensor([[1, 2, 3, 4, 5]])  # Simple test sequence

        # Get the QKV bridge from the first attention layer
        joint_qkv_attention_bridge = bridge.blocks[0].attn

        # Run a forward pass to populate the hooks
        with torch.no_grad():
            _ = bridge(test_input)

        # Test that hook aliases work correctly
        # These should all reference the same hook points
        assert (
            joint_qkv_attention_bridge.q.hook_out is bridge.blocks[0].attn.hook_q
        ), "Q hook alias should work"
        assert (
            joint_qkv_attention_bridge.k.hook_out is bridge.blocks[0].attn.hook_k
        ), "K hook alias should work"
        assert (
            joint_qkv_attention_bridge.v.hook_out is bridge.blocks[0].attn.hook_v
        ), "V hook alias should work"
