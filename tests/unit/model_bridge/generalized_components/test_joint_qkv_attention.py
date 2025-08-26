"""Unit tests for joint QKV attention bridge."""


from transformer_lens.hook_points import HookPoint
from transformer_lens.model_bridge.generalized_components.joint_qkv_attention import (
    JointQKVAttentionBridge,
)
from transformer_lens.model_bridge.generalized_components.qkv_bridge import QKVBridge
from transformer_lens.model_bridge.hook_point_wrapper import HookPointWrapper


class TestJointQKVAttention:
    """Test that QKV bridge hooks are properly shared with joint QKV attention bridge."""

    def test_qkv_hook_identity(self):
        """Test that q.hook_in in attention bridge is the same object as q_hook_in in QKV bridge."""

        # Create a simple config for testing
        class TestConfig:
            n_heads = 12
            d_model = 768

        # Initialize QKV bridge
        qkv_bridge = QKVBridge(
            name="qkv",
            config=TestConfig(),
        )

        # Initialize joint QKV attention bridge with QKV bridge as submodule
        joint_qkv_attention = JointQKVAttentionBridge(
            name="attn",
            config=TestConfig(),
            submodules={"qkv": qkv_bridge},
        )

        # Assert that the q.hook_in property in the attention bridge
        # is the same object as q_hook_in in the QKV bridge
        assert (
            joint_qkv_attention.q.hook_in is qkv_bridge.q_hook_in
        ), "q.hook_in in attention bridge should be the same object as q_hook_in in QKV bridge"

        # Also test that k and v hooks are properly shared
        assert (
            joint_qkv_attention.k.hook_in is qkv_bridge.k_hook_in
        ), "k.hook_in in attention bridge should be the same object as k_hook_in in QKV bridge"
        assert (
            joint_qkv_attention.v.hook_in is qkv_bridge.v_hook_in
        ), "v.hook_in in attention bridge should be the same object as v_hook_in in QKV bridge"

        # Test hook_out properties as well
        assert (
            joint_qkv_attention.q.hook_out is qkv_bridge.q_hook_out
        ), "q.hook_out in attention bridge should be the same object as q_hook_out in QKV bridge"
        assert (
            joint_qkv_attention.k.hook_out is qkv_bridge.k_hook_out
        ), "k.hook_out in attention bridge should be the same object as k_hook_out in QKV bridge"
        assert (
            joint_qkv_attention.v.hook_out is qkv_bridge.v_hook_out
        ), "v.hook_out in attention bridge should be the same object as v_hook_out in QKV bridge"

    def test_hook_point_wrapper_properties(self):
        """Test that HookPointWrapper properly exposes hook_in and hook_out."""

        # Create test hook points
        hook_in = HookPoint()
        hook_out = HookPoint()

        # Create wrapper
        wrapper = HookPointWrapper(hook_in=hook_in, hook_out=hook_out)

        # Test that properties are accessible
        assert wrapper.hook_in is hook_in
        assert wrapper.hook_out is hook_out

        # Test that they are the correct types
        assert isinstance(wrapper.hook_in, HookPoint)
        assert isinstance(wrapper.hook_out, HookPoint)
