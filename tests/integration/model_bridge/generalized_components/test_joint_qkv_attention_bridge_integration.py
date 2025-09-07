"""Lightweight integration tests for JointQKVAttentionBridge.

Tests the core functionality without loading large models to keep CI fast.
"""

import pytest
import torch

import transformer_lens.utils as utils


class TestJointQKVAttentionBridgeIntegration:
    """Minimal integration tests for JointQKVAttentionBridge."""

    def test_hook_alias_resolution(self):
        """Test that hook aliases are properly resolved."""
        # Test the hook alias resolution that caused the original issue
        hook_name = utils.get_act_name("v", 0)
        assert (
            hook_name == "blocks.0.attn.hook_v"
        ), f"Expected 'blocks.0.attn.hook_v', got '{hook_name}'"

        # Test other hook names
        assert utils.get_act_name("q", 1) == "blocks.1.attn.hook_q"
        assert utils.get_act_name("k", 2) == "blocks.2.attn.hook_k"

    def test_hook_point_has_hooks_method(self):
        """Test that HookPoint.has_hooks method works correctly."""
        from transformer_lens.hook_points import HookPoint

        hook_point = HookPoint()

        # Test initial state
        assert not hook_point.has_hooks()
        assert not hook_point.has_hooks(dir="fwd")
        assert not hook_point.has_hooks(dir="bwd")

        # Add a hook and test detection
        def dummy_hook(x, hook):
            return x

        hook_point.add_hook(dummy_hook)
        assert hook_point.has_hooks()
        assert hook_point.has_hooks(dir="fwd")
        assert not hook_point.has_hooks(dir="bwd")

        # Clean up
        hook_point.remove_hooks()
        assert not hook_point.has_hooks()

    def test_architecture_imports(self):
        """Test that architecture files can be imported and reference JointQKVAttentionBridge."""
        # Test that we can import the architecture files without errors
        # Test that JointQKVAttentionBridge is referenced in the source files
        import inspect

        from transformer_lens.model_bridge.supported_architectures import (
            bloom,
            gpt2,
            neox,
        )

        gpt2_source = inspect.getsource(gpt2)
        assert (
            "JointQKVAttentionBridge" in gpt2_source
        ), "GPT-2 architecture should reference JointQKVAttentionBridge"

        bloom_source = inspect.getsource(bloom)
        assert (
            "JointQKVAttentionBridge" in bloom_source
        ), "BLOOM architecture should reference JointQKVAttentionBridge"

        neox_source = inspect.getsource(neox)
        assert (
            "JointQKVAttentionBridge" in neox_source
        ), "NeoX architecture should reference JointQKVAttentionBridge"

    @pytest.mark.skip(reason="Requires model loading - too slow for CI")
    def test_distilgpt2_integration(self):
        """Full integration test with DistilGPT-2 (skipped in CI)."""
        # This test would load DistilGPT-2 and test full functionality
        # but is skipped by default to keep CI fast
        from transformer_lens.model_bridge import TransformerBridge

        torch.set_grad_enabled(False)
        model = TransformerBridge.boot_transformers("distilgpt2", device="cpu")

        # Verify JointQKVAttentionBridge usage
        joint_qkv_attention_bridge_modules = [
            name
            for name, module in model.named_modules()
            if "JointQKVAttentionBridge" in getattr(module, "__class__", {}).get("__name__", "")
        ]
        assert (
            len(joint_qkv_attention_bridge_modules) == 6
        ), f"Expected 6 JointQKVAttentionBridge modules, got {len(joint_qkv_attention_bridge_modules)}"

        # Test basic functionality
        tokens = model.to_tokens("Test")
        with torch.no_grad():
            loss = model(tokens, return_type="loss")
            assert torch.isfinite(loss) and loss > 0

        # Test hook integration
        def v_ablation_hook(value, hook):
            value[:, :, 0, :] = 0.0  # Ablate first head
            return value

        original_loss = model(tokens, return_type="loss")
        hooked_loss = model.run_with_hooks(
            tokens, return_type="loss", fwd_hooks=[(utils.get_act_name("v", 0), v_ablation_hook)]
        )
        assert not torch.isclose(original_loss, hooked_loss, atol=1e-6)
