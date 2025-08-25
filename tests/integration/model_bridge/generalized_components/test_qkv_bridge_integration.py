"""Lightweight integration tests for QKVBridge.

Tests the core functionality without loading large models to keep CI fast.
"""

import pytest
import torch

import transformer_lens.utils as utils
from transformer_lens.model_bridge.generalized_components.qkv_bridge import QKVBridge


class TestQKVBridgeIntegration:
    """Minimal integration tests for QKVBridge."""

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

    def test_joint_qkv_attention_bridge_properties(self):
        """Test that JointQKVAttentionBridge properties are properly resolved."""
        from transformer_lens.model_bridge.generalized_components.joint_qkv_attention import (
            JointQKVAttentionBridge,
        )

        class TestConfig:
            n_heads = 12

        qkv_bridge = QKVBridge(name="qkv", config=TestConfig())

        qkv_attention_bridge = JointQKVAttentionBridge(
            name="blocks.0.attn",
            config=TestConfig(),
            submodules={"qkv": qkv_bridge},
        )

        assert qkv_attention_bridge.q.hook_in == qkv_bridge.q_hook_in
        assert qkv_attention_bridge.q.hook_out == qkv_bridge.q_hook_out
        assert qkv_attention_bridge.k.hook_in == qkv_bridge.k_hook_in
        assert qkv_attention_bridge.k.hook_out == qkv_bridge.k_hook_out
        assert qkv_attention_bridge.v.hook_in == qkv_bridge.v_hook_in
        assert qkv_attention_bridge.v.hook_out == qkv_bridge.v_hook_out

    def test_component_class_exists(self):
        """Test that QKVBridge class can be imported."""

        # Verify the class exists and has expected methods
        assert hasattr(QKVBridge, "forward")
        assert hasattr(QKVBridge, "_create_qkv_conversion_rule")
        assert hasattr(QKVBridge, "_create_qkv_separation_rule")

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
        """Test that architecture files can be imported and reference QKVBridge."""
        # Test that we can import the architecture files without errors
        # Test that QKVBridge is referenced in the source files
        import inspect

        from transformer_lens.model_bridge.supported_architectures import (
            bloom,
            gpt2,
            neox,
        )

        gpt2_source = inspect.getsource(gpt2)
        assert "QKVBridge" in gpt2_source, "GPT-2 architecture should reference QKVBridge"

        bloom_source = inspect.getsource(bloom)
        assert "QKVBridge" in bloom_source, "BLOOM architecture should reference QKVBridge"

        neox_source = inspect.getsource(neox)
        assert "QKVBridge" in neox_source, "NeoX architecture should reference QKVBridge"

    @pytest.mark.skip(reason="Requires model loading - too slow for CI")
    def test_distilgpt2_integration(self):
        """Full integration test with DistilGPT-2 (skipped in CI)."""
        # This test would load DistilGPT-2 and test full functionality
        # but is skipped by default to keep CI fast
        from transformer_lens.model_bridge import TransformerBridge

        torch.set_grad_enabled(False)
        model = TransformerBridge.boot_transformers("distilgpt2", device="cpu")

        # Verify QKVBridge usage
        qkv_bridge_modules = [
            name
            for name, module in model.named_modules()
            if "QKVBridge" in getattr(module, "__class__", {}).get("__name__", "")
        ]
        assert (
            len(qkv_bridge_modules) == 6
        ), f"Expected 6 QKVBridge modules, got {len(qkv_bridge_modules)}"

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
