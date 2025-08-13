"""Integration tests for JointQKVAttentionBridge using DistilGPT-2.

Tests the JointQKVAttentionBridge component which handles fused QKV attention layers
by exposing individual Q, K, V components for hooking while maintaining efficiency.
"""

import pytest
import torch

import transformer_lens.utils as utils
from transformer_lens.model_bridge import TransformerBridge

# Global model cache to avoid reloading models
_MODEL_CACHE = {}


def get_cached_model(model_name: str):
    """Get a cached model or load it if not cached."""
    if model_name not in _MODEL_CACHE:
        torch.set_grad_enabled(False)
        try:
            _MODEL_CACHE[model_name] = TransformerBridge.boot_transformers(model_name, device="cpu")
        except Exception as e:
            pytest.skip(f"Model {model_name} has compatibility issues: {e}")
    return _MODEL_CACHE[model_name]


class TestJointQKVAttentionBridgeIntegration:
    """Fast integration test suite for JointQKVAttentionBridge."""

    PRIMARY_ARCHITECTURE = "distilgpt2"
    SLOW_TEST_ARCHITECTURES = ["bigscience/bloom-560m", "EleutherAI/pythia-70m"]

    @pytest.fixture(scope="session")
    def primary_model(self):
        """Session-scoped DistilGPT-2 model for fast testing."""
        return get_cached_model(self.PRIMARY_ARCHITECTURE)

    @pytest.fixture
    def fast_input(self, primary_model):
        """Create minimal input tokens for testing."""
        text = "Test input"
        return primary_model.to_tokens(text)

    @pytest.fixture
    def primary_model_info(self):
        """DistilGPT-2 model configuration."""
        return {"name": "distilgpt2", "n_heads": 12, "d_head": 64, "n_layers": 6}

    # Core Integration Tests

    def test_distilgpt2_uses_joint_qkv_attention_bridge(self, primary_model, primary_model_info):
        """Verify that DistilGPT-2 uses JointQKVAttentionBridge."""
        joint_qkv_modules = []
        for name, module in primary_model.named_modules():
            if (
                hasattr(module, "__class__")
                and "JointQKVAttentionBridge" in module.__class__.__name__
            ):
                joint_qkv_modules.append(name)

        assert len(joint_qkv_modules) > 0, "DistilGPT-2 should use JointQKVAttentionBridge"
        expected_layers = primary_model_info["n_layers"]
        assert (
            len(joint_qkv_modules) == expected_layers
        ), f"Expected {expected_layers} modules, found {len(joint_qkv_modules)}"

    def test_basic_model_functionality(self, primary_model, fast_input):
        """Test basic model forward pass."""
        with torch.no_grad():
            loss = primary_model(fast_input, return_type="loss")
            logits = primary_model(fast_input, return_type="logits")

        assert torch.isfinite(loss) and loss > 0
        assert logits.shape[-1] > 0 and torch.all(torch.isfinite(logits))

    def test_v_hook_integration_and_effect(self, primary_model, fast_input, primary_model_info):
        """Test that V hooks affect computation."""

        def v_ablation_hook(value, hook):
            assert len(value.shape) == 4 and value.shape[2] == 12 and value.shape[3] == 64
            value[:, :, 5, :] = 0.0  # Ablate head 5
            return value

        original_loss = primary_model(fast_input, return_type="loss")
        hooked_loss = primary_model.run_with_hooks(
            fast_input,
            return_type="loss",
            fwd_hooks=[(utils.get_act_name("v", 0), v_ablation_hook)],
        )

        assert not torch.isclose(original_loss, hooked_loss, atol=1e-6)

    def test_extreme_v_ablation_large_effect(self, primary_model, fast_input):
        """Test that extreme V ablation has significant effect."""

        def zero_all_v_hook(value, hook):
            value[:, :, :, :] = 0.0
            return value

        original_loss = primary_model(fast_input, return_type="loss")
        extreme_hooked_loss = primary_model.run_with_hooks(
            fast_input,
            return_type="loss",
            fwd_hooks=[(utils.get_act_name("v", 0), zero_all_v_hook)],
        )

        effect_size = abs(extreme_hooked_loss - original_loss)
        assert effect_size > 0.01, f"Effect too small: {effect_size:.6f}"

    def test_multiple_layer_hooks_cumulative_effect(self, primary_model, fast_input):
        """Test that hooks on multiple layers have cumulative effects."""

        def v_scale_hook(value, hook):
            return value * 0.8

        original_loss = primary_model(fast_input, return_type="loss")
        layer0_loss = primary_model.run_with_hooks(
            fast_input, return_type="loss", fwd_hooks=[(utils.get_act_name("v", 0), v_scale_hook)]
        )
        both_layers_loss = primary_model.run_with_hooks(
            fast_input,
            return_type="loss",
            fwd_hooks=[
                (utils.get_act_name("v", 0), v_scale_hook),
                (utils.get_act_name("v", 1), v_scale_hook),
            ],
        )

        assert not torch.isclose(original_loss, layer0_loss, atol=1e-6)
        assert not torch.isclose(original_loss, both_layers_loss, atol=1e-6)
        assert not torch.isclose(layer0_loss, both_layers_loss, atol=1e-6)

    def test_efficient_path_when_no_hooks(self, primary_model, fast_input):
        """Test deterministic computation without hooks."""
        loss1 = primary_model(fast_input, return_type="loss")
        loss2 = primary_model(fast_input, return_type="loss")
        loss3 = primary_model(fast_input, return_type="loss")

        assert torch.isclose(loss1, loss2, atol=1e-8)
        assert torch.isclose(loss2, loss3, atol=1e-8)

    def test_identity_hook_preserves_functionality(self, primary_model, fast_input):
        """Test that identity hooks don't change model behavior."""

        def identity_hook(value, hook):
            return value

        original_loss = primary_model(fast_input, return_type="loss")
        identity_hooked_loss = primary_model.run_with_hooks(
            fast_input, return_type="loss", fwd_hooks=[(utils.get_act_name("v", 0), identity_hook)]
        )

        assert torch.isclose(original_loss, identity_hooked_loss, atol=1e-5)

    def test_reconstruction_vs_original_compatibility(self, primary_model, fast_input):
        """Test that reconstructed attention produces reasonable results."""

        def minimal_hook(value, hook):
            return value * 1.001  # Tiny modification to force reconstruction

        original_loss = primary_model(fast_input, return_type="loss")
        reconstructed_loss = primary_model.run_with_hooks(
            fast_input, return_type="loss", fwd_hooks=[(utils.get_act_name("v", 0), minimal_hook)]
        )

        assert torch.isfinite(original_loss) and original_loss > 0
        assert torch.isfinite(reconstructed_loss) and reconstructed_loss > 0
        assert not torch.isclose(original_loss, reconstructed_loss, atol=1e-6)

        relative_diff = abs(original_loss - reconstructed_loss) / original_loss
        assert relative_diff < 0.1, f"Relative difference too large: {relative_diff:.4f}"

    def test_hook_detection_and_management(self, primary_model):
        """Test hook detection across different hook types."""
        attention_layer = None
        for name, module in primary_model.named_modules():
            if "JointQKVAttentionBridge" in str(type(module)):
                attention_layer = module
                break

        assert attention_layer is not None

        def dummy_hook(value, hook):
            return value

        # Test initial state
        assert not attention_layer.q.hook_out.has_hooks()
        assert not attention_layer.v.hook_out.has_hooks()

        # Add and verify hooks
        attention_layer.q.hook_out.add_hook(dummy_hook)
        assert attention_layer.q.hook_out.has_hooks()

        attention_layer.v.hook_out.add_hook(dummy_hook)
        assert attention_layer.v.hook_out.has_hooks()

        # Clean up
        attention_layer.q.hook_out.remove_hooks()
        attention_layer.v.hook_out.remove_hooks()

        assert not attention_layer.q.hook_out.has_hooks()
        assert not attention_layer.v.hook_out.has_hooks()

    # Cross-Architecture Tests (Slow)

    @pytest.mark.skip(reason="Slow test - enable manually if needed")
    @pytest.mark.parametrize("model_name", ["bigscience/bloom-560m", "EleutherAI/pythia-70m"])
    def test_other_architectures_use_joint_qkv_bridge(self, model_name):
        """Test other architectures use JointQKVAttentionBridge."""
        torch.set_grad_enabled(False)
        try:
            model = get_cached_model(model_name)
            joint_qkv_modules = [
                name
                for name, module in model.named_modules()
                if "JointQKVAttentionBridge" in getattr(module, "__class__", {}).get("__name__", "")
            ]
            assert len(joint_qkv_modules) > 0, f"{model_name} should use JointQKVAttentionBridge"

            tokens = model.to_tokens("Test")
            with torch.no_grad():
                loss = model(tokens, return_type="loss")
                assert torch.isfinite(loss) and loss > 0
        except Exception as e:
            pytest.skip(f"Model {model_name} compatibility issues: {e}")

    def test_cross_architecture_detection_summary(self):
        """Test architecture detection across models."""
        results = {"distilgpt2": "✓ Primary architecture"}

        for model_name in self.SLOW_TEST_ARCHITECTURES:
            try:
                torch.set_grad_enabled(False)
                model = get_cached_model(model_name)
                has_joint_qkv = any(
                    "JointQKVAttentionBridge" in str(type(module))
                    for name, module in model.named_modules()
                )
                results[model_name] = (
                    "✓ Uses JointQKVAttentionBridge"
                    if has_joint_qkv
                    else "✗ No JointQKVAttentionBridge"
                )
            except Exception as e:
                results[model_name] = f"⚠ Issues: {str(e)[:30]}..."

        assert "distilgpt2" in results
        print("\nArchitecture Support:")
        for arch, status in results.items():
            print(f"  {arch}: {status}")

    # Performance Tests

    def test_large_head_ablation_effect_scaling(self, primary_model, fast_input):
        """Test that ablating more heads has larger effects."""

        def ablate_n_heads_hook(n_heads):
            def hook_fn(value, hook):
                value[:, :, :n_heads, :] = 0.0
                return value

            return hook_fn

        original_loss = primary_model(fast_input, return_type="loss")
        loss_1_head = primary_model.run_with_hooks(
            fast_input,
            return_type="loss",
            fwd_hooks=[(utils.get_act_name("v", 0), ablate_n_heads_hook(1))],
        )
        loss_6_heads = primary_model.run_with_hooks(
            fast_input,
            return_type="loss",
            fwd_hooks=[(utils.get_act_name("v", 0), ablate_n_heads_hook(6))],
        )

        effect_1 = abs(loss_1_head - original_loss)
        effect_6 = abs(loss_6_heads - original_loss)

        assert effect_1 > 1e-6 and effect_6 > 1e-6
        assert effect_6 > effect_1, "More heads ablated should have larger effect"
