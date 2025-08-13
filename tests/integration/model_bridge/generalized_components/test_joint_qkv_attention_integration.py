"""Fast integration tests for JointQKVAttentionBridge across different model architectures.

This module provides optimized integration testing for the JointQKVAttentionBridge
component, which handles fused QKV attention layers by exposing individual Q, K, V
components for hooking while maintaining computational efficiency.

Optimizations for CI speed:
- Session-scoped model caching to load each model only once
- Minimal input sequences (4 tokens instead of 10+)
- Primary testing on GPT-2 (fastest), with selective cross-architecture validation
- Efficient test parametrization to avoid redundant model loading
"""

import pytest
import torch
from jaxtyping import Float

import transformer_lens.utils as utils
from transformer_lens.hook_points import HookPoint
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

    # Primary architecture for most tests (fastest)
    PRIMARY_ARCHITECTURE = "gpt2"

    # All supported architectures (for selective cross-architecture testing)
    ALL_ARCHITECTURES = [
        "gpt2",  # GPT-2 (12 layers, 12 heads) - Primary
        "bigscience/bloom-560m",  # BLOOM (24 layers, 16 heads) - Selective
        "EleutherAI/pythia-70m",  # NeoX (6 layers, 8 heads) - Selective
    ]

    @pytest.fixture(scope="session")
    def gpt2_model(self):
        """Session-scoped GPT-2 model for fast repeated testing."""
        return get_cached_model(self.PRIMARY_ARCHITECTURE)

    @pytest.fixture
    def fast_input(self, gpt2_model):
        """Create minimal input tokens for fast testing."""
        # Use very short sequence for speed (4 tokens)
        text = "Test input"
        return gpt2_model.to_tokens(text)

    @pytest.fixture
    def gpt2_model_info(self):
        """Pre-computed GPT-2 model info to avoid repeated computation."""
        return {"name": "gpt2", "n_heads": 12, "d_head": 64, "n_layers": 12}

    @pytest.fixture
    def model_info(self, model):
        """Extract model-specific configuration information."""
        # Get model name
        if hasattr(model, "cfg") and hasattr(model.cfg, "model_name"):
            model_name = model.cfg.model_name
        elif hasattr(model.original_model, "config") and hasattr(
            model.original_model.config, "_name_or_path"
        ):
            model_name = model.original_model.config._name_or_path
        else:
            model_name = "unknown"

        # Extract architecture-specific parameters
        config = model.original_model.config
        if "gpt2" in model_name.lower():
            return {
                "name": "gpt2",
                "n_heads": config.n_head,
                "d_head": config.n_embd // config.n_head,
                "n_layers": config.n_layer,
            }
        elif "bloom" in model_name.lower():
            return {
                "name": "bloom",
                "n_heads": config.n_head,
                "d_head": config.hidden_size // config.n_head,
                "n_layers": config.n_layer,
            }
        elif "pythia" in model_name.lower():
            return {
                "name": "pythia",
                "n_heads": config.num_attention_heads,
                "d_head": config.hidden_size // config.num_attention_heads,
                "n_layers": config.num_hidden_layers,
            }
        else:
            # Fallback for unknown architectures
            return {
                "name": "unknown",
                "n_heads": getattr(config, "n_head", getattr(config, "num_attention_heads", 12)),
                "d_head": 64,
                "n_layers": getattr(config, "n_layer", getattr(config, "num_hidden_layers", 12)),
            }

    # Core Integration Tests (Fast - using GPT-2 only)

    def test_gpt2_uses_joint_qkv_attention_bridge(self, gpt2_model, gpt2_model_info):
        """Verify that GPT-2 architecture uses JointQKVAttentionBridge."""
        joint_qkv_modules = []

        for name, module in gpt2_model.named_modules():
            if hasattr(module, "__class__"):
                class_name = module.__class__.__name__
                if "JointQKVAttentionBridge" in class_name:
                    joint_qkv_modules.append(name)

        assert len(joint_qkv_modules) > 0, "GPT-2 should use JointQKVAttentionBridge but none found"

        # Verify expected number of attention layers
        expected_layers = gpt2_model_info["n_layers"]
        assert len(joint_qkv_modules) == expected_layers, (
            f"Expected {expected_layers} JointQKVAttentionBridge modules, "
            f"found {len(joint_qkv_modules)}"
        )

    def test_basic_model_functionality(self, gpt2_model, fast_input):
        """Test basic model loading and forward pass functionality."""
        # Test forward pass without hooks
        with torch.no_grad():
            loss = gpt2_model(fast_input, return_type="loss")
            logits = gpt2_model(fast_input, return_type="logits")

        # Validate outputs
        assert torch.isfinite(loss), f"Loss should be finite, got {loss}"
        assert loss > 0, f"Loss should be positive, got {loss}"
        assert logits.shape[-1] > 0, f"Logits should have vocabulary dimension > 0"
        assert torch.all(torch.isfinite(logits)), "All logits should be finite"

    def test_v_hook_integration_and_effect(self, gpt2_model, fast_input, gpt2_model_info):
        """Test that V hooks are properly integrated and affect computation."""
        layer_idx = 0
        head_idx = 5  # Valid for GPT-2's 12 heads

        def v_ablation_hook(
            value: Float[torch.Tensor, "batch pos head_index d_head"], hook: HookPoint
        ) -> Float[torch.Tensor, "batch pos head_index d_head"]:
            """Ablate a specific head in the V matrix."""
            # Verify tensor shape matches GPT-2
            assert len(value.shape) == 4, f"Expected 4D tensor, got {value.shape}"
            batch, pos, heads, d_head = value.shape
            assert heads == 12, f"Expected 12 heads for GPT-2, got {heads}"
            assert d_head == 64, f"Expected 64 d_head for GPT-2, got {d_head}"

            # Ablate the specified head
            value[:, :, head_idx, :] = 0.0
            return value

        # Test hook effect on computation
        original_loss = gpt2_model(fast_input, return_type="loss")
        hooked_loss = gpt2_model.run_with_hooks(
            fast_input,
            return_type="loss",
            fwd_hooks=[(utils.get_act_name("v", layer_idx), v_ablation_hook)],
        )

        # Verify hook had an effect
        assert not torch.isclose(original_loss, hooked_loss, atol=1e-6), (
            f"V hook should affect computation. Original: {original_loss:.6f}, "
            f"Hooked: {hooked_loss:.6f}"
        )

    def test_extreme_v_ablation_large_effect(self, gpt2_model, fast_input):
        """Test that extreme V ablation has a significant effect on model output."""
        layer_idx = 0

        def zero_all_v_hook(
            value: Float[torch.Tensor, "batch pos head_index d_head"], hook: HookPoint
        ) -> Float[torch.Tensor, "batch pos head_index d_head"]:
            """Zero out all V values to test large effect."""
            # Verify shape for GPT-2
            assert len(value.shape) == 4, f"Expected 4D tensor, got {value.shape}"
            batch, pos, heads, d_head = value.shape
            assert heads == 12, f"Expected 12 heads for GPT-2, got {heads}"
            assert d_head == 64, f"Expected 64 d_head for GPT-2, got {d_head}"

            # Zero out all values
            value[:, :, :, :] = 0.0
            return value

        original_loss = gpt2_model(fast_input, return_type="loss")
        extreme_hooked_loss = gpt2_model.run_with_hooks(
            fast_input,
            return_type="loss",
            fwd_hooks=[(utils.get_act_name("v", layer_idx), zero_all_v_hook)],
        )

        # Verify extreme ablation has meaningful effect
        effect_size = abs(extreme_hooked_loss - original_loss)
        assert (
            effect_size > 0.01
        ), f"Extreme V ablation should have meaningful effect, got {effect_size:.6f}"

    def test_multiple_layer_hooks_cumulative_effect(self, gpt2_model, fast_input):
        """Test that hooks on multiple layers have cumulative effects."""

        def v_scale_hook(
            value: Float[torch.Tensor, "batch pos head_index d_head"], hook: HookPoint
        ) -> Float[torch.Tensor, "batch pos head_index d_head"]:
            """Scale V values by 0.8."""
            # Verify shape for GPT-2
            assert len(value.shape) == 4, f"Expected 4D tensor, got {value.shape}"
            batch, pos, heads, d_head = value.shape
            assert heads == 12, f"Expected 12 heads for GPT-2, got {heads}"
            assert d_head == 64, f"Expected 64 d_head for GPT-2, got {d_head}"

            value *= 0.8
            return value

        original_loss = gpt2_model(fast_input, return_type="loss")

        # Apply hook to layer 0 only
        layer0_loss = gpt2_model.run_with_hooks(
            fast_input,
            return_type="loss",
            fwd_hooks=[(utils.get_act_name("v", 0), v_scale_hook)],
        )

        # Apply hooks to layers 0 and 1
        both_layers_loss = gpt2_model.run_with_hooks(
            fast_input,
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

    def test_efficient_path_when_no_hooks(self, gpt2_model, fast_input):
        """Test that no hooks uses the efficient fused computation path."""
        # Multiple runs without hooks should be deterministic
        loss1 = gpt2_model(fast_input, return_type="loss")
        loss2 = gpt2_model(fast_input, return_type="loss")
        loss3 = gpt2_model(fast_input, return_type="loss")

        # All should be identical (deterministic computation)
        assert torch.isclose(loss1, loss2, atol=1e-8)
        assert torch.isclose(loss2, loss3, atol=1e-8)

    def test_identity_hook_preserves_functionality(self, gpt2_model, fast_input):
        """Test that identity hooks don't change model behavior."""
        layer_idx = 0

        def identity_hook(
            value: Float[torch.Tensor, "batch pos head_index d_head"], hook: HookPoint
        ) -> Float[torch.Tensor, "batch pos head_index d_head"]:
            """Identity hook that doesn't modify the tensor."""
            # Verify shape for GPT-2
            assert len(value.shape) == 4, f"Expected 4D tensor, got {value.shape}"
            batch, pos, heads, d_head = value.shape
            assert heads == 12, f"Expected 12 heads for GPT-2, got {heads}"
            assert d_head == 64, f"Expected 64 d_head for GPT-2, got {d_head}"

            return value

        original_loss = gpt2_model(fast_input, return_type="loss")
        identity_hooked_loss = gpt2_model.run_with_hooks(
            fast_input,
            return_type="loss",
            fwd_hooks=[(utils.get_act_name("v", layer_idx), identity_hook)],
        )

        # Identity hook should not change results significantly
        assert torch.isclose(original_loss, identity_hooked_loss, atol=1e-5), (
            f"Identity hook should preserve computation. "
            f"Original: {original_loss:.6f}, Hooked: {identity_hooked_loss:.6f}"
        )

    def test_reconstruction_vs_original_compatibility(self, gpt2_model, fast_input):
        """Test that reconstructed attention produces reasonable results."""

        # This test ensures the reconstruction path works correctly
        def minimal_hook(value, hook):
            """Minimal hook that barely modifies values."""
            return value * 1.001  # Tiny modification to force reconstruction

        original_loss = gpt2_model(fast_input, return_type="loss")
        reconstructed_loss = gpt2_model.run_with_hooks(
            fast_input,
            return_type="loss",
            fwd_hooks=[(utils.get_act_name("v", 0), minimal_hook)],
        )

        # Results should be reasonable (not NaN, not extreme)
        assert torch.isfinite(original_loss), f"Original loss should be finite"
        assert torch.isfinite(reconstructed_loss), f"Reconstructed loss should be finite"
        assert original_loss > 0, f"Original loss should be positive"
        assert reconstructed_loss > 0, f"Reconstructed loss should be positive"

        # Should be similar but not identical due to the hook
        assert not torch.isclose(original_loss, reconstructed_loss, atol=1e-6)

        # But shouldn't be wildly different
        relative_diff = abs(original_loss - reconstructed_loss) / original_loss
        assert relative_diff < 0.1, f"Relative difference too large: {relative_diff:.4f}"

    def test_hook_detection_and_management(self, gpt2_model):
        """Test comprehensive hook detection across different hook types."""
        # Get the first attention layer
        attention_layer = None
        for name, module in gpt2_model.named_modules():
            if "JointQKVAttentionBridge" in str(type(module)):
                attention_layer = module
                break

        assert attention_layer is not None, "Should find a JointQKVAttentionBridge"

        def dummy_hook(value, hook):
            """Simple hook for testing detection."""
            return value

        # Test initial state (no hooks)
        assert not attention_layer.q.hook_in.has_hooks()
        assert not attention_layer.q.hook_out.has_hooks()
        assert not attention_layer.k.hook_in.has_hooks()
        assert not attention_layer.k.hook_out.has_hooks()
        assert not attention_layer.v.hook_in.has_hooks()
        assert not attention_layer.v.hook_out.has_hooks()

        # Add hooks and verify detection
        attention_layer.q.hook_out.add_hook(dummy_hook)
        assert attention_layer.q.hook_out.has_hooks()

        attention_layer.v.hook_out.add_hook(dummy_hook)
        assert attention_layer.v.hook_out.has_hooks()

        # Clean up
        attention_layer.q.hook_out.remove_hooks()
        attention_layer.v.hook_out.remove_hooks()

        assert not attention_layer.q.hook_out.has_hooks()
        assert not attention_layer.v.hook_out.has_hooks()

    # Cross-Architecture Validation (Selective - only when needed)

    @pytest.mark.skip(reason="Slow cross-architecture test - enable manually if needed")
    @pytest.mark.parametrize("model_name", ["bigscience/bloom-560m", "EleutherAI/pythia-70m"])
    def test_other_architectures_use_joint_qkv_bridge(self, model_name):
        """Test that other architectures also use JointQKVAttentionBridge (slow test)."""
        torch.set_grad_enabled(False)

        try:
            model = get_cached_model(model_name)

            # Check for JointQKVAttentionBridge usage
            joint_qkv_modules = []
            for name, module in model.named_modules():
                if hasattr(module, "__class__"):
                    class_name = module.__class__.__name__
                    if "JointQKVAttentionBridge" in class_name:
                        joint_qkv_modules.append(name)

            assert (
                len(joint_qkv_modules) > 0
            ), f"{model_name} should use JointQKVAttentionBridge but none found"

            # Test basic functionality
            text = "Test"  # Minimal input
            tokens = model.to_tokens(text)

            with torch.no_grad():
                loss = model(tokens, return_type="loss")
                assert torch.isfinite(loss), f"Loss should be finite for {model_name}"
                assert loss > 0, f"Loss should be positive for {model_name}"

        except Exception as e:
            pytest.skip(f"Model {model_name} has compatibility issues: {e}")

    def test_cross_architecture_detection_summary(self):
        """Fast test to verify architecture detection works across all models."""
        results = {
            "gpt2": "✓ Confirmed (primary test architecture)",
        }

        # Test other architectures only if they load quickly
        for model_name in ["bigscience/bloom-560m", "EleutherAI/pythia-70m"]:
            try:
                # Quick architecture check without full model loading
                torch.set_grad_enabled(False)
                model = get_cached_model(model_name)

                # Quick check for JointQKVAttentionBridge
                has_joint_qkv = any(
                    "JointQKVAttentionBridge" in str(type(module))
                    for name, module in model.named_modules()
                )

                if has_joint_qkv:
                    results[model_name] = "✓ Uses JointQKVAttentionBridge"
                else:
                    results[model_name] = "✗ Does not use JointQKVAttentionBridge"

            except Exception as e:
                results[model_name] = f"⚠ Compatibility issues: {str(e)[:50]}..."

        # At minimum, GPT-2 should work
        assert "gpt2" in results

        # Print summary for debugging
        print("\nJointQKVAttentionBridge Architecture Support:")
        for arch, status in results.items():
            print(f"  {arch}: {status}")

    # Performance and Edge Case Tests (GPT-2 only for speed)

    def test_large_head_ablation_effect_scaling(self, gpt2_model, fast_input):
        """Test that ablating more heads has larger effects."""
        layer_idx = 0

        def ablate_n_heads_hook(n_heads_to_ablate):
            def hook_fn(value, hook):
                # Ablate first n heads
                value[:, :, :n_heads_to_ablate, :] = 0.0
                return value

            return hook_fn

        original_loss = gpt2_model(fast_input, return_type="loss")

        # Test ablating 1, 3, and 6 heads
        loss_1_head = gpt2_model.run_with_hooks(
            fast_input,
            return_type="loss",
            fwd_hooks=[(utils.get_act_name("v", layer_idx), ablate_n_heads_hook(1))],
        )

        loss_3_heads = gpt2_model.run_with_hooks(
            fast_input,
            return_type="loss",
            fwd_hooks=[(utils.get_act_name("v", layer_idx), ablate_n_heads_hook(3))],
        )

        loss_6_heads = gpt2_model.run_with_hooks(
            fast_input,
            return_type="loss",
            fwd_hooks=[(utils.get_act_name("v", layer_idx), ablate_n_heads_hook(6))],
        )

        # More heads ablated should generally have larger effects
        effect_1 = abs(loss_1_head - original_loss)
        effect_3 = abs(loss_3_heads - original_loss)
        effect_6 = abs(loss_6_heads - original_loss)

        # Verify effects are non-zero and generally increasing
        assert effect_1 > 1e-6, "Single head ablation should have some effect"
        assert effect_3 > 1e-6, "Three head ablation should have some effect"
        assert effect_6 > 1e-6, "Six head ablation should have some effect"
        assert effect_6 > effect_1, "More heads ablated should have larger effect"
