"""Legacy hook compatibility tests for TransformerBridge.

This module contains comprehensive tests that verify TransformerBridge provides all the hooks
that should be available from HookedTransformer for interpretability research, including
cache compatibility and hook availability tests.
"""

import pytest
import torch

from transformer_lens import HookedTransformer
from transformer_lens.model_bridge import TransformerBridge


class TestLegacyHookCompatibility:
    """Test suite to verify comprehensive hook compatibility for TransformerBridge."""

    @pytest.fixture
    def model_name(self):
        """Model name to use for testing."""
        return "gpt2"

    @pytest.fixture
    def prompt(self):
        """Test prompt for cache generation."""
        return "Hello World!"

    @pytest.fixture
    def transformer_bridge(self, model_name):
        """Create a TransformerBridge for testing."""
        model = TransformerBridge.boot_transformers(model_name, device="cpu")
        model.enable_compatibility_mode()
        return model

    @pytest.fixture
    def hooked_transformer(self, model_name):
        """Create a HookedTransformer for comparison testing."""
        return HookedTransformer.from_pretrained(model_name, device="cpu")

    @pytest.fixture
    def expected_hooks(self):
        """Get the unified list of hooks that should be available for TransformerBridge testing.

        This includes all hooks that should be present in activation caches and accessible
        on the model for interpretability research.
        """
        return [
            # Core embedding hooks
            "hook_embed",
            "hook_pos_embed",
            # Block 0 residual stream hooks
            "blocks.0.hook_resid_pre",
            "blocks.0.hook_resid_mid",
            "blocks.0.hook_resid_post",
            # Layer norm hooks
            "blocks.0.ln1.hook_scale",
            "blocks.0.ln1.hook_normalized",
            "blocks.0.ln2.hook_scale",
            "blocks.0.ln2.hook_normalized",
            # Attention hooks
            "blocks.0.attn.hook_q",
            "blocks.0.attn.hook_k",
            "blocks.0.attn.hook_v",
            "blocks.0.attn.hook_z",
            "blocks.0.attn.hook_attn_scores",
            "blocks.0.attn.hook_pattern",
            "blocks.0.attn.hook_result",
            # MLP hooks
            "blocks.0.mlp.hook_pre",
            "blocks.0.mlp.hook_post",
            # Output hooks
            "blocks.0.hook_attn_out",
            "blocks.0.hook_mlp_out",
            # Final layer norm hooks
            "ln_final.hook_scale",
            "ln_final.hook_normalized",
            # Hook aliases for commonly used patterns
            "blocks.0.hook_attn_in",
            "blocks.0.hook_mlp_in",
            "blocks.0.hook_q_input",
            "blocks.0.hook_k_input",
            "blocks.0.hook_v_input",
        ]

    def hook_exists_on_model(self, model, hook_path: str) -> bool:
        """Check if a hook path exists on the model by traversing attributes."""
        parts = hook_path.split(".")
        current = model

        try:
            for part in parts:
                if "[" in part and "]" in part:
                    # Handle array indexing like blocks[0]
                    attr_name = part.split("[")[0]
                    index = int(part.split("[")[1].split("]")[0])
                    current = getattr(current, attr_name)[index]
                else:
                    current = getattr(current, part)

            # Check if the final object is a HookPoint
            from transformer_lens.hook_points import HookPoint

            return isinstance(current, HookPoint)

        except (AttributeError, IndexError, TypeError):
            return False

    def test_cache_hook_names_present(self, transformer_bridge, prompt, expected_hooks):
        """Test that TransformerBridge cache contains all expected hook names."""
        _, cache = transformer_bridge.run_with_cache(prompt)

        # Get the actual cache keys
        actual_keys = list(cache.keys())

        print(f"\nExpected hooks: {len(expected_hooks)}")
        print(f"Actual hooks: {len(actual_keys)}")

        # Find missing and extra hooks
        expected_set = set(expected_hooks)
        actual_set = set(actual_keys)

        missing_hooks = expected_set - actual_set
        extra_hooks = actual_set - expected_set

        print(f"Missing hooks ({len(missing_hooks)}): {sorted(missing_hooks)}")
        print(
            f"Extra hooks ({len(extra_hooks)}): {sorted(list(extra_hooks)[:10])}{'...' if len(extra_hooks) > 10 else ''}"
        )

        # Check that all expected hooks are present (subset check)
        # It's okay to have extra hooks - that means more functionality is exposed
        assert len(missing_hooks) == 0, f"Missing expected hooks: {sorted(missing_hooks)}"

        # Verify we have at least the expected hooks
        assert all(
            hook in actual_set for hook in expected_set
        ), f"Some expected hooks are missing: {missing_hooks}"

    def test_cache_hook_equality_with_hooked_transformer(
        self, transformer_bridge, hooked_transformer, prompt, expected_hooks
    ):
        """Test that TransformerBridge cache values match HookedTransformer cache values."""
        _, bridge_cache = transformer_bridge.run_with_cache(prompt)
        _, hooked_transformer_cache = hooked_transformer.run_with_cache(prompt)

        for hook in expected_hooks:
            # Skip hooks that might not be present in both models
            if hook not in bridge_cache or hook not in hooked_transformer_cache:
                continue

            hooked_transformer_activation = hooked_transformer_cache[hook]
            bridge_activation = bridge_cache[hook]

            assert hooked_transformer_activation.shape == bridge_activation.shape, (
                f"Shape mismatch for hook {hook}: "
                f"HookedTransformer shape {hooked_transformer_activation.shape}, "
                f"TransformerBridge shape {bridge_activation.shape}"
            )

            # Allow for some numerical differences due to different implementations
            # Use nanmean to handle -inf values in attention scores (which produce nan when subtracted)
            mean_abs_diff = torch.nanmean(
                torch.abs(hooked_transformer_activation - bridge_activation)
            )
            assert mean_abs_diff < 0.5, (
                f"Hook {hook} does not match between HookedTransformer and TransformerBridge. "
                f"Mean absolute difference: {mean_abs_diff}"
            )

    def test_required_model_hooks_available(self, transformer_bridge, expected_hooks):
        """Test that TransformerBridge has all required TransformerLens hooks accessible on the model."""
        # Get expected hooks and assert each one exists

        missing_hooks = []
        for hook_name in expected_hooks:
            if not self.hook_exists_on_model(transformer_bridge, hook_name):
                missing_hooks.append(hook_name)

        assert (
            len(missing_hooks) == 0
        ), f"Required hooks are not accessible on TransformerBridge: {sorted(missing_hooks)}"

    def test_cache_completeness_vs_strict_equality(
        self, transformer_bridge, prompt, expected_hooks
    ):
        """Test cache completeness (allowing extra hooks) vs strict equality."""
        _, cache = transformer_bridge.run_with_cache(prompt)
        actual_keys = list(cache.keys())

        # Find missing and extra hooks
        expected_set = set(expected_hooks)
        actual_set = set(actual_keys)

        missing_hooks = expected_set - actual_set
        extra_hooks = actual_set - expected_set

        # This test documents the current behavior:
        # - We require all expected hooks to be present
        # - We allow extra hooks (they indicate additional functionality)
        assert len(missing_hooks) == 0, f"Missing expected hooks: {sorted(missing_hooks)}"

        # Log extra hooks for visibility but don't fail
        if extra_hooks:
            print(f"Note: Found {len(extra_hooks)} additional hooks beyond expected set")
            print(
                f"Additional hooks: {sorted(list(extra_hooks)[:5])}{'...' if len(extra_hooks) > 5 else ''}"
            )
