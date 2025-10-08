"""Hook compatibility tests for TransformerBridge.

This module contains tests that verify TransformerBridge provides all the hooks
that should be available from HookedTransformer for interpretability research.
"""

from typing import Set

import pytest

from transformer_lens.model_bridge import TransformerBridge


class TestHookCompatibility:
    """Test suite to verify hook compatibility for TransformerBridge."""

    @pytest.fixture(scope="class")
    def model_name(self):
        """Model name to use for testing."""
        return "distilgpt2"

    @pytest.fixture(scope="class")
    def transformer_bridge(self, model_name):
        """Create a TransformerBridge for testing."""
        return TransformerBridge.boot_transformers(model_name, device="cpu")

    def get_expected_hooks(self) -> Set[str]:
        """Get the list of hooks that should be available in a TransformerBridge."""
        expected_hooks = {
            # Core embedding hooks
            # "hook_embed",
            # "hook_pos_embed",
            # Final layer norm and unembedding
            # Block 0 hooks only
            # Residual stream hooks
            "blocks.0.hook_resid_pre",
            "blocks.0.hook_resid_mid",
            "blocks.0.hook_resid_post",
            # Attention hooks
            # "blocks.0.attn.hook_q",
            "blocks.0.attn.hook_k",
            "blocks.0.attn.hook_v",
            "blocks.0.attn.hook_z",
            # "blocks.0.attn.hook_attn_out",
            "blocks.0.attn.hook_pattern",
            "blocks.0.attn.hook_result",
            "blocks.0.attn.hook_attn_scores",
            # MLP hooks
            "blocks.0.mlp.hook_pre",
            # "blocks.0.mlp.hook_post",
            # Layer norm hooks
            "blocks.0.ln1.hook_normalized",
            "blocks.0.ln1.hook_scale",
            "blocks.0.ln2.hook_normalized",
            "blocks.0.ln2.hook_scale",
            # Hook aliases for commonly used patterns
            "blocks.0.hook_attn_in",
            "blocks.0.hook_attn_out",
            "blocks.0.hook_mlp_in",
            "blocks.0.hook_mlp_out",
            "blocks.0.hook_q_input",
            "blocks.0.hook_k_input",
            "blocks.0.hook_v_input",
        }

        return expected_hooks

    def test_required_hooks_available(self, transformer_bridge):
        """Test that TransformerBridge has all required TransformerLens hooks."""

        def hook_exists_on_model(model, hook_path: str) -> bool:
            """Check if a hook path exists on the model by traversing attributes."""
            parts = hook_path.split(".")
            model.enable_compatibility_mode(disable_warnings=False)
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

        # Get expected hooks and assert each one exists
        expected_hooks = self.get_expected_hooks()

        for hook_name in expected_hooks:
            assert hook_exists_on_model(
                transformer_bridge, hook_name
            ), f"Required hook '{hook_name}' is not accessible on TransformerBridge"
