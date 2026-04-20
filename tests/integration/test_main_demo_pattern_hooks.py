"""
Acceptance test for Main_Demo pattern hooks workflow.

This test ensures that the induction score calculation pattern from Main_Demo
continues to work correctly after changes to the hook system.
"""

import einops
import pytest
import torch

from transformer_lens import HookedTransformer


class TestMainDemoPatternHooks:
    """Test that Main_Demo pattern hooks workflow works correctly."""

    @pytest.fixture(scope="class")
    def model(self):
        """Create a small model for testing."""
        return HookedTransformer.from_pretrained("gpt2", device="cpu")

    @pytest.fixture
    def repeated_tokens(self, model):
        """Create repeated token sequence like in Main_Demo."""
        seq_len = 50
        # Create sequence: [BOS, 1, 2, ..., seq_len-1, 1, 2, ..., seq_len-1]
        return torch.tensor(
            [[model.tokenizer.bos_token_id] + list(range(1, seq_len)) * 2],
            device=model.cfg.device,
        )

    def test_pattern_filter_hook_works(self, model, repeated_tokens):
        """Test that pattern hooks with filters work (Main_Demo pattern)."""
        seq_len = 50

        # Create storage for induction scores
        induction_score_store = torch.zeros(
            (model.cfg.n_layers, model.cfg.n_heads), device=model.cfg.device
        )

        def induction_score_hook(pattern, hook):
            """Calculate induction scores like in Main_Demo."""
            # Take the diagonal of attention paid from each destination position
            # to source positions seq_len-1 tokens back
            induction_stripe = pattern.diagonal(dim1=-2, dim2=-1, offset=1 - seq_len)

            # Get an average score per head
            induction_score = einops.reduce(
                induction_stripe, "batch head_index position -> head_index", "mean"
            )

            # Store the result
            induction_score_store[hook.layer(), :] = induction_score

            return pattern

        # Filter for pattern hooks (like Main_Demo)
        pattern_hook_names_filter = lambda name: name.endswith("pattern")

        # Run with hooks (should not raise any errors)
        model.run_with_hooks(
            repeated_tokens,
            return_type=None,  # For efficiency, don't calculate logits
            fwd_hooks=[(pattern_hook_names_filter, induction_score_hook)],
        )

        # Verify that induction scores were computed
        assert induction_score_store.shape == (model.cfg.n_layers, model.cfg.n_heads)

        # Check that scores were actually written (not all zeros)
        non_zero_count = (induction_score_store != 0).sum().item()
        total_count = induction_score_store.numel()

        assert (
            non_zero_count > 0
        ), f"Expected some non-zero induction scores, got {non_zero_count}/{total_count}"

        # Verify that all layers and heads have scores
        assert (
            non_zero_count == total_count
        ), f"Expected all {total_count} scores to be computed, got {non_zero_count}"

    def test_pattern_hooks_called_once_per_layer(self, model):
        """Test that pattern hooks are called exactly once per layer."""
        tokens = torch.tensor([[1, 2, 3, 4, 5]], device=model.cfg.device)

        # Track hook calls
        hook_calls = {}

        def tracking_hook(pattern, hook):
            name = hook.name
            hook_calls[name] = hook_calls.get(name, 0) + 1
            return pattern

        # Filter for pattern hooks
        pattern_filter = lambda name: name.endswith("pattern")

        # Run with hooks
        model.run_with_hooks(tokens, return_type=None, fwd_hooks=[(pattern_filter, tracking_hook)])

        # Verify each pattern hook was called exactly once
        for name, count in hook_calls.items():
            assert (
                count == 1
            ), f"Hook {name} was called {count} times, expected 1 (possible duplicate hook registration)"

        # Verify we got hooks for all layers
        expected_hooks = model.cfg.n_layers
        actual_hooks = len(hook_calls)
        assert (
            actual_hooks == expected_hooks
        ), f"Expected {expected_hooks} pattern hooks, got {actual_hooks}"

    def test_hook_layer_method_works(self, model):
        """Test that hook.layer() method works correctly (used in Main_Demo)."""
        tokens = torch.tensor([[1, 2, 3, 4, 5]], device=model.cfg.device)

        # Track layer indices extracted from hooks
        layer_indices = []

        def layer_tracking_hook(pattern, hook):
            # This is what Main_Demo does - call hook.layer()
            layer_idx = hook.layer()
            layer_indices.append(layer_idx)
            return pattern

        # Filter for pattern hooks
        pattern_filter = lambda name: name.endswith("pattern")

        # Run with hooks
        model.run_with_hooks(
            tokens, return_type=None, fwd_hooks=[(pattern_filter, layer_tracking_hook)]
        )

        # Verify we got layer indices for all layers
        assert len(layer_indices) == model.cfg.n_layers

        # Verify layer indices are correct (0, 1, 2, ..., n_layers-1)
        expected_indices = list(range(model.cfg.n_layers))
        assert sorted(layer_indices) == expected_indices


class TestMainDemoPatternHooksWithBridge:
    """Test that Main_Demo pattern hooks also work with TransformerBridge."""

    @pytest.fixture(scope="class")
    def model(self):
        """Create a bridge model for testing."""
        from transformer_lens.model_bridge import TransformerBridge

        bridge = TransformerBridge.boot_transformers("gpt2", device="cpu")
        # Don't enable compatibility mode - Main_Demo doesn't use it
        return bridge

    @pytest.fixture
    def repeated_tokens(self, model):
        """Create repeated token sequence."""
        seq_len = 50
        return torch.tensor(
            [[model.tokenizer.bos_token_id] + list(range(1, seq_len)) * 2],
            device="cpu",
        )

    def test_pattern_filter_hook_works_with_bridge(self, model, repeated_tokens):
        """Test that pattern hooks work with TransformerBridge."""
        seq_len = 50

        # Create storage for induction scores
        induction_score_store = torch.zeros((model.cfg.n_layers, model.cfg.n_heads), device="cpu")

        def induction_score_hook(pattern, hook):
            """Calculate induction scores."""
            induction_stripe = pattern.diagonal(dim1=-2, dim2=-1, offset=1 - seq_len)
            induction_score = einops.reduce(
                induction_stripe, "batch head_index position -> head_index", "mean"
            )
            induction_score_store[hook.layer(), :] = induction_score
            return pattern

        # Filter for pattern hooks
        pattern_hook_names_filter = lambda name: name.endswith("pattern")

        # Run with hooks
        model.run_with_hooks(
            repeated_tokens,
            return_type=None,
            fwd_hooks=[(pattern_hook_names_filter, induction_score_hook)],
        )

        # Verify scores were computed
        non_zero_count = (induction_score_store != 0).sum().item()
        total_count = induction_score_store.numel()

        assert (
            non_zero_count == total_count
        ), f"Expected all {total_count} scores, got {non_zero_count}"
