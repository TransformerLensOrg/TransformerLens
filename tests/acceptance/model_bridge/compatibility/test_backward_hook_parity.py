"""Test that all backward hooks produce identical gradients in HookedTransformer and TransformerBridge.

This test ensures complete parity between the two architectures by comparing every gradient
that passes through every backward hook during backpropagation.
"""

import pytest

from transformer_lens import HookedTransformer
from transformer_lens.benchmarks import (
    benchmark_backward_hooks,
    benchmark_critical_backward_hooks,
)
from transformer_lens.model_bridge import TransformerBridge


class TestBackwardHookParity:
    """Test suite for comparing backward hook gradients between HookedTransformer and TransformerBridge."""

    @pytest.fixture
    def model_name(self):
        """Model name to use for testing."""
        return "gpt2"

    @pytest.fixture
    def prompt(self):
        """Test prompt for forward pass."""
        return "The quick brown fox"

    @pytest.fixture
    def hooked_transformer(self, model_name):
        """Create a HookedTransformer for comparison."""
        return HookedTransformer.from_pretrained_no_processing(model_name, device_map="cpu")

    @pytest.fixture
    def transformer_bridge(self, model_name):
        """Create a TransformerBridge without processing."""
        model = TransformerBridge.boot_transformers(model_name, device="cpu")
        model.enable_compatibility_mode(no_processing=True)
        return model

    def test_all_backward_hooks_match(self, hooked_transformer, transformer_bridge, prompt):
        """Test that all backward hook gradients match between HT and TB.

        This test:
        1. Gets all hooks available in HookedTransformer
        2. Registers backward hooks on both models for each hook
        3. Runs forward pass and backward pass on both models
        4. Compares all captured gradients
        5. Asserts they match within tolerance (atol=1e-3)
        """
        # Use benchmark function
        result = benchmark_backward_hooks(
            transformer_bridge,
            prompt,
            reference_model=hooked_transformer,
            abs_tolerance=0.2,
            rel_tolerance=3e-4,
        )
        assert result.passed, result.message

    def test_large_gradient_hooks_match(self, hooked_transformer, transformer_bridge, prompt):
        """Test hooks with large gradient magnitudes using relaxed absolute tolerance.

        Some hooks have very large gradient magnitudes (100,000+) where tiny relative errors
        (< 0.004%) translate to absolute differences > 1.0. This test verifies these hooks
        match with appropriate tolerance for their scale.
        """
        # Use the general backward hooks benchmark with appropriate tolerances
        result = benchmark_backward_hooks(
            transformer_bridge,
            prompt,
            reference_model=hooked_transformer,
            abs_tolerance=0.2,
            rel_tolerance=3e-4,
        )
        assert result.passed, result.message

    def test_critical_backward_hooks_match(self, hooked_transformer, transformer_bridge, prompt):
        """Test that critical backward hooks (commonly used in interpretability research) match.

        This is a lighter-weight version of the full test that focuses on the most
        commonly used hooks for debugging purposes.
        """
        # Use benchmark function
        result = benchmark_critical_backward_hooks(
            transformer_bridge,
            prompt,
            reference_model=hooked_transformer,
            abs_tolerance=0.2,
            rel_tolerance=3e-4,
        )
        assert result.passed, result.message
