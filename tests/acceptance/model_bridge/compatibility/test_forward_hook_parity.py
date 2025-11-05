"""Test that all forward hooks produce identical activations in HookedTransformer and TransformerBridge.

This test ensures complete parity between the two architectures by comparing every tensor
that passes through every hook during a forward pass.
"""

import pytest

from transformer_lens import HookedTransformer
from transformer_lens.benchmarks import (
    benchmark_critical_forward_hooks,
    benchmark_forward_hooks,
)
from transformer_lens.model_bridge import TransformerBridge


class TestForwardHookParity:
    """Test suite for comparing forward hook activations between HookedTransformer and TransformerBridge."""

    @pytest.fixture
    def model_name(self):
        """Model name to use for testing."""
        return "gpt2"

    @pytest.fixture
    def prompt(self):
        """Test prompt for forward pass."""
        return "The quick brown fox jumps over the lazy dog"

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

    def test_all_forward_hooks_match(self, hooked_transformer, transformer_bridge, prompt):
        """Test that all forward hook activations match between HT and TB.

        This test:
        1. Gets all hooks available in HookedTransformer
        2. Registers forward hooks on both models for each hook
        3. Runs forward pass on both models
        4. Compares all captured activations
        5. Asserts they match within tolerance (atol=1e-3)
        """
        # Use benchmark function
        result = benchmark_forward_hooks(
            transformer_bridge, prompt, reference_model=hooked_transformer, tolerance=1e-3
        )
        assert result.passed, result.message

    def test_critical_hooks_match(self, hooked_transformer, transformer_bridge, prompt):
        """Test that critical hooks (commonly used in interpretability research) match.

        This is a lighter-weight version of the full test that focuses on the most
        commonly used hooks for debugging purposes.
        """
        # Use benchmark function
        result = benchmark_critical_forward_hooks(
            transformer_bridge, prompt, reference_model=hooked_transformer, tolerance=1e-3
        )
        assert result.passed, result.message
