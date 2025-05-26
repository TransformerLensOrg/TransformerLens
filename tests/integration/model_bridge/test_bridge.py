"""Integration tests for the model bridge functionality.

This module contains tests that verify the core functionality of the model bridge,
including model initialization, text generation, hooks, and caching.
"""

import pytest
import torch

from transformer_lens.boot import boot


def test_model_initialization():
    """Test that the model can be initialized correctly."""
    model_name = "gpt2"  # Use a smaller model for testing
    bridge = boot(model_name)

    assert bridge is not None, "Bridge should be initialized"
    assert bridge.tokenizer is not None, "Tokenizer should be initialized"
    assert isinstance(bridge.model, torch.nn.Module), "Model should be a PyTorch module"


def test_text_generation():
    """Test basic text generation functionality."""
    model_name = "gpt2"  # Use a smaller model for testing
    bridge = boot(model_name)

    prompt = "The quick brown fox jumps over the lazy dog"
    output = bridge.generate(prompt, max_new_tokens=10)

    assert isinstance(output, str), "Output should be a string"
    assert len(output) > len(prompt), "Generated text should be longer than the prompt"


def test_hooks():
    """Test that hooks can be added and removed correctly."""
    model_name = "gpt2"  # Use a smaller model for testing
    bridge = boot(model_name)

    # Track if hook was called
    hook_called = False

    def test_hook(tensor, hook):
        nonlocal hook_called
        hook_called = True
        return tensor

    # Add hook to first attention layer
    hook_name = "blocks.0.attn"
    bridge.blocks[0].attn.add_hook(test_hook)

    # Run model
    prompt = "Test prompt"
    bridge.generate(prompt, max_new_tokens=1)

    # Verify hook was called
    assert hook_called, "Hook should have been called"

    # Remove hook
    bridge.blocks[0].attn.remove_hooks()
    hook_called = False

    # Run model again
    bridge.generate(prompt, max_new_tokens=1)

    # Verify hook was not called
    assert not hook_called, "Hook should not have been called after removal"


def test_cache():
    """Test that the cache functionality works correctly."""
    model_name = "gpt2"  # Use a smaller model for testing
    bridge = boot(model_name)

    prompt = "Test prompt"
    output, cache = bridge.run_with_cache(prompt)

    # Verify output and cache
    assert isinstance(output, torch.Tensor), "Output should be a tensor"
    assert isinstance(cache, dict), "Cache should be a dictionary"
    assert len(cache) > 0, "Cache should contain activations"

    # Verify cache contains some expected keys (using actual HuggingFace model structure)
    # The exact keys depend on the model architecture, but we should have some basic ones
    cache_keys = list(cache.keys())
    assert any("wte" in key for key in cache_keys), "Cache should contain word token embeddings"
    assert any("ln_f" in key for key in cache_keys), "Cache should contain final layer norm"
    assert any("lm_head" in key for key in cache_keys), "Cache should contain language model head"

    # Verify that cached tensors are actually tensors
    for key, value in cache.items():
        assert isinstance(value, torch.Tensor), f"Cache value for {key} should be a tensor"


def test_component_access():
    """Test that model components can be accessed correctly."""
    model_name = "gpt2"  # Use a smaller model for testing
    bridge = boot(model_name)

    # Test accessing various components
    assert hasattr(bridge, "embed"), "Bridge should have embed component"
    assert hasattr(bridge, "blocks"), "Bridge should have blocks component"
    assert hasattr(bridge, "unembed"), "Bridge should have unembed component"

    # Test accessing block components
    block = bridge.blocks[0]
    assert hasattr(block, "attn"), "Block should have attention component"
    assert hasattr(block, "mlp"), "Block should have MLP component"
    assert hasattr(block, "ln1"), "Block should have first layer norm"
    assert hasattr(block, "ln2"), "Block should have second layer norm"


if __name__ == "__main__":
    pytest.main([__file__])
