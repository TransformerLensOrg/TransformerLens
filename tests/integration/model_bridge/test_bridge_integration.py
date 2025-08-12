"""Integration tests for the model bridge functionality.

This module contains tests that verify the core functionality of the model bridge,
including model initialization, text generation, hooks, and caching.
"""

import pytest
import torch

from transformer_lens.ActivationCache import ActivationCache
from transformer_lens.model_bridge import TransformerBridge


def test_model_initialization():
    """Test that the model can be initialized correctly."""
    model_name = "gpt2"  # Use a smaller model for testing
    bridge = TransformerBridge.boot_transformers(model_name)

    assert bridge is not None, "Bridge should be initialized"
    assert bridge.tokenizer is not None, "Tokenizer should be initialized"
    assert isinstance(bridge.original_model, torch.nn.Module), "Model should be a PyTorch module"


def test_text_generation():
    """Test basic text generation functionality."""
    model_name = "gpt2"  # Use a smaller model for testing
    bridge = TransformerBridge.boot_transformers(model_name)

    if bridge.tokenizer.pad_token is None:
        bridge.tokenizer.pad_token = bridge.tokenizer.eos_token

    prompt = "The quick brown fox jumps over the lazy dog"
    output = bridge.generate(prompt, max_new_tokens=10)

    assert isinstance(output, str), "Output should be a string"
    assert len(output) > len(prompt), "Generated text should be longer than the prompt"


def test_hooks():
    """Test that hooks can be added and removed correctly."""
    model_name = "gpt2"  # Use a smaller model for testing
    bridge = TransformerBridge.boot_transformers(model_name)

    if bridge.tokenizer.pad_token is None:
        bridge.tokenizer.pad_token = bridge.tokenizer.eos_token

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
    bridge = TransformerBridge.boot_transformers(model_name)

    if bridge.tokenizer.pad_token is None:
        bridge.tokenizer.pad_token = bridge.tokenizer.eos_token

    prompt = "Test prompt"
    output, cache = bridge.run_with_cache(prompt)

    # Verify output and cache
    assert isinstance(output, torch.Tensor), "Output should be a tensor"
    assert isinstance(cache, ActivationCache), "Cache should be an ActivationCache object"
    assert len(cache) > 0, "Cache should contain activations"

    # Verify cache contains some expected keys (using TransformerLens naming convention)
    # The exact keys depend on the model architecture, but we should have some basic ones
    cache_keys = list(cache.keys())
    assert any("embed" in key for key in cache_keys), "Cache should contain word token embeddings"
    assert any("ln_final" in key for key in cache_keys), "Cache should contain final layer norm"
    assert any(
        "unembed" in key for key in cache_keys
    ), "Cache should contain unembedding/language model head"

    # Verify that cached tensors are actually tensors
    for key, value in cache.items():
        assert isinstance(value, torch.Tensor), f"Cache value for {key} should be a tensor"


def test_component_access():
    """Test that model components can be accessed correctly."""
    model_name = "gpt2"  # Use a smaller model for testing
    bridge = TransformerBridge.boot_transformers(model_name)

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


def test_joint_qkv_custom_conversion_rule():
    """Test that custom QKV conversion rules can be passed to JointQKVAttentionBridge."""
    from transformer_lens.conversion_utils.conversion_steps.rearrange_hook_conversion import (
        RearrangeHookConversion,
    )
    from transformer_lens.model_bridge.generalized_components.joint_qkv_attention import (
        JointQKVAttentionBridge,
    )
    from transformer_lens.model_bridge.generalized_components.linear import LinearBridge

    model_name = "gpt2"  # Use a smaller model for testing
    bridge = TransformerBridge.boot_transformers(model_name)

    # Create a custom QKV conversion rule
    custom_qkv_conversion = RearrangeHookConversion(
        "batch seq (num_attention_heads d_head) -> batch seq num_attention_heads d_head",
        num_attention_heads=12,  # GPT-2 small has 12 heads
    )

    # Create QKV config
    qkv_config = {
        "split_qkv_matrix": lambda x: (x, x, x),  # Dummy function for test
    }

    # Create submodules
    submodules = {
        "qkv": LinearBridge(name="c_attn"),
        "o": LinearBridge(name="c_proj"),
    }

    # This should not raise an error
    test_bridge = JointQKVAttentionBridge(
        name="test_joint_qkv",
        model_config=bridge.cfg,
        submodules=submodules,
        qkv_config=qkv_config,
        qkv_conversion_rule=custom_qkv_conversion,
    )

    # Verify the custom conversion rule was set on Q, K, V components
    assert (
        test_bridge.q.hook_out.hook_conversion is custom_qkv_conversion
    ), "Custom QKV conversion rule should be set on Q"
    assert (
        test_bridge.k.hook_out.hook_conversion is custom_qkv_conversion
    ), "Custom QKV conversion rule should be set on K"
    assert (
        test_bridge.v.hook_out.hook_conversion is custom_qkv_conversion
    ), "Custom QKV conversion rule should be set on V"


def test_attention_pattern_hook_shape_custom_conversion():
    """Test that custom pattern conversion rules can be passed to attention components."""
    from transformer_lens.conversion_utils.conversion_steps.rearrange_hook_conversion import (
        RearrangeHookConversion,
    )

    model_name = "gpt2"  # Use a smaller model for testing
    bridge = TransformerBridge.boot_transformers(model_name)

    if bridge.tokenizer.pad_token is None:
        bridge.tokenizer.pad_token = bridge.tokenizer.eos_token

    # Create a custom conversion rule (this is just for testing the parameter passing)
    custom_conversion = RearrangeHookConversion(
        "batch n_heads pos_q pos_k -> batch n_heads pos_q pos_k"  # Same as default but explicitly set
    )

    # Verify that the attention bridge accepts the custom conversion parameter
    # We can't easily test this with the existing bridge without recreating it,
    # but we can at least verify the parameter is accepted without error
    from transformer_lens.model_bridge.generalized_components.attention import (
        AttentionBridge,
    )

    # This should not raise an error
    test_bridge = AttentionBridge(
        name="test_attn", config=bridge.cfg, pattern_conversion_rule=custom_conversion
    )

    # Verify the conversion rule was set
    assert (
        test_bridge.hook_pattern.hook_conversion is custom_conversion
    ), "Custom conversion rule should be set"


def test_attention_pattern_hook_shape():
    """Test that the attention pattern hook produces the correct shape (batch, n_heads, pos, pos)."""
    model_name = "gpt2"  # Use a smaller model for testing
    bridge = TransformerBridge.boot_transformers(
        model_name,
        hf_config_overrides={
            "attn_implementation": "eager",
        },
    )

    if bridge.tokenizer.pad_token is None:
        bridge.tokenizer.pad_token = bridge.tokenizer.eos_token

    # Attention output enabled via hf_config_overrides

    # Variable to store captured attention patterns
    captured_patterns = {}

    def capture_pattern_hook(tensor, hook):
        """Hook to capture attention patterns."""
        captured_patterns[hook.name] = tensor.clone()
        return tensor

    # Add hook to capture attention patterns
    bridge.blocks[0].attn.hook_pattern.add_hook(capture_pattern_hook)

    try:
        # Run model with a prompt
        prompt = "The quick brown fox"
        tokens = bridge.to_tokens(prompt)
        batch_size, seq_len = tokens.shape

        # Run forward pass
        output = bridge(tokens)

        # Verify we captured attention patterns
        assert len(captured_patterns) > 0, "Should have captured attention patterns"

        # Get the captured pattern tensor
        pattern_tensor = list(captured_patterns.values())[0]

        # Verify the shape is (batch, n_heads, pos, pos)
        assert (
            len(pattern_tensor.shape) == 4
        ), f"Pattern tensor should be 4D, got {len(pattern_tensor.shape)}D"

        batch_dim, n_heads_dim, pos_q_dim, pos_k_dim = pattern_tensor.shape

        # Verify dimensions make sense
        assert batch_dim == batch_size, f"Batch dimension should be {batch_size}, got {batch_dim}"
        assert (
            n_heads_dim == bridge.cfg.n_heads
        ), f"Heads dimension should be {bridge.cfg.n_heads}, got {n_heads_dim}"
        assert (
            pos_q_dim == seq_len
        ), f"Query position dimension should be {seq_len}, got {pos_q_dim}"
        assert pos_k_dim == seq_len, f"Key position dimension should be {seq_len}, got {pos_k_dim}"

        # Verify it's actually attention weights (should be non-negative and roughly sum to 1 along last dim)
        assert torch.all(pattern_tensor >= 0), "Attention patterns should be non-negative"

        # Check that attention weights roughly sum to 1 along the last dimension (with some tolerance for numerical precision)
        attention_sums = pattern_tensor.sum(dim=-1)
        expected_sums = torch.ones_like(attention_sums)
        assert torch.allclose(
            attention_sums, expected_sums, atol=1e-5
        ), "Attention patterns should sum to ~1 along key dimension"

    finally:
        # Clean up hooks
        bridge.blocks[0].attn.hook_pattern.remove_hooks()


if __name__ == "__main__":
    pytest.main([__file__])
