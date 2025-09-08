"""Integration tests for the model bridge functionality.

This module contains tests that verify the core functionality of the model bridge,
including model initialization, text generation, hooks, and caching.
"""

import logging

import pytest
import torch

from transformer_lens.ActivationCache import ActivationCache
from transformer_lens.conversion_utils.conversion_steps.rearrange_hook_conversion import (
    RearrangeHookConversion,
)
from transformer_lens.model_bridge import TransformerBridge
from transformer_lens.model_bridge.generalized_components.attention import (
    AttentionBridge,
)
from transformer_lens.model_bridge.generalized_components.joint_qkv_attention import (
    JointQKVAttentionBridge,
)


def test_model_initialization():
    """Test that the model can be initialized correctly."""
    model_name = "gpt2"  # Use a smaller model for testing
    bridge = TransformerBridge.boot_transformers(model_name)

    assert bridge is not None, "Bridge should be initialized"
    assert bridge.tokenizer is not None, "Tokenizer should be initialized"
    assert isinstance(bridge.original_model, torch.nn.Module), "Model should be a PyTorch module"


def test_model_initialization_with_alias(caplog):
    """Test that the model can be initialized correctly with an alias and logs deprecation warning."""

    model_name = "gpt2-small"

    # Set logging level to capture warnings
    with caplog.at_level(logging.WARNING):
        bridge = TransformerBridge.boot_transformers(model_name)

        # Basic assertions
        assert bridge is not None, "Bridge should be initialized"
        assert bridge.tokenizer is not None, "Tokenizer should be initialized"
        assert isinstance(
            bridge.original_model, torch.nn.Module
        ), "Model should be a PyTorch module"

    # Check that a deprecation warning was logged
    deprecation_found = False
    for record in caplog.records:
        if "DEPRECATED" in record.message:
            deprecation_found = True
            # Verify the warning contains expected content
            assert "gpt2-small" in record.message, "Warning should mention the deprecated alias"
            assert "gpt2" in record.message, "Warning should mention the official name"
            break

    assert deprecation_found, "Expected deprecation warning for alias 'gpt2-small' was not logged"


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


def test_generate_with_kv_cache():
    """Test that generate works with use_past_kv_cache parameter."""
    model_name = "gpt2"  # Use a smaller model for testing
    bridge = TransformerBridge.boot_transformers(model_name)

    if bridge.tokenizer.pad_token is None:
        bridge.tokenizer.pad_token = bridge.tokenizer.eos_token

    prompt = "The quick brown fox jumps over the lazy dog"

    # Test with KV cache enabled
    output_with_cache = bridge.generate(prompt, max_new_tokens=5, use_past_kv_cache=True)

    # Test with KV cache disabled
    output_without_cache = bridge.generate(prompt, max_new_tokens=5, use_past_kv_cache=False)

    # Both should produce valid outputs
    assert isinstance(output_with_cache, str), "Output with KV cache should be a string"
    assert isinstance(output_without_cache, str), "Output without KV cache should be a string"
    assert len(output_with_cache) > len(
        prompt
    ), "Generated text with KV cache should be longer than the prompt"
    assert len(output_without_cache) > len(
        prompt
    ), "Generated text without KV cache should be longer than the prompt"

    # The outputs might be different due to sampling, but both should be valid
    assert len(output_with_cache) > 0, "Output with KV cache should not be empty"
    assert len(output_without_cache) > 0, "Output without KV cache should not be empty"


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

    # Enable compatibility mode to include hook aliases
    bridge.enable_compatibility_mode(disable_warnings=True)

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
    """Test that custom QKV conversion rules can be passed to QKVBridge."""

    model_name = "gpt2"  # Use a smaller model for testing
    bridge = TransformerBridge.boot_transformers(model_name)

    # Create a custom QKV conversion rule
    custom_qkv_conversion_rule = RearrangeHookConversion(
        "batch seq (num_attention_heads d_head) -> batch seq num_attention_heads d_head",
        num_attention_heads=12,  # GPT-2 small has 12 heads
    )

    # This should not raise an error
    test_bridge = JointQKVAttentionBridge(
        name="test_joint_qkv_attention_bridge",
        config=bridge.cfg,
        split_qkv_matrix=lambda x: (x, x, x),  # Dummy function for test
        submodules={},
        qkv_conversion_rule=custom_qkv_conversion_rule,
    )

    # Verify the custom conversion rule was set on Q, K, V components
    assert (
        test_bridge.q.hook_in.hook_conversion is custom_qkv_conversion_rule
    ), "Custom QKV conversion rule should be set on hook_in of Q"
    assert (
        test_bridge.k.hook_in.hook_conversion is custom_qkv_conversion_rule
    ), "Custom QKV conversion rule should be set on hook_in of K"
    assert (
        test_bridge.v.hook_in.hook_conversion is custom_qkv_conversion_rule
    ), "Custom QKV conversion rule should be set on hook_in of V"
    assert (
        test_bridge.q.hook_out.hook_conversion is custom_qkv_conversion_rule
    ), "Custom QKV conversion rule should be set on hook_out of Q"
    assert (
        test_bridge.k.hook_out.hook_conversion is custom_qkv_conversion_rule
    ), "Custom QKV conversion rule should be set on hook_out of K"
    assert (
        test_bridge.v.hook_out.hook_conversion is custom_qkv_conversion_rule
    ), "Custom QKV conversion rule should be set on hook_out of V"

    assert (
        test_bridge.qkv_conversion_rule is custom_qkv_conversion_rule
    ), "Custom QKV conversion rule should be set"


def test_attention_pattern_hook_shape_custom_conversion():
    """Test that custom pattern conversion rules can be passed to attention components."""

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

    # This should not raise an error
    test_bridge = AttentionBridge(
        name="test_attn", config=bridge.cfg, pattern_conversion_rule=custom_conversion
    )

    # Verify the conversion rule was set
    assert (
        test_bridge.hook_pattern.hook_conversion is custom_conversion
    ), "Custom conversion rule should be set"


def test_attention_pattern_hook_shape():
    """Test that the attention pattern hook produces the correct shape (n_heads, pos, pos)."""
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

        # Verify the shape is (n_heads, pos, pos) - attention patterns should not have batch dimension
        assert (
            len(pattern_tensor.shape) == 3
        ), f"Pattern tensor should be 3D, got {len(pattern_tensor.shape)}D"

        n_heads_dim, pos_q_dim, pos_k_dim = pattern_tensor.shape

        # Verify dimensions make sense
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


@pytest.mark.parametrize(
    "model_name",
    [
        "gpt2",  # GPT-2 architecture
        "distilgpt2",  # DistilGPT-2 architecture (smaller GPT-2)
        "EleutherAI/pythia-160m",  # Pythia architecture (GPT-NeoX family)
        "EleutherAI/gpt-neo-125M",  # GPT-Neo architecture
        "google/gemma-2b",  # Gemma architecture (Multi-Query Attention)
    ],
)
def test_get_params(model_name):
    """Test that get_params works correctly with different model architectures.

    This test verifies that the get_params function can successfully extract
    parameters from various model types (GPT-2, DistilGPT-2, Pythia, GPT-Neo, Gemma)
    without encountering attribute errors or missing component issues. This includes
    models with different attention architectures like Multi-Query Attention.

    Args:
        model_name: The model name to test (parameterized)
    """
    bridge = TransformerBridge.boot_transformers(model_name)

    # This should not raise any exceptions
    params_dict = bridge.get_params()

    # Verify that we got a dictionary with expected keys
    assert isinstance(params_dict, dict), "get_params should return a dictionary"
    assert len(params_dict) > 0, "Parameters dictionary should not be empty"

    # Check for expected embedding parameters
    assert "embed.W_E" in params_dict, "Should contain embedding weights"
    assert "pos_embed.W_pos" in params_dict, "Should contain positional embedding weights"

    # Check for expected layer parameters (at least layer 0)
    assert "blocks.0.attn.W_Q" in params_dict, "Should contain query weights for layer 0"
    assert "blocks.0.attn.W_K" in params_dict, "Should contain key weights for layer 0"
    assert "blocks.0.attn.W_V" in params_dict, "Should contain value weights for layer 0"
    assert "blocks.0.attn.W_O" in params_dict, "Should contain output weights for layer 0"

    # Check for attention biases
    assert "blocks.0.attn.b_Q" in params_dict, "Should contain query biases for layer 0"
    assert "blocks.0.attn.b_K" in params_dict, "Should contain key biases for layer 0"
    assert "blocks.0.attn.b_V" in params_dict, "Should contain value biases for layer 0"
    assert "blocks.0.attn.b_O" in params_dict, "Should contain output biases for layer 0"

    # Check for MLP parameters
    assert "blocks.0.mlp.W_in" in params_dict, "Should contain MLP input weights for layer 0"
    assert "blocks.0.mlp.W_out" in params_dict, "Should contain MLP output weights for layer 0"
    assert "blocks.0.mlp.b_in" in params_dict, "Should contain MLP input biases for layer 0"
    assert "blocks.0.mlp.b_out" in params_dict, "Should contain MLP output biases for layer 0"

    # Check for unembedding weights
    assert "unembed.W_U" in params_dict, "Should contain unembedding weights"

    # Verify that all parameter values are tensors
    for key, value in params_dict.items():
        assert isinstance(
            value, torch.Tensor
        ), f"Parameter {key} should be a tensor, got {type(value)}"
        assert value.numel() > 0, f"Parameter {key} should not be empty"

    # Verify tensor shapes are reasonable (not zero-dimensional)
    for key, value in params_dict.items():
        assert (
            len(value.shape) > 0
        ), f"Parameter {key} should have at least 1 dimension, got shape {value.shape}"

    # Check that we have parameters for all layers
    for layer_idx in range(bridge.cfg.n_layers):
        assert (
            f"blocks.{layer_idx}.attn.W_Q" in params_dict
        ), f"Should contain query weights for layer {layer_idx}"
        assert (
            f"blocks.{layer_idx}.attn.W_K" in params_dict
        ), f"Should contain key weights for layer {layer_idx}"
        assert (
            f"blocks.{layer_idx}.attn.W_V" in params_dict
        ), f"Should contain value weights for layer {layer_idx}"
        assert (
            f"blocks.{layer_idx}.attn.W_O" in params_dict
        ), f"Should contain output weights for layer {layer_idx}"


def test_get_params_parameter_shapes():
    """Test that get_params returns parameters with expected shapes for GPT-2."""
    model_name = "gpt2"
    bridge = TransformerBridge.boot_transformers(model_name)

    params_dict = bridge.get_params()

    # Check embedding shapes
    embed_weight = params_dict["embed.W_E"]
    assert embed_weight.shape == (
        bridge.cfg.d_vocab,
        bridge.cfg.d_model,
    ), f"Embedding weight shape should be ({bridge.cfg.d_vocab}, {bridge.cfg.d_model}), got {embed_weight.shape}"

    pos_embed_weight = params_dict["pos_embed.W_pos"]
    assert pos_embed_weight.shape == (
        bridge.cfg.n_ctx,
        bridge.cfg.d_model,
    ), f"Position embedding weight shape should be ({bridge.cfg.n_ctx}, {bridge.cfg.d_model}), got {pos_embed_weight.shape}"

    # Check attention weight shapes for first layer
    w_q = params_dict["blocks.0.attn.W_Q"]
    w_k = params_dict["blocks.0.attn.W_K"]
    w_v = params_dict["blocks.0.attn.W_V"]
    w_o = params_dict["blocks.0.attn.W_O"]

    expected_qkv_shape = (bridge.cfg.n_heads, bridge.cfg.d_model, bridge.cfg.d_head)
    expected_o_shape = (bridge.cfg.n_heads, bridge.cfg.d_head, bridge.cfg.d_model)

    assert (
        w_q.shape == expected_qkv_shape
    ), f"W_Q shape should be {expected_qkv_shape}, got {w_q.shape}"
    assert (
        w_k.shape == expected_qkv_shape
    ), f"W_K shape should be {expected_qkv_shape}, got {w_k.shape}"
    assert (
        w_v.shape == expected_qkv_shape
    ), f"W_V shape should be {expected_qkv_shape}, got {w_v.shape}"
    assert w_o.shape == expected_o_shape, f"W_O shape should be {expected_o_shape}, got {w_o.shape}"

    # Check attention bias shapes
    b_q = params_dict["blocks.0.attn.b_Q"]
    b_k = params_dict["blocks.0.attn.b_K"]
    b_v = params_dict["blocks.0.attn.b_V"]
    b_o = params_dict["blocks.0.attn.b_O"]

    expected_qkv_bias_shape = (bridge.cfg.n_heads, bridge.cfg.d_head)
    expected_o_bias_shape = (bridge.cfg.d_model,)

    assert (
        b_q.shape == expected_qkv_bias_shape
    ), f"b_Q shape should be {expected_qkv_bias_shape}, got {b_q.shape}"
    assert (
        b_k.shape == expected_qkv_bias_shape
    ), f"b_K shape should be {expected_qkv_bias_shape}, got {b_k.shape}"
    assert (
        b_v.shape == expected_qkv_bias_shape
    ), f"b_V shape should be {expected_qkv_bias_shape}, got {b_v.shape}"
    assert (
        b_o.shape == expected_o_bias_shape
    ), f"b_O shape should be {expected_o_bias_shape}, got {b_o.shape}"


def test_get_params_missing_components():
    """Test that get_params gracefully handles missing components with zero tensors."""
    model_name = "gpt2"
    bridge = TransformerBridge.boot_transformers(model_name)

    # Test that the method works normally first
    params_dict = bridge.get_params()
    assert isinstance(params_dict, dict)

    # Test handling of missing components - should return zero tensors instead of exceptions
    # Save original components
    original_embed = bridge.embed
    original_pos_embed = bridge.pos_embed
    original_unembed = bridge.unembed

    try:
        # Test missing embed component - should return zero tensor
        del bridge.embed
        params_dict = bridge.get_params()
        assert isinstance(params_dict, dict)
        assert "embed.W_E" in params_dict
        embed_weight = params_dict["embed.W_E"]
        assert torch.all(embed_weight == 0), "Missing embed should be filled with zeros"
        assert embed_weight.shape == (bridge.cfg.d_vocab, bridge.cfg.d_model)

        # Restore embed, test missing pos_embed
        bridge.embed = original_embed
        del bridge.pos_embed
        params_dict = bridge.get_params()
        assert isinstance(params_dict, dict)
        assert "pos_embed.W_pos" in params_dict
        pos_embed_weight = params_dict["pos_embed.W_pos"]
        assert torch.all(pos_embed_weight == 0), "Missing pos_embed should be filled with zeros"
        assert pos_embed_weight.shape == (bridge.cfg.n_ctx, bridge.cfg.d_model)

        # Restore pos_embed, test missing unembed
        bridge.pos_embed = original_pos_embed
        del bridge.unembed
        params_dict = bridge.get_params()
        assert isinstance(params_dict, dict)
        assert "unembed.W_U" in params_dict
        unembed_weight = params_dict["unembed.W_U"]
        assert torch.all(unembed_weight == 0), "Missing unembed should be filled with zeros"
        assert unembed_weight.shape == (bridge.cfg.d_model, bridge.cfg.d_vocab)

    finally:
        # Always restore components
        bridge.embed = original_embed
        bridge.pos_embed = original_pos_embed
        bridge.unembed = original_unembed


def test_get_params_consistency():
    """Test that get_params returns consistent results across multiple calls."""
    model_name = "gpt2"
    bridge = TransformerBridge.boot_transformers(model_name)

    # Get parameters twice
    params1 = bridge.get_params()
    params2 = bridge.get_params()

    # Should have same keys
    assert set(params1.keys()) == set(
        params2.keys()
    ), "Parameter keys should be consistent across calls"

    # Should have same tensor shapes and values
    for key in params1.keys():
        assert params1[key].shape == params2[key].shape, f"Shape mismatch for {key}"
        assert torch.equal(params1[key], params2[key]), f"Value mismatch for {key}"


def test_get_params_configuration_mismatch():
    """Test that get_params raises ValueError for configuration mismatches."""
    model_name = "gpt2"
    bridge = TransformerBridge.boot_transformers(model_name)

    # Test that the method works normally first
    params_dict = bridge.get_params()
    assert isinstance(params_dict, dict)

    # Save original configuration
    original_n_layers = bridge.cfg.n_layers

    try:
        # Simulate configuration mismatch - more layers in config than actual blocks
        bridge.cfg.n_layers = len(bridge.blocks) + 2

        with pytest.raises(ValueError, match="Configuration mismatch.*blocks found"):
            bridge.get_params()

    finally:
        # Always restore original configuration
        bridge.cfg.n_layers = original_n_layers


if __name__ == "__main__":
    pytest.main([__file__])
