"""
Tests for the stop_at_layer parameter in TransformerBridge.

This module tests stop_at_layer functionality across different configurations:
1. Default state (no processing, no compatibility mode)
2. With processed weights only
3. With compatibility mode only (no processing)
4. With compatibility mode and weight processing
"""

import pytest
import torch

from transformer_lens.model_bridge import TransformerBridge


@pytest.fixture(scope="module")
def bridge_default():
    """Load a small model in default state (no processing, no compat mode)."""
    bridge = TransformerBridge.boot_transformers("distilgpt2", device="cpu")
    if bridge.tokenizer.pad_token is None:
        bridge.tokenizer.pad_token = bridge.tokenizer.eos_token
    return bridge


@pytest.fixture(scope="module")
def bridge_with_processed_weights():
    """Load a small model with processed weights (no compat mode)."""
    bridge = TransformerBridge.boot_transformers("distilgpt2", device="cpu")
    bridge.process_compatibility_weights()
    if bridge.tokenizer.pad_token is None:
        bridge.tokenizer.pad_token = bridge.tokenizer.eos_token
    return bridge


@pytest.fixture(scope="module")
def bridge_with_compat_no_processing():
    """Load a small model with compatibility mode but no processing."""
    bridge = TransformerBridge.boot_transformers("distilgpt2", device="cpu")
    bridge.enable_compatibility_mode(no_processing=True, disable_warnings=True)
    if bridge.tokenizer.pad_token is None:
        bridge.tokenizer.pad_token = bridge.tokenizer.eos_token
    return bridge


@pytest.fixture(scope="module")
def bridge_with_compat_and_processing():
    """Load a small model with compatibility mode and weight processing."""
    bridge = TransformerBridge.boot_transformers("distilgpt2", device="cpu")
    bridge.enable_compatibility_mode(disable_warnings=True)
    if bridge.tokenizer.pad_token is None:
        bridge.tokenizer.pad_token = bridge.tokenizer.eos_token
    return bridge


# Test 1: Default state (no processing, no compat mode)


def test_stop_at_embed_default(bridge_default):
    """Test stop_at_layer=0 in default state (only embed, no blocks)."""
    rand_input = torch.randint(0, 100, (2, 10))

    # Run with stop_at_layer=0 (should stop before block 0)
    output, cache = bridge_default.run_with_cache(rand_input, stop_at_layer=0)

    # Run normally to get reference cache
    _, normal_cache = bridge_default.run_with_cache(rand_input)

    # Verify output matches the embedding output
    # Note: In bridge, hook names might be different
    # stop_at_layer=0 should give us the output before block 0 starts
    assert output.shape == (
        2,
        10,
        bridge_default.cfg.d_model,
    ), f"Output shape should be (2, 10, {bridge_default.cfg.d_model}), got {output.shape}"

    # Verify that embedding hooks are present
    assert any("embed" in key for key in cache.keys()), "Cache should contain embedding hooks"

    # Verify that block hooks are NOT present
    assert not any(
        "blocks.0.hook_resid_pre" in key for key in cache.keys()
    ), "Cache should not contain blocks.0.hook_resid_pre"
    assert not any(
        "ln_final" in key for key in cache.keys()
    ), "Cache should not contain ln_final hooks"


def test_stop_at_layer_1_default(bridge_default):
    """Test stop_at_layer=1 in default state (embed + block 0)."""
    rand_input = torch.randint(0, 100, (2, 10))

    # Run with stop_at_layer=1 (should run block 0, stop before block 1)
    output, cache = bridge_default.run_with_cache(rand_input, stop_at_layer=1)

    # Run normally to get reference cache
    _, normal_cache = bridge_default.run_with_cache(rand_input)

    # Verify output shape is correct
    assert output.shape == (
        2,
        10,
        bridge_default.cfg.d_model,
    ), f"Output shape should be (2, 10, {bridge_default.cfg.d_model}), got {output.shape}"

    # Verify that embedding and block 0 hooks are present
    assert any("embed" in key for key in cache.keys()), "Cache should contain embedding hooks"
    assert any("blocks.0" in key for key in cache.keys()), "Cache should contain block 0 hooks"

    # Verify that block 1 hooks are NOT present
    assert not any(
        "blocks.1" in key for key in cache.keys()
    ), "Cache should not contain block 1 hooks"
    assert not any(
        "ln_final" in key for key in cache.keys()
    ), "Cache should not contain ln_final hooks"


def test_stop_at_final_layer_default(bridge_default):
    """Test stop_at_layer=-1 in default state (all layers except last)."""
    rand_input = torch.randint(0, 100, (2, 10))

    # Run with stop_at_layer=-1 (should stop before the last layer)
    output, cache = bridge_default.run_with_cache(rand_input, stop_at_layer=-1)

    # Verify output shape is correct
    assert output.shape == (
        2,
        10,
        bridge_default.cfg.d_model,
    ), f"Output shape should be (2, 10, {bridge_default.cfg.d_model}), got {output.shape}"

    # Verify that embedding and most blocks are present
    assert any("embed" in key for key in cache.keys()), "Cache should contain embedding hooks"

    # Should contain all blocks except the last one
    num_layers = bridge_default.cfg.n_layers
    for layer_idx in range(num_layers - 1):
        assert any(
            f"blocks.{layer_idx}" in key for key in cache.keys()
        ), f"Cache should contain block {layer_idx} hooks"

    # Should NOT contain the last block
    assert not any(
        f"blocks.{num_layers - 1}" in key for key in cache.keys()
    ), f"Cache should not contain block {num_layers - 1} hooks"
    assert not any(
        "ln_final" in key for key in cache.keys()
    ), "Cache should not contain ln_final hooks"


def test_no_stop_default(bridge_default):
    """Test stop_at_layer=None in default state (full forward pass)."""
    rand_input = torch.randint(0, 100, (2, 10))

    # Run with stop_at_layer=None (full forward pass)
    output, cache = bridge_default.run_with_cache(rand_input, stop_at_layer=None)

    # Run normally to compare
    normal_output = bridge_default(rand_input)

    # Verify output shape is correct (logits)
    assert output.shape == (
        2,
        10,
        bridge_default.cfg.d_vocab,
    ), f"Output shape should be (2, 10, {bridge_default.cfg.d_vocab}), got {output.shape}"

    # Verify outputs match
    assert torch.allclose(
        output, normal_output, atol=1e-5
    ), "Output with stop_at_layer=None should match normal forward pass"

    # Verify that all hooks are present
    assert any("embed" in key for key in cache.keys()), "Cache should contain embedding hooks"
    num_layers = bridge_default.cfg.n_layers
    for layer_idx in range(num_layers):
        assert any(
            f"blocks.{layer_idx}" in key for key in cache.keys()
        ), f"Cache should contain block {layer_idx} hooks"
    assert any("ln_final" in key for key in cache.keys()), "Cache should contain ln_final hooks"


def test_run_with_hooks_stop_at_layer_default(bridge_default):
    """Test that run_with_hooks respects stop_at_layer in default state."""
    rand_input = torch.randint(0, 100, (2, 10))

    counting_list = []

    def count_hook(activation, hook):
        counting_list.append(len(counting_list))
        return None

    # Add hooks to different layers
    # Hook at embed should fire
    # Hook at blocks.0 should fire
    # Hook at blocks.1 should NOT fire (stop_at_layer=1)
    output = bridge_default.run_with_hooks(
        rand_input,
        stop_at_layer=1,
        fwd_hooks=[
            ("embed.hook_out", count_hook),
            ("blocks.0.attn.hook_out", count_hook),
            ("blocks.1.attn.hook_out", count_hook),
        ],
    )

    # Verify that only the first two hooks fired
    assert len(counting_list) == 2, f"Expected 2 hooks to fire, got {len(counting_list)}"


# Test 2: With processed weights only


def test_stop_at_embed_processed(bridge_with_processed_weights):
    """Test stop_at_layer=0 with processed weights."""
    rand_input = torch.randint(0, 100, (2, 10))

    # Run with stop_at_layer=0
    output, cache = bridge_with_processed_weights.run_with_cache(rand_input, stop_at_layer=0)

    # Verify output shape
    assert output.shape == (
        2,
        10,
        bridge_with_processed_weights.cfg.d_model,
    ), f"Output shape should be (2, 10, {bridge_with_processed_weights.cfg.d_model}), got {output.shape}"

    # Verify that embedding hooks are present but block hooks are not
    assert any("embed" in key for key in cache.keys()), "Cache should contain embedding hooks"
    assert not any(
        "blocks.0.hook_resid_pre" in key for key in cache.keys()
    ), "Cache should not contain blocks.0.hook_resid_pre"


def test_stop_at_layer_1_processed(bridge_with_processed_weights):
    """Test stop_at_layer=1 with processed weights."""
    rand_input = torch.randint(0, 100, (2, 10))

    # Run with stop_at_layer=1
    output, cache = bridge_with_processed_weights.run_with_cache(rand_input, stop_at_layer=1)

    # Verify output shape
    assert output.shape == (
        2,
        10,
        bridge_with_processed_weights.cfg.d_model,
    ), f"Output shape should be (2, 10, {bridge_with_processed_weights.cfg.d_model}), got {output.shape}"

    # Verify that embedding and block 0 hooks are present
    assert any("embed" in key for key in cache.keys()), "Cache should contain embedding hooks"
    assert any("blocks.0" in key for key in cache.keys()), "Cache should contain block 0 hooks"

    # Verify that block 1 hooks are NOT present
    assert not any(
        "blocks.1" in key for key in cache.keys()
    ), "Cache should not contain block 1 hooks"


def test_no_stop_processed(bridge_with_processed_weights):
    """Test stop_at_layer=None with processed weights (full forward pass)."""
    rand_input = torch.randint(0, 100, (2, 10))

    # Run with stop_at_layer=None
    output, cache = bridge_with_processed_weights.run_with_cache(rand_input, stop_at_layer=None)

    # Verify output shape is correct (logits)
    assert output.shape == (
        2,
        10,
        bridge_with_processed_weights.cfg.d_vocab,
    ), f"Output shape should be (2, 10, {bridge_with_processed_weights.cfg.d_vocab}), got {output.shape}"

    # Verify that all hooks are present
    assert any("embed" in key for key in cache.keys()), "Cache should contain embedding hooks"
    num_layers = bridge_with_processed_weights.cfg.n_layers
    for layer_idx in range(num_layers):
        assert any(
            f"blocks.{layer_idx}" in key for key in cache.keys()
        ), f"Cache should contain block {layer_idx} hooks"


# Test 3: With compatibility mode only (no processing)


def test_stop_at_embed_compat_no_processing(bridge_with_compat_no_processing):
    """Test stop_at_layer=0 with compatibility mode (no processing)."""
    rand_input = torch.randint(0, 100, (2, 10))

    # Run with stop_at_layer=0
    output, cache = bridge_with_compat_no_processing.run_with_cache(rand_input, stop_at_layer=0)

    # Verify output shape
    assert output.shape == (
        2,
        10,
        bridge_with_compat_no_processing.cfg.d_model,
    ), f"Output shape should be (2, 10, {bridge_with_compat_no_processing.cfg.d_model}), got {output.shape}"

    # Verify that embedding hooks are present (with compatibility aliases)
    # In compat mode, should have hook_embed
    assert "hook_embed" in cache.keys() or any(
        "embed" in key for key in cache.keys()
    ), "Cache should contain embedding hooks"

    # In compat mode, blocks.0.hook_resid_pre is an alias for blocks.0.hook_in
    # which is where we stop, so it WILL be in the cache as the stopping point
    # Verify that block INTERNAL hooks are NOT present (attn, mlp)
    assert not any(
        "blocks.0.attn" in key for key in cache.keys()
    ), "Cache should not contain blocks.0.attn hooks"
    assert not any(
        "blocks.0.mlp" in key for key in cache.keys()
    ), "Cache should not contain blocks.0.mlp hooks"


def test_stop_at_layer_1_compat_no_processing(bridge_with_compat_no_processing):
    """Test stop_at_layer=1 with compatibility mode (no processing)."""
    rand_input = torch.randint(0, 100, (2, 10))

    # Run with stop_at_layer=1
    output, cache = bridge_with_compat_no_processing.run_with_cache(rand_input, stop_at_layer=1)

    # Verify output shape
    assert output.shape == (
        2,
        10,
        bridge_with_compat_no_processing.cfg.d_model,
    ), f"Output shape should be (2, 10, {bridge_with_compat_no_processing.cfg.d_model}), got {output.shape}"

    # Verify that embedding and block 0 hooks are present
    assert "hook_embed" in cache.keys() or any(
        "embed" in key for key in cache.keys()
    ), "Cache should contain embedding hooks"
    assert any("blocks.0" in key for key in cache.keys()), "Cache should contain block 0 hooks"

    # Verify that block 1 hooks are NOT present
    assert not any(
        "blocks.1" in key for key in cache.keys()
    ), "Cache should not contain block 1 hooks"


def test_stop_at_final_compat_no_processing(bridge_with_compat_no_processing):
    """Test stop_at_layer=-1 with compatibility mode (no processing)."""
    rand_input = torch.randint(0, 100, (2, 10))

    # Run with stop_at_layer=-1
    output, cache = bridge_with_compat_no_processing.run_with_cache(rand_input, stop_at_layer=-1)

    # Verify output shape
    assert output.shape == (
        2,
        10,
        bridge_with_compat_no_processing.cfg.d_model,
    ), f"Output shape should be (2, 10, {bridge_with_compat_no_processing.cfg.d_model}), got {output.shape}"

    # Should contain all blocks except the last one
    num_layers = bridge_with_compat_no_processing.cfg.n_layers
    for layer_idx in range(num_layers - 1):
        assert any(
            f"blocks.{layer_idx}" in key for key in cache.keys()
        ), f"Cache should contain block {layer_idx} hooks"

    # Should NOT contain the last block
    assert not any(
        f"blocks.{num_layers - 1}" in key for key in cache.keys()
    ), f"Cache should not contain block {num_layers - 1} hooks"


def test_no_stop_compat_no_processing(bridge_with_compat_no_processing):
    """Test stop_at_layer=None with compatibility mode (no processing)."""
    rand_input = torch.randint(0, 100, (2, 10))

    # Run with stop_at_layer=None
    output, cache = bridge_with_compat_no_processing.run_with_cache(rand_input, stop_at_layer=None)

    # Verify output shape is correct (logits)
    assert output.shape == (
        2,
        10,
        bridge_with_compat_no_processing.cfg.d_vocab,
    ), f"Output shape should be (2, 10, {bridge_with_compat_no_processing.cfg.d_vocab}), got {output.shape}"

    # Verify that all hooks are present
    assert "hook_embed" in cache.keys() or any(
        "embed" in key for key in cache.keys()
    ), "Cache should contain embedding hooks"
    num_layers = bridge_with_compat_no_processing.cfg.n_layers
    for layer_idx in range(num_layers):
        assert any(
            f"blocks.{layer_idx}" in key for key in cache.keys()
        ), f"Cache should contain block {layer_idx} hooks"


# Test 4: With compatibility mode and weight processing


def test_stop_at_embed_compat_with_processing(bridge_with_compat_and_processing):
    """Test stop_at_layer=0 with compatibility mode and weight processing."""
    rand_input = torch.randint(0, 100, (2, 10))

    # Run with stop_at_layer=0
    output, cache = bridge_with_compat_and_processing.run_with_cache(rand_input, stop_at_layer=0)

    # Verify output shape
    assert output.shape == (
        2,
        10,
        bridge_with_compat_and_processing.cfg.d_model,
    ), f"Output shape should be (2, 10, {bridge_with_compat_and_processing.cfg.d_model}), got {output.shape}"

    # Verify that embedding hooks are present
    assert "hook_embed" in cache.keys() or any(
        "embed" in key for key in cache.keys()
    ), "Cache should contain embedding hooks"

    # In compat mode, blocks.0.hook_resid_pre is an alias for blocks.0.hook_in
    # which is where we stop, so it WILL be in the cache as the stopping point
    # Verify that block INTERNAL hooks are NOT present (attn, mlp)
    assert not any(
        "blocks.0.attn" in key for key in cache.keys()
    ), "Cache should not contain blocks.0.attn hooks"
    assert not any(
        "blocks.0.mlp" in key for key in cache.keys()
    ), "Cache should not contain blocks.0.mlp hooks"


def test_stop_at_layer_1_compat_with_processing(bridge_with_compat_and_processing):
    """Test stop_at_layer=1 with compatibility mode and weight processing."""
    rand_input = torch.randint(0, 100, (2, 10))

    # Run with stop_at_layer=1
    output, cache = bridge_with_compat_and_processing.run_with_cache(rand_input, stop_at_layer=1)

    # Verify output shape
    assert output.shape == (
        2,
        10,
        bridge_with_compat_and_processing.cfg.d_model,
    ), f"Output shape should be (2, 10, {bridge_with_compat_and_processing.cfg.d_model}), got {output.shape}"

    # Verify that embedding and block 0 hooks are present
    assert "hook_embed" in cache.keys() or any(
        "embed" in key for key in cache.keys()
    ), "Cache should contain embedding hooks"
    assert any("blocks.0" in key for key in cache.keys()), "Cache should contain block 0 hooks"

    # Verify that block 1 hooks are NOT present
    assert not any(
        "blocks.1" in key for key in cache.keys()
    ), "Cache should not contain block 1 hooks"


def test_stop_at_final_compat_with_processing(bridge_with_compat_and_processing):
    """Test stop_at_layer=-1 with compatibility mode and weight processing."""
    rand_input = torch.randint(0, 100, (2, 10))

    # Run with stop_at_layer=-1
    output, cache = bridge_with_compat_and_processing.run_with_cache(rand_input, stop_at_layer=-1)

    # Verify output shape
    assert output.shape == (
        2,
        10,
        bridge_with_compat_and_processing.cfg.d_model,
    ), f"Output shape should be (2, 10, {bridge_with_compat_and_processing.cfg.d_model}), got {output.shape}"

    # Should contain all blocks except the last one
    num_layers = bridge_with_compat_and_processing.cfg.n_layers
    for layer_idx in range(num_layers - 1):
        assert any(
            f"blocks.{layer_idx}" in key for key in cache.keys()
        ), f"Cache should contain block {layer_idx} hooks"

    # Should NOT contain the last block
    assert not any(
        f"blocks.{num_layers - 1}" in key for key in cache.keys()
    ), f"Cache should not contain block {num_layers - 1} hooks"


def test_no_stop_compat_with_processing(bridge_with_compat_and_processing):
    """Test stop_at_layer=None with compatibility mode and weight processing."""
    rand_input = torch.randint(0, 100, (2, 10))

    # Run with stop_at_layer=None
    output, cache = bridge_with_compat_and_processing.run_with_cache(rand_input, stop_at_layer=None)

    # Verify output shape is correct (logits)
    assert output.shape == (
        2,
        10,
        bridge_with_compat_and_processing.cfg.d_vocab,
    ), f"Output shape should be (2, 10, {bridge_with_compat_and_processing.cfg.d_vocab}), got {output.shape}"

    # Verify that all hooks are present
    assert "hook_embed" in cache.keys() or any(
        "embed" in key for key in cache.keys()
    ), "Cache should contain embedding hooks"
    num_layers = bridge_with_compat_and_processing.cfg.n_layers
    for layer_idx in range(num_layers):
        assert any(
            f"blocks.{layer_idx}" in key for key in cache.keys()
        ), f"Cache should contain block {layer_idx} hooks"


def test_run_with_hooks_stop_at_layer_compat_with_processing(
    bridge_with_compat_and_processing,
):
    """Test that run_with_hooks respects stop_at_layer with compat mode and processing."""
    rand_input = torch.randint(0, 100, (2, 10))

    counting_list = []

    def count_hook(activation, hook):
        counting_list.append(len(counting_list))
        return None

    # Add hooks to different layers using canonical names
    # (avoid using aliases like hook_embed as they may cause duplicate firings in compat mode)
    # Hook at embed should fire
    # Hook at blocks.0 should fire
    # Hook at blocks.1 should NOT fire (stop_at_layer=1)
    output = bridge_with_compat_and_processing.run_with_hooks(
        rand_input,
        stop_at_layer=1,
        fwd_hooks=[
            ("embed.hook_out", count_hook),
            ("blocks.0.attn.hook_out", count_hook),
            ("blocks.1.attn.hook_out", count_hook),
        ],
    )

    # Verify that only the first two hooks fired
    assert len(counting_list) == 2, f"Expected 2 hooks to fire, got {len(counting_list)}"


# Additional test: Manual hooks with stop_at_layer


def test_manual_hooks_stop_at_layer_compat_with_processing(
    bridge_with_compat_and_processing,
):
    """Test that manually added hooks respect stop_at_layer with compat mode and processing."""
    rand_input = torch.randint(0, 100, (2, 10))

    counting_list = []

    def count_hook(activation, hook):
        counting_list.append(len(counting_list))
        return None

    # Manually add hooks to different layers
    bridge_with_compat_and_processing.embed.add_hook(count_hook)
    bridge_with_compat_and_processing.blocks[0].attn.add_hook(count_hook)
    bridge_with_compat_and_processing.blocks[1].attn.add_hook(count_hook)

    try:
        # Run with stop_at_layer=1 (should only fire first two hooks)
        output = bridge_with_compat_and_processing(rand_input, stop_at_layer=1)

        # Verify that only the first two hooks fired
        assert len(counting_list) == 2, f"Expected 2 hooks to fire, got {len(counting_list)}"

    finally:
        # Clean up hooks
        bridge_with_compat_and_processing.embed.remove_hooks()
        bridge_with_compat_and_processing.blocks[0].attn.remove_hooks()
        bridge_with_compat_and_processing.blocks[1].attn.remove_hooks()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
