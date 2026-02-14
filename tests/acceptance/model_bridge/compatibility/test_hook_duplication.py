"""Test for hook duplication bug in compatibility mode.

This test verifies that hooks are called exactly once per forward pass in compatibility mode,
not multiple times due to aliasing.
"""

import torch

from transformer_lens import HookedTransformer
from transformer_lens.model_bridge import TransformerBridge


def test_TransformerBridge_compatibility_mode_calls_hooks_once():
    """Test that TransformerBridge compatibility mode calls hooks exactly once per forward pass.

    This is a regression test for a bug where the same HookPoint object was registered in hook_dict
    under multiple names (e.g., both "blocks.0.hook_mlp_out" and "blocks.0.mlp.hook_out").
    When hooks were added to this HookPoint, they got called once for each registered name,
    resulting in multiple executions per forward pass.

    This broke code that uses stateful closures (like cached dictionaries) and expects
    hooks to be called exactly once per forward pass.
    """
    # Create both models with the same configuration
    hooked_model = HookedTransformer.from_pretrained_no_processing("gpt2", device_map="cpu")
    bridge_model: TransformerBridge = TransformerBridge.boot_transformers("gpt2", device="cpu")  # type: ignore
    bridge_model.enable_compatibility_mode(no_processing=True)

    test_input = torch.tensor([[1, 2, 3]])

    # Test HookedTransformer - hooks should be called once
    hooked_call_count = 0

    def count_hooked_calls(acts, hook):
        nonlocal hooked_call_count
        hooked_call_count += 1
        return acts

    hooked_model.blocks[0].hook_mlp_out.add_hook(count_hooked_calls, is_permanent=True)
    _ = hooked_model(test_input)
    hooked_model.reset_hooks()

    # Test TransformerBridge - hooks should also be called once (after fix)
    bridge_call_count = 0

    def count_bridge_calls(acts, hook):
        nonlocal bridge_call_count
        bridge_call_count += 1
        return acts

    bridge_model.blocks[0].mlp.hook_out.add_hook(count_bridge_calls, is_permanent=True)
    _ = bridge_model(test_input)
    bridge_model.reset_hooks()

    # Verify call counts
    assert (
        hooked_call_count == 1
    ), f"HookedTransformer should call hook once, got {hooked_call_count}"

    # After the fix, TransformerBridge should also call the hook exactly once
    assert bridge_call_count == 1, (
        f"TransformerBridge should call hook once, got {bridge_call_count}. "
        f"Hooks should not be called multiple times even when the same HookPoint is "
        f"registered under multiple names (e.g., 'blocks.0.hook_mlp_out' and 'blocks.0.mlp.hook_out')."
    )


def test_hook_mlp_out_aliasing():
    """Test that hook_mlp_out is properly aliased to mlp.hook_out in compatibility mode."""
    bridge_model: TransformerBridge = TransformerBridge.boot_transformers("gpt2", device="cpu")  # type: ignore
    bridge_model.enable_compatibility_mode(no_processing=True)

    block0 = bridge_model.blocks[0]

    # Verify that hook_mlp_out and mlp.hook_out are the same object
    assert hasattr(block0, "hook_mlp_out"), "Block should have hook_mlp_out attribute"
    assert hasattr(block0.mlp, "hook_out"), "MLP should have hook_out attribute"
    assert id(block0.hook_mlp_out) == id(
        block0.mlp.hook_out
    ), "hook_mlp_out should be aliased to mlp.hook_out (same object)"


def test_stateful_hook_pattern():
    """Test the stateful closure pattern that was breaking due to hook duplication.

    This simulates the pattern used in circuit-tracer's ReplacementModel where a hook
    caches an activation and a later hook uses that cached value.
    """
    bridge_model: TransformerBridge = TransformerBridge.boot_transformers("gpt2", device="cpu")  # type: ignore
    bridge_model.enable_compatibility_mode(no_processing=True)

    test_input = torch.tensor([[1, 2, 3]])
    block = bridge_model.blocks[0]

    # Simulate the pattern from circuit-tracer
    cached = {}

    def cache_activations(acts, hook):
        """Cache activations for later use."""
        cached["acts"] = acts.clone()
        return acts

    def use_cached_activations(acts, hook):
        """Use cached activations - this will fail if hook is called twice."""
        # This pattern uses .pop() which will raise KeyError on second call
        skip_input_activation = cached.pop("acts")
        assert skip_input_activation is not None
        return acts

    # Set up hooks
    block.ln2.hook_in.add_hook(cache_activations, is_permanent=True)
    block.mlp.hook_out.add_hook(use_cached_activations, is_permanent=True)

    # Run forward pass - should not raise KeyError
    try:
        _ = bridge_model(test_input)
        success = True
    except KeyError:
        success = False
    finally:
        bridge_model.reset_hooks()

    assert success, (
        "Stateful hook pattern failed - hook was likely called multiple times, "
        "causing the second call to fail when trying to pop from empty dict"
    )
