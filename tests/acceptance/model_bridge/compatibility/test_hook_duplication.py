"""Test for hook duplication bug in compatibility mode."""

import torch

from transformer_lens import HookedTransformer
from transformer_lens.model_bridge import TransformerBridge


def test_TransformerBridge_compatibility_mode_calls_hooks_once():
    """Regression test: hooks fire exactly once even with aliased HookPoint names."""
    hooked_model = HookedTransformer.from_pretrained_no_processing("gpt2", device_map="cpu")
    bridge_model: TransformerBridge = TransformerBridge.boot_transformers("gpt2", device="cpu")  # type: ignore
    bridge_model.enable_compatibility_mode(no_processing=True)

    test_input = torch.tensor([[1, 2, 3]])

    hooked_call_count = 0

    def count_hooked_calls(acts, hook):
        nonlocal hooked_call_count
        hooked_call_count += 1
        return acts

    hooked_model.blocks[0].hook_mlp_out.add_hook(count_hooked_calls, is_permanent=True)
    _ = hooked_model(test_input)
    hooked_model.reset_hooks()

    bridge_call_count = 0

    def count_bridge_calls(acts, hook):
        nonlocal bridge_call_count
        bridge_call_count += 1
        return acts

    bridge_model.blocks[0].mlp.hook_out.add_hook(count_bridge_calls, is_permanent=True)
    _ = bridge_model(test_input)
    bridge_model.reset_hooks()

    assert (
        hooked_call_count == 1
    ), f"HookedTransformer should call hook once, got {hooked_call_count}"

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

    assert hasattr(block0, "hook_mlp_out"), "Block should have hook_mlp_out attribute"
    assert hasattr(block0.mlp, "hook_out"), "MLP should have hook_out attribute"
    assert id(block0.hook_mlp_out) == id(
        block0.mlp.hook_out
    ), "hook_mlp_out should be aliased to mlp.hook_out (same object)"


def test_stateful_hook_pattern():
    """Test stateful closure pattern (circuit-tracer's cache-then-pop) with aliased hooks."""
    bridge_model: TransformerBridge = TransformerBridge.boot_transformers("gpt2", device="cpu")  # type: ignore
    bridge_model.enable_compatibility_mode(no_processing=True)

    test_input = torch.tensor([[1, 2, 3]])
    block = bridge_model.blocks[0]

    cached = {}

    def cache_activations(acts, hook):
        """Cache activations for later use."""
        cached["acts"] = acts.clone()
        return acts

    def use_cached_activations(acts, hook):
        """Use cached activations; .pop() raises KeyError if called twice."""
        skip_input_activation = cached.pop("acts")
        assert skip_input_activation is not None
        return acts

    block.ln2.hook_in.add_hook(cache_activations, is_permanent=True)
    block.mlp.hook_out.add_hook(use_cached_activations, is_permanent=True)

    # Should not raise KeyError (hook called exactly once)
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
