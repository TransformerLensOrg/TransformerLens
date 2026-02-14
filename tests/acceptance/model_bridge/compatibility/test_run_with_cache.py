#!/usr/bin/env python3
"""Acceptance tests for run_with_cache compatibility between TransformerBridge and HookedTransformer.

This test suite ensures that run_with_cache works correctly and produces identical
results in both TransformerBridge and HookedTransformer implementations.
"""

import torch

from transformer_lens import HookedTransformer
from transformer_lens.model_bridge import TransformerBridge


class TestRunWithCacheCompatibility:
    """Test run_with_cache compatibility between TransformerBridge and HookedTransformer."""

    def test_run_with_cache_matches_forward_pass(self):
        """Test that run_with_cache produces identical results to a regular forward pass.

        This ensures that the caching mechanism doesn't alter the model's output.
        """
        bridge_model: TransformerBridge = TransformerBridge.boot_transformers(
            "gpt2", device="cpu"
        )  # type: ignore
        bridge_model.enable_compatibility_mode(no_processing=True)

        test_input = torch.tensor([[1, 2, 3]])
        bridge_logits_cache, _ = bridge_model.run_with_cache(test_input)
        bridge_logits_manual = bridge_model(test_input)

        print(f"Cache logits shape: {bridge_logits_cache.shape}")
        print(f"Manual logits shape: {bridge_logits_manual.shape}")
        print(
            f"Max difference: {torch.abs(bridge_logits_cache - bridge_logits_manual).max().item():.6f}"
        )

        assert torch.allclose(
            bridge_logits_cache, bridge_logits_manual, atol=1e-2
        ), "run_with_cache should produce identical results to forward pass"

    def test_run_with_cache_returns_correct_cached_values(self):
        """Test that run_with_cache returns correct cached activation values.

        This ensures that TransformerBridge.run_with_cache() returns the same
        cached activation values as manual hooks, matching HookedTransformer behavior.
        """
        # Create both models with the same configuration
        hooked_model = HookedTransformer.from_pretrained_no_processing("gpt2", device_map="cpu")
        bridge_model: TransformerBridge = TransformerBridge.boot_transformers(
            "gpt2", device="cpu"
        )  # type: ignore
        bridge_model.enable_compatibility_mode(no_processing=True)

        test_input = torch.tensor([[1, 2, 3]])

        # Method 1: run_with_cache
        _, hooked_cache = hooked_model.run_with_cache(test_input)
        _, bridge_cache = bridge_model.run_with_cache(test_input)

        # Method 2: Manual hooks (ground truth)
        manual_cache = {}

        def make_cache_hook(name):
            def hook_fn(acts, hook):
                manual_cache[name] = acts.clone()
                return acts

            return hook_fn

        hooked_model.reset_hooks()
        with hooked_model.hooks(fwd_hooks=[("blocks.0.hook_mlp_out", make_cache_hook("hooked"))]):
            hooked_model(test_input)

        bridge_model.reset_hooks()
        with bridge_model.hooks(fwd_hooks=[("blocks.0.hook_mlp_out", make_cache_hook("bridge"))]):
            bridge_model(test_input)

        # Verify cache values match manual hooks for HookedTransformer
        print(f"HookedTransformer cache sum: {hooked_cache['blocks.0.hook_mlp_out'].sum():.6f}")
        print(f"HookedTransformer manual sum: {manual_cache['hooked'].sum():.6f}")
        assert torch.allclose(
            hooked_cache["blocks.0.hook_mlp_out"], manual_cache["hooked"], atol=1e-5
        ), "HookedTransformer run_with_cache should match manual hooks"

        # Verify cache values match manual hooks for TransformerBridge
        print(f"TransformerBridge cache sum: {bridge_cache['blocks.0.hook_mlp_out'].sum():.6f}")
        print(f"TransformerBridge manual sum: {manual_cache['bridge'].sum():.6f}")
        cache_diff = (bridge_cache["blocks.0.hook_mlp_out"] - manual_cache["bridge"]).abs().max()
        print(f"Max difference: {cache_diff:.6f}")

        assert torch.allclose(
            bridge_cache["blocks.0.hook_mlp_out"], manual_cache["bridge"], atol=1e-2, rtol=1e-2
        ), (
            f"TransformerBridge run_with_cache should match manual hooks. "
            f"Cache sum: {bridge_cache['blocks.0.hook_mlp_out'].sum():.6f}, "
            f"Manual hooks sum: {manual_cache['bridge'].sum():.6f}, "
            f"Difference: {cache_diff:.6f}"
        )


if __name__ == "__main__":
    # Run tests when executed directly
    test = TestRunWithCacheCompatibility()
    test.test_run_with_cache_matches_forward_pass()
    print("✅ run_with_cache forward pass test passed!")
    test.test_run_with_cache_returns_correct_cached_values()
    print("✅ run_with_cache cached values test passed!")
