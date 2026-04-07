#!/usr/bin/env python3
"""Acceptance tests for run_with_cache compatibility between TransformerBridge and HookedTransformer."""

import torch


class TestRunWithCacheCompatibility:
    """Test run_with_cache compatibility between TransformerBridge and HookedTransformer."""

    def test_run_with_cache_matches_forward_pass(self, gpt2_bridge_compat_no_processing):
        """Test that run_with_cache produces identical results to a regular forward pass."""
        bridge_model = gpt2_bridge_compat_no_processing

        test_input = torch.tensor([[1, 2, 3]])
        bridge_logits_cache, _ = bridge_model.run_with_cache(test_input)
        bridge_logits_manual = bridge_model(test_input)

        assert torch.allclose(
            bridge_logits_cache, bridge_logits_manual, atol=1e-2
        ), "run_with_cache should produce identical results to forward pass"

    def test_run_with_cache_returns_correct_cached_values(
        self, gpt2_hooked_unprocessed, gpt2_bridge_compat_no_processing
    ):
        """Test that run_with_cache returns correct cached activation values."""
        hooked_model = gpt2_hooked_unprocessed
        bridge_model = gpt2_bridge_compat_no_processing

        test_input = torch.tensor([[1, 2, 3]])

        _, hooked_cache = hooked_model.run_with_cache(test_input)
        _, bridge_cache = bridge_model.run_with_cache(test_input)

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

        # Verify cache matches manual hooks
        assert torch.allclose(
            hooked_cache["blocks.0.hook_mlp_out"], manual_cache["hooked"], atol=1e-5
        ), "HookedTransformer run_with_cache should match manual hooks"

        # Same check for TransformerBridge
        cache_diff = (bridge_cache["blocks.0.hook_mlp_out"] - manual_cache["bridge"]).abs().max()
        assert torch.allclose(
            bridge_cache["blocks.0.hook_mlp_out"], manual_cache["bridge"], atol=1e-2, rtol=1e-2
        ), (
            f"TransformerBridge run_with_cache should match manual hooks. "
            f"Max difference: {cache_diff:.6f}"
        )
