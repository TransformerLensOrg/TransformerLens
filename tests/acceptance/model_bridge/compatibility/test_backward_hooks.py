#!/usr/bin/env python3
"""Acceptance tests for backward hook compatibility between TransformerBridge and HookedTransformer.

This test suite ensures that backward hooks produce identical gradient values
in both TransformerBridge and HookedTransformer implementations.
"""

import torch

from transformer_lens import HookedTransformer
from transformer_lens.model_bridge import TransformerBridge


class TestBackwardHookCompatibility:
    """Test backward hook compatibility between TransformerBridge and HookedTransformer."""

    def test_backward_hook_gradients_match_hooked_transformer(self):
        """Test that backward hook gradients match between TransformerBridge and HookedTransformer.

        This test ensures that backward hooks see identical gradient values in both
        TransformerBridge and HookedTransformer when using no_processing mode.
        """
        # Create both models with the same configuration
        hooked_model = HookedTransformer.from_pretrained_no_processing("gpt2", device_map="cpu")
        bridge_model: TransformerBridge = TransformerBridge.boot_transformers(
            "gpt2", device="cpu"
        )  # type: ignore
        bridge_model.enable_compatibility_mode(no_processing=True)

        test_input = torch.tensor([[1, 2, 3]])

        # Collect gradient sums from backward hooks
        hooked_grad_sum = torch.zeros(1)
        bridge_grad_sum = torch.zeros(1)

        def sum_hooked_grads(grad, hook=None):
            nonlocal hooked_grad_sum
            hooked_grad_sum = grad.sum()
            return None

        def sum_bridge_grads(grad, hook=None):
            nonlocal bridge_grad_sum
            bridge_grad_sum = grad.sum()
            return None

        # Run with HookedTransformer
        hooked_model.zero_grad()
        with hooked_model.hooks(bwd_hooks=[("blocks.0.hook_mlp_out", sum_hooked_grads)]):
            out = hooked_model(test_input)
            out.sum().backward()

        # Run with TransformerBridge
        bridge_model.zero_grad()
        with bridge_model.hooks(bwd_hooks=[("blocks.0.hook_mlp_out", sum_bridge_grads)]):
            out = bridge_model(test_input)
            out.sum().backward()

        # Verify gradient values match
        print(f"HookedTransformer gradient sum: {hooked_grad_sum.item():.6f}")
        print(f"TransformerBridge gradient sum: {bridge_grad_sum.item():.6f}")
        print(f"Difference: {abs(hooked_grad_sum - bridge_grad_sum).item():.6f}")
        assert torch.allclose(hooked_grad_sum, bridge_grad_sum, atol=1e-2, rtol=1e-2), (
            f"Gradient sums should be identical but differ by "
            f"{abs(hooked_grad_sum - bridge_grad_sum).item():.6f}"
        )


if __name__ == "__main__":
    # Run test when executed directly
    test = TestBackwardHookCompatibility()
    test.test_backward_hook_gradients_match_hooked_transformer()
    print("✅ Backward hook compatibility test passed!")
