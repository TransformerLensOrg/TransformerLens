#!/usr/bin/env python3
"""Acceptance tests for TransformerBridge backward-hook behavior."""

import torch


class TestBackwardHookCorrectness:
    """Backward hooks must report the true autograd gradient of the hooked activation."""

    def test_backward_hook_gradients_match_autograd(self, gpt2_bridge_compat_no_processing):
        """The gradient a bwd hook sees must equal retain_grad ground truth.

        This anchors the bwd-hook plumbing on autograd itself: capture the
        activation with retain_grad in a forward hook, capture the hook-reported
        gradient in a backward hook, and require them to be identical.
        """
        bridge_model = gpt2_bridge_compat_no_processing
        test_input = torch.tensor([[1, 2, 3]])

        captured: dict = {}

        def retain_activation(tensor, hook=None):
            tensor.retain_grad()
            captured["activation"] = tensor
            return tensor

        def capture_grad(grad, hook=None):
            captured["hook_grad"] = grad.detach().clone()
            return None

        bridge_model.zero_grad()
        with bridge_model.hooks(
            fwd_hooks=[("blocks.0.hook_mlp_out", retain_activation)],
            bwd_hooks=[("blocks.0.hook_mlp_out", capture_grad)],
        ):
            out = bridge_model(test_input)
            out.sum().backward()

        assert "hook_grad" in captured, "backward hook never fired"
        autograd_grad = captured["activation"].grad
        assert autograd_grad is not None, "retain_grad produced no gradient"
        max_diff = (captured["hook_grad"] - autograd_grad).abs().max().item()
        assert max_diff < 1e-6, (
            f"Backward hook reports a gradient {max_diff:.3e} away from the autograd "
            f"ground truth for the same activation"
        )


def test_transformer_bridge_hooks_context_cleans_up_backward_hooks(
    gpt2_bridge_compat_no_processing,
):
    """Regression test for backward-hook cleanup on context exit."""
    bridge_model = gpt2_bridge_compat_no_processing
    bridge_hook = bridge_model.blocks[0].hook_resid_post
    test_input = torch.tensor([[1, 2, 3]])

    def noop_backward_hook(grad, hook=None):
        return None

    bridge_model.zero_grad()
    with bridge_model.hooks(bwd_hooks=[("blocks.0.hook_resid_post", noop_backward_hook)]):
        bridge_model(test_input).sum().backward()

    assert not bridge_hook.has_hooks(dir="bwd", including_permanent=False)


def test_transformer_bridge_reset_hooks_removes_backward_hooks(gpt2_bridge_compat_no_processing):
    """Regression test for bridge reset_hooks removing backward hooks."""
    bridge_model = gpt2_bridge_compat_no_processing
    backward_hook = bridge_model.blocks[0].hook_resid_post

    backward_hook.add_hook(lambda grad, hook=None: None, dir="bwd")

    assert backward_hook.has_hooks(dir="bwd", including_permanent=False)

    bridge_model.reset_hooks()

    assert not backward_hook.has_hooks(dir="bwd", including_permanent=False)
