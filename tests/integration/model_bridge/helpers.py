"""Shared helpers for bridge integration tests."""

import torch


def assert_bridge_matches_hf(bridge, *args, atol: float = 1e-5, **kwargs) -> None:
    """Assert the bridge's logits match its wrapped HF model on the same inputs.

    Runs both under no_grad with identical args/kwargs; unwraps HF ModelOutput
    logits. atol is a max-abs-diff bound.
    """
    with torch.no_grad():
        bridge_out = bridge(*args, **kwargs)
        hf_out = bridge.original_model(*args, **kwargs)
    hf_logits = hf_out.logits if hasattr(hf_out, "logits") else hf_out
    max_diff = (bridge_out - hf_logits).abs().max().item()
    assert max_diff < atol, f"Bridge vs HF max diff = {max_diff}"
