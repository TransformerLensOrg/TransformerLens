"""Shared helpers for bridge integration tests."""

import copy

import torch


def make_tiny_pair(hf_config, arch_name, *, loader=None):
    """Seeded tiny (bridge, ref) pair sharing identical weights.

    ``loader(config) -> model`` builds each side (default
    ``AutoModelForCausalLM.from_config``); ref keeps the seeded init, hf gets a
    state-dict copy, and the bridge wraps hf with eager attention on cpu.
    """
    from transformers import AutoModelForCausalLM

    from transformer_lens.model_bridge.sources._bridge_builder import (
        build_bridge_from_module,
    )

    if loader is None:
        loader = AutoModelForCausalLM.from_config
    hf_config._attn_implementation = "eager"
    torch.manual_seed(42)
    ref = loader(hf_config).eval()
    hf = loader(copy.deepcopy(hf_config)).eval()
    hf.load_state_dict(ref.state_dict())
    bridge = build_bridge_from_module(
        hf, arch_name, hf_config=copy.deepcopy(hf_config), tokenizer=None, device="cpu"
    ).eval()
    return bridge, ref


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
