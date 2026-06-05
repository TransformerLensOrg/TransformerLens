"""Integration tests for the Qwen3.5 multimodal TransformerBridge.

Asserts the gated-q_proj gate signal (``hook_q_gate``) is hookable under the nested
``model.language_model.layers.*`` path — the verify suite's ``gated_hooks_fire`` benchmark
never exercises it, so this guards the gate as an interpretability surface.
"""

import torch

from transformer_lens.model_bridge import TransformerBridge

MODEL_NAME = "trl-internal-testing/tiny-Qwen3_5ForConditionalGeneration"


def _gate_hooks(bridge) -> list[str]:
    return sorted(n for n in bridge.hook_dict if n.rsplit(".", 1)[-1] == "hook_q_gate")


def test_hook_q_gate_fires_under_nested_language_model_path():
    """hook_q_gate must register AND fire on full-attention layers nested under
    model.language_model.layers.*, with the correct gate shape."""
    bridge = TransformerBridge.boot_transformers(MODEL_NAME, device="cpu", dtype=torch.float32)
    assert getattr(bridge.cfg, "gated_q_proj", False) is True

    gate_hooks = _gate_hooks(bridge)
    # Layer 1 is the full-attention layer; layer 0 is GatedDeltaNet (no full attn,
    # hence no gate hook). The tiny fixture has exactly one full-attention layer.
    assert gate_hooks == ["blocks.1.attn.hook_q_gate"], gate_hooks

    captured: dict[str, torch.Tensor] = {}

    def capture(name):
        def fn(tensor, hook):
            captured[name] = tensor.detach()
            return tensor

        return fn

    with torch.no_grad():
        bridge.run_with_hooks(
            "The quick brown fox", fwd_hooks=[(n, capture(n)) for n in gate_hooks]
        )

    gate = captured.get("blocks.1.attn.hook_q_gate")
    assert gate is not None, "hook_q_gate did not fire under the nested path"
    # Gate is per-head over the query: (batch, seq, n_heads * d_head).
    assert gate.shape[-1] == bridge.cfg.n_heads * bridge.cfg.d_head
    assert torch.isfinite(gate).all()
    # A real (non-degenerate) gate signal, not an all-zero placeholder.
    assert gate.float().std() > 0


def test_linear_attention_layer_has_no_gate_hook():
    """The GatedDeltaNet (linear-attention) layer must not expose a q_proj gate hook."""
    bridge = TransformerBridge.boot_transformers(MODEL_NAME, device="cpu", dtype=torch.float32)
    assert not any("blocks.0." in n for n in _gate_hooks(bridge))
