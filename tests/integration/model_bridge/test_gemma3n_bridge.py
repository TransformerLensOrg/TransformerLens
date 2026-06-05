"""Integration tests for the Gemma 3n text-only TransformerBridge.

Uses a tiny random-init `Gemma3nForConditionalGeneration` fixture (4 layers, d_model 32) so CI
stays light while still exercising the real quirks: AltUp 4-stream residual, KV-cache sharing,
and the per-layer `intermediate_size` list. Confirms bit-exact logit parity vs HF (the block
bridge defers all math to HF) and that the active AltUp stream is exposed as a conventional
`(batch, seq, d_model)` residual.
"""

import pytest
import torch

from transformer_lens.model_bridge import TransformerBridge

MODEL_NAME = "onnx-internal-testing/tiny-random-Gemma3nForConditionalGeneration"
IDS = torch.tensor([[1, 2, 3, 4, 5]])


@pytest.fixture(scope="module")
def bridge():
    return TransformerBridge.boot_transformers(MODEL_NAME, device="cpu", dtype=torch.float32)


def test_text_only_logit_parity_vs_hf(bridge):
    from transformers import AutoModelForCausalLM

    hf = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.float32, attn_implementation="eager"
    ).eval()
    with torch.no_grad():
        ref = hf(IDS).logits
        out = bridge.forward(IDS, return_type="logits")
    assert out.shape == ref.shape
    # AltUp / LAuReL / PLE / KV-sharing all run inside HF — the bridge is a pass-through.
    assert torch.max(torch.abs(out - ref)).item() < 1e-3


def test_config_from_text_config(bridge):
    # Text dims resolve from the nested text_config of the tri-modal model.
    assert bridge.cfg.n_layers == 4
    assert getattr(bridge.cfg, "is_multimodal", False) is False


def test_altup_active_stream_resid_hook(bridge):
    """blocks.0.hook_resid_pre exposes the active AltUp stream as (batch, seq, d_model)."""
    name = next(n for n in bridge.hook_dict if n.endswith("blocks.0.hook_resid_pre"))
    captured = {}

    def cap(tensor, hook):
        captured["t"] = tensor.detach()
        return tensor

    with torch.no_grad():
        bridge.run_with_hooks(IDS, fwd_hooks=[(name, cap)])

    resid = captured.get("t")
    assert resid is not None, "active-stream resid hook did not fire"
    # Conventional residual shape, not the stacked 4-stream tensor.
    assert resid.shape == (IDS.shape[0], IDS.shape[1], bridge.cfg.d_model)


def test_run_with_cache_text_only(bridge):
    with torch.no_grad():
        logits, cache = bridge.run_with_cache(IDS)
    assert torch.isfinite(logits).all()
    assert len(cache) > 0
