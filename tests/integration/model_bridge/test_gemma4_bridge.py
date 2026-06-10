"""Integration tests for the Gemma 4 text-only TransformerBridge.

Uses tiny random-init `Gemma4ForConditionalGeneration` fixtures (4 layers, d_model 8) so CI
stays light while still exercising the real per-layer heterogeneity across the family:

- ``tiny-random/gemma-4-e`` — Per-Layer Embeddings + KV-cache sharing (E2B/E4B shape)
- ``tiny-random/gemma-4-dense`` — K==V attention on global layers, no v_proj (31B shape)
- ``tiny-random/gemma-4-moe`` — router + batched experts beside the dense MLP (26B-A4B shape)

Confirms logit parity vs HF (the block bridge defers all math to HF) and that hooks fire on
the conventional single-stream residual.
"""

import pytest
import torch

from transformer_lens.model_bridge import TransformerBridge

MODEL_NAMES = {
    "ple_kv_shared": "tiny-random/gemma-4-e",
    "dense_k_eq_v": "tiny-random/gemma-4-dense",
    "moe": "tiny-random/gemma-4-moe",
}
IDS = torch.tensor([[1, 2, 3, 4, 5]])


@pytest.fixture(scope="module", params=list(MODEL_NAMES), ids=list(MODEL_NAMES))
def bridge(request):
    return TransformerBridge.boot_transformers(
        MODEL_NAMES[request.param], device="cpu", dtype=torch.float32
    )


def test_text_only_logit_parity_vs_hf(bridge):
    from transformers import AutoModelForCausalLM

    hf = AutoModelForCausalLM.from_pretrained(
        bridge.cfg.model_name, torch_dtype=torch.float32, attn_implementation="eager"
    ).eval()
    with torch.no_grad():
        ref = hf(IDS).logits
        out = bridge.forward(IDS, return_type="logits")
    assert out.shape == ref.shape
    # PLE / KV-sharing / K==V / MoE all run inside HF — the bridge is a pass-through.
    assert torch.max(torch.abs(out - ref)).item() < 1e-3


def test_config_from_text_config(bridge):
    # Text dims resolve from the nested text_config of the multimodal model.
    assert bridge.cfg.n_layers == 4
    assert getattr(bridge.cfg, "is_multimodal", False) is False


def test_resid_hooks_fire_with_conventional_shape(bridge):
    """The residual stream is a single conventional (batch, seq, d_model) tensor."""
    captured = {}

    def cap(tensor, hook):
        captured[hook.name] = tensor.detach()
        return tensor

    names = [
        n
        for n in bridge.hook_dict
        if n.endswith("blocks.0.hook_resid_pre") or n.endswith("blocks.0.hook_resid_post")
    ]
    assert names, "no residual hooks registered"
    with torch.no_grad():
        bridge.run_with_hooks(IDS, fwd_hooks=[(n, cap) for n in names])

    assert captured, "residual hooks did not fire"
    for tensor in captured.values():
        assert tensor.shape == (IDS.shape[0], IDS.shape[1], bridge.cfg.d_model)


def test_run_with_cache_text_only(bridge):
    with torch.no_grad():
        logits, cache = bridge.run_with_cache(IDS)
    assert torch.isfinite(logits).all()
    assert len(cache) > 0
