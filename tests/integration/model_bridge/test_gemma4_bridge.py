"""Integration tests for the Gemma 4 TransformerBridge.

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


def test_multimodal_effect_vision_pipeline_bridged(bridge):
    """is_multimodal drives the booted bridge to wire the HF vision pipeline as real
    submodules and to open the multimodal-input gate. Assert those effects, not the flag.

    Text dims still resolve from the nested text_config of the multimodal model.
    """
    assert bridge.cfg.n_layers == 4

    # EFFECT 1: the vision pipeline is bridged. All three fixtures are
    # Gemma4ForConditionalGeneration (vision_tower encoder + embed_vision projector),
    # so both map to real attached nn.Modules on the booted bridge.
    real = bridge.real_components
    assert real["vision_projector"][0] == "model.embed_vision"
    assert real["vision_encoder"][0] == "model.vision_tower"
    assert isinstance(bridge.vision_projector, torch.nn.Module)
    assert isinstance(bridge.vision_encoder, torch.nn.Module)

    # EFFECT 2: the is_multimodal gate is open. The "requires a multimodal model" guard
    # fires only when cfg.is_multimodal is False, so it must NOT fire here. The call may
    # succeed (processor present) or raise a different error (e.g. no processor) — both fine.
    try:
        bridge.prepare_multimodal_inputs("a")
    except ValueError as exc:
        assert "requires a multimodal model" not in str(exc)


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
