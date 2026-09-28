"""DLA must not silently ignore an unfolded learned final normalization."""

from __future__ import annotations

import pytest
import torch

from transformer_lens.config import TransformerBridgeConfig
from transformer_lens.model_bridge import TransformerBridge
from transformer_lens.tools.analysis import direct_logit_attribution


def _tiny_bridge(normalization_type: str) -> TransformerBridge:
    cfg = TransformerBridgeConfig(
        d_model=32,
        d_head=16,
        n_heads=2,
        n_layers=1,
        n_ctx=8,
        d_vocab=16,
        d_mlp=64,
        act_fn="gelu",
        normalization_type=normalization_type,
        seed=0,
    )
    model = TransformerBridge.boot_native(cfg)
    model.eval()
    weight = getattr(model.ln_final, "weight", None)
    if isinstance(weight, torch.Tensor):
        with torch.no_grad():
            weight.copy_(torch.linspace(0.5, 1.5, cfg.d_model))
    return model


@pytest.mark.parametrize("normalization_type", ["LN", "RMS"])
@pytest.mark.parametrize("processing", ["unfolded", "none"])
def test_dla_rejects_unfolded_final_norm(normalization_type: str, processing: str):
    model = _tiny_bridge(normalization_type)
    if processing == "none":
        model.enable_compatibility_mode(no_processing=True)
    else:
        model.enable_compatibility_mode(fold_ln=False)

    with pytest.raises(ValueError, match="final.*norm.*weight"):
        direct_logit_attribution(
            model,
            torch.tensor([[1, 2, 3, 4]]),
            answer_tokens=5,
            incorrect_tokens=6,
        )


@pytest.mark.parametrize("normalization_type", ["LN", "RMS"])
def test_dla_folded_final_norm_reconstructs_logit_difference(normalization_type: str):
    model = _tiny_bridge(normalization_type)
    model.enable_compatibility_mode()
    tokens = torch.tensor([[1, 2, 3, 4]])
    logits = model(tokens)
    attribution = direct_logit_attribution(
        model, tokens, answer_tokens=5, incorrect_tokens=6
    ).attribution.sum()
    expected = logits[0, -1, 5] - logits[0, -1, 6] - (model.b_U[5] - model.b_U[6])
    torch.testing.assert_close(attribution, expected, atol=1e-4, rtol=1e-4)


def test_dla_rejects_unfolded_final_norm_bias():
    model = _tiny_bridge("LN")
    with torch.no_grad():
        model.ln_final.weight.fill_(1.0)
        model.ln_final.bias.copy_(torch.linspace(-0.2, 0.2, model.cfg.d_model))
    model.enable_compatibility_mode(fold_ln=False)
    with pytest.raises(ValueError, match="final.*norm.*bias"):
        direct_logit_attribution(model, torch.tensor([[1, 2, 3, 4]]), answer_tokens=5)


def test_dla_rechecks_final_norm_after_weight_edit():
    model = _tiny_bridge("LN")
    model.enable_compatibility_mode()
    with torch.no_grad():
        model.ln_final.weight[0] = 2.0
    with pytest.raises(ValueError, match="final.*norm.*weight"):
        direct_logit_attribution(model, torch.tensor([[1, 2, 3, 4]]), answer_tokens=5)


@pytest.mark.parametrize("normalization_type", ["LNPre", "RMSPre"])
def test_dla_accepts_parameter_free_final_norm(normalization_type: str):
    model = _tiny_bridge(normalization_type)
    model.enable_compatibility_mode(no_processing=True)
    tokens = torch.tensor([[1, 2, 3, 4]])
    logits = model(tokens)
    attribution = direct_logit_attribution(
        model, tokens, answer_tokens=5, incorrect_tokens=6
    ).attribution.sum()
    expected = logits[0, -1, 5] - logits[0, -1, 6] - (model.b_U[5] - model.b_U[6])
    torch.testing.assert_close(attribution, expected, atol=1e-4, rtol=1e-4)
