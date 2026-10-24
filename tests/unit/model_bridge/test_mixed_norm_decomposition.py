"""Mixed block-LN/final-RMS predictions and cached decompositions must both reconcile."""

import pytest
import torch

from transformer_lens.config import TransformerBridgeConfig
from transformer_lens.model_bridge import TransformerBridge
from transformer_lens.tools.analysis import direct_logit_attribution


@pytest.fixture(params=["LN", "LNPre"])
def mixed_bridge(request):
    model = TransformerBridge.boot_native(
        TransformerBridgeConfig(
            d_model=16,
            d_head=8,
            n_heads=2,
            n_layers=2,
            n_ctx=8,
            d_vocab=32,
            d_mlp=24,
            act_fn="gelu",
            normalization_type=request.param,
            final_rms=True,
            seed=3,
        )
    ).eval()
    return model


@pytest.mark.parametrize("processing", ["none", "fold_only", "default"])
@pytest.mark.parametrize("batchless", [False, True])
def test_mixed_final_rms_decompositions_reconcile(mixed_bridge, processing, batchless):
    model = mixed_bridge
    tokens = torch.tensor([[2, 3, 4], [4, 5, 6]])
    with torch.no_grad():
        if processing != "none" and hasattr(model.ln_final, "weight"):
            model.ln_final.weight.copy_(torch.linspace(0.5, 1.5, model.cfg.d_model))
        before = model(tokens).log_softmax(-1)
    if processing == "none":
        model.enable_compatibility_mode(no_processing=True)
    elif processing == "fold_only":
        model.enable_compatibility_mode(
            center_writing_weights=False, center_unembed=False, fold_value_biases=False
        )
    else:
        model.enable_compatibility_mode()
    if batchless:
        tokens = tokens[:1]
    with torch.no_grad():
        logits, cache = model.run_with_cache(tokens, remove_batch_dim=batchless)
    torch.testing.assert_close(
        logits.log_softmax(-1), before[: tokens.shape[0]], atol=1e-5, rtol=1e-5
    )
    residual = cache["blocks.1.hook_resid_post"]
    normalized = cache.apply_ln_to_stack(
        residual.unsqueeze(0), layer=-1, has_batch_dim=not batchless
    )[0]
    torch.testing.assert_close(normalized, residual / cache["ln_final.hook_scale"])
    direction = model.W_U[:, 7] - model.W_U[:, 8]
    expected = logits[..., 7] - logits[..., 8] - (model.b_U[7] - model.b_U[8])
    if batchless:
        expected = expected[0]
    for expand_neurons in (False, True):
        dense = cache.get_full_resid_decomposition(expand_neurons=expand_neurons, apply_ln=True)
        projected = cache.get_full_resid_decomposition(
            expand_neurons=expand_neurons, apply_ln=True, project_output_onto=direction
        )
        torch.testing.assert_close(projected, dense @ direction, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(projected.sum(0), expected, atol=1e-5, rtol=1e-5)
    if not batchless:
        result = direct_logit_attribution(model, answer_tokens=7, incorrect_tokens=8, cache=cache)
        torch.testing.assert_close(result.attribution.sum(0), expected[:, -1], atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("mlp_input", [False, True])
def test_intermediate_layernorm_still_centers(mixed_bridge, mlp_input):
    model = mixed_bridge
    model.enable_compatibility_mode(no_processing=True)
    with torch.no_grad():
        _, cache = model.run_with_cache(torch.tensor([[2, 3, 4]]))
    residual = cache["blocks.1.hook_resid_mid" if mlp_input else "blocks.1.hook_resid_pre"]
    expected = (residual - residual.mean(-1, keepdim=True)) / cache[
        f"blocks.1.ln{2 if mlp_input else 1}.hook_scale"
    ]
    actual = cache.apply_ln_to_stack(residual.unsqueeze(0), layer=1, mlp_input=mlp_input)[0]
    torch.testing.assert_close(actual, expected)
    direction = torch.linspace(0.1, 1.0, model.cfg.d_model)
    dense = cache.stack_neuron_results(layer=1, apply_ln=True, mlp_input=mlp_input)
    projected = cache.stack_neuron_results(
        layer=1, apply_ln=True, mlp_input=mlp_input, project_output_onto=direction
    )
    torch.testing.assert_close(projected, dense @ direction, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("matrix_projection", [False, True])
def test_final_rms_projected_neurons_match_uncentered_oracle(mixed_bridge, matrix_projection):
    model = mixed_bridge
    model.enable_compatibility_mode(no_processing=True)
    with torch.no_grad():
        _, cache = model.run_with_cache(torch.tensor([[2, 3, 4], [4, 5, 6]]))
    direction = torch.linspace(0.1, 1.0, model.cfg.d_model)
    if matrix_projection:
        direction = torch.stack([direction, direction.square()], dim=-1)
    raw = cache.stack_neuron_results(layer=model.cfg.n_layers, pos_slice=-1, apply_ln=False)
    expected = (raw / cache["ln_final.hook_scale"][:, -1]) @ direction
    actual = cache.stack_neuron_results(
        layer=model.cfg.n_layers, pos_slice=-1, apply_ln=True, project_output_onto=direction
    )
    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)
