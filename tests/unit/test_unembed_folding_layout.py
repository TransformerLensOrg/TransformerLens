"""Unembedding folding follows converted layout rather than ambiguous dimensions."""

from unittest.mock import Mock

import pytest
import torch

from transformer_lens.config.transformer_bridge_config import TransformerBridgeConfig
from transformer_lens.conversion_utils.conversion_steps.transpose_tensor_conversion import (
    TransposeTensorConversion,
)
from transformer_lens.conversion_utils.param_processing_conversion import (
    ParamProcessingConversion,
)
from transformer_lens.model_bridge.architecture_adapter import ArchitectureAdapter
from transformer_lens.weight_processing import ProcessWeights


@pytest.mark.parametrize("layout", ["tl", "hf", "converted", "legacy"])
@pytest.mark.parametrize("vocab_size", [4, 7])
@pytest.mark.parametrize("center", [False, True])
@pytest.mark.parametrize("norm", ["ln", "rms", "offset_rms"])
@pytest.mark.parametrize("fold_biases", [False, True])
def test_fold_final_affine_uses_model_axis(layout, vocab_size, center, norm, fold_biases):
    weight = torch.arange(4 * vocab_size, dtype=torch.float64).reshape(4, vocab_size) / 10
    gamma = torch.tensor([0.3, 0.7, 1.4, 2.0], dtype=torch.float64)
    beta = torch.tensor([-0.2, 0.1, 0.3, -0.1], dtype=torch.float64)
    bias = torch.arange(vocab_size, dtype=torch.float64) / 10
    adapter = None if layout == "tl" else Mock(spec=ArchitectureAdapter)
    if adapter is not None:
        adapter.weight_processing_conversions = {}
    if layout == "converted":
        adapter.weight_processing_conversions["unembed.weight"] = ParamProcessingConversion(
            tensor_conversion=TransposeTensorConversion()
        )
    elif layout == "legacy":
        adapter.weight_processing_conversions["unembed.weight"] = "lm_head.weight"
    w_key = ProcessWeights._get_param_key("unembed.W_U", adapter)
    b_key = ProcessWeights._get_param_key("unembed.b_U", adapter)
    gamma_key = ProcessWeights._get_param_key("ln_final.w", adapter)
    beta_key = ProcessWeights._get_param_key("ln_final.b", adapter)
    state = {
        w_key: weight.clone() if layout == "tl" else weight.T.clone(),
        b_key: bias.clone(),
        gamma_key: gamma.clone(),
        beta_key: beta.clone(),
    }
    config = TransformerBridgeConfig(
        d_model=4,
        d_vocab=vocab_size,
        d_head=2,
        n_heads=2,
        n_layers=1,
        n_ctx=8,
        act_fn="relu",
        final_rms=norm != "ln",
        rmsnorm_uses_offset=norm == "offset_rms",
    )
    ProcessWeights._fold_final_rms_bias(state, config, fold_biases, adapter)
    ProcessWeights._fold_unembed_layer_norm(state, config, fold_biases, center, adapter)
    actual = state[w_key] if layout == "tl" else state[w_key].T
    effective_gamma = 1 + gamma if norm == "offset_rms" else gamma
    expected = effective_gamma[:, None] * weight
    if center and norm == "ln":
        expected -= expected.mean(0, keepdim=True)
    torch.testing.assert_close(actual, expected)
    folded_beta = fold_biases and norm == "ln"
    torch.testing.assert_close(state[b_key], bias + beta @ weight if folded_beta else bias)
    identity = torch.zeros_like(gamma) if norm == "offset_rms" else torch.ones_like(gamma)
    torch.testing.assert_close(state[gamma_key], identity)
    torch.testing.assert_close(state[beta_key], torch.zeros_like(beta) if folded_beta else beta)
