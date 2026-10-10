"""Native attention totals and explicit rejection of unsupported layouts."""

import pytest
import torch
import torch.nn.functional as F
from jaxtyping import TypeCheckError

from transformer_lens.tools.analysis.contextual_decomposition.rules import (
    _attention,
    _CDConstituents,
    _gelu,
    _layer_norm,
    _masked_softmax,
    _residual_add,
)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("causal", [True, False])
def test_attention_totals_match_ordinary_scaled_dot_product_attention(dtype, causal):
    generator = torch.Generator().manual_seed(17)
    beta = torch.randn(2, 3, 4, generator=generator, dtype=dtype)
    gamma = torch.randn(2, 3, 4, generator=generator, dtype=dtype)
    weights = [torch.randn(4, 4, generator=generator, dtype=dtype) for _ in range(4)]
    biases = [torch.randn(4, generator=generator, dtype=dtype) for _ in range(4)]
    parts = _CDConstituents(beta, gamma)
    result = _attention(
        parts,
        *weights,
        n_heads=2,
        q_bias=biases[0],
        k_bias=biases[1],
        v_bias=biases[2],
        o_bias=biases[3],
        causal=causal,
    )
    total = beta + gamma
    q, k, v = [
        (total @ w + b).reshape(2, 3, 2, 2).transpose(1, 2) for w, b in zip(weights[:3], biases[:3])
    ]
    context = F.scaled_dot_product_attention(q, k, v, is_causal=causal)
    expected = context.transpose(1, 2).reshape(2, 3, 4) @ weights[3] + biases[3]
    torch.testing.assert_close(
        result.output.beta + result.output.gamma, expected, rtol=2e-5, atol=2e-6
    )
    torch.testing.assert_close(
        result.context.beta + result.context.gamma, context, rtol=2e-5, atol=2e-6
    )


@pytest.mark.parametrize(
    "kwargs,match",
    [
        ({"q_weight": torch.ones(2)}, "q_weight.*shape"),
        ({"k_weight": torch.ones(2, 3)}, "projected widths"),
        ({"v_weight": torch.ones(2, 3)}, "projected widths"),
        ({"o_weight": torch.ones(2, 3)}, "o_weight.*shape"),
        ({"o_bias": torch.ones(3)}, "o_bias.*shape"),
        ({"key_mask": torch.ones(1, 2)}, "bool"),
        ({"key_mask": torch.ones(2, 1, dtype=torch.bool)}, "key_mask.*shape"),
        ({"q_bias": torch.ones(3)}, "bias.*shape"),
    ],
)
def test_attention_rejects_bad_projection_and_mask_shapes(kwargs, match):
    parts = _CDConstituents(torch.ones(1, 2, 2), torch.zeros(1, 2, 2))
    args = {name: torch.eye(2) for name in ("q_weight", "k_weight", "v_weight", "o_weight")}
    args.update(kwargs)
    with pytest.raises(ValueError, match=match):
        _attention(parts, **args, n_heads=1)


def test_attention_rejects_nonboolean_causal_flag():
    parts = _CDConstituents(torch.ones(1, 2, 2), torch.zeros(1, 2, 2))
    eye = torch.eye(2)
    with pytest.raises((ValueError, TypeCheckError), match="causal"):
        _attention(parts, eye, eye, eye, eye, n_heads=1, causal="false")


def test_attention_rejects_non_sequence_inputs():
    parts = _CDConstituents(torch.ones(1, 2), torch.zeros(1, 2))
    eye = torch.eye(2)
    with pytest.raises(ValueError, match="attention input.*shape"):
        _attention(parts, eye, eye, eye, eye, n_heads=1)


def test_attention_rejects_mask_on_different_device():
    parts = _CDConstituents(torch.ones(1, 2, 2), torch.zeros(1, 2, 2))
    eye = torch.eye(2)
    with pytest.raises(ValueError, match="device"):
        _attention(
            parts,
            eye,
            eye,
            eye,
            eye,
            n_heads=1,
            key_mask=torch.empty(1, 2, dtype=torch.bool, device="meta"),
        )
    with pytest.raises(ValueError, match="device"):
        _masked_softmax(
            _CDConstituents(torch.ones(1, 2), torch.zeros(1, 2)),
            torch.empty(1, 2, dtype=torch.bool, device="meta"),
        )


def test_attention_one_token_and_wider_projected_head_space():
    parts = _CDConstituents(torch.tensor([[[1.0, 2.0]]]), torch.tensor([[[3.0, -1.0]]]))
    projection = torch.tensor([[1.0, 0.0, 2.0, 0.0], [0.0, 1.0, 0.0, 2.0]])
    output = projection.T
    result = _attention(parts, projection, projection, projection, output, n_heads=2)
    torch.testing.assert_close(result.output.beta, parts.beta @ projection @ output)
    torch.testing.assert_close(result.output.gamma, parts.gamma @ projection @ output)
    assert result.head_writes.beta.shape == (1, 1, 2, 2)


def test_normalized_attention_residual_and_gelu_reconstruct_composed_total():
    parts = _CDConstituents(
        torch.tensor([[[1.0, -2.0], [0.5, 3.0]]], dtype=torch.float64),
        torch.tensor([[[0.25, 1.0], [-1.0, 0.5]]], dtype=torch.float64),
    )
    normalized = _layer_norm(parts)
    eye = torch.eye(2, dtype=torch.float64)
    attention = _attention(normalized, eye, eye, eye, eye, n_heads=1)
    result = _gelu(_residual_add(parts, attention.output), variant="gelu")
    total = parts.beta + parts.gamma
    native_norm = F.layer_norm(total, (2,))
    native_attention = F.scaled_dot_product_attention(
        native_norm[:, None], native_norm[:, None], native_norm[:, None], is_causal=True
    )[:, 0]
    torch.testing.assert_close(result.beta + result.gamma, F.gelu(total + native_attention))


def test_attention_inputs_and_diagnostics_do_not_alias():
    beta = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])
    gamma = torch.tensor([[[0.0, 1.0], [1.0, -1.0]]])
    originals = (beta.clone(), gamma.clone())
    eye = torch.eye(2)
    result = _attention(_CDConstituents(beta, gamma), eye, eye, eye, eye, n_heads=2)
    output = result.output.beta.clone()
    result.head_writes.beta.zero_()
    result.probabilities.gamma.zero_()
    assert torch.equal(result.output.beta, output)
    assert torch.equal(beta, originals[0]) and torch.equal(gamma, originals[1])
    assert torch.equal(eye, torch.eye(2))


def test_layer_norm_bias_completion_and_native_module_total():
    beta = torch.tensor([[0.0, 0.0, 0.0], [2.0, 1.0, -1.0]], dtype=torch.float64)
    gamma = torch.tensor([[0.0, 0.0, 0.0], [-1.0, -2.0, 3.0]], dtype=torch.float64)
    module = torch.nn.LayerNorm(3, eps=0.01, dtype=torch.float64)
    with torch.no_grad():
        module.weight.copy_(torch.tensor([2.0, -1.0, 3.0]))
        module.bias.copy_(torch.tensor([3.0, 4.0, -2.0]))
    result = _layer_norm(_CDConstituents(beta, gamma), module.weight, module.bias, eps=module.eps)
    torch.testing.assert_close(result.beta + result.gamma, module(beta + gamma))
    assert torch.equal(result.beta[0], torch.zeros(3, dtype=torch.float64))
    assert torch.equal(result.gamma[0], module.bias.detach())
