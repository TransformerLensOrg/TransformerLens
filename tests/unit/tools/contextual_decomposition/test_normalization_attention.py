"""Scalar allocation oracles for normalization, attention, and residual operations."""

import math

import pytest
import torch
import torch.nn.functional as F

from transformer_lens.tools.analysis.contextual_decomposition.rules import (
    _attention,
    _CDConstituents,
    _layer_norm,
    _masked_softmax,
    _residual_add,
    _slice,
)

DTYPES = (torch.float32, torch.float64)


def _parts(beta, gamma, dtype=torch.float64):
    return _CDConstituents(torch.tensor(beta, dtype=dtype), torch.tensor(gamma, dtype=dtype))


def _norm_oracle(beta, gamma, gain, bias, eps):
    total = [b + g for b, g in zip(beta, gamma)]
    mean = sum(total) / len(total)
    denominator = math.sqrt(sum((x - mean) ** 2 for x in total) / len(total) + eps)
    beta_mean = sum(beta) / len(beta)
    gamma_mean = sum(gamma) / len(gamma)
    relevant, remaining = [], []
    for b, g, w, offset in zip(beta, gamma, gain, bias):
        share = abs(b) / (abs(b) + abs(g)) if abs(b) + abs(g) else 0.0
        relevant.append((b - beta_mean) / denominator * w + share * offset)
        remaining.append((g - gamma_mean) / denominator * w + (1 - share) * offset)
    return relevant, remaining, denominator


@pytest.mark.parametrize("dtype", DTYPES)
def test_layer_norm_matches_independent_constituent_oracle(dtype):
    beta = [[1.0, -2.0, 0.0], [0.0, 0.0, 0.0], [2.0, -1.0, 4.0]]
    gamma = [[-0.5, 3.0, 0.0], [0.0, 0.0, 0.0], [-2.0, 1.0, -4.0]]
    gain, bias, eps = [2.0, -1.0, 0.5], [3.0, -2.0, 1.0], 0.25
    expected = [_norm_oracle(b, g, gain, bias, eps) for b, g in zip(beta, gamma)]
    parts = _parts(beta, gamma, dtype)
    result = _layer_norm(
        parts, torch.tensor(gain, dtype=dtype), torch.tensor(bias, dtype=dtype), eps=eps
    )
    torch.testing.assert_close(result.beta, torch.tensor([r[0] for r in expected], dtype=dtype))
    torch.testing.assert_close(result.gamma, torch.tensor([r[1] for r in expected], dtype=dtype))
    native = F.layer_norm(
        parts.beta + parts.gamma,
        (3,),
        torch.tensor(gain, dtype=dtype),
        torch.tensor(bias, dtype=dtype),
        eps,
    )
    torch.testing.assert_close(result.beta + result.gamma, native)


@pytest.mark.parametrize("dtype", DTYPES)
def test_layer_norm_cached_denominator_and_noncontiguous_batches(dtype):
    beta = torch.arange(24, dtype=dtype).reshape(2, 3, 4).transpose(0, 1)
    parts = _CDConstituents(beta, beta * -0.25)
    total = parts.beta + parts.gamma
    denominator = torch.sqrt(
        (total - total.mean(-1, keepdim=True)).square().mean(-1, keepdim=True) + 1e-5
    )
    expected = _layer_norm(parts)
    cached = _layer_norm(parts, denominator=denominator)
    torch.testing.assert_close(cached.beta, expected.beta)
    torch.testing.assert_close(cached.gamma, expected.gamma)
    torch.testing.assert_close(cached.beta + cached.gamma, F.layer_norm(total, (4,)))


def test_independently_normalizing_constituents_is_detected_by_share_oracle():
    parts = _parts([[1.0, 0.0, -2.0]], [[3.0, -1.0, 0.5]])
    result = _layer_norm(parts)
    wrong_beta = F.layer_norm(parts.beta, (3,))
    wrong_gamma = result.beta + result.gamma - wrong_beta
    torch.testing.assert_close(wrong_beta + wrong_gamma, result.beta + result.gamma)
    expected_beta, _, _ = _norm_oracle(
        parts.beta[0].tolist(), parts.gamma[0].tolist(), [1] * 3, [0] * 3, 1e-5
    )
    with pytest.raises(AssertionError):
        torch.testing.assert_close(wrong_beta, torch.tensor([expected_beta], dtype=torch.float64))


@pytest.mark.parametrize("eps", [0.0, -1.0, float("nan"), float("inf")])
def test_layer_norm_rejects_invalid_epsilon(eps):
    with pytest.raises(ValueError, match="eps"):
        _layer_norm(_parts([[1.0, 2.0]], [[0.0, 1.0]]), eps=eps)


@pytest.mark.parametrize(
    "denominator,match",
    [
        ([[0.0]], "positive"),
        ([[-1.0]], "positive"),
        ([[float("nan")]], "finite"),
        ([[1.0, 1.0]], "shape"),
        ([[100.0]], "full input"),
    ],
)
def test_layer_norm_rejects_invalid_cached_denominator(denominator, match):
    with pytest.raises(ValueError, match=match):
        _layer_norm(
            _parts([[1.0, 2.0]], [[0.0, 1.0]]),
            denominator=torch.tensor(denominator, dtype=torch.float64),
        )


@pytest.mark.parametrize("name", ["weight", "bias"])
def test_layer_norm_rejects_bad_affine_shapes(name):
    with pytest.raises(ValueError, match="shape"):
        _layer_norm(
            _parts([[1.0, 2.0]], [[0.0, 1.0]]), **{name: torch.ones(1, dtype=torch.float64)}
        )


def _scalar_softmax(scores, allowed):
    maximum = max(x for x, valid in zip(scores, allowed) if valid)
    exponents = [math.exp(x - maximum) if valid else 0.0 for x, valid in zip(scores, allowed)]
    denominator = sum(exponents)
    return [x / denominator for x in exponents]


def _projection(row, weights, bias):
    return [sum(x * w for x, w in zip(row, column)) + b for column, b in zip(zip(*weights), bias)]


def _project_parts(beta, gamma, weights, bias):
    b = _projection(beta, weights, [0.0] * len(bias))
    g = _projection(gamma, weights, [0.0] * len(bias))
    shares = [abs(x) / (abs(x) + abs(y)) if abs(x) + abs(y) else 0.0 for x, y in zip(b, g)]
    return [x + s * a for x, s, a in zip(b, shares, bias)], [
        y + (1 - s) * a for y, s, a in zip(g, shares, bias)
    ]


def _attention_oracle(beta, gamma, q_weight, k_weight, v_weight, o_weight, biases, heads, allowed):
    width, sequence = len(beta[0]), len(beta)
    head_width = width // heads
    projected = []
    for weights, bias in zip([q_weight, k_weight, v_weight], biases):
        projected.append([_project_parts(b, g, weights, bias) for b, g in zip(beta, gamma)])
    writes_beta = [[[0.0] * width for _ in range(heads)] for _ in range(sequence)]
    writes_gamma = [[[0.0] * width for _ in range(heads)] for _ in range(sequence)]
    beta_probs, gamma_probs = [], []
    for head in range(heads):
        start, end = head * head_width, (head + 1) * head_width
        head_beta_p, head_gamma_p = [], []
        for query in range(sequence):
            bq, gq = projected[0][query]
            beta_scores, total_scores = [], []
            for key in range(sequence):
                bk, gk = projected[1][key]
                beta_scores.append(
                    sum(bq[d] * bk[d] for d in range(start, end)) / math.sqrt(head_width)
                )
                # Enumerate every Q/K cross term independently of the tensor implementation.
                total_scores.append(
                    sum(
                        bq[d] * bk[d] + bq[d] * gk[d] + gq[d] * bk[d] + gq[d] * gk[d]
                        for d in range(start, end)
                    )
                    / math.sqrt(head_width)
                )
            bp = _scalar_softmax(beta_scores, allowed[query])
            tp = _scalar_softmax(total_scores, allowed[query])
            head_beta_p.append(bp)
            head_gamma_p.append([t - b for t, b in zip(tp, bp)])
            bz, tz = [], []
            for d in range(start, end):
                bz.append(sum(bp[key] * projected[2][key][0][d] for key in range(sequence)))
                tz.append(
                    sum(
                        tp[key] * (projected[2][key][0][d] + projected[2][key][1][d])
                        for key in range(sequence)
                    )
                )
            for output in range(width):
                writes_beta[query][head][output] = sum(
                    x * o_weight[d][output] for x, d in zip(bz, range(start, end))
                )
                writes_gamma[query][head][output] = sum(
                    (t - b) * o_weight[d][output] for t, b, d in zip(tz, bz, range(start, end))
                )
        beta_probs.append(head_beta_p)
        gamma_probs.append(head_gamma_p)
    return writes_beta, writes_gamma, beta_probs, gamma_probs


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("heads", [1, 2])
def test_attention_matches_independent_cross_term_oracle(dtype, heads):
    beta = [[1.0, -0.5], [0.25, 2.0]]
    gamma = [[-0.5, 1.0], [1.0, -1.0]]
    weights = [
        [[1.0, 0.5], [-0.25, 1.0]],
        [[0.5, -1.0], [1.0, 0.25]],
        [[1.0, 0.0], [0.5, -1.0]],
        [[2.0, -0.5], [0.25, 1.0]],
    ]
    biases = [[0.25, -0.5], [-0.25, 0.5], [0.5, 0.25]]
    expected = _attention_oracle(
        beta, gamma, *weights, biases, heads, [[True, False], [True, True]]
    )
    parts = _parts([beta], [gamma], dtype)
    q, k, v, o = [torch.tensor(w, dtype=dtype) for w in weights]
    qb, kb, vb = [torch.tensor(b, dtype=dtype) for b in biases]
    result = _attention(parts, q, k, v, o, n_heads=heads, q_bias=qb, k_bias=kb, v_bias=vb)
    torch.testing.assert_close(result.head_writes.beta, torch.tensor([expected[0]], dtype=dtype))
    torch.testing.assert_close(result.head_writes.gamma, torch.tensor([expected[1]], dtype=dtype))
    torch.testing.assert_close(result.probabilities.beta, torch.tensor([expected[2]], dtype=dtype))
    torch.testing.assert_close(result.probabilities.gamma, torch.tensor([expected[3]], dtype=dtype))
    torch.testing.assert_close(result.output.beta, result.head_writes.beta.sum(2))
    torch.testing.assert_close(result.output.gamma, result.head_writes.gamma.sum(2))


def test_frozen_pattern_wrong_share_is_detected_even_with_reconstruction():
    parts = _parts([[[1.0, -0.5], [0.25, 2.0]]], [[[-0.5, 1.0], [1.0, -1.0]]])
    eye = torch.eye(2, dtype=torch.float64)
    result = _attention(parts, eye, eye, eye, eye, n_heads=1, causal=False)
    total_p = result.probabilities.beta + result.probabilities.gamma
    wrong_beta = total_p @ result.value.beta
    assert not torch.allclose(wrong_beta, result.context.beta)
    wrong_gamma = result.context.beta + result.context.gamma - wrong_beta
    torch.testing.assert_close(wrong_beta + wrong_gamma, result.context.beta + result.context.gamma)


def test_shared_output_bias_is_not_part_of_individual_head_writes():
    parts = _parts([[[1.0, 2.0], [3.0, 4.0]]], [[[0.5, -1.0], [1.0, 0.0]]])
    eye = torch.eye(2, dtype=torch.float64)
    base = _attention(parts, eye, eye, eye, eye, n_heads=2)
    bias = torch.tensor([7.0, -2.0], dtype=torch.float64)
    biased = _attention(parts, eye, eye, eye, eye, n_heads=2, o_bias=bias)
    torch.testing.assert_close(base.head_writes.beta, biased.head_writes.beta)
    torch.testing.assert_close(base.head_writes.gamma, biased.head_writes.gamma)
    torch.testing.assert_close(
        biased.output.beta + biased.output.gamma, base.output.beta + base.output.gamma + bias
    )
    expected_beta, expected_gamma = [], []
    for relevant, remaining in zip(base.output.beta[0].tolist(), base.output.gamma[0].tolist()):
        b_row, g_row = [], []
        for b, g, offset in zip(relevant, remaining, bias.tolist()):
            share = abs(b) / (abs(b) + abs(g)) if abs(b) + abs(g) else 0.0
            b_row.append(b + share * offset)
            g_row.append(g + (1 - share) * offset)
        expected_beta.append(b_row)
        expected_gamma.append(g_row)
    torch.testing.assert_close(
        biased.output.beta, torch.tensor([expected_beta], dtype=torch.float64)
    )
    torch.testing.assert_close(
        biased.output.gamma, torch.tensor([expected_gamma], dtype=torch.float64)
    )


@pytest.mark.parametrize("causal", [True, False])
def test_attention_padding_keys_do_not_contribute(causal):
    parts = _parts(
        [[[1.0, 2.0], [3.0, 4.0], [100.0, -100.0]]], [[[1.0, 0.0], [0.0, 1.0], [50.0, 50.0]]]
    )
    eye = torch.eye(2, dtype=torch.float64)
    mask = torch.tensor([[True, True, False]])
    result = _attention(parts, eye, eye, eye, eye, n_heads=2, causal=causal, key_mask=mask)
    assert torch.equal(
        result.probabilities.beta[..., 2], torch.zeros_like(result.probabilities.beta[..., 2])
    )
    assert torch.equal(
        result.probabilities.gamma[..., 2], torch.zeros_like(result.probabilities.gamma[..., 2])
    )
    changed = _parts(
        [[[1.0, 2.0], [3.0, 4.0], [-100.0, 100.0]]], [[[1.0, 0.0], [0.0, 1.0], [-50.0, -50.0]]]
    )
    other = _attention(changed, eye, eye, eye, eye, n_heads=2, causal=causal, key_mask=mask)
    # Key padding does not suppress padded queries; compare only unchanged queries.
    torch.testing.assert_close(other.output.beta[:, :2], result.output.beta[:, :2])
    torch.testing.assert_close(other.output.gamma[:, :2], result.output.gamma[:, :2])


def test_masked_softmax_zero_beta_scores_are_not_zero_probabilities():
    result = _masked_softmax(_parts([[0.0, 0.0]], [[2.0, -1.0]]), torch.tensor([[True, True]]))
    torch.testing.assert_close(result.beta, torch.tensor([[0.5, 0.5]], dtype=torch.float64))
    assert (result.gamma < 0).any()
    torch.testing.assert_close(
        result.gamma.sum(-1), torch.zeros(1, dtype=torch.float64), atol=1e-15, rtol=0
    )


@pytest.mark.parametrize(
    "allowed,match",
    [
        (torch.tensor([[False, False]]), "fully masked"),
        (torch.ones(1, 2), "bool"),
        (torch.ones(3, 2, dtype=torch.bool), "shape"),
    ],
)
def test_masked_softmax_rejects_invalid_mask(allowed, match):
    with pytest.raises(ValueError, match=match):
        _masked_softmax(_parts([[1.0, 2.0]], [[0.0, 1.0]]), allowed)


def test_attention_rejects_fully_masked_causal_rows():
    parts = _parts([[[1.0, 2.0], [3.0, 4.0]]], [[[0.0, 1.0], [1.0, 0.0]]])
    eye = torch.eye(2, dtype=torch.float64)
    with pytest.raises(ValueError, match="fully masked"):
        _attention(parts, eye, eye, eye, eye, n_heads=1, key_mask=torch.tensor([[False, True]]))


@pytest.mark.parametrize("heads", [0, -1, 3, True])
def test_attention_rejects_invalid_head_layout(heads):
    parts = _parts([[[1.0, 2.0]]], [[[0.0, 1.0]]])
    eye = torch.eye(2, dtype=torch.float64)
    with pytest.raises(ValueError, match="n_heads"):
        _attention(parts, eye, eye, eye, eye, n_heads=heads)


def test_residual_merge_and_slicing_copy_inputs_without_autograd():
    a = _CDConstituents(torch.ones(2, 3, requires_grad=True), torch.zeros(2, 3, requires_grad=True))
    b = _CDConstituents(torch.ones(2, 3), -torch.ones(2, 3))
    merged = _residual_add(a, b)
    torch.testing.assert_close(merged.beta, torch.full((2, 3), 2.0))
    selected = _slice(merged, (0, slice(None)))
    assert not selected.beta.requires_grad
    selected.beta.zero_()
    assert torch.equal(merged.beta, torch.full((2, 3), 2.0))
    with pytest.raises(ValueError, match="shape"):
        _residual_add(a, _CDConstituents(torch.ones(1, 3), torch.zeros(1, 3)))
    with pytest.raises(ValueError, match="empty"):
        _slice(a, (slice(0, 0), slice(None)))


def test_normalization_and_attention_outputs_are_detached():
    beta = torch.ones(1, 2, 2, requires_grad=True)
    gamma = torch.tensor([[[0.0, 1.0], [2.0, 0.0]]], requires_grad=True)
    parts = _CDConstituents(beta, gamma)
    eye = torch.eye(2, requires_grad=True)
    result = _attention(parts, eye, eye, eye, eye, n_heads=1)
    norm = _layer_norm(parts, torch.ones(2, requires_grad=True))
    for constituents in (
        result.output,
        result.head_writes,
        result.scores,
        result.probabilities,
        result.context,
        norm,
    ):
        assert not constituents.beta.requires_grad and not constituents.gamma.requires_grad
    assert beta.grad is None and gamma.grad is None and eye.grad is None
