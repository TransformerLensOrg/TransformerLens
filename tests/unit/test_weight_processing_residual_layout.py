"""Layout-driven weight processing (ResidualLayout) leaves outputs unchanged.

Each function moves a factor between weights in a way that is exact under the
condition it states: readers see the LayerNorm output, residual streams are read
only through LayerNorms, attention weights sum to 1. These tests check exactly that,
on random tensors with HF nn.Linear layout ([out, in]).
"""
import pytest
import torch
import torch.nn.functional as F

from transformer_lens.model_bridge.architecture_adapter import ResidualLayout
from transformer_lens.weight_processing import ProcessWeights

D_MODEL = 8


def _linear(out_dim, in_dim, bias=True):
    weights = {"weight": torch.randn(out_dim, in_dim)}
    if bias:
        weights["bias"] = torch.randn(out_dim)
    return weights


def _state_dict(**components):
    return {
        f"{name}.{param}": tensor
        for name, params in components.items()
        for param, tensor in params.items()
    }


def _apply(state_dict, name, x):
    out = x @ state_dict[f"{name}.weight"].T
    bias = state_dict.get(f"{name}.bias")
    return out if bias is None else out + bias


def _layer_norm(state_dict, name, x):
    return F.layer_norm(x, (D_MODEL,), state_dict[f"{name}.weight"], state_dict.get(f"{name}.bias"))


@pytest.fixture
def folded():
    torch.manual_seed(0)
    state_dict = _state_dict(
        ln={"weight": torch.randn(D_MODEL) + 1.0, "bias": torch.randn(D_MODEL)},
        q=_linear(D_MODEL, D_MODEL),
        k=_linear(D_MODEL, D_MODEL, bias=False),
        mlp_in=_linear(2 * D_MODEL, D_MODEL),
    )
    layout = ResidualLayout(norm_readers={"ln": ["q", "k", "mlp_in"]})
    return state_dict, ProcessWeights.fold_layer_norm_with_layout(state_dict, layout)


class TestFoldLayerNormWithLayout:
    def test_readers_are_unchanged(self, folded):
        before, after = folded
        x = torch.randn(5, D_MODEL)
        for reader in ("q", "k", "mlp_in"):
            expected = _apply(before, reader, _layer_norm(before, "ln", x))
            actual = _apply(after, reader, _layer_norm(after, "ln", x))
            torch.testing.assert_close(actual, expected, rtol=0, atol=1e-5)

    def test_bias_free_reader_receives_the_folded_bias(self, folded):
        before, after = folded
        torch.testing.assert_close(
            after["k.bias"], before["k.weight"] @ before["ln.bias"], rtol=0, atol=1e-5
        )

    def test_norm_keeps_only_the_normalization(self, folded):
        _, after = folded
        assert torch.equal(after["ln.weight"], torch.ones(D_MODEL))
        assert torch.equal(after["ln.bias"], torch.zeros(D_MODEL))

    def test_reader_weights_centered_over_input(self, folded):
        _, after = folded
        for reader in ("q", "k", "mlp_in"):
            assert after[f"{reader}.weight"].mean(dim=1).abs().max() < 1e-6


class TestCenterWritingWeightsWithLayout:
    def test_layer_norm_reads_are_unchanged(self):
        torch.manual_seed(1)
        state_dict = _state_dict(o=_linear(D_MODEL, 2 * D_MODEL), embed=_linear(10, D_MODEL, False))
        layout = ResidualLayout(
            norm_readers={}, residual_writers=["o"], embedding_writers=["embed"]
        )
        after = ProcessWeights.center_writing_weights_with_layout(state_dict, layout)
        residual, hidden, token = torch.randn(4, D_MODEL), torch.randn(4, 2 * D_MODEL), 3

        def stream(sd):
            return residual + _apply(sd, "o", hidden) + sd["embed.weight"][token]

        torch.testing.assert_close(
            F.layer_norm(stream(after), (D_MODEL,)),
            F.layer_norm(stream(state_dict), (D_MODEL,)),
            rtol=0,
            atol=1e-5,
        )
        assert after["o.weight"].mean(dim=0).abs().max() < 1e-6
        assert after["o.bias"].mean().abs() < 1e-6
        assert after["embed.weight"].mean(dim=1).abs().max() < 1e-6


class TestFoldValueBiasesWithLayout:
    def test_attention_output_is_unchanged(self):
        torch.manual_seed(2)
        state_dict = _state_dict(v=_linear(D_MODEL, D_MODEL), o=_linear(D_MODEL, D_MODEL))
        layout = ResidualLayout(norm_readers={}, value_output_pairs=[("v", "o")])
        after = ProcessWeights.fold_value_biases_with_layout(state_dict, layout)
        x = torch.randn(6, D_MODEL)
        pattern = torch.softmax(torch.randn(6, 6), dim=-1)

        def attend(sd):
            return _apply(sd, "o", pattern @ _apply(sd, "v", x))

        torch.testing.assert_close(attend(after), attend(state_dict), rtol=0, atol=1e-5)
        assert torch.equal(after["v.bias"], torch.zeros(D_MODEL))

    def test_center_outputs_centers_the_output_bias(self):
        torch.manual_seed(3)
        state_dict = _state_dict(v=_linear(D_MODEL, D_MODEL), o=_linear(D_MODEL, D_MODEL))
        layout = ResidualLayout(norm_readers={}, value_output_pairs=[("v", "o")])
        after = ProcessWeights.fold_value_biases_with_layout(
            state_dict, layout, center_outputs=True
        )
        assert after["o.bias"].mean().abs() < 1e-6

    def test_output_without_bias_is_rejected(self):
        state_dict = _state_dict(v=_linear(D_MODEL, D_MODEL), o=_linear(D_MODEL, D_MODEL, False))
        layout = ResidualLayout(norm_readers={}, value_output_pairs=[("v", "o")])
        with pytest.raises(ValueError, match="o has no bias"):
            ProcessWeights.fold_value_biases_with_layout(state_dict, layout)
