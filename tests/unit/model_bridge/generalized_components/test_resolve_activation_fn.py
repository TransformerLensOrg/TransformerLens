"""Tests for resolve_activation_fn utility."""
import torch

from transformer_lens.model_bridge.generalized_components.gated_mlp import resolve_activation_fn


class _FakeCfg:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


class TestResolveActivationFn:
    def test_silu_produces_correct_output(self):
        fn = resolve_activation_fn(_FakeCfg(hidden_act="silu"))
        x = torch.tensor([-1.0, 0.0, 1.0, 2.0])
        result = fn(x)
        # silu(x) = x * sigmoid(x); silu(0) = 0, silu(1) ~ 0.7311
        assert result[1].item() == 0.0
        assert 0.7 < result[2].item() < 0.8

    def test_gelu_new_uses_tanh_approximation(self):
        fn = resolve_activation_fn(_FakeCfg(hidden_act="gelu_new"))
        x = torch.tensor([1.0])
        exact_gelu = torch.nn.functional.gelu(x, approximate="none")
        tanh_gelu = torch.nn.functional.gelu(x, approximate="tanh")
        result = fn(x)
        # tanh approx differs from exact — verify we got the tanh variant
        assert not torch.allclose(exact_gelu, tanh_gelu)
        assert torch.allclose(result, tanh_gelu)

    def test_gelu_vs_relu_differ(self):
        fn_gelu = resolve_activation_fn(_FakeCfg(activation_function="gelu"))
        fn_relu = resolve_activation_fn(_FakeCfg(activation_function="relu"))
        x = torch.tensor([-0.5, 0.5])
        # gelu(-0.5) is small negative; relu(-0.5) is exactly 0
        assert fn_gelu(x)[0].item() != 0.0
        assert fn_relu(x)[0].item() == 0.0

    def test_relu_zeroes_negatives(self):
        fn = resolve_activation_fn(_FakeCfg(act_fn="relu"))
        x = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
        result = fn(x)
        assert result[0].item() == 0.0
        assert result[1].item() == 0.0
        assert result[3].item() == 1.0

    def test_none_config_defaults_to_silu(self):
        fn = resolve_activation_fn(None)
        x = torch.tensor([1.0])
        # silu(1) ~ 0.7311, relu(1) = 1.0 — verify it's not relu
        assert fn(x).item() != 1.0
        assert 0.7 < fn(x).item() < 0.8

    def test_priority_order(self):
        # activation_function takes priority over hidden_act
        fn = resolve_activation_fn(_FakeCfg(activation_function="relu", hidden_act="gelu"))
        x = torch.tensor([-0.5, 1.0])
        assert fn(x)[0].item() == 0.0  # relu zeros negatives; gelu wouldn't

    def test_swish_alias_maps_to_silu(self):
        fn = resolve_activation_fn(_FakeCfg(hidden_act="swish"))
        x = torch.tensor([1.0])
        fn_silu = resolve_activation_fn(_FakeCfg(hidden_act="silu"))
        assert torch.allclose(fn(x), fn_silu(x))

    def test_unknown_activation_defaults_to_silu(self):
        fn = resolve_activation_fn(_FakeCfg(hidden_act="some_unknown_fn"))
        x = torch.tensor([1.0])
        assert 0.7 < fn(x).item() < 0.8  # silu(1) ~ 0.7311
