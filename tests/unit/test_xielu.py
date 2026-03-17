import pytest
import torch
import torch.nn as nn

from transformer_lens.utils import ACTIVATION_FN_DICT, XIELU, xielu


class TestXIELUClass:
    def test_default_parameters(self):
        act = XIELU()
        assert act.alpha_p.item() == pytest.approx(0.8)
        assert act.alpha_n.item() == pytest.approx(0.8)
        assert act.beta.item() == pytest.approx(0.5)

    def test_custom_parameters(self):
        act = XIELU(alpha_p_init=1.0, alpha_n_init=0.5, beta_init=0.25)
        assert act.alpha_p.item() == pytest.approx(1.0)
        assert act.alpha_n.item() == pytest.approx(0.5)
        assert act.beta.item() == pytest.approx(0.25)

    def test_parameters_are_trainable(self):
        act = XIELU()
        assert isinstance(act.alpha_p, nn.Parameter)
        assert isinstance(act.alpha_n, nn.Parameter)
        assert isinstance(act.beta, nn.Parameter)
        assert act.alpha_p.requires_grad
        assert act.alpha_n.requires_grad
        assert act.beta.requires_grad

    def test_output_shape_preserved(self):
        act = XIELU()
        x = torch.randn(2, 10, 32)
        assert act(x).shape == x.shape

    def test_positive_branch(self):
        """For x > 0: f(x) = alpha_p * x^2 + beta * x."""
        act = XIELU(alpha_p_init=1.0, alpha_n_init=1.0, beta_init=1.0)
        x = torch.tensor([[[1.0, 2.0, 3.0]]])  # (1, 1, 3)
        expected = 1.0 * x ** 2 + 1.0 * x  # alpha_p * x^2 + beta * x
        torch.testing.assert_close(act(x), expected)

    def test_negative_branch(self):
        """For x <= 0: f(x) = alpha_n * (exp(clamp(x, eps)) - 1) - alpha_n * x + beta * x."""
        alpha_n, beta, eps = 1.0, 1.0, -1e-6
        act = XIELU(alpha_p_init=1.0, alpha_n_init=alpha_n, beta_init=beta, eps=eps)
        x = torch.tensor([[[-1.0, -2.0, -3.0]]])  # (1, 1, 3)
        expected = (
            alpha_n * torch.expm1(torch.clamp_max(x, eps))
            - alpha_n * x
            + beta * x
        )
        torch.testing.assert_close(act(x), expected)

    def test_zero_input(self):
        """x = 0 falls into the negative branch."""
        act = XIELU(alpha_p_init=1.0, alpha_n_init=1.0, beta_init=1.0, eps=-1e-6)
        x = torch.tensor([[[0.0]]])  # (1, 1, 1)
        expected = (
            1.0 * torch.expm1(torch.clamp_max(x, -1e-6))
            - 1.0 * x
            + 1.0 * x
        )
        torch.testing.assert_close(act(x), expected)

    def test_gradients_flow_through(self):
        act = XIELU()
        x = torch.randn(4, 8, 16, requires_grad=True)
        out = act(x).sum()
        out.backward()
        assert x.grad is not None
        assert act.alpha_p.grad is not None
        assert act.alpha_n.grad is not None
        assert act.beta.grad is not None

    def test_class_and_function_agree_at_defaults(self):
        """XIELU class with default params should match the fixed xielu function."""
        act = XIELU()  # defaults match xielu() fixed values
        x = torch.randn(2, 5, 16)
        torch.testing.assert_close(act(x), xielu(x))


class TestXIELUFunction:
    def test_output_shape_preserved(self):
        x = torch.randn(2, 10, 32)
        assert xielu(x).shape == x.shape

    def test_positive_values(self):
        """For x > 0: f(x) = 0.8*x^2 + 0.5*x."""
        x = torch.tensor([[[1.0, 2.0]]])  # (1, 1, 2)
        expected = 0.8 * x ** 2 + 0.5 * x
        torch.testing.assert_close(xielu(x), expected)

    def test_negative_values(self):
        """For x <= 0: f(x) = 0.8*(exp(clamp(x,-1e-6))-1) - 0.8*x + 0.5*x."""
        x = torch.tensor([[[-1.0, -2.0]]])  # (1, 1, 2)
        expected = (
            0.8 * torch.expm1(torch.clamp_max(x, -1e-6))
            - 0.8 * x
            + 0.5 * x
        )
        torch.testing.assert_close(xielu(x), expected)

    def test_registered_in_activation_dict(self):
        assert "xielu" in ACTIVATION_FN_DICT
        assert ACTIVATION_FN_DICT["xielu"] is xielu
