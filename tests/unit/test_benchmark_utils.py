"""Tests for benchmark utility functions."""
import torch

from transformer_lens.benchmarks.utils import (
    compare_activation_dicts,
    make_capture_hook,
    make_grad_capture_hook,
)


class TestMakeCaptureHook:
    def test_stores_detached_clone(self):
        storage = {}
        hook = make_capture_hook(storage, "test")
        tensor = torch.randn(2, 3, requires_grad=True)
        hook(tensor, None)
        assert "test" in storage
        # Must be a clone (different object) and detached (no grad)
        assert storage["test"] is not tensor
        assert not storage["test"].requires_grad
        assert torch.equal(storage["test"], tensor.detach())

    def test_extracts_first_from_tuple(self):
        storage = {}
        hook = make_capture_hook(storage, "test")
        tensor = torch.randn(2, 3)
        other = torch.randn(4, 5)
        hook((tensor, other), None)
        assert "test" in storage
        assert storage["test"].shape == (2, 3)  # got first element, not second

    def test_passthrough_returns_input(self):
        storage = {}
        hook = make_capture_hook(storage, "test")
        tensor = torch.randn(2, 3)
        result = hook(tensor, None)
        # Hook must return the original tensor (not the clone) for the forward pass
        assert result is tensor

    def test_ignores_empty_tuple(self):
        storage = {}
        hook = make_capture_hook(storage, "test")
        hook((), None)
        assert "test" not in storage

    def test_ignores_non_tensor_in_tuple(self):
        storage = {}
        hook = make_capture_hook(storage, "test")
        hook(("not_a_tensor",), None)
        assert "test" not in storage


class TestMakeGradCaptureHook:
    def test_captures_gradient_clone(self):
        storage = {}
        hook = make_grad_capture_hook(storage, "grad")
        grad = torch.randn(4, 5)
        hook(grad)
        assert "grad" in storage
        assert storage["grad"] is not grad  # must be a clone
        assert torch.equal(storage["grad"], grad)

    def test_return_none_mode(self):
        storage = {}
        hook = make_grad_capture_hook(storage, "grad", return_none=True)
        result = hook(torch.randn(2, 3))
        assert result is None
        assert "grad" in storage  # still captured even though returned None

    def test_return_tensor_mode(self):
        storage = {}
        hook = make_grad_capture_hook(storage, "grad", return_none=False)
        tensor = torch.randn(2, 3)
        result = hook(tensor)
        assert result is tensor  # passthrough

    def test_handles_tuple_gradient(self):
        storage = {}
        hook = make_grad_capture_hook(storage, "grad")
        grad = torch.randn(3, 3)
        hook((grad, None))
        assert "grad" in storage
        assert storage["grad"].shape == (3, 3)


class TestCompareActivationDicts:
    def test_detects_value_difference(self):
        d1 = {"a": torch.zeros(2, 3)}
        d2 = {"a": torch.ones(2, 3)}
        mismatches = compare_activation_dicts(d1, d2, atol=0.1)
        assert len(mismatches) == 1
        assert "Value mismatch" in mismatches[0]
        assert "max_diff=1.0" in mismatches[0]

    def test_detects_shape_mismatch(self):
        d1 = {"a": torch.ones(2, 3)}
        d2 = {"a": torch.ones(3, 2)}
        mismatches = compare_activation_dicts(d1, d2)
        assert len(mismatches) == 1
        assert "Shape mismatch" in mismatches[0]

    def test_within_tolerance_passes(self):
        d1 = {"a": torch.tensor([1.0, 2.0, 3.0])}
        d2 = {"a": torch.tensor([1.001, 2.001, 3.001])}
        assert compare_activation_dicts(d1, d2, atol=0.01) == []

    def test_exceeds_tolerance_fails(self):
        d1 = {"a": torch.tensor([1.0, 2.0, 3.0])}
        d2 = {"a": torch.tensor([1.1, 2.0, 3.0])}
        mismatches = compare_activation_dicts(d1, d2, atol=0.01)
        assert len(mismatches) == 1

    def test_batch_dim_squeeze_2d_vs_3d(self):
        # [seq, dim] vs [1, seq, dim] — should squeeze and match
        vals = torch.randn(3, 4)
        d1 = {"a": vals}
        d2 = {"a": vals.unsqueeze(0)}
        assert compare_activation_dicts(d1, d2) == []

    def test_batch_dim_squeeze_incompatible_fails(self):
        # [2, 3, 4] vs [3, 4] — batch dim is 2, not 1, so can't squeeze
        d1 = {"a": torch.randn(2, 3, 4)}
        d2 = {"a": torch.randn(3, 4)}
        mismatches = compare_activation_dicts(d1, d2)
        assert len(mismatches) == 1
        assert "Shape mismatch" in mismatches[0]

    def test_only_compares_common_keys(self):
        d1 = {"a": torch.ones(2), "b": torch.ones(3)}
        d2 = {"a": torch.ones(2), "c": torch.ones(4)}
        # "b" and "c" are unique — only "a" is compared, and it matches
        assert compare_activation_dicts(d1, d2) == []

    def test_multiple_mismatches_reported(self):
        d1 = {"a": torch.zeros(2), "b": torch.zeros(3)}
        d2 = {"a": torch.ones(2), "b": torch.ones(3)}
        mismatches = compare_activation_dicts(d1, d2, atol=0.01)
        assert len(mismatches) == 2
