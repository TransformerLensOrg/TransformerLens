"""Tests for the Recursive DLA Functionality."""
# pylint: disable=missing-function-docstring,missing-class-docstring
import pytest
import torch

from transformer_lens.attribution.recursive_dla import pad_tensor_dimension


class TestPadTensorDimension:
    def test_basic_expansion(self):
        x = torch.tensor([[1, 2], [3, 4]])
        expanded = pad_tensor_dimension(x, 1, 5)
        expected = torch.tensor([[1, 2, 0, 0, 0], [3, 4, 0, 0, 0]])
        assert torch.equal(expanded, expected)

    def test_negative_dimension_expansion(self):
        x = torch.tensor([[1, 2], [3, 4]])
        expanded = pad_tensor_dimension(x, -1, 5)
        expected = torch.tensor([[1, 2, 0, 0, 0], [3, 4, 0, 0, 0]])
        assert torch.equal(expanded, expected)

    def test_no_expansion_needed(self):
        x = torch.tensor([[1, 2], [3, 4]])
        expanded = pad_tensor_dimension(x, 1, 2)
        expected = x
        assert torch.equal(expanded, expected)

    def test_invalid_expansion_size(self):
        x = torch.tensor([[1, 2], [3, 4]])
        with pytest.raises(ValueError, match=r"Expansion to size 1 not possible.*"):
            pad_tensor_dimension(x, 1, 1)

    def test_3d_tensor_expansion(self):
        x = torch.tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
        expanded = pad_tensor_dimension(x, 2, 4)
        expected = torch.tensor(
            [[[1, 2, 0, 0], [3, 4, 0, 0]], [[5, 6, 0, 0], [7, 8, 0, 0]]]
        )
        assert torch.equal(expanded, expected)
