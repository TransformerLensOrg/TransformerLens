"""Tests for int-like scalar normalization in Slice."""

import numpy as np
import pytest
import torch
from torch.testing import assert_close

from transformer_lens.utilities.slice import Slice, to_python_int


@pytest.fixture
def tensor():
    return torch.arange(24).reshape(4, 6)


@pytest.mark.parametrize(
    "scalar",
    [np.int64(2), np.int32(2), torch.tensor(2), np.array(2)],
    ids=["np-int64", "np-int32", "0d-tensor", "0d-ndarray"],
)
class TestIntLikeScalars:
    def test_init_matches_plain_int(self, scalar, tensor):
        sliced = Slice(scalar)
        assert sliced.mode == "int"
        assert sliced.slice == 2
        assert_close(sliced.apply(tensor), Slice(2).apply(tensor))

    def test_indices_matches_plain_int(self, scalar, tensor):
        np.testing.assert_array_equal(Slice(scalar).indices(), Slice(2).indices())

    def test_unwrap_matches_plain_int(self, scalar, tensor):
        unwrapped = Slice.unwrap(scalar)
        reference = Slice.unwrap(2)
        assert unwrapped.mode == reference.mode == "array"
        assert_close(unwrapped.apply(tensor), reference.apply(tensor))


def test_bool_scalars_are_not_coerced():
    # Python bool keeps its legacy int-mode path; tensor/ndarray bools stay masks.
    assert to_python_int(True) is None
    assert to_python_int(torch.tensor(True)) is None
    assert to_python_int(np.array(True)) is None
    assert Slice(torch.tensor([True, False])).mode == "array"


def test_plain_int_unchanged(tensor):
    sliced = Slice(3)
    assert sliced.mode == "int"
    assert_close(sliced.apply(tensor), tensor[3])
