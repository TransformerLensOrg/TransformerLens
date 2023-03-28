import pytest
import torch
import numpy as np

import transformer_lens.utils as utils

ref_tensor = torch.tensor([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]])
shape = ref_tensor.shape
tensor_row0_dim2 = torch.tensor([[1, 2, 3, 4, 5]])
shape_2 = tensor_row0_dim2.shape
tensor_row0_dim1 = torch.tensor([1, 2, 3, 4, 5])
shape_1 = tensor_row0_dim1.shape

class TestSlice:
    @pytest.mark.parametrize("input_slice, expected_shape", [
        ([0,], shape_2),
        ((1,), shape_2),
        (torch.tensor([0,]), shape_2),
        (np.array([0,]), shape_2),
        (0, shape_1),
        (torch.tensor(0), shape_1),
        (None, shape),
    ])
    def test_modularity_shape(self, input_slice, expected_shape):
        slc = utils.Slice(input_slice=input_slice)
        sliced_tensor = slc.apply(ref_tensor)
        assert sliced_tensor.shape == expected_shape

    @pytest.mark.parametrize("input_slice, expected_tensor", [
        ([0,], tensor_row0_dim2),
        (torch.tensor([0,]), tensor_row0_dim2),
        (np.array([0,]), tensor_row0_dim1),
        (0, tensor_row0_dim1),
        (torch.tensor(0), tensor_row0_dim1),
        (None, ref_tensor),
    ])
    def test_modularity_tensor(self, input_slice, expected_tensor):
        slc = utils.Slice(input_slice=input_slice)
        sliced_tensor = slc.apply(ref_tensor)
        assert (sliced_tensor == expected_tensor).all()
    
    @pytest.mark.parametrize("input_slice, expected_indices", [
        ([0,1], np.array([0, 1])),
        (0, np.array(0)),
        (torch.tensor(0), np.array(0)),
    ])
    def test_indices(self, input_slice, expected_indices):
        slc = utils.Slice(input_slice=input_slice)
        indices = slc.indices(2)
        assert (indices == expected_indices).all()

    def test_indices_error(self):
        with pytest.raises(ValueError):
            _ = utils.Slice(input_slice=[0, 2, 5]).indices()
