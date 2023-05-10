import numpy as np
import pytest
import torch

import transformer_lens.utils as utils
from transformer_lens import HookedTransformer

ref_tensor = torch.tensor([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]])
shape = ref_tensor.shape
tensor_row0_dim2 = torch.tensor([[1, 2, 3, 4, 5]])
shape_2 = tensor_row0_dim2.shape
tensor_row0_dim1 = torch.tensor([1, 2, 3, 4, 5])
shape_1 = tensor_row0_dim1.shape


class TestSlice:
    @pytest.mark.parametrize(
        "input_slice, expected_shape",
        [
            (
                [
                    0,
                ],
                shape_2,
            ),
            ((1,), shape_2),
            (
                torch.tensor(
                    [
                        0,
                    ]
                ),
                shape_2,
            ),
            (
                np.array(
                    [
                        0,
                    ]
                ),
                shape_2,
            ),
            (0, shape_1),
            (torch.tensor(0), shape_1),
            (None, shape),
        ],
    )
    def test_modularity_shape(self, input_slice, expected_shape):
        slc = utils.Slice(input_slice=input_slice)
        sliced_tensor = slc.apply(ref_tensor)
        assert sliced_tensor.shape == expected_shape

    @pytest.mark.parametrize(
        "input_slice, expected_tensor",
        [
            (
                [
                    0,
                ],
                tensor_row0_dim2,
            ),
            (
                torch.tensor(
                    [
                        0,
                    ]
                ),
                tensor_row0_dim2,
            ),
            (
                np.array(
                    [
                        0,
                    ]
                ),
                tensor_row0_dim1,
            ),
            (0, tensor_row0_dim1),
            (torch.tensor(0), tensor_row0_dim1),
            (None, ref_tensor),
        ],
    )
    def test_modularity_tensor(self, input_slice, expected_tensor):
        slc = utils.Slice(input_slice=input_slice)
        sliced_tensor = slc.apply(ref_tensor)
        assert (sliced_tensor == expected_tensor).all()

    @pytest.mark.parametrize(
        "input_slice, expected_indices",
        [
            ([0, 1], np.array([0, 1])),
            (0, np.array(0)),
            (torch.tensor(0), np.array(0)),
        ],
    )
    def test_indices(self, input_slice, expected_indices):
        slc = utils.Slice(input_slice=input_slice)
        indices = slc.indices(2)
        assert (indices == expected_indices).all()

    def test_indices_error(self):
        with pytest.raises(ValueError):
            _ = utils.Slice(input_slice=[0, 2, 5]).indices()


MODEL = "solu-1l"

model = HookedTransformer.from_pretrained(MODEL)


@pytest.fixture
def nested_list_1():
    return [1]


@pytest.fixture
def nested_list_1x1():
    return [[6]]


@pytest.fixture
def nested_list_1x3():
    return [[1, 2, 3]]


def test_to_str_tokens(nested_list_1, nested_list_1x1, nested_list_1x3):
    tensor_1_to_str_tokens = model.to_str_tokens(torch.tensor(nested_list_1))
    assert isinstance(tensor_1_to_str_tokens, list)
    assert len(tensor_1_to_str_tokens) == 1
    assert isinstance(tensor_1_to_str_tokens[0], str)

    tensor_1x1_to_str_tokens = model.to_str_tokens(torch.tensor(nested_list_1x1))
    assert isinstance(tensor_1x1_to_str_tokens, list)
    assert len(tensor_1x1_to_str_tokens) == 1
    assert isinstance(tensor_1x1_to_str_tokens[0], str)

    ndarray_1_to_str_tokens = model.to_str_tokens(np.array(nested_list_1))
    assert isinstance(ndarray_1_to_str_tokens, list)
    assert len(ndarray_1_to_str_tokens) == 1
    assert isinstance(ndarray_1_to_str_tokens[0], str)

    ndarray_1x1_to_str_tokens = model.to_str_tokens(np.array(nested_list_1x1))
    assert isinstance(ndarray_1x1_to_str_tokens, list)
    assert len(ndarray_1x1_to_str_tokens) == 1
    assert isinstance(ndarray_1x1_to_str_tokens[0], str)

    single_int_to_single_str_token = model.to_single_str_token(3)
    assert isinstance(single_int_to_single_str_token, str)

    squeezable_tensor_to_str_tokens = model.to_str_tokens(torch.tensor(nested_list_1x3))
    assert isinstance(squeezable_tensor_to_str_tokens, list)
    assert len(squeezable_tensor_to_str_tokens) == 3
    assert isinstance(squeezable_tensor_to_str_tokens[0], str)
    assert isinstance(squeezable_tensor_to_str_tokens[1], str)
    assert isinstance(squeezable_tensor_to_str_tokens[2], str)

    squeezable_ndarray_to_str_tokens = model.to_str_tokens(np.array(nested_list_1x3))
    assert isinstance(squeezable_ndarray_to_str_tokens, list)
    assert len(squeezable_ndarray_to_str_tokens) == 3
    assert isinstance(squeezable_ndarray_to_str_tokens[0], str)
    assert isinstance(squeezable_ndarray_to_str_tokens[1], str)
    assert isinstance(squeezable_ndarray_to_str_tokens[2], str)


class Test_is_square:
    failed_cases = (
        torch.tensor([]),
        torch.tensor(2),
        torch.ones(2, 3),
        torch.zeros(2),
        torch.ones(3, 3, 3),
        torch.ones(1, 1, 1, 1),
    )

    @pytest.mark.parametrize("x", (torch.ones(3, 3), torch.zeros(1, 1)))
    def test_pass(self, x: torch.Tensor):
        assert utils.is_square(x)

    @pytest.mark.parametrize("x", failed_cases)
    def test_fail(self, x: torch.Tensor):
        assert not utils.is_square(x)


class Test_lower_triangular:
    @pytest.mark.parametrize(
        "x",
        (
            torch.eye(4),
            torch.ones(4, 4).tril(),
            torch.ones(4, 4).triu().T,
            torch.zeros(4, 4),
        ),
    )
    def test_pass(self, x: torch.Tensor):
        assert utils.is_lower_triangular(x)

    @pytest.mark.parametrize(
        "x",
        (
            *Test_is_square.failed_cases,
            torch.ones(4, 4).triu(),
            torch.ones(4, 4).tril().T,
            torch.ones(3, 3),
        ),
    )
    def test_fail(self, x: torch.Tensor):
        assert not utils.is_lower_triangular(x)
