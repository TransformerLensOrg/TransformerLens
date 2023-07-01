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


@pytest.mark.parametrize(
        "prepend_space_to_answer, expected_console_output",
        [(
            True,
            """Tokenized prompt: ['<|BOS|>', 'The', ' circumference', ' is', ' the', ' perimeter', ' of', ' the', ' circ']
Tokenized answer: [' le', '.']
Performance on answer token:
Rank: 5284     Logit:  3.81 Prob:  0.00% Token: | le|
Top 0th token. Logit: 16.69 Prob: 38.00% Token: |a|
Top 1th token. Logit: 15.71 Prob: 14.30% Token: |let|
Top 2th token. Logit: 15.48 Prob: 11.37% Token: |ump|
Top 3th token. Logit: 14.94 Prob:  6.64% Token: |u|
Top 4th token. Logit: 13.96 Prob:  2.49% Token: |lets|
Top 5th token. Logit: 13.23 Prob:  1.19% Token: |ut|
Top 6th token. Logit: 13.17 Prob:  1.13% Token: |umn|
Top 7th token. Logit: 13.01 Prob:  0.96% Token: |us|
Top 8th token. Logit: 12.96 Prob:  0.91% Token: | is|
Top 9th token. Logit: 12.89 Prob:  0.85% Token: |umb|
Performance on answer token:
Rank: 340      Logit:  8.94 Prob:  0.01% Token: |.|
Top 0th token. Logit: 16.19 Prob: 16.71% Token: |vy|
Top 1th token. Logit: 15.80 Prob: 11.24% Token: |vers|
Top 2th token. Logit: 15.33 Prob:  7.03% Token: |aps|
Top 3th token. Logit: 14.63 Prob:  3.48% Token: |vens|
Top 4th token. Logit: 14.62 Prob:  3.45% Token: |av|
Top 5th token. Logit: 14.43 Prob:  2.87% Token: |opard|
Top 6th token. Logit: 14.30 Prob:  2.52% Token: |as|
Top 7th token. Logit: 14.26 Prob:  2.41% Token: |ew|
Top 8th token. Logit: 14.23 Prob:  2.33% Token: |on|
Top 9th token. Logit: 13.98 Prob:  1.82% Token: |gged|
Ranks of the answer tokens: [(' le', 5284), ('.', 340)]
"""
        ), (
            False,
            """Tokenized prompt: ['<|BOS|>', 'The', ' circumference', ' is', ' the', ' perimeter', ' of', ' the', ' circ']
Tokenized answer: ['le', '.']
Performance on answer token:
Rank: 93       Logit:  9.99 Prob:  0.05% Token: |le|
Top 0th token. Logit: 16.69 Prob: 38.00% Token: |a|
Top 1th token. Logit: 15.71 Prob: 14.30% Token: |let|
Top 2th token. Logit: 15.48 Prob: 11.37% Token: |ump|
Top 3th token. Logit: 14.94 Prob:  6.64% Token: |u|
Top 4th token. Logit: 13.96 Prob:  2.49% Token: |lets|
Top 5th token. Logit: 13.23 Prob:  1.19% Token: |ut|
Top 6th token. Logit: 13.17 Prob:  1.13% Token: |umn|
Top 7th token. Logit: 13.01 Prob:  0.96% Token: |us|
Top 8th token. Logit: 12.96 Prob:  0.91% Token: | is|
Top 9th token. Logit: 12.89 Prob:  0.85% Token: |umb|
Performance on answer token:
Rank: 2        Logit: 12.46 Prob:  2.60% Token: |.|
Top 0th token. Logit: 15.30 Prob: 44.91% Token: |th|
Top 1th token. Logit: 12.67 Prob:  3.22% Token: | 1|
Top 2th token. Logit: 12.46 Prob:  2.60% Token: |.|
Top 3th token. Logit: 12.33 Prob:  2.30% Token: | 2|
Top 4th token. Logit: 11.98 Prob:  1.62% Token: |,|
Top 5th token. Logit: 11.85 Prob:  1.41% Token: |-|
Top 6th token. Logit: 11.79 Prob:  1.33% Token: | and|
Top 7th token. Logit: 11.62 Prob:  1.13% Token: | 3|
Top 8th token. Logit: 11.56 Prob:  1.06% Token: |thal|
Top 9th token. Logit: 11.48 Prob:  0.98% Token: |an|
Ranks of the answer tokens: [('le', 93), ('.', 2)]
"""
        )]
    )
def test_test_prompt(
    prepend_space_to_answer, 
    expected_console_output,
    capfd 
):
    utils.test_prompt(
        "The circumference is the perimeter of the circ", 
        "le.", 
        model,
        prepend_space_to_answer=prepend_space_to_answer
    )
    out, err = capfd.readouterr()
    assert out == expected_console_output
