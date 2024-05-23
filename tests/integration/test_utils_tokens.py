from unittest import mock

import numpy as np
import pytest
import torch

import transformer_lens.utils as utils
from transformer_lens import HookedTransformer

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


@pytest.mark.parametrize(
    "prepend_space_to_answer, tokenized_prompt, tokenized_answer",
    [
        (
            True,
            [
                "<|BOS|>",
                "The",
                " circumference",
                " is",
                " the",
                " perimeter",
                " of",
                " the",
                " circ",
            ],
            [" le", "."],
        ),
        (
            False,
            [
                "<|BOS|>",
                "The",
                " circumference",
                " is",
                " the",
                " perimeter",
                " of",
                " the",
                " circ",
            ],
            ["le", "."],
        ),
    ],
)
@mock.patch("builtins.print")
def test_test_prompt(
    mocked_print,
    prepend_space_to_answer,
    tokenized_prompt,
    tokenized_answer,
):
    """
    Tests that utils.test_prompt produces the correct tokenization. In particular, when prepend_space_to_answer = False, the last token of the prompt
    and the first answer token should not be turned into one token (e.g. 'circ' and 'le' don't become 'circle'). See https://github.com/TransformerLensOrg/TransformerLens/issues/271
    for a more detailed explanation.
    """
    utils.test_prompt(
        "The circumference is the perimeter of the circ",
        "le.",
        model,
        prepend_space_to_answer=prepend_space_to_answer,
    )

    printed_tokenized_prompt = mock.call("Tokenized prompt:", tokenized_prompt)
    printed_tokenized_answer = mock.call("Tokenized answer:", tokenized_answer)

    assert mocked_print.mock_calls[0] == printed_tokenized_prompt
    assert mocked_print.mock_calls[1] == printed_tokenized_answer
