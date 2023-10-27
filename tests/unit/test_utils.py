from unittest import mock

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
    and the first answer token should not be turned into one token (e.g. 'circ' and 'le' don't become 'circle'). See https://github.com/neelnanda-io/TransformerLens/issues/271
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


def test_override_or_use_default_value():
    # Case when override is not None
    assert utils.override_or_use_default_value(default_flag=True, override=True) == True
    assert (
        utils.override_or_use_default_value(default_flag=True, override=False) == False
    )
    assert (
        utils.override_or_use_default_value(default_flag=False, override=True) == True
    )
    assert (
        utils.override_or_use_default_value(default_flag=False, override=False) == False
    )

    # Case when override is None
    assert utils.override_or_use_default_value(default_flag=True, override=None) == True
    assert (
        utils.override_or_use_default_value(default_flag=False, override=None) == False
    )

    # Case when override is not passed
    assert utils.override_or_use_default_value(default_flag=True) == True
    assert utils.override_or_use_default_value(default_flag=False) == False


class TestAttentionMask:
    prompts = [
        "Hello world!",
        "How are you today?",
        "I'm fine, thank you.",
        "I am happy.",
    ]

    prompts_with_sep = [
        "I like cats<|endoftext|>Cats are so cute",
        "Hello world!",
        "How are you<|endoftext|>I am fine, thanks",
    ]

    # fixtures
    @pytest.fixture(scope="class", params=["gpt2-small", "facebook/opt-125m"])
    def model_name(self, request):
        return request.param

    @pytest.fixture(scope="class")
    def model(self, model_name):
        return HookedTransformer.from_pretrained(model_name)

    # tests
    @pytest.mark.parametrize("padding_side", ["left", "right"])
    @pytest.mark.parametrize("prepend_bos", [True, False])
    @pytest.mark.parametrize("prompts_with_sep", [True, False])
    def test_get_attention_mask(
        self, model, padding_side, prepend_bos, prompts_with_sep
    ):
        # setup
        model.tokenizer.padding_side = padding_side
        model.tokenizer.sep_token_id = model.tokenizer.pad_token_id
        prepend_bos = prepend_bos

        prompts = self.prompts_with_sep if prompts_with_sep else self.prompts
        tokens = model.to_tokens(prompts, prepend_bos=prepend_bos)

        attention_mask = utils.get_attention_mask(
            model.tokenizer, tokens, prepend_bos=prepend_bos
        )  # [batch pos]

        # dimension should be the same
        assert attention_mask.shape == tokens.shape

        # number of attended tokens for each sequence
        # should be the same as the number of 1s in the attention mask for that sequence
        str_tokens = model.to_str_tokens(prompts, prepend_bos=prepend_bos)
        intended_num_attended_tokens = torch.tensor(
            [len(t) for t in str_tokens], device=attention_mask.device
        )
        assert (intended_num_attended_tokens == attention_mask.sum(dim=1)).all()

        # all the masked tokens should be the padding token
        assert (tokens[attention_mask == 0] == model.tokenizer.pad_token_id).all()

        if padding_side == "right":
            # the first token is always attended
            assert (attention_mask[:, 0] == 1).all()

            # attended tokens are at the beginning of the sequence
            for i, num in enumerate(intended_num_attended_tokens.tolist()):
                assert (attention_mask[i, 0:num] == 1).all()

        else:  # left padding case
            # the last token is always attended
            assert (attention_mask[:, -1] == 1).all()

            # attended tokens are at the end of the sequence
            for i, num in enumerate(intended_num_attended_tokens.tolist()):
                assert (attention_mask[i, -num:] == 1).all()

        # the following tests make sense only when the prompts do not contain the separator token
        if not prompts_with_sep:
            non_pad_token_mask = (tokens != model.tokenizer.pad_token_id).int()
            attended_but_non_pad_mask = attention_mask != non_pad_token_mask
            if model.tokenizer.bos_token == model.tokenizer.pad_token and prepend_bos:
                # if bos_token is the same as pad_token and prepend_bos is True,
                # then there is one attended but non-pad token (bos token) in each sequence
                assert attended_but_non_pad_mask.sum() == tokens.shape[0]
            else:
                # otherwise, there should be no attended but non-pad token
                assert attended_but_non_pad_mask.sum() == 0
