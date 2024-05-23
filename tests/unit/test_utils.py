import numpy as np
import pytest
import torch
from torch import nn

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


def test_override_or_use_default_value():
    # Case when override is not None
    assert utils.override_or_use_default_value(default_flag=True, override=True) == True
    assert utils.override_or_use_default_value(default_flag=True, override=False) == False
    assert utils.override_or_use_default_value(default_flag=False, override=True) == True
    assert utils.override_or_use_default_value(default_flag=False, override=False) == False

    # Case when override is None
    assert utils.override_or_use_default_value(default_flag=True, override=None) == True
    assert utils.override_or_use_default_value(default_flag=False, override=None) == False

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
    def test_get_attention_mask(self, model, padding_side, prepend_bos, prompts_with_sep):
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


def test_calc_fan_in_fan_out():
    """
    Test for the calc_fan_in_and_fan_out function in the utils module.
    """
    # Test for the case when the tensor is 1D
    tensor_1d = torch.tensor([1, 2, 3, 4, 5])
    fan_in, fan_out = utils.calc_fan_in_and_fan_out(tensor_1d)
    assert fan_in == 1
    assert fan_out == 5

    # Test for the case when the tensor is 2D
    tensor_2d = torch.tensor([[1, 2, 3], [4, 5, 6]])
    fan_in, fan_out = utils.calc_fan_in_and_fan_out(tensor_2d)
    assert fan_in == 2
    assert fan_out == 3

    # Test for the case when the tensor is 3D
    tensor_3d = nn.Parameter(torch.rand(2, 25, 5))  # 2 x 25 x 5, I'm not writing this out
    fan_in, fan_out = utils.calc_fan_in_and_fan_out(tensor_3d)
    assert fan_in == 25
    assert fan_out == 10

    # Test for the case when the tensor is 4D (should raise a ValueError)
    tensor_4d = torch.tensor([[[[1, 2], [3, 4]], [[5, 6], [7, 8]]]])
    with pytest.raises(ValueError):
        fan_in, fan_out = utils.calc_fan_in_and_fan_out(tensor_4d)

    # Test for the case when the tensor is 0D (also should raise a ValueError)
    tensor_0d = torch.tensor(1)
    with pytest.raises(ValueError):
        fan_in, fan_out = utils.calc_fan_in_and_fan_out(tensor_0d)


class TestInitKaiming:
    """Test cases for kaiming init."""

    @pytest.mark.parametrize(
        "d_model", [4096, 10_000]
    )  # this needs to be large so std and min/max estimates are accurate
    @pytest.mark.parametrize("d_mlp", [256, 512])
    @pytest.mark.parametrize("nonlinearity", ["linear", "relu"])
    def test_init_kaiming_uniform(self, d_model, d_mlp, nonlinearity):
        """
        Test init_kaiming_uniform function in the utils module on 3/2/1D tensors.
        """
        torch.manual_seed(1234)

        gain = np.sqrt(2.0) if nonlinearity == "relu" else 1.0

        x = nn.Parameter(torch.empty(2, d_model, 137))  # n_head and d_head don't matter
        utils.init_kaiming_uniform_(x, nonlinearity=nonlinearity)
        std = gain / np.sqrt(d_model)
        assert np.isclose(x.std().detach().numpy(), std, rtol=1e-2)
        # for uniform distributions, min/max is sqrt(3) times the std
        assert np.isclose(x.max().detach().numpy(), np.sqrt(3) * std, rtol=1e-2)
        assert np.isclose(x.min().detach().numpy(), -np.sqrt(3) * std, rtol=1e-2)

        y = nn.Parameter(torch.empty(d_mlp, d_model))
        utils.init_kaiming_uniform_(y, nonlinearity=nonlinearity)
        std = gain / np.sqrt(d_mlp)
        assert np.isclose(y.std().detach().numpy(), std, rtol=1e-2)
        # for uniform distributions, min/max is sqrt(3) times the std
        assert np.isclose(y.max().detach().numpy(), np.sqrt(3) * std, rtol=1e-2)
        assert np.isclose(y.min().detach().numpy(), -np.sqrt(3) * std, rtol=1e-2)

        z = nn.Parameter(torch.empty(d_model * 123))
        utils.init_kaiming_uniform_(z, nonlinearity=nonlinearity)
        std = gain  # bias has fan_in 1
        assert np.isclose(z.std().detach().numpy(), std, rtol=1e-2)
        # for uniform distributions, min/max is sqrt(3) times the std
        assert np.isclose(z.max().detach().numpy(), np.sqrt(3) * std, rtol=1e-2)
        assert np.isclose(z.min().detach().numpy(), -np.sqrt(3) * std, rtol=1e-2)

        torch.manual_seed(1234)
        x_new = nn.Parameter(torch.empty(2, d_model, 137))
        utils.init_kaiming_uniform_(x_new, nonlinearity=nonlinearity)
        assert torch.allclose(x_new, x, rtol=1e-2)

    @pytest.mark.parametrize("d_model", [4096, 10_000])
    @pytest.mark.parametrize("d_mlp", [256, 512])
    @pytest.mark.parametrize("nonlinearity", ["linear", "relu"])
    def test_init_kaiming_normal(self, d_model, d_mlp, nonlinearity):
        """
        Test init_kaiming_normal function in the utils module on 3/2/1D tensors.
        """
        torch.manual_seed(1234)

        gain = np.sqrt(2.0) if nonlinearity == "relu" else 1.0

        x = nn.Parameter(torch.empty(2, d_model, 137))
        utils.init_kaiming_normal_(x, nonlinearity=nonlinearity)
        std = gain / np.sqrt(d_model)
        assert np.isclose(x.std().detach().numpy(), std, rtol=1e-2)

        y = nn.Parameter(torch.empty(d_mlp, d_model))
        utils.init_kaiming_normal_(y, nonlinearity=nonlinearity)
        std = gain / np.sqrt(d_mlp)
        assert np.isclose(y.std().detach().numpy(), std, rtol=1e-2)

        z = nn.Parameter(torch.empty(d_model * 123))
        utils.init_kaiming_normal_(z, nonlinearity=nonlinearity)
        std = gain  # bias has fan_in 1
        assert np.isclose(z.std().detach().numpy(), std, rtol=1e-2)

        torch.manual_seed(1234)
        x_new = nn.Parameter(torch.empty(2, d_model, 137))
        utils.init_kaiming_normal_(x_new, nonlinearity=nonlinearity)
        assert torch.allclose(x_new, x, rtol=1e-2)


class TestInitXavier:
    """Test cases for Xavier init. Std of distribution should be scaled to sqrt(2/(fan_in + fan_out))."""

    @pytest.mark.parametrize("d_model", [4096, 10_000])
    @pytest.mark.parametrize("d_mlp", [256, 512])
    def test_init_xavier_uniform(self, d_model, d_mlp):
        """Test init_xavier_uniform function in the utils module on 3/2/1D tensors."""
        torch.manual_seed(1234)

        x = nn.Parameter(torch.empty(2, d_model, 137))
        utils.init_xavier_uniform_(x)
        std = np.sqrt(2 / (d_model + 137 * 2))
        assert np.isclose(x.std().detach().numpy(), std, rtol=1e-2)
        # for uniform distributions, min/max is sqrt(3) times the std
        assert np.isclose(x.max().detach().numpy(), np.sqrt(3) * std, rtol=1e-2)
        assert np.isclose(x.min().detach().numpy(), -np.sqrt(3) * std, rtol=1e-2)

        y = nn.Parameter(torch.empty(d_mlp, d_model))
        utils.init_xavier_uniform_(y)
        std = np.sqrt(2 / (d_mlp + d_model))
        assert np.isclose(y.std().detach().numpy(), std, rtol=1e-2)
        # for uniform distributions, min/max is sqrt(3) times the std
        assert np.isclose(y.max().detach().numpy(), np.sqrt(3) * std, rtol=1e-2)
        assert np.isclose(y.min().detach().numpy(), -np.sqrt(3) * std, rtol=1e-2)

        z = nn.Parameter(torch.empty(d_model * 123))
        utils.init_xavier_uniform_(z)
        std = np.sqrt(2 / (1 + d_model * 123))
        assert np.isclose(z.std().detach().numpy(), std, rtol=1e-2)
        # for uniform distributions, min/max is sqrt(3) times the std
        assert np.isclose(z.max().detach().numpy(), np.sqrt(3) * std, rtol=1e-2)
        assert np.isclose(z.min().detach().numpy(), -np.sqrt(3) * std, rtol=1e-2)

        torch.manual_seed(1234)
        x_new = nn.Parameter(torch.empty(2, d_model, 137))
        utils.init_xavier_uniform_(x_new)
        assert torch.allclose(x_new, x, rtol=1e-2)

    @pytest.mark.parametrize("d_model", [4096, 10_000])
    @pytest.mark.parametrize("d_mlp", [256, 512])
    def test_init_xavier_normal(self, d_model, d_mlp):
        """Test init_xavier_normal function in the utils module on 3/2/1D tensors."""
        torch.manual_seed(1234)

        x = nn.Parameter(torch.empty(2, d_model, 137))
        utils.init_xavier_normal_(x)
        std = np.sqrt(2 / (d_model + 137 * 2))
        assert np.isclose(x.std().detach().numpy(), std, rtol=1e-2)

        y = nn.Parameter(torch.empty(d_mlp, d_model))
        utils.init_xavier_normal_(y)
        std = np.sqrt(2 / (d_mlp + d_model))
        assert np.isclose(y.std().detach().numpy(), std, rtol=1e-2)

        z = nn.Parameter(torch.empty(d_model * 123))  # need to make this larger so std is accurate
        utils.init_xavier_normal_(z)
        std = np.sqrt(2 / (1 + d_model * 123))
        assert np.isclose(z.std().detach().numpy(), std, rtol=1e-2)

        torch.manual_seed(1234)
        x_new = nn.Parameter(torch.empty(2, d_model, 137))
        utils.init_xavier_normal_(x_new)
        assert torch.allclose(x_new, x, rtol=1e-2)
