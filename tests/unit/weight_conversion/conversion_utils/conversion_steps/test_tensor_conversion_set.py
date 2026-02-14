import torch

from transformer_lens.conversion_utils.conversion_steps.base_tensor_conversion import (
    BaseTensorConversion,
)

# These imports reflect your code structure
from transformer_lens.conversion_utils.conversion_steps.tensor_conversion_set import (
    TensorConversionSet,
)


class MockSubConversion(BaseTensorConversion):
    """
    A mock sub-conversion that emulates deeper logic.
    For demonstration, it just adds 10 to each element if given a tensor,
    or returns a known tensor if input is None.
    """

    def __init__(self):
        super().__init__()
        self.was_called = False

    def handle_conversion(self, input_value, *full_context):
        self.was_called = True
        if isinstance(input_value, torch.Tensor):
            return input_value + 10
        else:
            return torch.tensor([999.0])


def test_tensor_conversion_set_basic_tensor_and_str():
    """
    Tests a field set that includes a direct Tensor and a str (property).
    Ensures handle_conversion returns a dict with the correct resolved values.
    """
    # We'll pass 'pos_embed.weight' as a string property to find in an input structure
    field_set = {
        "embed.W_E": torch.ones(2, 2),  # direct tensor
        "pos_embed": "pos_embed.weight",  # property to look up in input_value
    }

    conversion_set = TensorConversionSet(fields=field_set)

    # Suppose the input_value is a dictionary or object that find_property can parse
    input_value = {
        "pos_embed": {"weight": torch.tensor([3.0, 4.0])}  # the property we want to find
    }

    output = conversion_set.convert(input_value)

    # 1) Check embed.W_E is the same tensor
    assert "embed.W_E" in output
    assert torch.allclose(output["embed.W_E"], torch.ones(2, 2))

    # 2) Check pos_embed is the retrieved property
    assert "pos_embed" in output
    assert torch.allclose(output["pos_embed"], torch.tensor([3.0, 4.0]))


def test_tensor_conversion_set_tuple_subconversion():
    """
    Tests a field set containing a tuple (remote_field, conversion).
    The sub-conversion is a mock that we ensure gets called with the correct input.
    """
    mock_sub = MockSubConversion()
    field_set = {"layer_0_attn": ("attn_proj.weight", mock_sub)}
    conversion_set = TensorConversionSet(fields=field_set)

    # We'll store "attn_proj.weight" in our input_value so find_property can retrieve it
    input_value = {"attn_proj": {"weight": torch.tensor([1.0, 2.0])}}

    output = conversion_set.convert(input_value)
    assert "layer_0_attn" in output, "Expected output key for the field set."

    # The sub-conversion 'MockSubConversion' adds 10
    expected_tensor = torch.tensor([1.0, 2.0]) + 10
    result_tensor = output["layer_0_attn"]

    assert mock_sub.was_called, "Expected mock_sub to be invoked."
    assert torch.allclose(result_tensor, expected_tensor), f"Expected [11, 12], got {result_tensor}"


def test_tensor_conversion_set_process_conversion_action_tensor():
    """
    Tests process_conversion_action for a direct tensor.
    """
    field_set = {"some_weight": torch.tensor([5.0])}
    conversion_set = TensorConversionSet(fields=field_set)

    input_value = {}  # not used for direct tensor
    output = conversion_set.convert(input_value)
    assert "some_weight" in output
    assert torch.allclose(output["some_weight"], torch.tensor([5.0]))


def test_tensor_conversion_set_process_conversion_action_str_property():
    """
    Tests process_conversion_action for a str property lookup
    with a nested dict.
    """
    field_set = {"some_str_key": "my_field.data"}
    conversion_set = TensorConversionSet(fields=field_set)

    input_value = {"my_field": {"data": torch.tensor([42.0])}}
    output = conversion_set.convert(input_value)
    assert torch.allclose(output["some_str_key"], torch.tensor([42.0]))


def test_tensor_conversion_set_process_conversion_action_tuple():
    """
    Tests process_conversion_action for a tuple (remote_field, sub_conversion).
    We'll reuse MockSubConversion for demonstration.
    """
    mock_sub = MockSubConversion()
    field_set = {"my_tuple_key": ("fieldA", mock_sub)}
    conversion_set = TensorConversionSet(fields=field_set)

    input_value = {"fieldA": torch.tensor([0.0])}
    output = conversion_set.convert(input_value)

    assert mock_sub.was_called, "Expected the sub-conversion to be used."
    assert torch.allclose(output["my_tuple_key"], torch.tensor([10.0]))


def test_tensor_conversion_set_repr():
    """
    Checks that __repr__ includes a string of the nested conversions,
    based on WeightConversionUtils.create_conversion_string output.
    """
    # We'll just define some fields to see that it formats them
    mock_sub = MockSubConversion()
    field_set = {
        "embed.W_E": torch.zeros(2, 2),
        "pos_embed": "pos_embed.weight",
        "layer_0_attn": ("attn_proj.weight", mock_sub),
    }

    conversion_set = TensorConversionSet(fields=field_set)
    rep_str = repr(conversion_set).lower()

    assert (
        "is composed of a set of nested conversions" in rep_str
    ), f"Expected reference to 'nested conversions', got {rep_str}"
    assert "embed.w_e" in rep_str, "Expected mention of embed.W_E key."
    assert "pos_embed" in rep_str, "Expected mention of pos_embed key."
    assert "layer_0_attn" in rep_str, "Expected mention of layer_0_attn key."
