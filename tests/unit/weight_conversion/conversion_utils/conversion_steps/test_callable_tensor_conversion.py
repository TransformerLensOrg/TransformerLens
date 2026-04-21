import torch

from transformer_lens.conversion_utils.conversion_steps.callable_tensor_conversion import (
    CallableTensorConversion,
)


def test_callable_tensor_conversion_basic():
    """
    Verifies that the given callable is invoked on a dict of tensors
    and the result is returned as expected.
    """

    def my_callable(tensor_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        # Example transformation: add 1 to each tensor value
        new_dict = {}
        for k, v in tensor_dict.items():
            new_dict[k] = v + 1
        return new_dict

    conversion = CallableTensorConversion(my_callable)

    input_data = {
        "a": torch.tensor([1.0, 2.0]),
        "b": torch.tensor([[3.0, 4.0], [5.0, 6.0]]),
    }
    # Original copies to verify no in-place modification
    original_a = input_data["a"].clone()
    original_b = input_data["b"].clone()

    output_data = conversion.convert(input_data)

    # Check the callable was applied
    assert torch.allclose(output_data["a"], input_data["a"] + 1)
    assert torch.allclose(output_data["b"], input_data["b"] + 1)

    # Confirm the original dict is not modified (unless that's desired)
    assert torch.allclose(input_data["a"], original_a), "Input data 'a' was modified in place!"
    assert torch.allclose(input_data["b"], original_b), "Input data 'b' was modified in place!"


def test_callable_tensor_conversion_input_filter():
    """
    Ensures BaseTensorConversion's input_filter is applied
    to the dict of tensors before the main callable runs.
    """

    def my_callable(tensor_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        new_dict = {}
        for k, v in tensor_dict.items():
            new_dict[k] = v + 10
        return new_dict

    def my_input_filter(tensor_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        # Multiply each tensor by 2
        filtered_dict = {}
        for k, v in tensor_dict.items():
            filtered_dict[k] = v * 2
        return filtered_dict

    conversion = CallableTensorConversion(my_callable)
    conversion.input_filter = my_input_filter

    input_data = {
        "x": torch.tensor([1.0, 2.0, 3.0]),
        "y": torch.tensor([4.0]),
    }
    output_data = conversion.convert(input_data)

    # The input_filter doubles each value => x=[2,4,6], y=[8],
    # then my_callable adds 10 => x=[12,14,16], y=[18].
    assert torch.allclose(output_data["x"], torch.tensor([12.0, 14.0, 16.0]))
    assert torch.allclose(output_data["y"], torch.tensor([18.0]))


def test_callable_tensor_conversion_output_filter():
    """
    Ensures BaseTensorConversion's output_filter is applied
    to the dict of tensors after the main callable runs.
    """

    def my_callable(tensor_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        new_dict = {}
        for k, v in tensor_dict.items():
            new_dict[k] = v**2
        return new_dict

    def my_output_filter(tensor_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        # Subtract 5 from each element
        filtered_dict = {}
        for k, v in tensor_dict.items():
            filtered_dict[k] = v - 5
        return filtered_dict

    conversion = CallableTensorConversion(my_callable)
    conversion.output_filter = my_output_filter

    input_data = {
        "first": torch.tensor([2.0, 3.0]),
        "second": torch.tensor([4.0]),
    }
    output_data = conversion.convert(input_data)
    # After callable: first=[4.0,9.0], second=[16.0]
    # After output_filter: first=[-1,4], second=[11]
    assert torch.allclose(output_data["first"], torch.tensor([-1.0, 4.0]))
    assert torch.allclose(output_data["second"], torch.tensor([11.0]))


def test_callable_tensor_conversion_input_and_output_filters():
    """
    Tests that input_filter and output_filter both run in the correct order
    around the callable, for a dict of tensors.
    """

    def my_callable(tensor_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        new_dict = {}
        for k, v in tensor_dict.items():
            new_dict[k] = v + 10
        return new_dict

    def my_input_filter(tensor_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        new_dict = {}
        for k, v in tensor_dict.items():
            new_dict[k] = v * 2
        return new_dict

    def my_output_filter(tensor_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        new_dict = {}
        for k, v in tensor_dict.items():
            new_dict[k] = v - 3
        return new_dict

    conversion = CallableTensorConversion(my_callable)
    conversion.input_filter = my_input_filter
    conversion.output_filter = my_output_filter

    input_data = {
        "stuff": torch.tensor([1.0, 2.0]),
        "things": torch.tensor([3.0]),
    }
    output_data = conversion.convert(input_data)
    # Steps:
    #   1) input_filter => stuff=[2,4], things=[6]
    #   2) callable => +10 => stuff=[12,14], things=[16]
    #   3) output_filter => -3 => stuff=[9,11], things=[13]
    expected_stuff = torch.tensor([9.0, 11.0])
    expected_things = torch.tensor([13.0])
    assert torch.allclose(output_data["stuff"], expected_stuff)
    assert torch.allclose(output_data["things"], expected_things)


def test_callable_tensor_conversion_repr():
    """
    Simple test confirming __repr__ returns the expected info
    about the callable operation.
    """

    def my_callable(tensor_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return tensor_dict

    conversion = CallableTensorConversion(my_callable)
    rep = repr(conversion).lower()
    assert (
        "callable operation" in rep
    ), f"Expected '__repr__' to mention 'callable operation', got '{rep}'"
