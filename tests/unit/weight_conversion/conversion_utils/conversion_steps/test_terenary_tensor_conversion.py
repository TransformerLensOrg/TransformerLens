import torch

from transformer_lens.conversion_utils.conversion_steps.base_tensor_conversion import (
    BaseTensorConversion,
)
from transformer_lens.conversion_utils.conversion_steps.ternary_tensor_conversion import (
    TernaryTensorConversion,
)

###################################
# Mock Conversions for Testing
###################################


class MockPrimaryConversion(BaseTensorConversion):
    """
    A primary conversion class that modifies the input tensor
    (e.g., adds 42) to show it's been called.
    """

    def handle_conversion(self, input_value: torch.Tensor, **unused) -> torch.Tensor:
        return input_value + 42


class MockFallbackConversion(BaseTensorConversion):
    """
    A fallback conversion that returns a known tensor if input_value is None,
    or modifies the existing tensor if not None.
    """

    def __init__(self):
        super().__init__()
        self.was_called = False

    def handle_conversion(self, input_value: torch.Tensor | None, **unused) -> torch.Tensor:
        self.was_called = True
        if input_value is None:
            return torch.tensor([999.0])
        else:
            return input_value + 999


###################################
# Ternary Tests
###################################


def test_ternary_tensor_conversion_primary_none():
    """
    If primary_conversion is None, it should just return the input_value unmodified.
    """
    fallback = torch.tensor([10.0])  # Not used if input_value is not None
    ternary = TernaryTensorConversion(fallback_conversion=fallback, primary_conversion=None)
    input_tensor = torch.tensor([1.0, 2.0])
    output_tensor = ternary.convert(input_tensor)
    assert torch.allclose(
        output_tensor, input_tensor
    ), "Expected primary=None to return input_value as is."


def test_ternary_tensor_conversion_primary_tensor():
    """
    If primary_conversion is a torch.Tensor, ignore input_value and return that tensor.
    """
    fallback = torch.tensor([10.0])  # Not used if input_value is not None
    primary_tensor = torch.tensor([100.0, 200.0])
    ternary = TernaryTensorConversion(
        fallback_conversion=fallback, primary_conversion=primary_tensor
    )
    input_tensor = torch.tensor([1.0, 2.0])
    output_tensor = ternary.convert(input_tensor)
    assert torch.allclose(
        output_tensor, primary_tensor
    ), "Expected to return primary tensor, ignoring input_value."


def test_ternary_tensor_conversion_primary_conversion_class():
    """
    If primary_conversion is a BaseTensorConversion, .convert() should be called.
    """
    fallback = torch.tensor([10.0])
    mock_primary = MockPrimaryConversion()
    ternary = TernaryTensorConversion(fallback_conversion=fallback, primary_conversion=mock_primary)
    input_tensor = torch.tensor([1.0, 2.0])
    output_tensor = ternary.convert(input_tensor)

    # MockPrimaryConversion adds 42
    expected = input_tensor + 42
    assert torch.allclose(output_tensor, expected)


def test_ternary_tensor_conversion_fallback_tensor():
    """
    If input_value is None and fallback is a Tensor, return that tensor directly.
    """
    fallback_tensor = torch.tensor([999.0])
    ternary = TernaryTensorConversion(fallback_conversion=fallback_tensor, primary_conversion=None)
    # Trigger fallback by passing input_value=None
    output = ternary.convert(None)
    assert torch.allclose(output, fallback_tensor), "Expected fallback tensor to be returned."


def test_ternary_tensor_conversion_fallback_str():
    """
    If input_value is None and fallback is a str, we do a context lookup
    with find_context_field().
    """
    # Suppose the context dict has "my_remote_field" -> [123.0, 456.0]
    ternary = TernaryTensorConversion(
        fallback_conversion="my_remote_field",  # str fallback
    )

    # We'll pass a dictionary as context
    context = {"my_remote_field": torch.tensor([123.0, 456.0])}
    # Convert(None, context=...) => fallback => find_context_field("my_remote_field") => returns [123.0, 456.0]
    output = ternary.convert(None, context)

    assert torch.allclose(
        output, torch.tensor([123.0, 456.0])
    ), "Expected fallback str to be found in context dictionary."


def test_ternary_tensor_conversion_fallback_tuple():
    """
    If fallback is a tuple (REMOTE_FIELD, BaseTensorConversion),
    we do a context lookup for the REMOTE_FIELD,
    then pass that to the second item, which is a conversion class.
    """

    class MockTupleConversion(BaseTensorConversion):
        def __init__(self):
            super().__init__()
            self.was_called = False

        def handle_conversion(self, input_value: torch.Tensor, *unused) -> torch.Tensor:
            self.was_called = True
            return input_value + 1000

    mock_fallback_conv = MockTupleConversion()
    fallback_tuple = ("my_backup_field", mock_fallback_conv)  # type: ignore

    ternary = TernaryTensorConversion(fallback_conversion=fallback_tuple, primary_conversion=None)

    # Suppose the context dictionary has "my_backup_field" => [5,6,7]
    context = {"my_backup_field": torch.tensor([5.0, 6.0, 7.0])}

    output = ternary.convert(None, context)

    assert mock_fallback_conv.was_called, "Expected fallback conversion in the tuple to be called."
    # The fallback conv adds 1000
    assert torch.allclose(output, torch.tensor([1005.0, 1006.0, 1007.0]))


def test_ternary_tensor_conversion_find_context_field_failure():
    """
    If fallback is a str, but the context doesn't contain that key,
    find_context_field returns None, and so fallback is effectively None =>
    This might cause an error or we accept returning None.
    We'll see how your code handles that scenario.
    """
    ternary = TernaryTensorConversion(fallback_conversion="missing_key", primary_conversion=None)
    # No context => won't find 'missing_key'
    output = ternary.convert(None, *{})
    # According to your code, if nothing is found, we return None from find_context_field.
    # handle_fallback_conversion will return that None from the function.
    # => This might break if the calling code expects a tensor.
    # We'll just check it's None.
    assert output is None, "Expected None if the fallback str wasn't found in the provided context."


########################################
# Tests for Input/Output Filters
########################################


def test_ternary_input_filter():
    """
    If there's an input_filter, it should run before handle_primary_conversion.
    """

    def input_filter(x: torch.Tensor) -> torch.Tensor:
        return x * 10

    mock_primary = MockPrimaryConversion()  # adds 42
    ternary = TernaryTensorConversion(
        fallback_conversion=torch.tensor([999.0]),
        primary_conversion=mock_primary,
        input_filter=input_filter,
    )
    # Steps: input_filter => *10 => [10,20], primary => +42 => [52,62]
    inp = torch.tensor([1.0, 2.0])
    out = ternary.convert(inp)
    expected = torch.tensor([52.0, 62.0])
    assert torch.allclose(out, expected)


def test_ternary_output_filter():
    """
    Output filter is applied after the main logic,
    i.e. after handle_primary_conversion or fallback.
    """

    def output_filter(x: torch.Tensor) -> torch.Tensor:
        return x - 100

    mock_primary = MockPrimaryConversion()  # adds 42
    ternary = TernaryTensorConversion(
        fallback_conversion=torch.tensor([999.0]),
        primary_conversion=mock_primary,
        output_filter=output_filter,
    )
    # primary => +42 => [43,44], then output_filter => -100 => [-57, -56]
    inp = torch.tensor([1.0, 2.0])
    out = ternary.convert(inp)
    expected = torch.tensor([-57.0, -56.0])
    assert torch.allclose(out, expected), "Output filter didn't apply after primary conversion."


def test_ternary_both_filters():
    """
    Combination: input_filter -> primary -> output_filter
    """

    def input_filter(x: torch.Tensor) -> torch.Tensor:
        return x * 3

    def output_filter(x: torch.Tensor) -> torch.Tensor:
        return x + 5

    mock_primary = MockPrimaryConversion()  # +42
    ternary = TernaryTensorConversion(
        fallback_conversion=("backup_field", MockFallbackConversion()),
        primary_conversion=mock_primary,
        input_filter=input_filter,
        output_filter=output_filter,
    )
    # Steps:
    #   1) input_filter => *3 => [3,6]
    #   2) primary => +42 => [45,48]
    #   3) output_filter => +5 => [50,53]
    inp = torch.tensor([1.0, 2.0])
    out = ternary.convert(inp)
    expected = torch.tensor([50.0, 53.0])
    assert torch.allclose(out, expected), "Filter pipeline or primary logic is off."


def test_ternary_repr():
    """
    Check that __repr__ includes mention of primary and fallback.
    """
    mock_primary = MockPrimaryConversion()
    fallback_tuple = ("my_field", MockFallbackConversion())  # type: ignore
    ternary = TernaryTensorConversion(
        fallback_conversion=fallback_tuple, primary_conversion=mock_primary
    )
    rep = repr(ternary).lower()
    assert "ternary operation" in rep, f"Expected mention of 'ternary operation', got {rep}"
    assert "primary conversion" in rep, f"Expected mention of 'primary conversion', got {rep}"
    assert "fallback conversion" in rep, f"Expected mention of 'fallback conversion', got {rep}"
