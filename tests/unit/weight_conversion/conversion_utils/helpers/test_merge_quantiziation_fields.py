import pytest
import torch

from transformer_lens.conversion_utils.conversion_steps.tensor_conversion_set import (
    TensorConversionSet,
)
from transformer_lens.conversion_utils.helpers.merge_quantiziation_fields import (
    merge_quantization_fields,
)


def test_merge_quantization_fields_simple_overwrite():
    """
    If the new field is a non-tuple (like torch.Tensor) and
    there is an existing field, we simply overwrite the existing field.
    """
    original_fields = {
        "layer_0": torch.tensor([1.0]),
        "layer_1": torch.tensor([2.0]),
    }
    new_fields = {
        "layer_0": torch.tensor([999.0]),  # Overwrites existing
    }

    conversion_set = TensorConversionSet(original_fields)
    result = merge_quantization_fields(conversion_set, new_fields)

    # Check we got the same object back
    assert result is conversion_set, "Expected the same TensorConversionSet returned (in-place)."

    # layer_0 overwritten
    assert torch.allclose(result.fields["layer_0"], torch.tensor([999.0]))
    # layer_1 unchanged
    assert torch.allclose(result.fields["layer_1"], torch.tensor([2.0]))


def test_merge_quantization_fields_missing_original_field():
    """
    If existing_field is None, raise a RuntimeError.
    """
    original_fields = {
        "layer_0": None,
    }
    new_fields = {
        "layer_0": torch.tensor([10.0]),
    }
    conversion_set = TensorConversionSet(original_fields)

    with pytest.raises(
        RuntimeError,
        match="Attempted to merge quantization field into existing conversion without original field configured",
    ):
        merge_quantization_fields(conversion_set, new_fields)


def test_merge_quantization_fields_complex_subfield_structure():
    """
    Both existing and new fields are (str, TensorConversionSet).
    We recursively merge subfields in place.
    """
    # Sub-WCS for existing
    existing_subfields = {
        "sub_0": torch.tensor([1.0]),
        "sub_1": torch.tensor([2.0]),
        "sub_2": torch.tensor([3.0]),
    }
    existing_sub_wcs = TensorConversionSet(existing_subfields)

    # Sub-WCS for new, overwrites sub_1, adds sub_2
    new_subfields = {
        "sub_1": torch.tensor([999.0]),
        "sub_2": torch.tensor([123.0]),
    }
    new_sub_wcs = TensorConversionSet(new_subfields)

    original_fields = {
        "layer_0": ("old_remote", existing_sub_wcs),
    }
    new_fields = {
        "layer_0": ("new_remote", new_sub_wcs),
    }

    conversion_set = TensorConversionSet(original_fields)
    merge_quantization_fields(conversion_set, new_fields)

    merged_field = conversion_set.fields["layer_0"]
    assert isinstance(merged_field, tuple), "Expected the merged field to remain a tuple."
    assert isinstance(
        merged_field[1], TensorConversionSet
    ), "Expected the merged field to remain a tuple."
    assert merged_field[0] == "new_remote", "Remote field should be updated."

    merged_sub_wcs = merged_field[1]
    # Check updated sub_1
    assert torch.allclose(merged_sub_wcs.fields["sub_1"], torch.tensor([999.0]))
    # Check newly added sub_2
    assert torch.allclose(merged_sub_wcs.fields["sub_2"], torch.tensor([123.0]))
    # Check original sub_0 remains
    assert torch.allclose(merged_sub_wcs.fields["sub_0"], torch.tensor([1.0]))


def test_merge_quantization_fields_new_field_tuple_existing_not_tuple():
    """
    If new_field is (str, TensorConversionSet), but the existing field
    is not also a tuple with a TensorConversionSet, raise RuntimeError.
    """
    original_fields = {
        "layer_0": torch.tensor([1.0]),
    }
    new_sub_wcs = TensorConversionSet({"sub_0": torch.tensor([2.0])})
    new_fields = {
        "layer_0": ("some_remote", new_sub_wcs),
    }

    conversion_set = TensorConversionSet(original_fields)

    with pytest.raises(
        RuntimeError,
        match="Attempted to merge TensorConversionSet into a field that is not configured as a TensorConversionSet",
    ):
        merge_quantization_fields(conversion_set, new_fields)


def test_merge_quantization_fields_existing_tuple_new_is_not_tuple():
    """
    If the existing field is a tuple with TensorConversionSet, but the new field
    isn't a tuple with a TensorConversionSet, we simply overwrite the entire field.
    """
    existing_sub = TensorConversionSet({"old_sub": torch.tensor([3.0])})
    original_fields = {
        "layer_0": ("old_remote", existing_sub),
    }
    new_fields = {
        "layer_0": torch.tensor([999.0]),
    }

    conversion_set = TensorConversionSet(original_fields)
    ret = merge_quantization_fields(conversion_set, new_fields)
    assert ret is conversion_set
    # Overwritten with [999.0]
    assert torch.allclose(ret.fields["layer_0"], torch.tensor([999.0]))


def test_merge_quantization_fields_returns_same_object():
    """Check the function returns the same TensorConversionSet for in-place merges."""
    original_fields = {"fieldA": torch.tensor([1.0])}
    new_fields = {"fieldA": torch.tensor([2.0])}

    conversion_set = TensorConversionSet(original_fields)
    ret = merge_quantization_fields(conversion_set, new_fields)
    assert ret is conversion_set, "Expected in-place merge to return the same object."
    assert torch.allclose(ret.fields["fieldA"], torch.tensor([2.0]))
