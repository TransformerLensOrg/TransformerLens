"""Tests for get_params_util helper functions."""
from unittest.mock import Mock

import torch

from transformer_lens.model_bridge.get_params_util import _tensor_attr


class TestTensorAttr:
    def test_returns_first_tensor_among_names(self):
        obj = Mock()
        obj.w = torch.ones(3)
        obj.weight = torch.zeros(3)
        assert torch.equal(_tensor_attr(obj, "w", "weight"), torch.ones(3))

    def test_falls_through_non_tensor_values(self):
        # Mock auto-attributes return Mocks, which must not be mistaken for weights.
        obj = Mock()
        obj.weight = torch.full((2,), 5.0)
        result = _tensor_attr(obj, "w", "weight")
        assert torch.equal(result, torch.full((2,), 5.0))

    def test_none_object_returns_none(self):
        assert _tensor_attr(None, "weight") is None

    def test_missing_and_none_attrs_return_none(self):
        class Holder:
            bias = None

        assert _tensor_attr(Holder(), "nonexistent", "bias") is None

    def test_property_raising_is_skipped(self):
        class Flaky:
            @property
            def w(self):
                raise AttributeError("not materialized")

            weight = torch.ones(2)

        assert torch.equal(_tensor_attr(Flaky(), "w", "weight"), torch.ones(2))
