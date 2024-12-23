from unittest import mock

import torch

from transformer_lens.weight_conversion.conversion_utils.model_search import (
    find_property,
)


def test_find_property():
    expected = torch.rand((1, 1, 1))

    transformers_model = mock.Mock()
    transformers_model.transformer = mock.Mock()
    transformers_model.transformer.wpe = mock.Mock()
    transformers_model.transformer.wpe.weight = expected

    result = find_property("transformer.wpe.weight", transformers_model)

    assert result == expected
