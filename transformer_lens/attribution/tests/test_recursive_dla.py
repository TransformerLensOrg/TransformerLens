"""Tests for the Recursive DLA Functionality."""
# pylint: disable=missing-function-docstring,missing-class-docstring
import pytest
import torch
from jaxtyping import Float
from torch import Tensor

from transformer_lens import HookedTransformer
from transformer_lens.attribution.recursive_dla import (
    dla_attn_head_breakdown_source_component,
    pad_tensor_dimension,
)


class TestPadTensorDimension:
    def test_basic_expansion(self):
        x = torch.tensor([[1, 2], [3, 4]])
        expanded = pad_tensor_dimension(x, 1, 5)
        expected = torch.tensor([[1, 2, 0, 0, 0], [3, 4, 0, 0, 0]])
        assert torch.equal(expanded, expected)

    def test_negative_dimension_expansion(self):
        x = torch.tensor([[1, 2], [3, 4]])
        expanded = pad_tensor_dimension(x, -1, 5)
        expected = torch.tensor([[1, 2, 0, 0, 0], [3, 4, 0, 0, 0]])
        assert torch.equal(expanded, expected)

    def test_no_expansion_needed(self):
        x = torch.tensor([[1, 2], [3, 4]])
        expanded = pad_tensor_dimension(x, 1, 2)
        expected = x
        assert torch.equal(expanded, expected)

    def test_invalid_expansion_size(self):
        x = torch.tensor([[1, 2], [3, 4]])
        with pytest.raises(ValueError, match=r"Expansion to size 1 not possible.*"):
            pad_tensor_dimension(x, 1, 1)

    def test_3d_tensor_expansion(self):
        x = torch.tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
        expanded = pad_tensor_dimension(x, 2, 4)
        expected = torch.tensor(
            [[[1, 2, 0, 0], [3, 4, 0, 0]], [[5, 6, 0, 0], [7, 8, 0, 0]]]
        )
        assert torch.equal(expanded, expected)


class TestAttentionHeadBreakdownSourceComponent:
    def matches_current_lib_dla(self):
        # Load model
        torch.set_grad_enabled(False)
        model = HookedTransformer.from_pretrained("tiny-stories-instruct-1M")
        model.set_use_attn_result(True)
        model.eval()

        # Run
        prompt = "Why did the elephant cross the"
        answer = " road"
        _logits, cache = model.run_with_cache(prompt)

        # Get DLA using existing functionality (for all attention heads)
        stacked_heads: Float[
            Tensor, "head batch pos d_model"
        ] = cache.stack_head_results()
        dla_heads: Float[Tensor, "head_idx batch pos"] = cache.logit_attrs(
            stacked_heads, tokens=answer
        )
        existing_dla = dla_heads[:, 0, -1].sum()  # Batch 0, last token

        # Get DLA using new functionality (for all attention heads)
        answer_token = model.tokenizer.encode(answer)
        dla_breakdown = dla_attn_head_breakdown_source_component(
            cache, model, answer_token
        )
        new_dla = dla_breakdown[0].sum(dim=-1).sum()

        assert existing_dla == new_dla
