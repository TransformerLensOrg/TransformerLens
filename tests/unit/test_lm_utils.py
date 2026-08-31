"""Unit tests for language-model loss and accuracy helpers."""

from __future__ import annotations

import pytest
import torch
from beartype.roar import BeartypeCallHintParamViolation

from transformer_lens.utilities.lm_utils import lm_accuracy, lm_cross_entropy_loss
from tests.typecheck_errors import TYPECHECK_ERRORS


def test_lm_cross_entropy_loss_rejects_mismatched_attention_mask() -> None:
    logits = torch.zeros(1, 2, 3)
    tokens = torch.tensor([[0, 1]])
    attention_mask = torch.ones(1, 5, dtype=torch.long)

    with pytest.raises(
        (AssertionError, *TYPECHECK_ERRORS),
        match="attention_mask|axis 'pos'",
    ):
        lm_cross_entropy_loss(logits, tokens, attention_mask)


def test_lm_cross_entropy_loss_masks_nan_transition() -> None:
    logits = torch.tensor(
        [
            [
                [torch.nan, torch.nan],
                [0.0, 0.0],
                [0.0, 0.0],
            ]
        ]
    )
    tokens = torch.tensor([[0, 1, 0]])
    attention_mask = torch.tensor([[0, 1, 1]])

    per_token = lm_cross_entropy_loss(logits, tokens, attention_mask, per_token=True)
    scalar = lm_cross_entropy_loss(logits, tokens, attention_mask)
    expected = torch.log(torch.tensor(2.0))

    torch.testing.assert_close(per_token, torch.stack((expected.new_zeros(()), expected))[None])
    torch.testing.assert_close(scalar, expected)
    assert torch.isfinite(per_token).all()
    assert torch.isfinite(scalar)


def test_lm_accuracy_per_token_returns_bool_pos_minus_one() -> None:
    logits = torch.zeros(2, 4, 3)
    tokens = torch.tensor([[0, 1, 2, 0], [2, 1, 0, 2]])

    accuracy = lm_accuracy(logits, tokens, per_token=True)

    assert accuracy.dtype is torch.bool
    assert accuracy.shape == (2, 3)
