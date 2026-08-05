"""Regression tests for padding-aware TransformerBridge causal loss."""

from __future__ import annotations

import pytest
import torch

from transformer_lens.config import TransformerBridgeConfig
from transformer_lens.model_bridge import TransformerBridge


def _bridge() -> TransformerBridge:
    cfg = TransformerBridgeConfig(
        d_model=32,
        d_head=8,
        n_heads=4,
        n_layers=2,
        n_ctx=6,
        d_vocab=32,
        d_mlp=64,
        act_fn="gelu",
        normalization_type="LN",
        seed=7,
        initializer_range=0.2,
    )
    return TransformerBridge.boot_native(cfg)


def _extract_loss(output: torch.Tensor | tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
    return output[1] if isinstance(output, tuple) else output


@pytest.mark.parametrize("return_type", ["loss", "both"])
def test_forward_loss_ignores_masked_padding_tokens(return_type: str) -> None:
    bridge = _bridge()
    attention_mask = torch.tensor(
        [
            [1, 1, 1, 0, 0, 0],
            [1, 1, 1, 1, 1, 1],
        ]
    )
    token_batches = (
        torch.tensor(
            [
                [1, 2, 3, 0, 0, 0],
                [4, 5, 6, 7, 8, 9],
            ]
        ),
        torch.tensor(
            [
                [1, 2, 3, 31, 30, 29],
                [4, 5, 6, 7, 8, 9],
            ]
        ),
    )

    losses = []
    for tokens in token_batches:
        output = bridge(tokens, attention_mask=attention_mask, return_type=return_type)
        loss = _extract_loss(output)
        logits = bridge(tokens, attention_mask=attention_mask, return_type="logits")
        expected = bridge.loss_fn(logits, tokens, attention_mask=attention_mask)

        torch.testing.assert_close(loss, expected)
        losses.append(loss)

    torch.testing.assert_close(losses[0], losses[1])


def test_forward_loss_per_token_zeros_masked_transitions() -> None:
    bridge = _bridge()
    tokens = torch.tensor(
        [
            [1, 2, 3, 0, 0, 0],
            [4, 5, 6, 7, 8, 9],
        ]
    )
    attention_mask = torch.tensor(
        [
            [1, 1, 1, 0, 0, 0],
            [1, 1, 1, 1, 1, 1],
        ]
    )

    loss = bridge(
        tokens,
        attention_mask=attention_mask,
        return_type="loss",
        loss_per_token=True,
    )
    next_token_mask = torch.logical_and(attention_mask[:, :-1], attention_mask[:, 1:])

    assert torch.count_nonzero(loss[~next_token_mask]) == 0


def test_forward_loss_is_finite_with_left_padding() -> None:
    bridge = _bridge()
    tokens = torch.tensor(
        [
            [0, 0, 0, 1, 2, 3],
            [4, 5, 6, 7, 8, 9],
        ]
    )
    attention_mask = torch.tensor(
        [
            [0, 0, 0, 1, 1, 1],
            [1, 1, 1, 1, 1, 1],
        ]
    )
    position_ids = attention_mask.long().cumsum(-1) - 1
    position_ids.masked_fill_(attention_mask == 0, 1)

    logits = bridge(
        tokens,
        attention_mask=attention_mask,
        position_ids=position_ids,
        return_type="logits",
    )
    loss = bridge(
        tokens,
        attention_mask=attention_mask,
        position_ids=position_ids,
        return_type="loss",
    )

    assert torch.isfinite(logits).all()
    assert torch.isfinite(loss)
