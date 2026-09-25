"""Regression tests for padding-aware TransformerBridge causal loss."""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F
from transformers import (
    GPT2Config,
    GPT2LMHeadModel,
    LlamaConfig,
    LlamaForCausalLM,
    MptConfig,
)
from transformers.models.mpt.modeling_mpt import MptForCausalLM

from transformer_lens.config import TransformerBridgeConfig
from transformer_lens.model_bridge import TransformerBridge
from transformer_lens.model_bridge.sources._bridge_builder import (
    build_bridge_from_module,
)


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


def _manual_masked_loss(
    logits: torch.Tensor, tokens: torch.Tensor, attention_mask: torch.Tensor
) -> torch.Tensor:
    transition_mask = attention_mask[:, :-1].bool() & attention_mask[:, 1:].bool()
    return F.cross_entropy(
        logits[:, :-1][transition_mask],
        tokens[:, 1:][transition_mask],
    )


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
        expected = _manual_masked_loss(logits, tokens, attention_mask)

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


@pytest.mark.parametrize("mask_kind", ["bool", "additive"])
@pytest.mark.parametrize("mask_layout", ["key_only", "causal"])
def test_forward_loss_accepts_equivalent_4d_attention_mask(
    mask_kind: str, mask_layout: str
) -> None:
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
    blocked = ~attention_mask.bool()[:, None, None, :]
    if mask_layout == "causal":
        blocked = blocked | torch.ones(6, 6, dtype=torch.bool).triu(1)[None, None]
    attention_mask_4d = blocked if mask_kind == "bool" else blocked.float() * -10_000.0

    logits_2d, loss_2d = bridge(
        tokens,
        attention_mask=attention_mask,
        return_type="both",
    )
    logits_4d, loss_4d = bridge(
        tokens,
        attention_mask=attention_mask_4d,
        return_type="both",
    )

    torch.testing.assert_close(logits_4d, logits_2d, rtol=0, atol=0)
    torch.testing.assert_close(loss_4d, loss_2d)
    torch.testing.assert_close(loss_4d, _manual_masked_loss(logits_4d, tokens, attention_mask))


def test_loss_fn_reduces_rectangular_cached_4d_attention_mask() -> None:
    bridge = _bridge()
    tokens = torch.tensor([[4, 5]])
    logits = torch.zeros(1, 2, 32)
    logits[0, 0, 5] = 2.0
    cache_and_new_mask = torch.tensor([[0, 1, 1, 1, 1, 1]])
    key_blocked = ~cache_and_new_mask.bool()[:, None, None, :]
    query_positions = torch.tensor([4, 5])
    causal = torch.arange(6)[None, None, None, :] > query_positions[None, None, :, None]
    attention_mask_4d = key_blocked | causal

    loss = bridge.loss_fn(
        logits,
        tokens,
        attention_mask=attention_mask_4d,
        per_token=True,
    )

    expected = F.cross_entropy(logits[:, 0], tokens[:, 1])
    torch.testing.assert_close(loss, expected.reshape(1, 1))
    assert loss.shape == (1, 1)


@pytest.fixture(scope="module", params=["gpt2", "llama", "mpt"])
def hf_bool_mask_bridge(request: pytest.FixtureRequest) -> tuple[TransformerBridge, bool]:
    if request.param == "gpt2":
        config = GPT2Config(
            vocab_size=32,
            n_positions=8,
            n_embd=16,
            n_layer=1,
            n_head=2,
            _attn_implementation="eager",
        )
        model_class = GPT2LMHeadModel
        architecture = "GPT2LMHeadModel"
        is_keep = True
    elif request.param == "llama":
        config = LlamaConfig(
            vocab_size=32,
            max_position_embeddings=8,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=2,
            _attn_implementation="eager",
        )
        model_class = LlamaForCausalLM
        architecture = "LlamaForCausalLM"
        is_keep = True
    else:
        config = MptConfig(
            vocab_size=32,
            max_seq_len=8,
            d_model=16,
            expansion_ratio=2,
            n_layers=1,
            n_heads=2,
            no_bias=True,
            _attn_implementation="eager",
        )
        model_class = MptForCausalLM
        architecture = "MPTForCausalLM"
        is_keep = False

    torch.manual_seed(1)
    bridge = build_bridge_from_module(
        model_class(config).eval(), architecture=architecture, hf_config=config
    )
    assert bridge.adapter.bool_4d_mask_is_keep is is_keep
    return bridge, is_keep


def _bool_4d_mask(token_mask: torch.Tensor, *, is_keep: bool) -> torch.Tensor:
    pos = token_mask.shape[1]
    blocked = ~token_mask.bool()[:, None, None, :] | torch.ones(pos, pos, dtype=torch.bool).triu(1)
    return ~blocked if is_keep else blocked


@pytest.mark.parametrize("padding", [False, True])
@pytest.mark.parametrize("return_type", ["loss", "both"])
def test_hf_bool_4d_mask_loss_matches_2d(
    hf_bool_mask_bridge: tuple[TransformerBridge, bool], padding: bool, return_type: str
) -> None:
    bridge, is_keep = hf_bool_mask_bridge
    tokens = torch.tensor([[1, 2, 3, 0, 0]] if padding else [[1, 2, 3, 4]])
    token_mask = torch.tensor([[1, 1, 1, 0, 0]] if padding else [[1, 1, 1, 1]])
    mask_4d = _bool_4d_mask(token_mask, is_keep=is_keep)

    logits_2d, loss_2d = bridge(tokens, attention_mask=token_mask, return_type="both")
    output = bridge(tokens, attention_mask=mask_4d, return_type=return_type)
    loss_4d = _extract_loss(output)
    logits_4d = bridge(tokens, attention_mask=mask_4d, return_type="logits")

    torch.testing.assert_close(logits_4d, logits_2d, rtol=0, atol=0)
    torch.testing.assert_close(loss_4d, loss_2d)
    torch.testing.assert_close(loss_4d, _manual_masked_loss(logits_4d, tokens, token_mask))


def test_hf_bool_4d_mask_respects_explicit_labels(
    hf_bool_mask_bridge: tuple[TransformerBridge, bool],
) -> None:
    bridge, is_keep = hf_bool_mask_bridge
    tokens = torch.tensor([[1, 2, 3, 0, 0]])
    labels = torch.tensor([[-100, 2, 3, -100, -100]])
    token_mask = torch.tensor([[1, 1, 1, 0, 0]])
    mask_4d = _bool_4d_mask(token_mask, is_keep=is_keep)

    expected = bridge(
        tokens, labels=labels, attention_mask=token_mask, return_type="loss", loss_per_token=True
    )
    actual = bridge(
        tokens, labels=labels, attention_mask=mask_4d, return_type="loss", loss_per_token=True
    )

    torch.testing.assert_close(actual, expected)
    assert torch.count_nonzero(actual[:, 2:]) == 0
