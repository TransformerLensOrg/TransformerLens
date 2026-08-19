"""Regression tests for TransformerBridge explicit-label loss semantics."""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F
from transformers import (
    BartConfig,
    BartForConditionalGeneration,
    GPT2Config,
    GPT2LMHeadModel,
    T5Config,
    T5ForConditionalGeneration,
)

from tests.integration.model_bridge.helpers import make_tiny_pair
from transformer_lens.model_bridge import TransformerBridge


@pytest.fixture(scope="module")
def tiny_gpt2_pair() -> tuple[TransformerBridge, torch.nn.Module]:
    config = GPT2Config(
        vocab_size=32,
        n_embd=16,
        n_layer=1,
        n_head=2,
        n_positions=16,
        n_ctx=16,
        bos_token_id=1,
        eos_token_id=2,
        pad_token_id=0,
    )
    return make_tiny_pair(config, "GPT2LMHeadModel", loader=GPT2LMHeadModel)


@pytest.fixture(scope="module", params=("bart", "t5"))
def tiny_seq2seq_pair(request) -> tuple[TransformerBridge, torch.nn.Module]:
    if request.param == "bart":
        config = BartConfig(
            vocab_size=32,
            d_model=16,
            encoder_layers=1,
            decoder_layers=1,
            encoder_attention_heads=2,
            decoder_attention_heads=2,
            encoder_ffn_dim=32,
            decoder_ffn_dim=32,
            max_position_embeddings=32,
            pad_token_id=0,
            bos_token_id=1,
            eos_token_id=2,
            decoder_start_token_id=2,
        )
        return make_tiny_pair(
            config,
            "BartForConditionalGeneration",
            loader=BartForConditionalGeneration,
        )

    config = T5Config(
        vocab_size=32,
        d_model=16,
        d_kv=8,
        d_ff=32,
        num_layers=1,
        num_decoder_layers=1,
        num_heads=2,
        pad_token_id=0,
        eos_token_id=1,
        decoder_start_token_id=0,
    )
    return make_tiny_pair(
        config,
        "T5ForConditionalGeneration",
        loader=T5ForConditionalGeneration,
    )


def test_seq2seq_loss_and_logits_follow_labels(
    tiny_seq2seq_pair: tuple[TransformerBridge, torch.nn.Module],
) -> None:
    bridge, reference = tiny_seq2seq_pair
    source = torch.tensor([[4, 5, 6, 7, 2]])
    label_batches = (
        torch.tensor([[8, 9, 10, 2, -100]]),
        torch.tensor([[11, 12, 13, 2, -100]]),
    )

    bridge_losses = []
    bridge_logits = []
    for labels in label_batches:
        with torch.no_grad():
            loss = bridge(source, labels=labels, return_type="loss")
            logits = bridge(source, labels=labels, return_type="logits")
            reference_output = reference(input_ids=source, labels=labels)

        torch.testing.assert_close(loss, reference_output.loss)
        torch.testing.assert_close(logits, reference_output.logits)
        bridge_losses.append(loss)
        bridge_logits.append(logits)

    assert not torch.allclose(bridge_losses[0], bridge_losses[1])
    assert not torch.allclose(bridge_logits[0], bridge_logits[1])


def test_seq2seq_both_allows_shorter_target(
    tiny_seq2seq_pair: tuple[TransformerBridge, torch.nn.Module],
) -> None:
    bridge, reference = tiny_seq2seq_pair
    source = torch.tensor([[4, 5, 6, 7, 2]])
    labels = torch.tensor([[14, 15, 2]])

    with torch.no_grad():
        logits, loss = bridge(source, labels=labels, return_type="both")
        reference_output = reference(input_ids=source, labels=labels)

    assert logits.shape[:2] == labels.shape
    torch.testing.assert_close(logits, reference_output.logits)
    torch.testing.assert_close(loss, reference_output.loss)


def test_seq2seq_loss_supports_hf_tuple_output(
    tiny_seq2seq_pair: tuple[TransformerBridge, torch.nn.Module],
) -> None:
    bridge, reference = tiny_seq2seq_pair
    source = torch.tensor([[4, 5, 6, 7, 2]])
    labels = torch.tensor([[14, 15, 2]])

    with torch.no_grad():
        logits, loss = bridge(
            source,
            labels=labels,
            return_type="both",
            return_dict=False,
        )
        reference_loss, reference_logits, *_ = reference(
            input_ids=source,
            labels=labels,
            return_dict=False,
        )

    torch.testing.assert_close(logits, reference_logits)
    torch.testing.assert_close(loss, reference_loss)


@pytest.mark.parametrize("return_type", ("loss", "both"))
def test_seq2seq_per_token_loss_matches_unshifted_labels(
    tiny_seq2seq_pair: tuple[TransformerBridge, torch.nn.Module],
    return_type: str,
) -> None:
    bridge, reference = tiny_seq2seq_pair
    source = torch.tensor([[4, 5, 6, 7, 2]])
    labels = torch.tensor([[14, 15, 2, -100]])

    with torch.no_grad():
        output = bridge(
            source,
            labels=labels,
            return_type=return_type,
            loss_per_token=True,
        )
        reference_logits = reference(input_ids=source, labels=labels).logits

    loss = output[1] if isinstance(output, tuple) else output
    expected = F.cross_entropy(
        reference_logits.flatten(0, 1),
        labels.flatten(),
        reduction="none",
        ignore_index=-100,
    ).view_as(labels)

    assert loss.shape == labels.shape
    assert loss[0, -1] == 0
    torch.testing.assert_close(loss, expected)


@pytest.mark.parametrize("return_type", ("loss", "both"))
def test_seq2seq_loss_requires_labels(
    tiny_seq2seq_pair: tuple[TransformerBridge, torch.nn.Module],
    return_type: str,
) -> None:
    bridge, _ = tiny_seq2seq_pair
    source = torch.tensor([[4, 5, 6, 7, 2]])

    with pytest.raises(ValueError, match="labels are required"):
        bridge(source, return_type=return_type)


def test_causal_loss_uses_explicit_labels(
    tiny_gpt2_pair: tuple[TransformerBridge, torch.nn.Module],
) -> None:
    bridge, reference = tiny_gpt2_pair
    input_ids = torch.tensor([[4, 5, 6, 7, 2]])
    labels = torch.full_like(input_ids, 9)

    with torch.no_grad():
        default_loss = bridge(input_ids, return_type="loss")
        logits, labeled_loss = bridge(input_ids, labels=labels, return_type="both")
        reference_output = reference(input_ids=input_ids, labels=labels)

    assert not torch.allclose(labeled_loss, default_loss)
    torch.testing.assert_close(logits, reference_output.logits)
    torch.testing.assert_close(labeled_loss, reference_output.loss)


def test_causal_labels_support_hf_tuple_output(
    tiny_gpt2_pair: tuple[TransformerBridge, torch.nn.Module],
) -> None:
    bridge, reference = tiny_gpt2_pair
    input_ids = torch.tensor([[4, 5, 6, 7, 2]])
    labels = torch.full_like(input_ids, 9)

    with torch.no_grad():
        logits, loss = bridge(
            input_ids,
            labels=labels,
            return_type="both",
            return_dict=False,
        )
        reference_loss, reference_logits, *_ = reference(
            input_ids=input_ids,
            labels=labels,
            return_dict=False,
        )

    torch.testing.assert_close(logits, reference_logits)
    torch.testing.assert_close(loss, reference_loss)


def test_causal_per_token_loss_uses_labels_and_ignore_index(
    tiny_gpt2_pair: tuple[TransformerBridge, torch.nn.Module],
) -> None:
    bridge, reference = tiny_gpt2_pair
    input_ids = torch.tensor([[4, 5, 6, 7, 2]])
    labels = torch.tensor([[9, 8, 7, -100, -100]])

    with torch.no_grad():
        loss = bridge(
            input_ids,
            labels=labels,
            return_type="loss",
            loss_per_token=True,
        )
        reference_logits = reference(input_ids=input_ids).logits

    expected = F.cross_entropy(
        reference_logits[:, :-1].flatten(0, 1),
        labels[:, 1:].flatten(),
        reduction="none",
        ignore_index=-100,
    ).view_as(labels[:, 1:])

    assert loss.shape == labels[:, 1:].shape
    assert torch.count_nonzero(loss[:, 2:]) == 0
    torch.testing.assert_close(loss, expected)


def test_causal_explicit_labels_preserve_attention_mask_contract(
    tiny_gpt2_pair: tuple[TransformerBridge, torch.nn.Module],
) -> None:
    bridge, _ = tiny_gpt2_pair
    input_ids = torch.tensor([[4, 5, 6, 0, 0]])
    labels = torch.tensor([[9, 8, 7, 6, 5]])
    attention_mask = torch.tensor([[1, 1, 1, 0, 0]])

    with torch.no_grad():
        logits = bridge(input_ids, attention_mask=attention_mask, return_type="logits")
        loss = bridge(
            input_ids,
            labels=labels,
            attention_mask=attention_mask,
            return_type="loss",
        )

    transition_mask = attention_mask[:, :-1].bool() & attention_mask[:, 1:].bool()
    expected = F.cross_entropy(
        logits[:, :-1][transition_mask],
        labels[:, 1:][transition_mask],
    )
    torch.testing.assert_close(loss, expected)
