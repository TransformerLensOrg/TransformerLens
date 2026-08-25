"""Stacked enc-dec weights on the bridge match HookedEncoderDecoder.

Deletion evidence for HookedEncoderDecoder's stacked-weight properties: the
bridge must produce the same tensors over chain(encoder, decoder), and the same
head labels, before the legacy class can go.
"""

from __future__ import annotations

import pytest
import torch

from transformer_lens import HookedEncoderDecoder
from transformer_lens.model_bridge.bridge import TransformerBridge

MODEL = "google-t5/t5-small"
STACKED = ["W_Q", "W_K", "W_V", "W_O", "W_in", "W_out"]


@pytest.fixture(scope="module")
def hooked() -> HookedEncoderDecoder:
    return HookedEncoderDecoder.from_pretrained(MODEL, device="cpu")


@pytest.fixture(scope="module")
def bridge() -> TransformerBridge:
    return TransformerBridge.boot_transformers(MODEL, device="cpu")


@pytest.mark.parametrize("name", STACKED)
def test_stacked_weights_match_hooked_encoder_decoder(name, hooked, bridge):
    """HookedEncoderDecoder does no weight processing, so these are directly comparable."""
    expected = getattr(hooked, name)
    actual = getattr(bridge, name)
    assert actual.shape == expected.shape
    torch.testing.assert_close(actual, expected, atol=0.0, rtol=0.0)


def test_head_labels_match_hooked_encoder_decoder(hooked, bridge):
    """all_head_labels is a property on the bridge; HT exposes it as a method."""
    assert bridge.all_head_labels == hooked.all_head_labels()
