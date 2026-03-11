"""Integration tests for refactor_factored_attn_matrices with TransformerBridge."""

import pytest
import torch

from transformer_lens import HookedTransformer
from transformer_lens.model_bridge import TransformerBridge


@pytest.fixture(scope="module")
def model_name():
    return "distilgpt2"


@pytest.fixture(scope="module")
def device():
    return "cpu"


@pytest.fixture(scope="module")
def test_text():
    return "Natural language processing"


@pytest.fixture(scope="module")
def reference_ht(model_name, device):
    """HookedTransformer with refactored attention matrices enabled."""
    return HookedTransformer.from_pretrained(
        model_name,
        device=device,
        refactor_factored_attn_matrices=True,
    )


def test_refactor_factored_attn_matrices_loss_matches(
    model_name, device, test_text, reference_ht
):
    """Bridge loss should stay close to HookedTransformer after refactoring."""
    ref_loss = reference_ht(test_text, return_type="loss")

    bridge = TransformerBridge.boot_transformers(model_name, device=device)
    bridge.enable_compatibility_mode(refactor_factored_attn_matrices=True)
    bridge_loss = bridge(test_text, return_type="loss")

    assert not torch.isnan(bridge_loss), "Bridge produced NaN loss"
    assert not torch.isinf(bridge_loss), "Bridge produced infinite loss"

    loss_diff = abs(bridge_loss.item() - ref_loss.item())
    assert loss_diff < 1.0, (
        f"Loss difference too large: {loss_diff:.6f} "
        f"(bridge={bridge_loss.item():.4f}, reference={ref_loss.item():.4f})"
    )


def test_refactor_factored_attn_matrices_logits_match(
    model_name, device, test_text, reference_ht
):
    """Bridge logits should stay close to HookedTransformer after refactoring."""
    tokens = reference_ht.to_tokens(test_text)
    ref_logits = reference_ht(tokens)

    bridge = TransformerBridge.boot_transformers(model_name, device=device)
    bridge.enable_compatibility_mode(refactor_factored_attn_matrices=True)
    bridge_logits = bridge(tokens)

    assert (
        ref_logits.shape == bridge_logits.shape
    ), f"Shape mismatch: ref={ref_logits.shape}, bridge={bridge_logits.shape}"

    max_diff = (ref_logits - bridge_logits).abs().max().item()
    assert max_diff < 1.0, f"Max logit difference too large: {max_diff:.6f}"


def test_refactor_preserves_fold_ln(model_name, device, test_text):
    """Refactoring must not discard fold_ln processing."""
    ref = HookedTransformer.from_pretrained(
        model_name,
        device=device,
        fold_ln=True,
        refactor_factored_attn_matrices=True,
    )
    ref_loss = ref(test_text, return_type="loss")

    bridge = TransformerBridge.boot_transformers(model_name, device=device)
    bridge.enable_compatibility_mode(
        fold_ln=True,
        refactor_factored_attn_matrices=True,
    )
    bridge_loss = bridge(test_text, return_type="loss")

    loss_diff = abs(bridge_loss.item() - ref_loss.item())
    assert loss_diff < 1.0, (
        f"fold_ln + refactor mismatch: {loss_diff:.6f} "
        f"(bridge={bridge_loss.item():.4f}, ref={ref_loss.item():.4f})"
    )
