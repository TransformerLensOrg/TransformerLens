"""Test refactor_factored_attn_matrices with TransformerBridge.

Verifies that the refactored attention matrices produce correct results when
used via TransformerBridge, matching HookedTransformer output.
"""

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
    """HookedTransformer with refactor_factored_attn_matrices=True."""
    return HookedTransformer.from_pretrained(
        model_name,
        device=device,
        refactor_factored_attn_matrices=True,
    )


def test_refactor_factored_attn_matrices_loss_matches(model_name, device, test_text, reference_ht):
    """Bridge with refactor_factored_attn_matrices should match HookedTransformer."""
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


def test_refactor_factored_attn_matrices_logits_match(model_name, device, test_text, reference_ht):
    """Bridge logits should closely match HookedTransformer logits after refactoring."""
    tokens = reference_ht.to_tokens(test_text)
    ref_logits = reference_ht(tokens)

    bridge = TransformerBridge.boot_transformers(model_name, device=device)
    bridge.enable_compatibility_mode(refactor_factored_attn_matrices=True)
    bridge_logits = bridge(tokens)

    # Check shapes match
    assert (
        ref_logits.shape == bridge_logits.shape
    ), f"Shape mismatch: ref={ref_logits.shape}, bridge={bridge_logits.shape}"

    # Check values are close
    max_diff = (ref_logits - bridge_logits).abs().max().item()
    assert max_diff < 1.0, f"Max logit difference too large: {max_diff:.6f}"


def test_refactor_preserves_fold_ln(model_name, device, test_text):
    """Refactoring should not undo fold_ln — both should be applied together."""
    # Reference: fold_ln=True + refactor=True
    ref = HookedTransformer.from_pretrained(
        model_name,
        device=device,
        fold_ln=True,
        refactor_factored_attn_matrices=True,
    )
    ref_loss = ref(test_text, return_type="loss")

    # Bridge: same settings
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
