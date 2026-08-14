"""Test refactor_factored_attn_matrices with TransformerBridge.

Verifies that the refactored attention matrices produce correct results when
used via TransformerBridge, matching the frozen HookedTransformer goldens
(gpt2 is the golden refactor model — see scripts/capture_ht_goldens.py).
"""

import pytest
import torch

from tests import goldens
from transformer_lens.model_bridge import TransformerBridge


@pytest.fixture(scope="module")
def golden():
    if not goldens.goldens_available("gpt2", "refactor_factored"):
        pytest.skip("TL goldens dataset unavailable (set TL_GOLDENS_DIR or enable network)")
    return goldens.GoldenCell("gpt2", "refactor_factored")


@pytest.fixture(scope="module")
def refactored_bridge():
    """gpt2 bridge with full default processing + refactor (the golden cell's config)."""
    bridge = TransformerBridge.boot_transformers("gpt2", device="cpu")
    bridge.enable_compatibility_mode(refactor_factored_attn_matrices=True)
    return bridge


def test_refactor_factored_attn_matrices_loss_matches(golden, refactored_bridge):
    """Bridge with refactor_factored_attn_matrices should match the frozen HT loss."""
    ref_loss = golden.scalars["long_text_ce_loss"]
    text = golden.scalars["ablation"]["text"]

    bridge_loss = refactored_bridge(text, return_type="loss")

    assert not torch.isnan(bridge_loss), "Bridge produced NaN loss"
    assert not torch.isinf(bridge_loss), "Bridge produced infinite loss"

    loss_diff = abs(bridge_loss.item() - ref_loss)
    assert loss_diff < 0.01, (
        f"Loss difference too large: {loss_diff:.6f} "
        f"(bridge={bridge_loss.item():.4f}, golden={ref_loss:.4f})"
    )


def test_refactor_factored_attn_matrices_logits_match(golden, refactored_bridge):
    """Bridge logits should closely match the frozen HT logits after refactoring."""
    ref_logits = golden.tensors("logits_short")["logits"]

    with torch.no_grad():
        bridge_logits = refactored_bridge(golden.scalars["short_prompt"], return_type="logits")

    assert (
        ref_logits.shape == bridge_logits.shape
    ), f"Shape mismatch: golden={ref_logits.shape}, bridge={bridge_logits.shape}"

    max_diff = (ref_logits - bridge_logits).abs().max().item()
    assert max_diff < 0.01, f"Max logit difference too large: {max_diff:.6f}"


def test_refactor_preserves_fold_ln(golden):
    """Refactoring must not undo fold_ln: the model function stays intact.

    Weight processing is function-preserving, so a bridge with fold_ln + refactor
    must still reproduce the frozen fold_ln-only loss on the golden text.
    """
    if not goldens.goldens_available("gpt2", "fold_ln_only"):
        pytest.skip("TL goldens dataset unavailable (set TL_GOLDENS_DIR or enable network)")
    fold_ln_golden = goldens.GoldenCell("gpt2", "fold_ln_only")

    bridge = TransformerBridge.boot_transformers("gpt2", device="cpu")
    bridge.enable_compatibility_mode(
        fold_ln=True,
        center_writing_weights=False,
        center_unembed=False,
        fold_value_biases=False,
        refactor_factored_attn_matrices=True,
    )
    bridge_loss = bridge(fold_ln_golden.scalars["ablation"]["text"], return_type="loss")

    ref_loss = fold_ln_golden.scalars["long_text_ce_loss"]
    loss_diff = abs(bridge_loss.item() - ref_loss)
    assert loss_diff < 0.01, (
        f"fold_ln + refactor mismatch: {loss_diff:.6f} "
        f"(bridge={bridge_loss.item():.4f}, golden={ref_loss:.4f})"
    )
