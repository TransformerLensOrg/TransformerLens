"""boot_tl_legacy: legacy TransformerLens-format repos on the bridge.

Deletion evidence for the last HookedTransformer-only load path (NeelNanda/
ArthurConmy/Baidicoot repos: TL config.json + *.pth, no HF model_type).
Value anchor is the frozen solu-1l golden captured from HookedTransformer.
"""

from __future__ import annotations

import pytest
import torch

from tests import goldens
from transformer_lens.model_bridge.bridge import TransformerBridge

SOLU_1L = "NeelNanda/SoLU_1L512W_C4_Code"


@pytest.fixture(scope="module")
def solu_bridge() -> TransformerBridge:
    return TransformerBridge.boot_tl_legacy(SOLU_1L, device="cpu")


@pytest.fixture(scope="module")
def golden():
    if not goldens.goldens_available("solu-1l", "no_processing"):
        pytest.skip("TL goldens dataset unavailable (set TL_GOLDENS_DIR or enable network)")
    return goldens.GoldenCell("solu-1l", "no_processing")


def test_logits_match_the_frozen_hooked_transformer_golden(solu_bridge, golden):
    """The whole chain — TL config derivation, .pth fetch, weight conversion,
    native solu_ln forward, tokenizer wiring — reproduces the golden logits.

    Bit-exact on the hardware that captured the goldens; ~2.5e-5 absolute on
    other platforms (fp32 accumulation). Same 1e-4 tolerance every golden
    comparison in the suite uses — never assert exactness against tensors
    captured on different hardware."""
    with torch.no_grad():
        logits = solu_bridge(golden.scalars["short_prompt"], return_type="logits")
    torch.testing.assert_close(
        logits, golden.tensors("logits_short")["logits"], atol=1e-4, rtol=1e-4
    )


def test_long_text_loss_matches_the_golden_scalar(solu_bridge, golden):
    """Cross-platform: observed 1.5e-5 delta on Linux CI vs the capture host."""
    loss = solu_bridge(golden.scalars["ablation"]["text"], return_type="loss")
    assert abs(float(loss) - golden.scalars["ablation"]["orig_loss"]) < 1e-4


def test_mid_mlp_layernorm_is_load_bearing(solu_bridge, golden):
    """SoLU-LN's defining structure: zeroing the mid-MLP LN weight must change
    the output — guards against the LN silently dropping out of the forward."""
    prompt = golden.scalars["short_prompt"]
    with torch.no_grad():
        base = solu_bridge(prompt, return_type="logits")
        ln = solu_bridge.blocks[0].mlp.ln._original_component
        keep = ln.weight.clone()
        ln.weight.zero_()
        zeroed = solu_bridge(prompt, return_type="logits")
        ln.weight.copy_(keep)
    assert not torch.allclose(base, zeroed)


def test_attn_only_repo_boots_and_runs():
    bridge = TransformerBridge.boot_tl_legacy("NeelNanda/Attn_Only_2L512W_C4_Code", device="cpu")
    assert bridge.cfg.attn_only
    with torch.no_grad():
        out = bridge("Hello world", return_type="logits")
    assert torch.isfinite(out).all()


def test_checkpointed_load_stamps_cfg_and_differs_from_final(solu_bridge):
    early = TransformerBridge.boot_tl_legacy(SOLU_1L, checkpoint_index=0, device="cpu")
    assert early.cfg.checkpoint_index == 0
    assert early.cfg.checkpoint_value is not None
    final_w = solu_bridge.state_dict()["embed.weight"]
    early_w = early.state_dict()["embed.weight"]
    assert not torch.equal(final_w, early_w), "checkpoint 0 loaded the final weights"


def test_checkpoint_index_out_of_range_raises():
    with pytest.raises(ValueError, match="out of range"):
        TransformerBridge.boot_tl_legacy(SOLU_1L, checkpoint_index=10_000)


def test_non_legacy_repo_is_refused_with_a_pointer():
    with pytest.raises(ValueError, match="boot_transformers"):
        TransformerBridge.boot_tl_legacy("gpt2")


def test_boot_transformers_stamps_checkpoint_metadata():
    bridge = TransformerBridge.boot_transformers(
        "EleutherAI/pythia-14m", device="cpu", checkpoint_index=2
    )
    assert bridge.cfg.checkpoint_index == 2
    assert bridge.cfg.checkpoint_value is not None
