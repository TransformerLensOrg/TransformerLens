"""Cohere2 NoPE layers must not warn about missing position_embeddings.

Full-attention (global) layers deliberately null position_embeddings — RoPE is
sliding-window-only on Cohere2 — so the base bridge's missing-RoPE RuntimeWarning
is spurious there.
"""

from __future__ import annotations

import warnings

import pytest
import torch
from transformers import AutoModelForCausalLM, Cohere2Config

from transformer_lens.model_bridge.sources import build_bridge_from_module


@pytest.fixture(scope="module")
def cohere2_bridge():
    cfg = Cohere2Config(
        vocab_size=64,
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        intermediate_size=64,
        max_position_embeddings=32,
        sliding_window=4,
        sliding_window_pattern=2,
        layer_types=["sliding_attention", "full_attention"],
    )
    cfg._attn_implementation = "eager"
    torch.manual_seed(0)
    hf = AutoModelForCausalLM.from_config(cfg).eval()
    return build_bridge_from_module(
        hf, "Cohere2ForCausalLM", hf_config=cfg, tokenizer=None, device="cpu"
    ).eval()


def test_nope_layer_forward_emits_no_rope_warning(cohere2_bridge) -> None:
    tokens = torch.tensor([[1, 2, 3, 4, 5]])
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with torch.no_grad():
            cohere2_bridge.run_with_cache(tokens, names_filter=["blocks.1.attn.hook_pattern"])
    rope_warnings = [
        w for w in caught if issubclass(w.category, RuntimeWarning) and "RoPE" in str(w.message)
    ]
    assert rope_warnings == [], [str(w.message) for w in rope_warnings]
