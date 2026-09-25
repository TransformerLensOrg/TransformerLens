"""Tests for MuseGlimmerArchitectureAdapter on a tiny random config."""

import copy

import pytest
import torch

pytest.importorskip("transformers.models.muse_glimmer")

from transformers import AutoModelForImageTextToText
from transformers.models.muse_glimmer import MuseGlimmerConfig

from transformer_lens.model_bridge.bridge import TransformerBridge
from transformer_lens.model_bridge.sources._bridge_builder import (
    build_bridge_config_from_hf,
)
from transformer_lens.model_bridge.supported_architectures.muse_glimmer import (
    MuseGlimmerArchitectureAdapter,
)

ARCH = "MuseGlimmerForConditionalGeneration"
N_LAYERS = 4
TOKENS = torch.tensor([[5, 17, 29, 3, 11, 42, 7, 23], [8, 9, 10, 11, 12, 13, 14, 15]])


class _Tok:
    pass


@pytest.fixture(scope="module")
def models():
    torch.manual_seed(0)
    cfg = MuseGlimmerConfig(
        text_config=dict(
            vocab_size=64,
            hidden_size=64,
            intermediate_size=96,
            num_hidden_layers=N_LAYERS,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=16,
            max_position_embeddings=64,
            sliding_window=4,
        ),
        vision_config=dict(
            hidden_size=32,
            intermediate_size=48,
            num_hidden_layers=1,
            num_attention_heads=2,
            pos_emb_height=4,
            pos_emb_width=4,
        ),
        out_hidden_size=128,
        projector_hidden_size=32,
    )
    cfg.architectures = [ARCH]
    hf = AutoModelForImageTextToText.from_config(cfg, attn_implementation="eager")
    hf = hf.to(torch.float32).eval()
    reference = copy.deepcopy(hf)
    bridge_cfg = build_bridge_config_from_hf(hf.config, ARCH, "muse-glimmer-tiny", torch.float32)
    bridge = TransformerBridge(hf, MuseGlimmerArchitectureAdapter(bridge_cfg), tokenizer=_Tok())
    return bridge, reference


def test_forward_matches_hf(models):
    bridge, reference = models
    with torch.no_grad():
        bridge_logits = bridge(TOKENS)
        hf_logits = reference(input_ids=TOKENS).logits
    assert torch.allclose(bridge_logits, hf_logits, atol=1e-4, rtol=0)


def test_run_with_cache_hooks(models):
    bridge, reference = models
    with torch.no_grad():
        _, cache = bridge.run_with_cache(TOKENS)
        hf_attn = reference(input_ids=TOKENS, output_attentions=True).attentions
    batch, seq = TOKENS.shape
    for i in range(N_LAYERS):
        for name in ("hook_resid_pre", "hook_attn_out", "hook_mlp_out", "hook_resid_post"):
            assert cache[f"blocks.{i}.{name}"].shape == (batch, seq, 64)
        assert cache[f"blocks.{i}.attn.hook_z"].shape == (batch, seq, 4, 16)
        torch.testing.assert_close(cache[f"blocks.{i}.attn.hook_pattern"], hf_attn[i])
