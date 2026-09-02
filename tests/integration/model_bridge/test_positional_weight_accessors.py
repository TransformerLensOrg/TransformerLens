"""W_pos / W_E_pos on TransformerBridge — deletion evidence for the
HookedTransformer/HookedEncoder accessors certified only on the legacy side."""

from __future__ import annotations

import pytest
import torch

from transformer_lens.model_bridge.bridge import TransformerBridge


@pytest.fixture(scope="module")
def gpt2_bridge_np():
    return TransformerBridge.boot_transformers("gpt2", device="cpu")


def test_w_pos_matches_raw_hf(gpt2_bridge_np):
    """Unprocessed bridge W_pos is HF's wpe weight exactly."""
    hf_wpe = gpt2_bridge_np.original_model.transformer.wpe.weight
    torch.testing.assert_close(gpt2_bridge_np.W_pos, hf_wpe, atol=0.0, rtol=0.0)
    assert gpt2_bridge_np.W_pos.shape == (
        gpt2_bridge_np.cfg.n_ctx,
        gpt2_bridge_np.cfg.d_model,
    )


def test_w_e_pos_is_the_concatenation(gpt2_bridge_np):
    b = gpt2_bridge_np
    assert b.W_E_pos.shape == (b.cfg.d_vocab + b.cfg.n_ctx, b.cfg.d_model)
    torch.testing.assert_close(b.W_E_pos[: b.cfg.d_vocab], b.W_E, atol=0.0, rtol=0.0)
    torch.testing.assert_close(b.W_E_pos[b.cfg.d_vocab :], b.W_pos, atol=0.0, rtol=0.0)


def test_w_pos_raises_cleanly_on_rotary():
    """A rotary model has no absolute positional matrix; say so, don't guess."""
    import copy

    from transformers import LlamaConfig, LlamaForCausalLM

    from transformer_lens.model_bridge.sources._bridge_builder import (
        build_bridge_from_module,
    )

    torch.manual_seed(0)
    cfg = LlamaConfig(
        vocab_size=128,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=4,
        max_position_embeddings=32,
    )
    bridge = build_bridge_from_module(
        LlamaForCausalLM(cfg).eval(),
        "LlamaForCausalLM",
        hf_config=copy.deepcopy(cfg),
        tokenizer=None,
        device="cpu",
    )
    with pytest.raises(AttributeError, match="absolute positional"):
        _ = bridge.W_pos
    with pytest.raises(AttributeError, match="absolute positional"):
        _ = bridge.W_E_pos


def test_w_pos_refuses_t5_relative_bias():
    """T5 maps pos_embed to the relative attention bias; W_pos must refuse it,
    as the legacy accessor did, not return a [num_buckets, n_heads] table."""
    import copy

    from transformers import T5Config, T5ForConditionalGeneration

    from transformer_lens.model_bridge.sources._bridge_builder import (
        build_bridge_from_module,
    )

    torch.manual_seed(0)
    cfg = T5Config(
        vocab_size=128, d_model=32, d_kv=8, d_ff=64, num_layers=2, num_decoder_layers=2, num_heads=4
    )
    bridge = build_bridge_from_module(
        T5ForConditionalGeneration(cfg).eval(),
        "T5ForConditionalGeneration",
        hf_config=copy.deepcopy(cfg),
        tokenizer=None,
        device="cpu",
    )
    with pytest.raises(AttributeError, match="relative attention bias|table"):
        _ = bridge.W_pos


def test_w_pos_slices_the_opt_offset():
    """OPT allocates n_ctx + 2 positional rows; HT's converter slices them off,
    and the bridge accessor must agree with HT on both shape and values."""
    import copy

    from transformers import OPTConfig, OPTForCausalLM

    from transformer_lens.model_bridge.sources._bridge_builder import (
        build_bridge_from_module,
    )

    torch.manual_seed(0)
    cfg = OPTConfig(
        vocab_size=128,
        hidden_size=32,
        ffn_dim=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        max_position_embeddings=32,
        word_embed_proj_dim=32,
    )
    model = OPTForCausalLM(cfg).eval()
    bridge = build_bridge_from_module(
        model,
        "OPTForCausalLM",
        hf_config=copy.deepcopy(cfg),
        tokenizer=None,
        device="cpu",
    )
    assert bridge.W_pos.shape == (bridge.cfg.n_ctx, bridge.cfg.d_model)
    raw = model.model.decoder.embed_positions.weight
    torch.testing.assert_close(bridge.W_pos, raw[2:], atol=0.0, rtol=0.0)


def test_w_pos_works_on_bert():
    """HookedEncoder certifies W_pos on BERT; the bridge accessor must too."""
    import copy

    from transformers import BertConfig, BertForMaskedLM

    from transformer_lens.model_bridge.sources._bridge_builder import (
        build_bridge_from_module,
    )

    torch.manual_seed(0)
    cfg = BertConfig(
        vocab_size=64,
        hidden_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        intermediate_size=64,
        max_position_embeddings=32,
    )
    model = BertForMaskedLM(cfg).eval()
    bridge = build_bridge_from_module(
        model,
        "BertForMaskedLM",
        hf_config=copy.deepcopy(cfg),
        tokenizer=None,
        device="cpu",
    )
    torch.testing.assert_close(
        bridge.W_pos, model.bert.embeddings.position_embeddings.weight, atol=0.0, rtol=0.0
    )
