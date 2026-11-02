"""Generation diagnostics retain model logits before decoding constraints."""

import pytest
import torch
from transformers import BartConfig, BartForConditionalGeneration, GPT2Config

from tests.integration.model_bridge.helpers import make_tiny_pair


@pytest.mark.parametrize("processor", ["none", "forced_eos", "min_length", "no_repeat_ngram"])
@pytest.mark.parametrize("do_sample", [False, True])
@pytest.mark.parametrize("batch_size", [1, 2])
def test_seq2seq_output_logits_match_forward_before_constraints(processor, do_sample, batch_size):
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
        eos_token_id=31,
        decoder_start_token_id=2,
        forced_eos_token_id=3 if processor == "forced_eos" else None,
    )
    bridge, ref = make_tiny_pair(
        config, "BartForConditionalGeneration", loader=BartForConditionalGeneration
    )
    generation_config = bridge.original_model.generation_config
    generation_config.min_length = 8 if processor == "min_length" else 0
    generation_config.no_repeat_ngram_size = 1 if processor == "no_repeat_ngram" else 0
    source = torch.tensor([[4, 5, 6, 2], [8, 9, 10, 2]])[:batch_size]
    prefix = torch.full((batch_size, 1), config.decoder_start_token_id)
    options = dict(
        max_new_tokens=3,
        do_sample=do_sample,
        eos_token_id=31,
        verbose=False,
        return_type="tokens",
    )
    torch.manual_seed(1234)
    output = bridge.generate(source, output_logits=True, **options)
    torch.testing.assert_close(output.sequences[:, :1], prefix, rtol=0, atol=0)
    torch.manual_seed(1234)
    tokens_only = bridge.generate(source, **options)
    torch.testing.assert_close(output.sequences, tokens_only, rtol=0, atol=0)
    assert len(output.logits) == 3
    with torch.no_grad():
        for step, actual in enumerate(output.logits):
            expected = ref(
                source, decoder_input_ids=output.sequences[:, : prefix.shape[1] + step]
            ).logits[:, -1]
            assert torch.isfinite(actual).all()
            torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
    generated = output.sequences[:, prefix.shape[1] :]
    if processor == "forced_eos":
        assert (generated[:, -1] == 3).all()
    elif processor == "min_length":
        assert not (generated == 31).any()
    elif processor == "no_repeat_ngram":
        assert not torch.isin(generated, prefix).any()
        for row in generated:
            assert row.unique().numel() == row.numel()


@pytest.mark.parametrize("use_cache", [False, True])
@pytest.mark.parametrize("do_sample", [False, True])
def test_decoder_only_output_logits_ignore_sampling_transforms(use_cache, do_sample):
    config = GPT2Config(
        vocab_size=32,
        n_embd=16,
        n_layer=1,
        n_head=2,
        n_positions=32,
        attn_pdrop=0.0,
        resid_pdrop=0.0,
        embd_pdrop=0.0,
    )
    bridge, ref = make_tiny_pair(config, "GPT2LMHeadModel")
    prompt = torch.tensor([[4, 5, 4]])
    output = bridge.generate(
        prompt,
        max_new_tokens=3,
        do_sample=do_sample,
        stop_at_eos=False,
        verbose=False,
        use_past_kv_cache=use_cache,
        output_logits=True,
        temperature=0.7,
        top_k=4,
        freq_penalty=0.2,
        repetition_penalty=1.3,
    )
    assert len(output.logits) == 3
    with torch.no_grad():
        for step, actual in enumerate(output.logits):
            expected = ref(output.sequences[:, : prompt.shape[1] + step]).logits[:, -1]
            torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
