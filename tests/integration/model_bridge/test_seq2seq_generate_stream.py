"""Offline seq2seq streaming regressions using tiny random models."""

import pytest
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import (
    BartConfig,
    BartForConditionalGeneration,
    PreTrainedTokenizerFast,
    StoppingCriteriaList,
    T5Config,
    T5ForConditionalGeneration,
)

from tests.integration.model_bridge.helpers import make_tiny_pair
from transformer_lens.model_bridge import TransformerBridge


@pytest.fixture(scope="module", params=("bart", "t5"))
def seq2seq_bridge(request) -> TransformerBridge:
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
        bridge, _ = make_tiny_pair(
            config, "BartForConditionalGeneration", loader=BartForConditionalGeneration
        )
    else:
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
        bridge, _ = make_tiny_pair(
            config, "T5ForConditionalGeneration", loader=T5ForConditionalGeneration
        )
    return bridge


@pytest.mark.parametrize("batch_size", (1, 2))
@pytest.mark.parametrize("use_cache", (False, True))
def test_stream_matches_generate(seq2seq_bridge, batch_size, use_cache):
    source = torch.tensor([[4, 5, 6, 2], [7, 8, 9, 2]])[:batch_size]
    options = dict(
        max_new_tokens=3,
        do_sample=False,
        stop_at_eos=False,
        use_past_kv_cache=use_cache,
        return_type="tokens",
        verbose=False,
    )
    expected = seq2seq_bridge.generate(source, **options)
    chunks = list(seq2seq_bridge.generate_stream(source, max_tokens_per_yield=2, **options))
    torch.testing.assert_close(torch.cat(chunks, dim=1), expected, rtol=0, atol=0)


def test_encoder_stays_fixed_while_decoder_grows(seq2seq_bridge):
    source = torch.tensor([[4, 5, 6, 2], [7, 8, 9, 2]])
    calls = []

    def record(module, args, kwargs):
        calls.append((args[0].clone(), kwargs["decoder_input"].clone()))

    handle = seq2seq_bridge.register_forward_pre_hook(record, with_kwargs=True)
    try:
        list(
            seq2seq_bridge.generate_stream(
                source, max_new_tokens=3, do_sample=False, stop_at_eos=False, verbose=False
            )
        )
    finally:
        handle.remove()
    assert len(calls) == 3
    for step, (encoder, decoder) in enumerate(calls):
        torch.testing.assert_close(encoder, source, rtol=0, atol=0)
        assert decoder.shape == (2, step + 1)
        assert (decoder[:, 0] == seq2seq_bridge.original_model.config.decoder_start_token_id).all()


@pytest.mark.parametrize("prompts", ("a b c", ["a b c", "d"]))
def test_text_stream_preserves_encoder_mask_and_decoder_text(seq2seq_bridge, prompts, monkeypatch):
    tokenizer = Tokenizer(
        WordLevel(
            {"<pad>": 0, "<eos>": 1, "<unk>": 3, "a": 4, "b": 5, "c": 6, "d": 7}, unk_token="<unk>"
        )
    )
    tokenizer.pre_tokenizer = Whitespace()
    fast = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer, pad_token="<pad>", eos_token="<eos>", unk_token="<unk>"
    )
    monkeypatch.setattr(seq2seq_bridge, "tokenizer", fast)
    encoded = fast(prompts, return_tensors="pt", padding=True)
    expected = seq2seq_bridge.generate(
        encoded["input_ids"],
        attention_mask=encoded["attention_mask"],
        max_new_tokens=3,
        do_sample=False,
        stop_at_eos=False,
        verbose=False,
        return_type="tokens",
    )
    masks = []

    def record(module, args, kwargs):
        masks.append(kwargs["attention_mask"].clone())

    handle = seq2seq_bridge.register_forward_pre_hook(record, with_kwargs=True)
    try:
        chunks = list(
            seq2seq_bridge.generate_stream(
                prompts,
                max_new_tokens=3,
                max_tokens_per_yield=2,
                do_sample=False,
                stop_at_eos=False,
                verbose=False,
            )
        )
    finally:
        handle.remove()
    assert len(masks) == 3
    for mask in masks:
        torch.testing.assert_close(mask, encoded["attention_mask"], rtol=0, atol=0)
    if isinstance(prompts, str):
        assert "".join(chunks) == fast.decode(expected[0], skip_special_tokens=True)
    else:
        assert ["".join(chunk[row] for chunk in chunks) for row in range(2)] == fast.batch_decode(
            expected, skip_special_tokens=True
        )


def test_stream_honors_seq2seq_generation_defaults(seq2seq_bridge, monkeypatch):
    config = seq2seq_bridge.original_model.generation_config
    monkeypatch.setattr(config, "forced_bos_token_id", 8)
    monkeypatch.setattr(config, "min_length", 4)
    monkeypatch.setattr(config, "no_repeat_ngram_size", 1)
    source = torch.tensor([[4, 5, 6, 2]])
    options = dict(
        max_new_tokens=4, do_sample=False, eos_token_id=2, verbose=False, return_type="tokens"
    )
    expected = seq2seq_bridge.generate(source, **options)
    chunks = list(seq2seq_bridge.generate_stream(source, max_tokens_per_yield=2, **options))
    torch.testing.assert_close(torch.cat(chunks, dim=1), expected, rtol=0, atol=0)
    assert chunks[0][0, 1] == 8
    assert (expected[:, 2:4] != 2).all()
    assert len(set(expected[0].tolist())) == expected.shape[1]


def test_stream_stops_at_eos(seq2seq_bridge):
    source = torch.tensor([[4, 5, 6, 2]])
    first = seq2seq_bridge.generate(
        source,
        max_new_tokens=1,
        do_sample=False,
        stop_at_eos=False,
        return_type="tokens",
        verbose=False,
    )
    options = dict(
        max_new_tokens=3,
        do_sample=False,
        eos_token_id=int(first[0, -1]),
        verbose=False,
        return_type="tokens",
    )
    expected = seq2seq_bridge.generate(source, **options)
    chunks = list(seq2seq_bridge.generate_stream(source, **options))
    torch.testing.assert_close(torch.cat(chunks, dim=1), expected, rtol=0, atol=0)
    assert expected.shape == (1, 2)


@pytest.mark.parametrize(
    "bos,eos,expected", ((9, 2, 9), (None, 2, 2), (None, [2, 3], 2), (None, None, 0))
)
def test_shared_decoder_start_fallback(seq2seq_bridge, monkeypatch, bos, eos, expected):
    config = seq2seq_bridge.original_model.config
    monkeypatch.setattr(config, "decoder_start_token_id", None)
    monkeypatch.setattr(config, "bos_token_id", bos, raising=False)
    monkeypatch.setattr(config, "eos_token_id", eos)
    source = torch.tensor([[4, 5], [6, 7]])
    seed = seq2seq_bridge._encoder_decoder_seed(source, 8)
    torch.testing.assert_close(seed, torch.tensor([[expected, 8], [expected, 8]]), rtol=0, atol=0)


@pytest.mark.parametrize(
    "kwargs", ({"stop_strings": "a"}, {"stopping_criteria": StoppingCriteriaList()})
)
def test_stream_rejects_unsupported_seq2seq_stopping(seq2seq_bridge, kwargs):
    with pytest.raises(NotImplementedError, match="encoder-decoder"):
        list(seq2seq_bridge.generate_stream(torch.tensor([[4, 5, 2]]), **kwargs))
