"""Offline generation-cache mask parity with a tiny real GPT-2."""

import pytest
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast

from transformer_lens.model_bridge.sources import build_bridge_from_module


@pytest.fixture
def tiny_bridge():
    config = GPT2Config(
        vocab_size=32,
        n_positions=16,
        n_embd=16,
        n_layer=1,
        n_head=2,
        bos_token_id=0,
        eos_token_id=1,
    )
    config._attn_implementation = "eager"
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(2)
        model = GPT2LMHeadModel(config).eval()
    return build_bridge_from_module(model, architecture="GPT2LMHeadModel", hf_config=config)


@pytest.mark.parametrize("mask_values", [[0, 0, 1, 1], [1, 0, 1, 1], [1, 1, 1, 1], None])
@pytest.mark.parametrize("mask_dtype", [torch.long, torch.bool])
@pytest.mark.parametrize("use_past", [False, True])
@pytest.mark.parametrize("early_eos", [False, True])
def test_returned_cache_preserves_generation_context(
    tiny_bridge, mask_values, mask_dtype, use_past, early_eos
):
    bridge = tiny_bridge
    tokens = torch.tensor([[3, 4, 7, 8]])
    mask = None if mask_values is None else torch.tensor([mask_values], dtype=mask_dtype)
    original_mask = None if mask is None else mask.clone()
    with torch.no_grad():
        first_logits = bridge(tokens, attention_mask=mask)[:, -1]
        eos = first_logits.argmax(-1).item() if early_eos else None
        output, cache, input_tokens = bridge.generate(
            tokens,
            attention_mask=mask,
            max_new_tokens=3,
            do_sample=False,
            stop_at_eos=early_eos,
            eos_token_id=eos,
            use_past_kv_cache=use_past,
            output_logits=True,
            return_cache=True,
            return_input_tokens=True,
            verbose=False,
        )
        generated = output.sequences.shape[1] - tokens.shape[1]
        assert generated == (1 if early_eos else 3)
        extended_mask = (
            None if mask is None else torch.cat([mask, mask.new_ones((1, generated))], dim=1)
        )
        _, reference = bridge.run_with_cache(output.sequences, attention_mask=extended_mask)
    assert set(cache.cache_dict) == set(reference.cache_dict)
    for name in cache.cache_dict:
        torch.testing.assert_close(cache[name], reference[name])
    for step, logits in enumerate(output.logits):
        torch.testing.assert_close(
            cache["unembed.hook_out"][:, tokens.shape[1] - 1 + step],
            logits,
            atol=1e-6,
            rtol=1e-5,
        )
    torch.testing.assert_close(input_tokens, tokens)
    if mask is not None:
        torch.testing.assert_close(mask, original_mask)
        masked_keys = mask[0] == 0
        pattern = cache["blocks.0.attn.hook_pattern"]
        assert (
            torch.count_nonzero(
                pattern[:, :, tokens.shape[1] - 1 :, : tokens.shape[1]][..., masked_keys]
            )
            == 0
        )


def test_inferred_left_padding_mask_survives_filtered_cache(tiny_bridge):
    bridge = tiny_bridge
    vocab = {"<pad>": 0, "<eos>": 1, "<unk>": 2, **{f"t{i}": i for i in range(3, 32)}}
    bridge.tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer(WordLevel(vocab, unk_token="<unk>")),
        pad_token="<pad>",
        eos_token="<eos>",
        unk_token="<unk>",
    )
    tokens = torch.tensor([[0, 0, 7, 8]])
    key = "blocks.0.hook_resid_post"
    with torch.no_grad():
        output, cache = bridge.generate(
            tokens,
            padding_side="left",
            max_new_tokens=2,
            do_sample=False,
            stop_at_eos=False,
            use_past_kv_cache=False,
            return_cache=True,
            names_filter=key,
            device="cpu",
            verbose=False,
        )
        _, reference = bridge.run_with_cache(
            output,
            attention_mask=torch.tensor([[0, 0, 1, 1, 1, 1]]),
            names_filter=key,
        )
    assert set(cache.cache_dict) == set(reference.cache_dict)
    torch.testing.assert_close(cache[key], reference[key])
    assert cache[key].device.type == "cpu"
