"""generate_stream() must honor attention_mask and forced_bos_token_id like generate()."""

import pytest
import torch
from transformers import GPT2Config, GPT2LMHeadModel

from transformer_lens.model_bridge.sources._bridge_builder import (
    build_bridge_from_module,
)


@pytest.fixture()
def gpt2_bridge():
    config = GPT2Config(vocab_size=50, n_positions=32, n_embd=16, n_layer=1, n_head=2)
    config._attn_implementation = "eager"
    torch.manual_seed(42)
    hf = GPT2LMHeadModel(config).eval()
    return build_bridge_from_module(
        hf, "GPT2LMHeadModel", hf_config=config, tokenizer=None, device="cpu"
    )


@pytest.mark.parametrize("use_cache", (False, True))
def test_stream_threads_attention_mask_for_padded_tensor(gpt2_bridge, use_cache):
    tokens = torch.tensor([[9, 9, 5, 6], [3, 4, 5, 6]])
    mask = torch.tensor([[0, 0, 1, 1], [1, 1, 1, 1]])
    options = dict(
        max_new_tokens=3,
        do_sample=False,
        stop_at_eos=False,
        use_past_kv_cache=use_cache,
        verbose=False,
        return_type="tokens",
    )
    expected = gpt2_bridge.generate(tokens, attention_mask=mask, **options)
    masks = []

    def record(module, args, kwargs):
        if "attention_mask" in kwargs:
            masks.append(kwargs["attention_mask"].clone())

    handle = gpt2_bridge.register_forward_pre_hook(record, with_kwargs=True)
    try:
        chunks = list(gpt2_bridge.generate_stream(tokens, attention_mask=mask, **options))
    finally:
        handle.remove()
    # The prompt mask must reach step 0 and grow one attended column per token.
    assert len(masks) == 3
    for step, seen in enumerate(masks):
        grown = torch.cat([mask, torch.ones((2, step), dtype=mask.dtype)], dim=1)
        torch.testing.assert_close(seen, grown, rtol=0, atol=0)
    torch.testing.assert_close(torch.cat(chunks, dim=1), expected, rtol=0, atol=0)


def test_stream_mask_shape_mismatch_raises(gpt2_bridge):
    with pytest.raises(ValueError, match="does not match the prompt shape"):
        list(
            gpt2_bridge.generate_stream(
                torch.tensor([[1, 2, 3]]),
                attention_mask=torch.tensor([[1, 1]]),
                stop_at_eos=False,
                verbose=False,
            )
        )


def test_stream_forced_bos_rejected_on_decoder_only(gpt2_bridge):
    with pytest.raises(ValueError, match="encoder-decoder"):
        list(
            gpt2_bridge.generate_stream(
                torch.tensor([[1, 2, 3]]),
                forced_bos_token_id=5,
                stop_at_eos=False,
                verbose=False,
            )
        )
