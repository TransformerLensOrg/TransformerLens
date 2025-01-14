import pytest
import torch as t

from transformer_lens import HookedTransformer
from transformer_lens.past_key_value_caching import HookedTransformerKeyValueCache


# Pythia models seem to have some kind of numerical stability issue.
# See: https://github.com/TransformerLensOrg/TransformerLens/issues/385
@pytest.fixture(scope="session", params=[("gpt2-small", 1e-4), ("pythia-14m", 1e-2)])
def model_and_atol(request):
    return request.param


@pytest.fixture(scope="session")
def pretrained(model_and_atol):
    name, atol = model_and_atol
    model = HookedTransformer.from_pretrained(name, default_padding_side="left")
    return model, atol


def test_single_new_token(pretrained):
    model, atol = pretrained
    pre_prompt = "I went to Staten Island,"
    pre_prompt_tokens = model.to_tokens(pre_prompt)
    pre_prompt_tokens_len = pre_prompt_tokens.shape[-1]
    single_token_post_prompt = " Sharon"
    single_new_token = model.to_tokens(single_token_post_prompt, prepend_bos=False)
    full_prompt_tokens = t.cat([pre_prompt_tokens, single_new_token], dim=-1)
    no_cache_logits = model(full_prompt_tokens)
    assert full_prompt_tokens.shape[-1] == pre_prompt_tokens_len + 1

    past_kv_cache = HookedTransformerKeyValueCache.init_cache(
        model.cfg, model.cfg.device, pre_prompt_tokens.shape[0]
    )
    model(
        pre_prompt_tokens,
        past_kv_cache=past_kv_cache,
    )
    with_cache_logits = model(
        single_new_token,
        past_kv_cache=past_kv_cache,
    )
    assert t.allclose(no_cache_logits[:, -1], with_cache_logits[:, -1], atol=atol)
    assert t.allclose(no_cache_logits[:, -1:], with_cache_logits, atol=atol)


def test_multiple_new_tokens(pretrained):
    model, atol = pretrained
    pre_prompt = "I went to Staten Island,"
    pre_prompt_tokens = model.to_tokens(pre_prompt)
    pre_prompt_tokens_len = pre_prompt_tokens.shape[-1]
    post_prompt = " to buy myself a mandolin"
    new_tokens = model.to_tokens(post_prompt, prepend_bos=False)
    new_tokens_len = new_tokens.shape[-1]
    full_prompt_tokens = t.cat([pre_prompt_tokens, new_tokens], dim=-1)
    assert full_prompt_tokens.shape[-1] == pre_prompt_tokens_len + new_tokens_len
    no_cache_logits = model(full_prompt_tokens)

    past_kv_cache = HookedTransformerKeyValueCache.init_cache(
        model.cfg, model.cfg.device, pre_prompt_tokens.shape[0]
    )
    model(
        pre_prompt_tokens,
        past_kv_cache=past_kv_cache,
    )
    with_cache_logits = model(
        new_tokens,
        past_kv_cache=past_kv_cache,
    )
    assert t.allclose(no_cache_logits[:, -1], with_cache_logits[:, -1], atol=atol)
    assert t.allclose(no_cache_logits[:, -new_tokens_len:], with_cache_logits, atol=atol)


@pytest.mark.parametrize("pre_padding", ["left", "right", None])
@pytest.mark.parametrize("post_padding", ["left", "right", None])
def test_multi_token_batch(pretrained, pre_padding, post_padding):
    model, atol = pretrained
    padded_batch_pre_prompts = [
        "It's always locked",
        "I'd rather be burned in Canada",
    ]
    unpadded_batch_pre_prompts = [
        "It's always locked",
        "It's always blocked",
    ]
    padded_batch_post_prompts = [
        " by the magistrate",
        " than to freeze here in the South",
    ]
    unpadded_batch_post_prompts = [
        " by the magistrate",
        " by the candidate",
    ]

    first_post_prompt_tokens = model.to_tokens(padded_batch_post_prompts[0], prepend_bos=False)
    first_full_prompt_tokens = t.cat(
        [model.to_tokens(padded_batch_pre_prompts[0]), first_post_prompt_tokens], dim=-1
    )
    first_post_prompt_len = first_post_prompt_tokens.shape[-1]
    first_prompt_no_cache_logits = model(first_full_prompt_tokens)
    first_post_prompt_no_cache_logits = first_prompt_no_cache_logits[0, -first_post_prompt_len:]

    if pre_padding is None:
        batch_pre_prompt_tokens = model.to_tokens(unpadded_batch_pre_prompts)
    else:
        assert pre_padding == "left" or pre_padding == "right"
        batch_pre_prompt_tokens = model.to_tokens(
            padded_batch_pre_prompts, padding_side=pre_padding
        )

    if post_padding is None:
        batch_post_prompt_tokens = model.to_tokens(unpadded_batch_post_prompts, prepend_bos=False)
    else:
        assert post_padding == "left" or post_padding == "right"
        batch_post_prompt_tokens = model.to_tokens(
            padded_batch_post_prompts,
            prepend_bos=False,
            padding_side=post_padding,
        )

    past_kv_cache = HookedTransformerKeyValueCache.init_cache(
        model.cfg, model.cfg.device, batch_pre_prompt_tokens.shape[0]
    )
    model(batch_pre_prompt_tokens, past_kv_cache=past_kv_cache, padding_side=pre_padding)
    past_kv_cache.freeze()
    with_cache_logits = model(
        batch_post_prompt_tokens,
        past_kv_cache=past_kv_cache,
        padding_side=post_padding,
        prepend_bos=False,
    )
    if post_padding == "left" or post_padding is None:
        first_post_prompt_with_cache_logits = with_cache_logits[0, -first_post_prompt_len:]
    else:
        assert post_padding == "right"
        first_post_prompt_with_cache_logits = with_cache_logits[0, :first_post_prompt_len]

    no_cache_probs = t.softmax(first_post_prompt_no_cache_logits, dim=-1)
    with_cache_probs = t.softmax(first_post_prompt_with_cache_logits, dim=-1)
    assert t.allclose(no_cache_probs, with_cache_probs, atol=atol)


def test_freeze_cache(pretrained):
    model, atol = pretrained
    pre_prompt = "I went to Staten Island,"
    pre_prompt_tokens = model.to_tokens(pre_prompt)
    post_prompt_1 = " I'm headed to the church to play bingo."
    new_tokens_1 = model.to_tokens(post_prompt_1, prepend_bos=False)
    past_kv_cache_1 = HookedTransformerKeyValueCache.init_cache(
        model.cfg, model.cfg.device, pre_prompt_tokens.shape[0]
    )

    post_prompt_2 = " shine your light on me, Miss Liberty"
    new_tokens_2 = model.to_tokens(post_prompt_2, prepend_bos=False)
    past_kv_cache_2 = HookedTransformerKeyValueCache.init_cache(
        model.cfg, model.cfg.device, pre_prompt_tokens.shape[0]
    )

    model(
        pre_prompt_tokens,
        past_kv_cache=past_kv_cache_1,
    )
    past_kv_cache_1.freeze()
    with_cache_logits_1 = model(
        new_tokens_1,
        past_kv_cache=past_kv_cache_1,
    )

    model(
        pre_prompt_tokens,
        past_kv_cache=past_kv_cache_2,
    )
    past_kv_cache_2.freeze()
    model(
        new_tokens_2,
        past_kv_cache=past_kv_cache_2,
    )

    # Caches frozen at the same point should be identical
    assert len(past_kv_cache_1.entries) == len(past_kv_cache_2.entries)
    for entry_1, entry_2 in zip(past_kv_cache_1.entries, past_kv_cache_2.entries):
        assert entry_1.past_keys.shape == entry_2.past_keys.shape
        assert entry_1.past_values.shape == entry_2.past_values.shape
        assert t.allclose(entry_1.past_keys, entry_2.past_keys, atol=1e-3)
        assert t.allclose(entry_1.past_values, entry_2.past_values, atol=1e-3)

    # Rerunning the same prompt with a different cache that was frozen at the same
    # point should give the same results
    with_cache_2_logits_1 = model(
        new_tokens_1,
        past_kv_cache=past_kv_cache_2,
    )
    assert t.allclose(with_cache_logits_1, with_cache_2_logits_1, atol=atol)

    # Test unfreeze
    past_kv_cache_2.unfreeze()
    with_cache_2_logits_1 = model(
        new_tokens_1,
        past_kv_cache=past_kv_cache_2,
    )
    for entry_1, entry_2 in zip(past_kv_cache_1.entries, past_kv_cache_2.entries):
        assert entry_1.past_keys.shape[1] < entry_2.past_keys.shape[1]
        assert entry_1.past_values.shape[1] < entry_2.past_values.shape[1]

    # Rerunning the same prompt with a different cache should give different
    # results
    assert t.allclose(with_cache_logits_1, with_cache_2_logits_1, atol=atol)
    with_cache_2_logits_1 = model(
        new_tokens_1,
        past_kv_cache=past_kv_cache_2,
    )
    assert not t.allclose(with_cache_logits_1, with_cache_2_logits_1, atol=atol)


def test_kv_cache_with_custom_attention_mask(pretrained):
    model, atol = pretrained
    prompt_pre = "An apple"
    prompt_post = " a day keeps junk the"
    prompt_whole = "An apple a day keeps the"
    tokens_pre = model.to_tokens(prompt_pre)
    tokens_post = model.to_tokens(prompt_post, prepend_bos=False)
    tokens_whole = model.to_tokens(prompt_whole)
    correct_logits = model(tokens_whole)

    past_kv_cache = HookedTransformerKeyValueCache.init_cache(
        model.cfg, model.cfg.device, tokens_pre.shape[0]
    )
    model(tokens_pre, past_kv_cache=past_kv_cache)
    exp_logits = model(
        tokens_post,
        attention_mask=t.tensor([[1, 1, 1, 0, 1]], device=model.cfg.device),
        past_kv_cache=past_kv_cache,
    )
    assert t.allclose(correct_logits[:, -1], exp_logits[:, -1], atol=atol)


def test_kv_cache_and_start_at_layer(pretrained):
    model, atol = pretrained
    pre_prompt = "I went to Staten Island,"
    pre_prompt_tokens = model.to_tokens(pre_prompt)
    pre_prompt_tokens_len = pre_prompt_tokens.shape[-1]
    single_token_post_prompt = " Sharon"
    single_new_token = model.to_tokens(single_token_post_prompt, prepend_bos=False)
    full_prompt_tokens = t.cat([pre_prompt_tokens, single_new_token], dim=-1)
    no_cache_logits = model(full_prompt_tokens)
    assert full_prompt_tokens.shape[-1] == pre_prompt_tokens_len + 1

    past_kv_cache = HookedTransformerKeyValueCache.init_cache(
        model.cfg, model.cfg.device, pre_prompt_tokens.shape[0]
    )
    model(
        pre_prompt_tokens,
        past_kv_cache=past_kv_cache,
    )
    past_kv_cache.freeze()
    _, toks, shortformer_pos_embed, attn_mask = model.input_to_embed(
        single_new_token, past_kv_cache=past_kv_cache
    )
    _, cache = model.run_with_cache(single_new_token, stop_at_layer=4, past_kv_cache=past_kv_cache)
    resid_3 = cache["blocks.3.hook_resid_pre"]
    with_cache_logits = model(
        resid_3,
        start_at_layer=3,
        tokens=toks,
        shortformer_pos_embed=shortformer_pos_embed,
        attention_mask=attn_mask,
        past_kv_cache=past_kv_cache,
    )
    assert t.allclose(no_cache_logits[:, -1], with_cache_logits[:, -1], atol=atol)
    assert t.allclose(no_cache_logits[:, -1:], with_cache_logits, atol=atol)
