# %%
import torch as t

from transformer_lens import HookedTransformer, utils
from transformer_lens.past_key_value_caching import HookedTransformerKeyValueCache

MODEL = "solu-1l"
model = HookedTransformer.from_pretrained(MODEL)

pre_prompt = "I went to Staten Island,"
padding_side = "left"
prepend_bos = True
pre_prompt_tokens = model.to_tokens(
    pre_prompt, prepend_bos=prepend_bos, padding_side=padding_side
)


def test_single_new_token():
    post_prompt = " Sharon"
    new_token = model.to_tokens(post_prompt, prepend_bos=False)
    full_prompt_tokens = t.cat([pre_prompt_tokens, new_token], dim=-1)
    assert full_prompt_tokens.shape[-1] == pre_prompt_tokens.shape[-1] + 1
    no_cache_logits = model(full_prompt_tokens, padding_side=padding_side)

    past_kv_cache = HookedTransformerKeyValueCache.init_cache(
        model.cfg, model.cfg.device, pre_prompt_tokens.shape[0]
    )
    model(
        pre_prompt_tokens,
        padding_side=padding_side,
        past_kv_cache=past_kv_cache,
        past_left_attention_mask=None,
    )
    past_left_attention_mask = utils.get_attention_mask(
        model.tokenizer,
        pre_prompt_tokens,
        model.cfg.default_prepend_bos,
    )
    with_cache_logits = model(
        new_token,
        padding_side=padding_side,
        past_kv_cache=past_kv_cache,
        past_left_attention_mask=past_left_attention_mask,
    )
    print("no_cache_logits", no_cache_logits[:, -1])
    print("with_cache_logits", with_cache_logits[:, -1])
    assert t.allclose(no_cache_logits[:, -1], with_cache_logits[:, -1], atol=1e-3)
    assert t.allclose(no_cache_logits[:, -1:], with_cache_logits, atol=1e-3)


def test_multiple_new_tokens():
    post_prompt = " to buy myself a mandolin"
    new_tokens = model.to_tokens(post_prompt, prepend_bos=False)
    new_tokens_len = new_tokens.shape[-1]
    full_prompt_tokens = t.cat([pre_prompt_tokens, new_tokens], dim=-1)
    assert full_prompt_tokens.shape[-1] == pre_prompt_tokens.shape[-1] + new_tokens_len
    no_cache_logits = model(full_prompt_tokens, padding_side=padding_side)

    past_kv_cache = HookedTransformerKeyValueCache.init_cache(
        model.cfg, model.cfg.device, pre_prompt_tokens.shape[0]
    )
    model(
        pre_prompt_tokens,
        padding_side=padding_side,
        past_kv_cache=past_kv_cache,
        past_left_attention_mask=None,
    )
    past_left_attention_mask = utils.get_attention_mask(
        model.tokenizer,
        pre_prompt_tokens,
        model.cfg.default_prepend_bos,
    )
    with_cache_logits = model(
        new_tokens,
        padding_side=padding_side,
        past_kv_cache=past_kv_cache,
        past_left_attention_mask=past_left_attention_mask,
    )
    assert t.allclose(no_cache_logits[:, -1], with_cache_logits[:, -1], atol=1e-3)
    assert t.allclose(
        no_cache_logits[:, -new_tokens_len:], with_cache_logits, atol=1e-3
    )


def test_freeze_cache():
    past_left_attention_mask = utils.get_attention_mask(
        model.tokenizer,
        pre_prompt_tokens,
        model.cfg.default_prepend_bos,
    )

    post_prompt_1 = " I'm headed to the church to play bingo."
    new_tokens_1 = model.to_tokens(post_prompt_1, prepend_bos=False)
    full_prompt_tokens_1 = t.cat([pre_prompt_tokens, new_tokens_1], dim=-1)
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
        padding_side=padding_side,
        past_kv_cache=past_kv_cache_1,
        past_left_attention_mask=None,
    )
    past_kv_cache_1.freeze()
    with_cache_logits_1 = model(
        new_tokens_1,
        padding_side=padding_side,
        past_kv_cache=past_kv_cache_1,
        past_left_attention_mask=past_left_attention_mask,
    )

    model(
        pre_prompt_tokens,
        padding_side=padding_side,
        past_kv_cache=past_kv_cache_2,
        past_left_attention_mask=None,
    )
    past_kv_cache_2.freeze()
    model(
        new_tokens_2,
        padding_side=padding_side,
        past_kv_cache=past_kv_cache_2,
        past_left_attention_mask=past_left_attention_mask,
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
        padding_side=padding_side,
        past_kv_cache=past_kv_cache_2,
        past_left_attention_mask=past_left_attention_mask,
    )
    assert t.allclose(with_cache_logits_1, with_cache_2_logits_1, atol=1e-3)

    # Test unfreeze
    past_kv_cache_2.unfreeze()
    with_cache_2_logits_1 = model(
        new_tokens_1,
        padding_side=padding_side,
        past_kv_cache=past_kv_cache_2,
        past_left_attention_mask=past_left_attention_mask,
    )
    for entry_1, entry_2 in zip(past_kv_cache_1.entries, past_kv_cache_2.entries):
        assert entry_1.past_keys.shape[1] < entry_2.past_keys.shape[1]
        assert entry_1.past_values.shape[1] < entry_2.past_values.shape[1]

    # Rerunning the same prompt with a different cache should give different
    # results
    assert t.allclose(with_cache_logits_1, with_cache_2_logits_1, atol=1e-3)
    past_left_attention_mask = utils.get_attention_mask(
        model.tokenizer,
        full_prompt_tokens_1,
        model.cfg.default_prepend_bos,
    )
    with_cache_2_logits_1 = model(
        new_tokens_1,
        padding_side=padding_side,
        past_kv_cache=past_kv_cache_2,
        past_left_attention_mask=past_left_attention_mask,
    )
    assert not t.allclose(with_cache_logits_1, with_cache_2_logits_1, atol=1e-3)
