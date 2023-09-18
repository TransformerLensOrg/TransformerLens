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
