"""Cached generation through bridges that rebuild attention, on every transformers version.

transformers 5.24 passes the KV cache to attention modules (and T5 blocks) as `cache`
instead of `past_key_values`. Bridges that rebuild attention read only the old names, so
the cache stayed empty and `generate()` (`use_past_kv_cache=True` is the default) gave
different tokens from Hugging Face's own generation. Tiny programmatic configs, no download.
"""

import copy

import pytest
import torch
from transformers import AutoConfig, AutoModelForCausalLM, DynamicCache

from transformer_lens.model_bridge.sources import build_bridge_from_module

_SHARED = dict(vocab_size=128, pad_token_id=0, bos_token_id=1, eos_token_id=2)

# (architecture, model_type, config): joint QKV, rotary with GQA, MLA.
CASES = {
    "gpt2": (
        "GPT2LMHeadModel",
        "gpt2",
        dict(n_embd=64, n_layer=2, n_head=4, n_positions=64, **_SHARED),
    ),
    "llama": (
        "LlamaForCausalLM",
        "llama",
        dict(
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            **_SHARED,
        ),
    ),
    "deepseek_v3": (
        "DeepseekV3ForCausalLM",
        "deepseek_v3",
        dict(
            hidden_size=64,
            intermediate_size=128,
            moe_intermediate_size=32,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=4,
            q_lora_rank=32,
            kv_lora_rank=16,
            qk_rope_head_dim=8,
            qk_nope_head_dim=8,
            v_head_dim=12,
            n_routed_experts=4,
            num_experts_per_tok=2,
            n_shared_experts=1,
            first_k_dense_replace=1,
            n_group=1,
            topk_group=1,
            **_SHARED,
        ),
    ),
}
PROMPT = torch.tensor([[5, 17, 42, 9, 77, 31]])
NEW_TOKENS = 6


@pytest.fixture(scope="module", params=sorted(CASES), ids=sorted(CASES))
def models(request):
    architecture, model_type, kwargs = CASES[request.param]
    hf_config = AutoConfig.for_model(model_type, **kwargs)
    torch.manual_seed(0)
    model = AutoModelForCausalLM.from_config(hf_config, attn_implementation="eager").eval()
    with torch.no_grad():
        # Large enough weights that greedy decoding does not repeat one token.
        for param in model.parameters():
            param.add_(0.2 * torch.randn_like(param))
    reference = copy.deepcopy(model)
    bridge = build_bridge_from_module(model, architecture, hf_config=hf_config).eval()
    return reference, bridge


def test_forward_fills_the_cache(models):
    _, bridge = models
    cache = DynamicCache()
    with torch.no_grad():
        bridge(PROMPT, past_key_values=cache, use_cache=True)
    assert cache.get_seq_length() == PROMPT.shape[1]


def test_cached_generation_matches_uncached_and_hf(models):
    reference, bridge = models
    with torch.no_grad():
        hf_tokens = reference.generate(
            PROMPT, max_new_tokens=NEW_TOKENS, min_new_tokens=NEW_TOKENS, do_sample=False
        )[0, PROMPT.shape[1] :].tolist()
    generated = {}
    for use_cache in (True, False):
        with torch.no_grad():
            tokens = bridge.generate(
                PROMPT,
                max_new_tokens=NEW_TOKENS,
                do_sample=False,
                use_past_kv_cache=use_cache,
                stop_at_eos=False,
                verbose=False,
                return_type="tokens",
            )
        generated[use_cache] = tokens[0, PROMPT.shape[1] :].tolist()
    assert len(set(hf_tokens)) > 1, "degenerate reference generation"
    assert generated[True] == generated[False] == hf_tokens


def test_t5_cached_generation_matches_uncached_and_hf():
    """The patched T5 block passes the cache on to its self- and cross-attention sublayers."""
    from transformers import T5ForConditionalGeneration

    hf_config = AutoConfig.for_model(
        "t5",
        vocab_size=128,
        d_model=64,
        d_kv=16,
        d_ff=128,
        num_layers=2,
        num_decoder_layers=2,
        num_heads=4,
        pad_token_id=0,
        eos_token_id=1,
        decoder_start_token_id=0,
    )
    torch.manual_seed(0)
    model = T5ForConditionalGeneration(hf_config).eval()
    with torch.no_grad():
        for param in model.parameters():
            param.add_(0.5 * torch.randn_like(param))
    reference = copy.deepcopy(model)
    source = torch.randint(3, 128, (1, 8), generator=torch.Generator().manual_seed(1))
    bridge = build_bridge_from_module(
        model, "T5ForConditionalGeneration", hf_config=hf_config
    ).eval()
    with torch.no_grad():
        hf_tokens = reference.generate(
            source, max_new_tokens=NEW_TOKENS, min_new_tokens=NEW_TOKENS, do_sample=False
        )[0].tolist()
    generated = {}
    for use_cache in (True, False):
        with torch.no_grad():
            # The bridged HF model runs the patched T5 blocks.
            tokens = bridge.original_model.generate(
                source,
                max_new_tokens=NEW_TOKENS,
                min_new_tokens=NEW_TOKENS,
                do_sample=False,
                use_cache=use_cache,
            )
        generated[use_cache] = tokens[0].tolist()
    assert len(set(hf_tokens[1:])) > 1, "degenerate reference generation"
    assert generated[True] == generated[False] == hf_tokens
