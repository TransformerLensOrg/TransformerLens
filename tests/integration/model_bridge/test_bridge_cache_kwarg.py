"""Cached generation through bridges that rebuild attention, on every transformers version.

transformers 5.24 passes the KV cache to attention modules (and T5 blocks) as `cache`
instead of `past_key_values`, and to recurrent mixers and SSM models as `cache` instead of
`cache_params`. Bridges that read only the old names left the cache empty or bypassed it,
so `generate()` (`use_past_kv_cache=True` is the default) gave different tokens from
Hugging Face's own generation. Tiny programmatic configs, no download.
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


# Recurrent layers: Qwen3-Next's gated delta-net next to attention, and pure Mamba-2, which
# `generate()` drives through its stateful-cache loop.
RECURRENT_CASES = {
    "mamba2": (
        "Mamba2ForCausalLM",
        "mamba2",
        dict(
            hidden_size=64,
            num_hidden_layers=2,
            num_heads=8,
            head_dim=16,
            n_groups=1,
            state_size=16,
            conv_kernel=4,
            chunk_size=8,
            **_SHARED,
        ),
    ),
    "qwen3_next": (
        "Qwen3NextForCausalLM",
        "qwen3_next",
        dict(
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=16,
            linear_num_key_heads=2,
            linear_num_value_heads=4,
            linear_key_head_dim=16,
            linear_value_head_dim=16,
            num_experts=4,
            num_experts_per_tok=2,
            moe_intermediate_size=32,
            shared_expert_intermediate_size=32,
            layer_types=["linear_attention", "full_attention"],
            **_SHARED,
        ),
    ),
}


def _build(architecture, model_type, kwargs):
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


def _hf_greedy(reference):
    with torch.no_grad():
        return reference.generate(
            PROMPT, max_new_tokens=NEW_TOKENS, min_new_tokens=NEW_TOKENS, do_sample=False
        )[0, PROMPT.shape[1] :].tolist()


def _bridge_greedy(bridge, use_cache):
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
    return tokens[0, PROMPT.shape[1] :].tolist()


@pytest.fixture(scope="module", params=sorted(CASES), ids=sorted(CASES))
def models(request):
    return _build(*CASES[request.param])


def test_forward_fills_the_cache(models):
    _, bridge = models
    cache = DynamicCache()
    with torch.no_grad():
        bridge(PROMPT, past_key_values=cache, use_cache=True)
    assert cache.get_seq_length() == PROMPT.shape[1]


def test_cached_generation_matches_uncached_and_hf(models):
    reference, bridge = models
    hf_tokens = _hf_greedy(reference)
    assert len(set(hf_tokens)) > 1, "degenerate reference generation"
    assert _bridge_greedy(bridge, True) == _bridge_greedy(bridge, False) == hf_tokens


@pytest.mark.parametrize("name", sorted(RECURRENT_CASES))
def test_recurrent_cached_generation_matches_hf(name):
    reference, bridge = _build(*RECURRENT_CASES[name])
    hf_tokens = _hf_greedy(reference)
    assert len(set(hf_tokens)) > 1, "degenerate reference generation"
    assert _bridge_greedy(bridge, True) == hf_tokens


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
