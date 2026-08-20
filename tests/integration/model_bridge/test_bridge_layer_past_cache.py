"""Architectures that name the KV cache `layer_past` must still populate it.

GPT-NeoX, GPT-J, Bloom, Falcon, MPT, CodeGen and GPT-BigCode take the cache as
`layer_past`; everything modern takes `past_key_values`. Reading only the latter
left the cache empty, so each decode step attended to itself alone and generation
silently ignored the prompt — pythia-1.4b answered every prompt with
" the first time.\\n\\n\\n...".
"""

import pytest
import torch
from transformers import DynamicCache

from transformer_lens.model_bridge import TransformerBridge

MODEL = "EleutherAI/pythia-70m"  # GPTNeoX: takes `layer_past`
PROMPT = "The theory of relativity explains that"


@pytest.fixture(scope="module")
def bridge():
    return TransformerBridge.boot_transformers(MODEL, device="cpu")


def test_forward_populates_a_layer_past_cache(bridge) -> None:
    cache = DynamicCache()
    tokens = bridge.to_tokens(PROMPT)
    with torch.no_grad():
        bridge(tokens, past_key_values=cache, use_cache=True)
    assert (
        cache.get_seq_length() == tokens.shape[1]
    ), f"cache holds {cache.get_seq_length()} of {tokens.shape[1]} tokens"


def test_cached_generation_matches_uncached(bridge) -> None:
    """The cache is an optimization: it must not change what is generated."""
    outputs = {}
    for use_cache in (True, False):
        torch.manual_seed(42)
        outputs[use_cache] = bridge.generate(
            PROMPT,
            max_new_tokens=15,
            do_sample=False,
            verbose=False,
            use_past_kv_cache=use_cache,
        )
    assert outputs[True] == outputs[False], f"cached={outputs[True]!r}\nuncached={outputs[False]!r}"


def test_generation_depends_on_the_prompt(bridge) -> None:
    """Engagement check: an empty cache made every prompt yield the same text,
    which prompt-independent output is the loudest symptom of."""
    torch.manual_seed(42)
    a = bridge.generate(
        "The theory of relativity explains that",
        max_new_tokens=12,
        do_sample=False,
        verbose=False,
        use_past_kv_cache=True,
    )
    torch.manual_seed(42)
    b = bridge.generate(
        "Modern computing relies heavily on",
        max_new_tokens=12,
        do_sample=False,
        verbose=False,
        use_past_kv_cache=True,
    )
    assert (
        a[len("The theory of relativity explains that") :]
        != b[len("Modern computing relies heavily on") :]
    )
