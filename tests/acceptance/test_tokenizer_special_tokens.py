from transformers import AutoTokenizer

import transformer_lens.loading_from_pretrained as loading
from transformer_lens import HookedTransformer, HookedTransformerConfig

# Get's tedious typing these out everytime I want to sweep over all the distinct small models
MODEL_TESTING_LIST = [
    "solu-1l",
    "gpt2-small",
    "gpt-neo-125M",
    "opt-125m",
    "opt-30b",
    "stanford-gpt2-small-a",
    "pythia-70m",
]


def test_d_vocab_from_tokenizer():
    cfg = HookedTransformerConfig(
        n_layers=1, d_mlp=10, d_model=10, d_head=5, n_heads=2, n_ctx=20, act_fn="relu"
    )
    test_string = "a fish."
    # Test tokenizers for different models
    for model_name in MODEL_TESTING_LIST:
        if model_name == "solu-1l":
            tokenizer_name = "NeelNanda/gpt-neox-tokenizer-digits"
        else:
            tokenizer_name = loading.get_official_model_name(model_name)

        model = HookedTransformer(cfg=cfg, tokenizer=AutoTokenizer.from_pretrained(tokenizer_name))

        tokens_with_bos = model.to_tokens(test_string)
        tokens_without_bos = model.to_tokens(test_string, prepend_bos=False)

        # Check that the lengths are different by one
        assert (
            tokens_with_bos.shape[-1] == tokens_without_bos.shape[-1] + 1
        ), "BOS Token not added when expected"
        # Check that we don't have BOS when we disable the flag
        assert (
            tokens_without_bos.squeeze()[0] != model.tokenizer.bos_token_id
        ), "BOS token is present when it shouldn't be"
