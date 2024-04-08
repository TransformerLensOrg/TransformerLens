# %%

import functools

import torch as t
from jaxtyping import Int

from transformer_lens import HookedTransformer, HookedTransformerConfig
from transformer_lens.hook_points import HookPoint


def test_patch_tokens():
    # Define small transformer
    cfg = HookedTransformerConfig(
        n_layers=1,
        d_mlp=10,
        d_model=10,
        d_head=5,
        n_heads=2,
        n_ctx=20,
        act_fn="relu",
        tokenizer_name="gpt2",
        use_hook_tokens=True,
    )
    model = HookedTransformer(cfg=cfg)

    # Define short prompt, and a token to replace the first token with (note this is index 1, because BOS)
    prompt = "Hello World!"
    modified_prompt = "Hi World!"
    new_first_token = model.to_single_token("Hi")

    # Define hook function to alter the first token
    def hook_fn(tokens: Int[t.Tensor, "batch seq"], hook: HookPoint, new_first_token: int):
        assert (
            tokens[0, 0].item() != new_first_token
        )  # Need new_first_token to be different from original
        tokens[0, 0] = new_first_token
        return tokens

    # Run with hooks
    out_from_hook = model.run_with_hooks(
        prompt,
        prepend_bos=False,
        fwd_hooks=[("hook_tokens", functools.partial(hook_fn, new_first_token=new_first_token))],
    )

    out_direct = model(modified_prompt, prepend_bos=False)

    t.testing.assert_close(out_from_hook, out_direct)
