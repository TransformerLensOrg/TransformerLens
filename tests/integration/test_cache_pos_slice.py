# %%

import torch

from transformer_lens import HookedTransformer

MODEL = "tiny-stories-1M"

prompt = "Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum."
model = HookedTransformer.from_pretrained(MODEL)
# %%
d_model = model.cfg.d_model
d_head = model.cfg.d_head
n_heads = model.cfg.n_heads
n_layers = model.cfg.n_layers
# %%


def test_run_with_cache_pos_slice_keep_batch():
    _, cache_no_slice = model.run_with_cache(prompt, return_type=None)
    num_tokens = len(model.tokenizer.encode(prompt))

    for i in range(-1, num_tokens + 1):
        _, cache_with_slice = model.run_with_cache(prompt, return_type=None, pos_slice=i)

        assert cache_with_slice["embed"].shape == torch.Size([1, 1, d_model])
        assert cache_with_slice["q", 0].shape == torch.Size([1, 1, n_heads, d_head])

        assert torch.equal(cache_no_slice["embed"][0, i, :], cache_with_slice["embed"][0, 0, :])
        assert torch.equal(
            cache_no_slice["pos_embed"][0, i, :], cache_with_slice["pos_embed"][0, 0, :]
        )

        for layer in range(n_layers):
            assert torch.equal(
                cache_no_slice["resid_pre", layer][0, i, :],
                cache_with_slice["resid_pre", layer][0, 0, :],
            )
            assert torch.equal(
                cache_no_slice["resid_post", layer][0, i, :],
                cache_with_slice["resid_post", layer][0, 0, :],
            )
            assert torch.equal(
                cache_no_slice["resid_mid", layer][0, i, :],
                cache_with_slice["resid_mid", layer][0, 0, :],
            )
            assert torch.equal(
                cache_no_slice["scale", layer, "ln1"][0, i, :],
                cache_with_slice["scale", layer, "ln1"][0, 0, :],
            )
            assert torch.equal(
                cache_no_slice["scale", layer, "ln2"][0, i, :],
                cache_with_slice["scale", layer, "ln2"][0, 0, :],
            )
            assert torch.equal(
                cache_no_slice["normalized", layer, "ln1"][0, i, :],
                cache_with_slice["normalized", layer, "ln1"][0, 0, :],
            )
            assert torch.equal(
                cache_no_slice["normalized", layer, "ln2"][0, i, :],
                cache_with_slice["normalized", layer, "ln2"][0, 0, :],
            )
            assert torch.equal(
                cache_no_slice[
                    "q",
                    layer,
                ][0, i, :, :],
                cache_with_slice[
                    "q",
                    layer,
                ][0, 0, :, :],
            )
            assert torch.equal(
                cache_no_slice[
                    "k",
                    layer,
                ][0, i, :, :],
                cache_with_slice[
                    "k",
                    layer,
                ][0, 0, :, :],
            )
            assert torch.equal(
                cache_no_slice[
                    "v",
                    layer,
                ][0, i, :, :],
                cache_with_slice[
                    "v",
                    layer,
                ][0, 0, :, :],
            )
            assert torch.equal(
                cache_no_slice[
                    "z",
                    layer,
                ][0, i, :, :],
                cache_with_slice[
                    "z",
                    layer,
                ][0, 0, :, :],
            )
            assert torch.equal(
                cache_no_slice[
                    "attn_scores",
                    layer,
                ][0, :, i, :],
                cache_with_slice[
                    "attn_scores",
                    layer,
                ][0, :, 0, :],
            )
            assert torch.equal(
                cache_no_slice[
                    "pattern",
                    layer,
                ][0, :, i, :],
                cache_with_slice[
                    "pattern",
                    layer,
                ][0, :, 0, :],
            )
            assert torch.equal(
                cache_no_slice["attn_out", layer][0, i, :],
                cache_with_slice["attn_out", layer][0, 0, :],
            )
            assert torch.equal(
                cache_no_slice["pre", layer][0, i, :],
                cache_with_slice["pre", layer][0, 0, :],
            )
            assert torch.equal(
                cache_no_slice["post", layer][0, i, :],
                cache_with_slice["post", layer][0, 0, :],
            )
            assert torch.equal(
                cache_no_slice["mlp_out", layer][0, i, :],
                cache_with_slice["mlp_out", layer][0, 0, :],
            )


def test_run_with_cache_pos_slice_remove_batch():
    _, cache_no_slice = model.run_with_cache(prompt, remove_batch_dim=True, return_type=None)
    num_tokens = len(model.tokenizer.encode(prompt))

    for i in range(-1, num_tokens + 1):
        _, cache_with_slice = model.run_with_cache(prompt, remove_batch_dim=True, pos_slice=i)

        assert cache_with_slice["embed"].shape == torch.Size([1, d_model])
        assert cache_with_slice["q", 0].shape == torch.Size([1, n_heads, d_head])

        assert torch.equal(cache_no_slice["embed"][i, :], cache_with_slice["embed"][0, :])
        assert torch.equal(cache_no_slice["pos_embed"][i, :], cache_with_slice["pos_embed"][0, :])

        for layer in range(n_layers):
            assert torch.equal(
                cache_no_slice["resid_pre", layer][i, :],
                cache_with_slice["resid_pre", layer][0, :],
            )
            assert torch.equal(
                cache_no_slice["resid_post", layer][i, :],
                cache_with_slice["resid_post", layer][0, :],
            )
            assert torch.equal(
                cache_no_slice["resid_mid", layer][i, :],
                cache_with_slice["resid_mid", layer][0, :],
            )
            assert torch.equal(
                cache_no_slice["scale", layer, "ln1"][i, :],
                cache_with_slice["scale", layer, "ln1"][0, :],
            )
            assert torch.equal(
                cache_no_slice["scale", layer, "ln2"][i, :],
                cache_with_slice["scale", layer, "ln2"][0, :],
            )
            assert torch.equal(
                cache_no_slice["normalized", layer, "ln1"][i, :],
                cache_with_slice["normalized", layer, "ln1"][0, :],
            )
            assert torch.equal(
                cache_no_slice["normalized", layer, "ln2"][i, :],
                cache_with_slice["normalized", layer, "ln2"][0, :],
            )
            assert torch.equal(
                cache_no_slice[
                    "q",
                    layer,
                ][i, :, :],
                cache_with_slice[
                    "q",
                    layer,
                ][0, :, :],
            )
            assert torch.equal(
                cache_no_slice[
                    "k",
                    layer,
                ][i, :, :],
                cache_with_slice[
                    "k",
                    layer,
                ][0, :, :],
            )
            assert torch.equal(
                cache_no_slice[
                    "v",
                    layer,
                ][i, :, :],
                cache_with_slice[
                    "v",
                    layer,
                ][0, :, :],
            )
            assert torch.equal(
                cache_no_slice[
                    "z",
                    layer,
                ][i, :, :],
                cache_with_slice[
                    "z",
                    layer,
                ][0, :, :],
            )
            assert torch.equal(
                cache_no_slice[
                    "attn_scores",
                    layer,
                ][:, i, :],
                cache_with_slice[
                    "attn_scores",
                    layer,
                ][:, 0, :],
            )
            assert torch.equal(
                cache_no_slice[
                    "pattern",
                    layer,
                ][:, i, :],
                cache_with_slice[
                    "pattern",
                    layer,
                ][:, 0, :],
            )
            assert torch.equal(
                cache_no_slice["attn_out", layer][i, :],
                cache_with_slice["attn_out", layer][0, :],
            )
            assert torch.equal(
                cache_no_slice["pre", layer][i, :], cache_with_slice["pre", layer][0, :]
            )
            assert torch.equal(
                cache_no_slice["post", layer][i, :],
                cache_with_slice["post", layer][0, :],
            )
            assert torch.equal(
                cache_no_slice["mlp_out", layer][i, :],
                cache_with_slice["mlp_out", layer][0, :],
            )
