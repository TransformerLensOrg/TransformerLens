# %%

import pytest
import torch

from transformer_lens import HookedTransformer

MODEL = "tiny-stories-1M"

# Use shorter prompt to reduce test time
prompt = "Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor."


@pytest.fixture(scope="module")
def model():
    """Load model once per module."""
    return HookedTransformer.from_pretrained(MODEL)


@pytest.fixture(scope="module")
def model_config(model):
    """Extract model config once."""
    return {
        "d_model": model.cfg.d_model,
        "d_head": model.cfg.d_head,
        "n_heads": model.cfg.n_heads,
        "n_layers": model.cfg.n_layers,
    }


def test_run_with_cache_pos_slice_keep_batch(model, model_config):
    _, cache_no_slice = model.run_with_cache(prompt, return_type=None)
    num_tokens = len(model.tokenizer.encode(prompt))

    d_model = model_config["d_model"]
    d_head = model_config["d_head"]
    n_heads = model_config["n_heads"]
    n_layers = model_config["n_layers"]

    # Test only a sample of positions to reduce test time
    test_positions = [0, num_tokens // 2, num_tokens - 1, -1]

    for i in test_positions:
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


def test_run_with_cache_pos_slice_remove_batch(model, model_config):
    _, cache_no_slice = model.run_with_cache(prompt, remove_batch_dim=True, return_type=None)
    num_tokens = len(model.tokenizer.encode(prompt))

    d_model = model_config["d_model"]
    d_head = model_config["d_head"]
    n_heads = model_config["n_heads"]
    n_layers = model_config["n_layers"]

    # Test only a sample of positions to reduce test time
    test_positions = [0, num_tokens // 2, num_tokens - 1, -1]

    for i in test_positions:
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
