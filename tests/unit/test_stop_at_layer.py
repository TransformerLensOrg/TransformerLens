"""
Tests for the stop_at_layer parameter in HookedTransformer
"""

import torch

from transformer_lens import HookedTransformer, HookedTransformerConfig


def test_stop_at_embed():
    cfg = HookedTransformerConfig(
        n_layers=3,
        d_mlp=8,
        d_model=10,
        d_head=5,
        n_heads=2,
        n_ctx=20,
        d_vocab=50,
        act_fn="relu",
    )
    model = HookedTransformer(
        cfg=cfg,
    )
    rand_input = torch.randint(0, 20, (2, 10))

    output, cache = model.run_with_cache(rand_input, stop_at_layer=0)

    assert output is None
    assert "hook_embed" in cache.keys()
    assert "hook_pos_embed" in cache.keys()
    assert "blocks.0.hook_resid_pre" not in cache.keys()
    assert "ln_final.hook_normalized" not in cache.keys()


def test_run_with_hooks():
    cfg = HookedTransformerConfig(
        n_layers=3,
        d_mlp=8,
        d_model=10,
        d_head=5,
        n_heads=2,
        n_ctx=20,
        d_vocab=50,
        act_fn="relu",
    )
    model = HookedTransformer(
        cfg=cfg,
    )
    rand_input = torch.randint(0, 20, (2, 10))

    counting_list = []

    def count_hook(activation, hook):
        counting_list.append(len(counting_list))
        return None

    output = model.run_with_hooks(
        rand_input,
        stop_at_layer=1,
        fwd_hooks=[
            ("hook_embed", count_hook),
            ("blocks.0.attn.hook_k", count_hook),
            ("blocks.1.mlp.hook_pre", count_hook),
        ],
    )

    assert output is None
    assert len(counting_list) == 2


def test_manual_hooks():
    cfg = HookedTransformerConfig(
        n_layers=3,
        d_mlp=8,
        d_model=10,
        d_head=5,
        n_heads=2,
        n_ctx=20,
        d_vocab=50,
        act_fn="relu",
    )
    model = HookedTransformer(
        cfg=cfg,
    )
    rand_input = torch.randint(0, 20, (2, 10))

    counting_list = []

    def count_hook(activation, hook):
        counting_list.append(len(counting_list))
        return None

    model.hook_embed.add_hook(count_hook)
    model.blocks[0].hook_mlp_out.add_hook(count_hook)
    model.blocks[1].hook_resid_mid.add_hook(count_hook)
    model.blocks[2].attn.hook_z.add_hook(count_hook)

    output = model(rand_input, stop_at_layer=-1)
    assert output is None
    assert len(counting_list) == 3


def test_stop_at_layer_1():
    cfg = HookedTransformerConfig(
        n_layers=3,
        d_mlp=8,
        d_model=10,
        d_head=5,
        n_heads=2,
        n_ctx=20,
        d_vocab=50,
        act_fn="relu",
    )
    model = HookedTransformer(
        cfg=cfg,
    )
    rand_input = torch.randint(0, 20, (2, 10))

    output, cache = model.run_with_cache(rand_input, stop_at_layer=1)

    assert output is None
    assert "hook_embed" in cache.keys()
    assert "hook_pos_embed" in cache.keys()
    assert "blocks.0.hook_resid_pre" in cache.keys()
    assert "blocks.0.hook_resid_post" in cache.keys()
    assert "blocks.1.hook_resid_pre" not in cache.keys()
    assert "ln_final.hook_normalized" not in cache.keys()
    for key in cache.keys():
        if key.startswith("blocks.0"):
            continue
        elif key in ["hook_embed", "hook_pos_embed"]:
            continue
        else:
            assert False, f"Unexpected key {key} in cache."


def test_stop_at_final():
    cfg = HookedTransformerConfig(
        n_layers=4,
        d_mlp=8,
        d_model=10,
        d_head=5,
        n_heads=2,
        n_ctx=20,
        d_vocab=50,
        act_fn="relu",
    )
    model = HookedTransformer(
        cfg=cfg,
    )
    rand_input = torch.randint(0, 20, (2, 10))

    output, cache = model.run_with_cache(rand_input, stop_at_layer=-1)

    assert output is None
    assert "hook_embed" in cache.keys()
    assert "hook_pos_embed" in cache.keys()
    assert "blocks.0.hook_resid_pre" in cache.keys()
    assert "blocks.0.hook_resid_post" in cache.keys()
    assert "blocks.1.hook_resid_pre" in cache.keys()
    assert "blocks.1.hook_resid_post" in cache.keys()
    assert "blocks.2.hook_resid_pre" in cache.keys()
    assert "blocks.2.hook_resid_post" in cache.keys()
    assert "blocks.3.hook_resid_pre" not in cache.keys()
    assert "ln_final.hook_normalized" not in cache.keys()


def test_no_stop_logit_output():
    cfg = HookedTransformerConfig(
        n_layers=3,
        d_mlp=8,
        d_model=10,
        d_head=5,
        n_heads=2,
        n_ctx=20,
        d_vocab=50,
        act_fn="relu",
    )
    model = HookedTransformer(
        cfg=cfg,
    )
    rand_input = torch.randint(0, 20, (2, 10))

    output, cache = model.run_with_cache(rand_input, stop_at_layer=None)

    assert output is not None
    assert isinstance(output, torch.Tensor)
    assert output.shape == (2, 10, 50)

    assert "hook_embed" in cache.keys()
    assert "hook_pos_embed" in cache.keys()
    assert "blocks.0.hook_resid_pre" in cache.keys()
    assert "blocks.0.hook_resid_post" in cache.keys()
    assert "blocks.1.hook_resid_pre" in cache.keys()
    assert "blocks.1.hook_resid_post" in cache.keys()
    assert "blocks.2.hook_resid_pre" in cache.keys()
    assert "blocks.2.hook_resid_post" in cache.keys()
    assert "ln_final.hook_normalized" in cache.keys()


def test_no_stop_no_output():
    cfg = HookedTransformerConfig(
        n_layers=3,
        d_mlp=8,
        d_model=10,
        d_head=5,
        n_heads=2,
        n_ctx=20,
        d_vocab=50,
        act_fn="relu",
    )
    model = HookedTransformer(
        cfg=cfg,
    )
    rand_input = torch.randint(0, 20, (2, 10))

    output, cache = model.run_with_cache(
        rand_input, stop_at_layer=None, return_type=None
    )

    assert output is None
    assert "hook_embed" in cache.keys()
    assert "hook_pos_embed" in cache.keys()
    assert "blocks.0.hook_resid_pre" in cache.keys()
    assert "blocks.0.hook_resid_post" in cache.keys()
    assert "blocks.1.hook_resid_pre" in cache.keys()
    assert "blocks.1.hook_resid_post" in cache.keys()
    assert "blocks.2.hook_resid_pre" in cache.keys()
    assert "blocks.2.hook_resid_post" in cache.keys()
    assert "ln_final.hook_normalized" in cache.keys()
