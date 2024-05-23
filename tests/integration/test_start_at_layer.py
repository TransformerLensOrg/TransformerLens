"""
Tests for the start_at_layer parameter in HookedTransformer
"""

from typing import Any, Dict

import pytest
import torch

from transformer_lens import HookedTransformer, HookedTransformerConfig


@pytest.fixture
def setup_data() -> Dict[str, Any]:
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
    rand_embed, _, _, _ = model.input_to_embed(rand_input)
    return {"model": model, "rand_input": rand_input, "rand_embed": rand_embed}


def test_start_at_layer_1(setup_data: Dict[str, Any]):
    model, rand_embed = setup_data["model"], setup_data["rand_embed"]
    output, cache = model.run_with_cache(rand_embed, start_at_layer=1)

    assert output is not None
    assert "hook_embed" not in cache.keys()
    assert "hook_pos_embed" not in cache.keys()
    assert "blocks.0.hook_resid_pre" not in cache.keys()
    assert "blocks.1.hook_resid_pre" in cache.keys()
    assert "ln_final.hook_normalized" in cache.keys()


def test_run_with_hooks(setup_data: Dict[str, Any]):
    model, rand_embed = setup_data["model"], setup_data["rand_embed"]

    counting_list = []

    def count_hook(activation, hook):
        counting_list.append(len(counting_list))
        return None

    output = model.run_with_hooks(
        rand_embed,
        start_at_layer=1,
        fwd_hooks=[
            ("hook_embed", count_hook),
            ("blocks.0.attn.hook_k", count_hook),
            ("blocks.1.mlp.hook_pre", count_hook),
            ("blocks.2.attn.hook_k", count_hook),
            ("blocks.2.mlp.hook_pre", count_hook)
            # ("blocks.2.mlp.hook_mid", count_hook),
        ],
    )

    assert output is not None
    assert len(counting_list) == 3


def test_manual_hooks(setup_data: Dict[str, Any]):
    model, rand_embed = setup_data["model"], setup_data["rand_embed"]
    counting_list = []

    def count_hook(activation, hook):
        counting_list.append(len(counting_list))
        return None

    model.hook_embed.add_hook(count_hook)
    model.blocks[0].hook_mlp_out.add_hook(count_hook)
    model.blocks[1].hook_resid_mid.add_hook(count_hook)
    model.blocks[2].attn.hook_z.add_hook(count_hook)

    output = model(rand_embed, start_at_layer=-2)
    assert output is not None
    assert len(counting_list) == 2


def test_start_at_final(setup_data: Dict[str, Any]):
    model, rand_embed = setup_data["model"], setup_data["rand_embed"]
    output, cache = model.run_with_cache(rand_embed, start_at_layer=-1)

    assert output is not None
    assert "hook_embed" not in cache.keys()
    assert "hook_pos_embed" not in cache.keys()
    assert "blocks.0.hook_resid_pre" not in cache.keys()
    assert "blocks.0.hook_resid_post" not in cache.keys()
    assert "blocks.1.hook_resid_pre" not in cache.keys()
    assert "blocks.1.hook_resid_post" not in cache.keys()
    assert "blocks.2.hook_resid_pre" in cache.keys()
    assert "blocks.2.hook_resid_post" in cache.keys()
    assert "blocks.3.hook_resid_pre" not in cache.keys()
    assert "ln_final.hook_normalized" in cache.keys()


def test_no_start_logit_output(setup_data: Dict[str, Any]):
    model, rand_input = setup_data["model"], setup_data["rand_input"]
    output, cache = model.run_with_cache(rand_input, start_at_layer=None)

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


def test_no_start_none_output(setup_data: Dict[str, Any]):
    model, rand_input = setup_data["model"], setup_data["rand_input"]
    output, cache = model.run_with_cache(rand_input, start_at_layer=None, return_type=None)

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


def test_start_and_stop(setup_data: Dict[str, Any]):
    model, rand_embed = setup_data["model"], setup_data["rand_embed"]
    output, cache = model.run_with_cache(rand_embed, start_at_layer=1, stop_at_layer=2)

    assert torch.allclose(output, cache["blocks.1.hook_resid_post"])
    assert "hook_embed" not in cache.keys()
    assert "hook_pos_embed" not in cache.keys()
    assert "blocks.0.hook_resid_pre" not in cache.keys()
    assert "blocks.0.hook_resid_post" not in cache.keys()
    assert "blocks.1.hook_resid_pre" in cache.keys()
    assert "blocks.1.hook_resid_post" in cache.keys()
    assert "blocks.2.hook_resid_pre" not in cache.keys()
    assert "blocks.2.hook_resid_post" not in cache.keys()
    assert "blocks.3.hook_resid_pre" not in cache.keys()
    assert "ln_final.hook_normalized" not in cache.keys()


def test_start_at_layer_kwargs():
    cfg = HookedTransformerConfig(
        n_layers=3,
        d_mlp=8,
        d_model=10,
        d_head=5,
        n_heads=2,
        n_ctx=20,
        d_vocab=50257,
        act_fn="relu",
        positional_embedding_type="shortformer",
        tokenizer_name="gpt2",
    )
    model = HookedTransformer(
        cfg=cfg,
    )
    assert model.tokenizer is not None
    model.tokenizer.padding_side = "left"
    input = "As soon as this ferry boat docks I'm headed to the church to play bingo."

    (
        rand_embed,
        tokens,
        shortformer_pos_embed,
        attention_mask,
    ) = model.input_to_embed(input)
    assert tokens is not None and shortformer_pos_embed is not None and attention_mask is not None

    start_at_layer_output = model(
        rand_embed,
        tokens=tokens,
        shortformer_pos_embed=shortformer_pos_embed,
        attention_mask=attention_mask,
        start_at_layer=0,
        return_type="loss",
    )
    normal_output = model(input, return_type="loss")

    assert start_at_layer_output is not None
    assert normal_output is not None
    assert torch.allclose(start_at_layer_output, normal_output)
