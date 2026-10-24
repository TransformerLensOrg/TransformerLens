"""BlockBridge derives post-norm wiring from the canonical norm keys."""

import pytest

from transformer_lens.model_bridge.generalized_components import BlockBridge
from transformer_lens.model_bridge.generalized_components.base import (
    GeneralizedComponent,
)


def _block(*norm_keys: str) -> BlockBridge:
    submodules = {key: GeneralizedComponent(name=key) for key in norm_keys}
    submodules["attn"] = GeneralizedComponent(name="self_attn")
    submodules["mlp"] = GeneralizedComponent(name="mlp")
    return BlockBridge(name="model.layers", submodules=submodules)


def test_post_norm_only_block_reads_resid_mid_at_the_mlp() -> None:
    block = _block("ln1_post", "ln2_post")
    assert block.hook_aliases["hook_resid_mid"] == "mlp.hook_in"
    assert block.hook_aliases["hook_attn_out"] == "ln1_post.hook_out"
    assert block.hook_aliases["hook_mlp_out"] == "ln2_post.hook_out"
    assert block.mlp_reads_resid_directly is True


def test_sandwich_block_keeps_the_pre_mlp_norm_as_resid_mid() -> None:
    block = _block("ln1", "ln1_post", "ln2", "ln2_post")
    assert block.hook_aliases["hook_resid_mid"] == "ln2.hook_in"
    assert block.hook_aliases["hook_mlp_out"] == "ln2_post.hook_out"
    assert block.mlp_reads_resid_directly is False


def test_sequential_block_without_an_mlp_side_norm_is_rejected() -> None:
    with pytest.raises(ValueError, match="neither 'ln2' nor 'ln2_post'"):
        _block("ln1")
