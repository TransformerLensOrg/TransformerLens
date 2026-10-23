"""_enumerate_blocks() discovery of non-standard block-list names (#1791).

``_BLOCK_LIST_ATTRS`` is a fixed tuple of known block-list attribute names
("blocks", "encoder_blocks", "decoder_blocks", "L_blocks", "H_blocks").
Architectures that register their blocks under any other name -- e.g.
Raven/Huginn's ``prelude`` / ``core_block`` / ``coda`` -- silently enumerated
zero blocks, which in turn made ``blocks_with()``, ``composition_layer_indices()``,
and ``attn_head_labels`` silently return empty results instead of the real
per-block data, with no error.

``_enumerate_blocks()`` now falls back to structural discovery for any
``nn.ModuleList`` in ``_modules`` not already covered by the known names,
picking up lists whose items are all ``is_list_item`` components (what
``setup_components`` sets on every block of a stack it builds, and exactly what
the named path accepts) -- so a new adapter's block lists are found
automatically, including block types that never set
``hook_out_is_single_residual_stream`` (T5, AltUp, T5Gemma, DeepSeek V4).
"""
from __future__ import annotations

from types import SimpleNamespace

import torch
import torch.nn as nn

from transformer_lens.model_bridge.generalized_components.base import (
    GeneralizedComponent,
)
from transformer_lens.model_bridge.generalized_components.linear import LinearBridge
from transformer_lens.model_bridge.generalized_components.opaque_block import (
    OpaqueBlockBridge,
)
from transformer_lens.model_bridge.transformer_bridge import TransformerBridge

_BORROWED_HELPERS = (
    "_enumerate_blocks",
    "blocks_with",
    "_resolve_submodule_name",
    "composition_layer_indices",
    "_reject_encoder_decoder_composition",
)


def _stub(modules: dict) -> SimpleNamespace:
    stub = SimpleNamespace(_modules=dict(modules), cfg=SimpleNamespace(n_heads=2))
    for helper in _BORROWED_HELPERS:
        setattr(stub, helper, getattr(TransformerBridge, helper).__get__(stub))
    for name, module in modules.items():
        setattr(stub, name, module)
    return stub


class _MockAttn(nn.Module):
    def forward(self, hidden_states: torch.Tensor, **kwargs) -> torch.Tensor:
        return hidden_states


class _MultiStreamBlock(GeneralizedComponent):
    """A list-item block that is not a single-residual-stream block: it sets
    ``is_list_item`` but not ``hook_out_is_single_residual_stream``, like
    ``T5BlockBridge``, ``AltUpBlockBridge`` and DeepSeek V4's blocks."""

    is_list_item = True


def _opaque_block_with_attn(name: str) -> OpaqueBlockBridge:
    """An OpaqueBlockBridge with a bridged "attn" submodule registered, matching
    how a real adapter (e.g. Raven's) wires a residual-stream block with
    attention -- _resolve_submodule_name only needs "attn" present in
    block._modules, not a fully wired/functional submodule."""
    block = OpaqueBlockBridge(name=name, submodules={})
    block.set_original_component(_MockAttn())
    block.add_module("attn", LinearBridge(name="Wqkv", config=None))
    return block


def test_enumerate_blocks_discovers_non_standard_block_lists():
    """Raven's shape: three separate OpaqueBlockBridge lists named prelude /
    core_block / coda -- none of them "blocks", "encoder_blocks", etc."""
    stub = _stub(
        {
            "prelude": nn.ModuleList(
                [_opaque_block_with_attn(f"transformer.prelude.{i}") for i in range(2)]
            ),
            "core_block": nn.ModuleList(
                [_opaque_block_with_attn(f"transformer.core_block.{i}") for i in range(4)]
            ),
            "coda": nn.ModuleList(
                [_opaque_block_with_attn(f"transformer.coda.{i}") for i in range(2)]
            ),
        }
    )

    result = stub._enumerate_blocks()

    assert [idx for idx, _ in result] == list(range(8))
    # Registration order preserved: prelude, then core_block, then coda.
    assert result[0][1].name == "transformer.prelude.0"
    assert result[2][1].name == "transformer.core_block.0"
    assert result[6][1].name == "transformer.coda.0"


def test_enumerate_blocks_ignores_modulelists_that_are_not_list_items():
    """A ModuleList whose items aren't list-item components must not be swept
    up as a block list -- otherwise any incidental ModuleList on the bridge
    would get misread as layers."""
    stub = _stub({"some_unrelated_list": nn.ModuleList([nn.Linear(2, 2), nn.Linear(2, 2)])})

    assert stub._enumerate_blocks() == []


def test_enumerate_blocks_accepts_list_items_without_the_single_stream_marker():
    """Pins the discovery criterion: ``is_list_item``, not
    ``hook_out_is_single_residual_stream``. The named path enumerates T5 / AltUp /
    T5Gemma / DeepSeek V4 blocks, none of which set the single-stream marker, so
    the same stack under an unknown name must enumerate too instead of silently
    yielding zero blocks."""
    items = [_MultiStreamBlock(name=f"layers.{i}") for i in range(3)]
    assert not any(b.hook_out_is_single_residual_stream for b in items)
    stub = _stub({"layers": nn.ModuleList(items)})

    result = stub._enumerate_blocks()

    assert [block.name for _, block in result] == ["layers.0", "layers.1", "layers.2"]


def test_enumerate_blocks_ignores_components_that_are_not_list_items():
    """The other side of the criterion: a ``GeneralizedComponent`` list that does
    not set ``is_list_item`` (e.g. a stack of non-layer helpers) is excluded."""
    items = [GeneralizedComponent(name=f"helpers.{i}") for i in range(2)]
    stub = _stub({"helpers": nn.ModuleList(items)})

    assert stub._enumerate_blocks() == []


def test_enumerate_blocks_skips_mixed_lists_instead_of_leaking_non_blocks():
    """A ModuleList is only a block list if *every* item is a residual-stream
    block. Checking just the first item let a mixed list through whole, leaking
    unrelated modules into every consumer of _enumerate_blocks()."""
    first_is_block = nn.ModuleList([_opaque_block_with_attn("mixed.0"), nn.Linear(2, 2)])
    last_is_block = nn.ModuleList([nn.Linear(2, 2), _opaque_block_with_attn("mixed2.1")])
    stub = _stub({"first_is_block": first_is_block, "last_is_block": last_is_block})

    assert stub._enumerate_blocks() == []


def test_enumerate_blocks_known_names_keep_priority_and_order():
    """Known names (here just "blocks") are still enumerated via the fixed
    priority list, ahead of any structurally-discovered extra list, and the
    existing decoder-only behavior is unchanged."""
    known = nn.ModuleList([_opaque_block_with_attn(f"blocks.{i}") for i in range(2)])
    extra = nn.ModuleList([_opaque_block_with_attn(f"extra.{i}") for i in range(2)])
    stub = _stub({"extra": extra, "blocks": known})

    result = stub._enumerate_blocks()

    assert [block.name for _, block in result] == [
        "blocks.0",
        "blocks.1",
        "extra.0",
        "extra.1",
    ]


def test_composition_layer_indices_finds_raven_shaped_attention_blocks():
    """End-to-end through blocks_with()/composition_layer_indices(): previously
    this silently returned [] for a Raven-shaped bridge; it must now report
    every attention-bearing block."""
    stub = _stub(
        {
            "prelude": nn.ModuleList(
                [_opaque_block_with_attn(f"transformer.prelude.{i}") for i in range(2)]
            ),
            "core_block": nn.ModuleList(
                [_opaque_block_with_attn(f"transformer.core_block.{i}") for i in range(4)]
            ),
            "coda": nn.ModuleList(
                [_opaque_block_with_attn(f"transformer.coda.{i}") for i in range(2)]
            ),
        }
    )

    assert stub.composition_layer_indices() == list(range(8))
