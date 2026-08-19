"""T5BlockBridge's replacement forward vs HF's own T5Block.

The patch replaces T5Block.forward wholesale, so anything HF does there that
the patch omits is silently dropped for every T5-family model (longt5
inherits). The fp16 inf-clamp was one such omission; the sublayer-output
aliases another.
"""

from __future__ import annotations

import copy

import pytest
import torch
from transformers.models.t5.configuration_t5 import T5Config
from transformers.models.t5.modeling_t5 import T5Block

from transformer_lens.factories.architecture_adapter_factory import (
    ArchitectureAdapterFactory,
)
from transformer_lens.model_bridge.component_setup import setup_submodules


def _tiny_config() -> T5Config:
    return T5Config(
        d_model=16,
        d_kv=4,
        d_ff=32,
        num_layers=2,
        num_heads=4,
        vocab_size=32,
        dropout_rate=0.0,
    )


def _adapter():
    from tests.unit.model_bridge.supported_architectures.helpers import make_bridge_cfg

    cfg = make_bridge_cfg("T5ForConditionalGeneration", d_model=16, n_heads=4, d_head=4, d_vocab=32)
    return ArchitectureAdapterFactory.select_architecture_adapter(cfg)


def _wired_encoder_block(hf_block):
    adapter = _adapter()
    bridge = copy.deepcopy(adapter.component_mapping["encoder_blocks"])
    bridge.set_original_component(hf_block)
    setup_submodules(bridge, adapter, hf_block)
    return bridge


# HF's clamp value, as fp16 actually stores it (65504 - 1000 rounds to 64512).
CLAMP_BOUNDARY = torch.tensor(torch.finfo(torch.float16).max - 1000, dtype=torch.float16).item()


def _blown_block(sublayer: str) -> T5Block:
    """A block whose named sublayer output overflows fp16 without the clamp.

    Both matrices of the sublayer scale by 600: enough for the PRODUCT to
    overflow while the weights stay finite in fp16 — a larger factor turns the
    weights themselves inf and NaN-poisons the block before any clamp.
    """
    torch.manual_seed(0)
    block = T5Block(_tiny_config(), has_relative_attention_bias=True).eval()
    with torch.no_grad():
        if sublayer == "ff":
            block.layer[1].DenseReluDense.wi.weight.mul_(600.0)
            block.layer[1].DenseReluDense.wo.weight.mul_(600.0)
        else:
            block.layer[0].SelfAttention.v.weight.mul_(600.0)
            block.layer[0].SelfAttention.o.weight.mul_(600.0)
    return block


@pytest.mark.parametrize("sublayer", ["attn", "ff"])
def test_fp16_inf_clamp_matches_hf(sublayer) -> None:
    """HF clamps fp16 hidden states after every sublayer; the patched forward
    dropped all of them, so fp16 runs overflowed exactly where HF's do not.

    Clamp ENGAGEMENT is asserted via the boundary value: clamped elements sit
    exactly at ±(finfo.max - 1000), so a fixture that never overflows cannot
    pass the boundary check — the flaw the first draft of this test had.
    """
    hf_block = _blown_block(sublayer)
    reference = copy.deepcopy(hf_block).half()

    _ = _wired_encoder_block(hf_block)
    hf_block.half()

    torch.manual_seed(1)
    hidden = torch.randn(1, 5, 16, dtype=torch.float16)
    with torch.no_grad():
        patched_out = hf_block(hidden)[0]
        # The reference runs HF's ORIGINAL forward (never patched).
        reference_out = reference(hidden)[0]

    assert (
        patched_out.abs() == CLAMP_BOUNDARY
    ).any(), "fixture did not engage the clamp — the parity assert below is vacuous"
    assert not torch.isinf(patched_out).any(), "clamp missing: fp16 overflow escaped"
    torch.testing.assert_close(patched_out, reference_out)


def test_sublayer_output_aliases_resolve_and_fire() -> None:
    """hook_attn_out / hook_mlp_out were absent on every T5-family block; the
    residual add happens inside the sublayer, so the wrapped module's output is
    the added contribution (exactly, in eval with dropout 0)."""
    torch.manual_seed(1)
    hf_block = T5Block(_tiny_config(), has_relative_attention_bias=True).eval()
    bridge = _wired_encoder_block(hf_block)

    assert bridge.hook_aliases["hook_attn_out"] == "attn.hook_out"
    assert bridge.hook_aliases["hook_mlp_out"] == "mlp.hook_out"

    seen: dict = {}
    bridge.hook_in.add_hook(lambda t, hook: seen.__setitem__("pre", t.clone()))
    bridge.hook_resid_mid.add_hook(lambda t, hook: seen.__setitem__("mid", t.clone()))
    bridge.submodules["attn"].hook_out.add_hook(
        lambda t, hook: seen.__setitem__("attn_out", t.clone())
    )
    hidden = torch.randn(1, 5, 16)
    with torch.no_grad():
        hf_block(hidden)

    assert set(seen) == {"pre", "mid", "attn_out"}
    # Contribution identity: resid_mid == resid_pre + attn_out (dropout=0).
    torch.testing.assert_close(seen["mid"], seen["pre"] + seen["attn_out"])


def test_decoder_gets_the_cross_attention_alias() -> None:
    adapter = _adapter()
    decoder = adapter.component_mapping["decoder_blocks"]
    assert decoder.hook_aliases["hook_attn_out"] == "self_attn.hook_out"
    assert decoder.hook_aliases["hook_cross_attn_out"] == "cross_attn.hook_out"
    assert decoder.hook_aliases["hook_mlp_out"] == "mlp.hook_out"
