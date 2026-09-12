"""Integration test: OV vocab readout and the causal patch gate on a real GPT-2 head.

The only file in the ``svd_circuits`` suite that downloads a pretrained model. Drives the
whole read-then-patch path against layer 9 head 9 (the paper's canonical name-mover head) on
a minimal IOI-style prompt, to check that the readout and the causal gate agree on a real
Bridge rather than only on the unit suite's synthetic and tiny-model fixtures.
"""

from __future__ import annotations

import math

import pytest
import torch

from transformer_lens.model_bridge import TransformerBridge
from transformer_lens.tools.analysis.svd_circuits import (
    decompose_head,
    patch_along_directions,
    vocab_readout,
)

CLEAN_PROMPT = "When Mary and John went to the store, John gave a drink to"
LAYER, HEAD = 9, 9


@pytest.fixture(scope="module")
def gpt2_bridge():
    model = TransformerBridge.boot_transformers("gpt2", device="cpu", dtype=torch.float32)
    model.enable_compatibility_mode()
    return model


def _logit_diff_metric(model):
    mary_token = model.to_single_token(" Mary")
    john_token = model.to_single_token(" John")

    def metric(logits: torch.Tensor) -> float:
        return float(logits[0, -1, mary_token] - logits[0, -1, john_token])

    return metric


def test_svd_circuits_readout_and_patch_gate_on_name_mover_head(gpt2_bridge) -> None:
    decomposition = decompose_head(gpt2_bridge, layer=LAYER, head=HEAD, which=("OV",))
    ov = decomposition.OV
    assert ov is not None

    readout = vocab_readout(gpt2_bridge, ov, k=10)
    assert readout.shape == (gpt2_bridge.cfg.d_vocab, 10)
    assert torch.isfinite(readout).all()

    # Pick the first non-degenerate direction rather than assuming index 0 is isolated.
    top_direction = next(row.idx for row in ov.rank_report if not row.is_degenerate)

    prompt = gpt2_bridge.to_tokens(CLEAN_PROMPT)
    metric = _logit_diff_metric(gpt2_bridge)

    result = patch_along_directions(gpt2_bridge, ov, prompt, metric, keep=[top_direction])

    assert result.retained == [top_direction]
    assert math.isfinite(result.original_metric)
    assert math.isfinite(result.patched_metric)
    assert math.isfinite(result.delta_metric)
    assert math.isfinite(result.baseline_delta_metric)
    assert isinstance(result.gated, bool)
