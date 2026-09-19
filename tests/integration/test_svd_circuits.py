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
from transformer_lens.SVDInterpreter import SVDInterpreter
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

    # readout reproduces the shipped SVDInterpreter reference up to sign, so a U-for-V swap
    # or an all-zeros return in vocab_readout fails here rather than passing a finiteness check.
    reference = SVDInterpreter(gpt2_bridge).get_singular_vectors(
        "OV", LAYER, head_index=HEAD, num_vectors=top_direction + 1
    )[:, 0, top_direction]
    assert torch.allclose(readout[:, top_direction], reference, atol=1e-4) or torch.allclose(
        readout[:, top_direction], -reference, atol=1e-4
    )

    prompt = gpt2_bridge.to_tokens(CLEAN_PROMPT)
    metric = _logit_diff_metric(gpt2_bridge)

    result = patch_along_directions(gpt2_bridge, ov, prompt, metric, keep=[top_direction])

    assert result.retained == [top_direction]
    assert math.isfinite(result.original_metric)
    assert math.isfinite(result.patched_metric)
    assert math.isfinite(result.delta_metric)
    assert math.isfinite(result.baseline_delta_metric)
    assert isinstance(result.gated, bool)

    rank = ov.V.shape[1]

    # Keeping every direction reconstructs the head onto its own full OV span: a no-op, so the
    # metric barely moves. A width-rank in-span control also spans span(V), so it ties this
    # kept projector by construction and the gate needs an explicit threshold. A hook never
    # installed, or one projecting the wrong basis, breaks the delta assertion.
    full_keep = patch_along_directions(
        gpt2_bridge, ov, prompt, metric, keep=list(range(rank)), threshold=1e-4
    )
    assert abs(full_keep.delta_metric) < 1e-4

    # Ablating every direction zeroes the head's whole output, which must move the logit diff.
    # An empty retained set ties the baseline by construction, so gate it with an explicit
    # threshold; only delta_metric's magnitude is asserted here.
    full_ablate = patch_along_directions(
        gpt2_bridge, ov, prompt, metric, ablate=list(range(rank)), threshold=0.0
    )
    assert abs(full_ablate.delta_metric) > 1e-2
