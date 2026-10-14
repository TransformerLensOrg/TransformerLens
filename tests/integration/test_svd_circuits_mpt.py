"""Integration test: SVD Circuits' activation tools on MPT's per-head attention output.

MPT's ALiBi attention reimplements the forward, so these pin that ``project_activations``
reads a real ``hook_result`` and that ``patch_along_directions`` edits it, rather than the
patch silently leaving the forward untouched.
"""

from __future__ import annotations

import pytest
import torch

from transformer_lens.model_bridge import TransformerBridge
from transformer_lens.tools.analysis.svd_circuits import (
    decompose_head,
    patch_along_directions,
    project_activations,
)

MODEL = "hf-internal-testing/tiny-random-MptForCausalLM"
LAYER = 0
PROMPT = torch.tensor([[1, 5, 7, 9, 11]])


@pytest.fixture(scope="module")
def mpt_bridge():
    model = TransformerBridge.boot_transformers(MODEL, device="cpu", dtype=torch.float32)
    model.eval()
    return model


@pytest.mark.parametrize("head", [0, 1])
def test_project_activations_reconstructs_mpt_head_output(mpt_bridge, head) -> None:
    ov = decompose_head(mpt_bridge, LAYER, head, which=("OV",)).OV
    assert ov is not None
    projection = project_activations(mpt_bridge, ov, PROMPT)

    mpt_bridge.set_use_attn_result(True)
    try:
        _, cache = mpt_bridge.run_with_cache(PROMPT)
    finally:
        mpt_bridge.set_use_attn_result(False)
    expected = cache[f"blocks.{LAYER}.attn.hook_result"][0, :, head, :]

    assert projection.coefficients.shape == (PROMPT.shape[1], ov.V.shape[1])
    reconstructed = projection.coefficients @ ov.V.transpose(-2, -1)
    torch.testing.assert_close(reconstructed, expected, atol=1e-5, rtol=1e-5)
    assert mpt_bridge.cfg.use_attn_result is False


def test_patch_along_directions_edits_mpt_head_output(mpt_bridge) -> None:
    """Ablating the head's whole OV span must move the metric; a patch hook that never ran
    would leave it exactly unchanged."""
    ov = decompose_head(mpt_bridge, LAYER, 0, which=("OV",)).OV
    assert ov is not None
    metric = lambda logits: float(logits[0, -1, 0] - logits[0, -1, 1])

    ablated = patch_along_directions(
        mpt_bridge, ov, PROMPT, metric, ablate=list(range(ov.V.shape[1])), threshold=0.0
    )
    assert abs(ablated.delta_metric) > 1e-6
    assert mpt_bridge.cfg.use_attn_result is False
