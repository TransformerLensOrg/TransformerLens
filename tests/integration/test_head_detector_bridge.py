"""Smoke coverage for head_detector on a TransformerBridge.

head_detector.py is kept post-4.0 and typed against the model_protocol surface,
but shipped without a test. This pins that the detectors run end-to-end on a
real bridge (the only model system after the Hooked* removal).
"""

from __future__ import annotations

import pytest
import torch

from transformer_lens.head_detector import HEAD_NAMES, detect_head
from transformer_lens.model_bridge.bridge import TransformerBridge


@pytest.fixture(scope="module")
def bridge() -> TransformerBridge:
    b = TransformerBridge.boot_transformers("gpt2", device="cpu")
    b.enable_compatibility_mode()
    return b


@pytest.mark.parametrize(
    "head_name", ["previous_token_head", "duplicate_token_head", "induction_head"]
)
def test_detect_head_runs_on_bridge(bridge, head_name):
    assert head_name in HEAD_NAMES
    scores = detect_head(bridge, "The cat sat on the cat sat on the mat", head_name)
    assert scores.shape == (bridge.cfg.n_layers, bridge.cfg.n_heads)
    assert torch.isfinite(scores).all()
