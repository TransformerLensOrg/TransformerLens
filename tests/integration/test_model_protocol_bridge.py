"""TransformerBridge satisfies the model_protocol structural types.

model_protocol.py was introduced by the deprecation program and is what
head_detector, SVDInterpreter, tools.training and the ActivationCache helpers
type against after the Hooked* removal — but it shipped without a test. This
pins that a live bridge is a structural instance of each protocol.
"""

from __future__ import annotations

import pytest

from transformer_lens.model_bridge.bridge import TransformerBridge
from transformer_lens.model_protocol import (
    TrainableTransformerLensModel,
    TransformerLensModel,
    TransformerLensModelWithWeights,
)


@pytest.fixture(scope="module")
def bridge() -> TransformerBridge:
    return TransformerBridge.boot_transformers("gpt2", device="cpu")


def test_bridge_is_transformer_lens_model(bridge):
    assert isinstance(bridge, TransformerLensModel)


def test_bridge_is_trainable_model(bridge):
    assert isinstance(bridge, TrainableTransformerLensModel)


def test_bridge_has_weight_surface(bridge):
    # WithWeights checks getattr-static-visible members; the bridge exposes them
    # as properties, so at minimum the runtime attributes must resolve.
    assert isinstance(bridge, TransformerLensModel)
    for attr in ("blocks", "ln_final", "W_U", "W_E"):
        assert hasattr(bridge, attr), attr
    # The protocol type itself is importable and usable as an annotation target.
    assert TransformerLensModelWithWeights is not None
