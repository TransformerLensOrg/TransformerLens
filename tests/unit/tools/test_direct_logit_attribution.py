"""Unit tests for Direct Logit Attribution guards and argument validation.

These exercise the fast-failing checks (argument validation, the Bridge
compatibility-mode requirement, and the hybrid-architecture refusal) without
loading a real model — using a ``spec``-ed mock TransformerBridge so the checks
fire before any forward pass.
"""

from unittest.mock import MagicMock

import pytest

from transformer_lens.model_bridge import TransformerBridge
from transformer_lens.tools.analysis import direct_logit_attribution


def _mock_bridge(compatibility_mode=True, layer_types=("attn+mlp",)):
    """A mock TransformerBridge that satisfies isinstance checks."""
    bridge = MagicMock(spec=TransformerBridge)
    bridge.compatibility_mode = compatibility_mode
    bridge.layer_types.return_value = list(layer_types)
    return bridge


def test_invalid_unit_raises():
    bridge = _mock_bridge()
    with pytest.raises(ValueError, match="unit must be one of"):
        direct_logit_attribution(bridge, "hi", answer_tokens=" world", unit="neuron")


def test_missing_answer_tokens_raises():
    bridge = _mock_bridge()
    with pytest.raises(ValueError, match="answer_tokens is required"):
        direct_logit_attribution(bridge, "hi")


def test_requires_compatibility_mode():
    bridge = _mock_bridge(compatibility_mode=False)
    with pytest.raises(ValueError, match="compatibility mode"):
        direct_logit_attribution(bridge, "hi", answer_tokens=" world")


def test_rejects_hybrid_architecture():
    bridge = _mock_bridge(layer_types=("attn+mlp", "mamba+mlp"))
    with pytest.raises(NotImplementedError, match="hybrid"):
        direct_logit_attribution(bridge, "hi", answer_tokens=" world")


def test_direct_logit_attribution_is_exported_from_analysis_package():
    from transformer_lens.tools import analysis

    assert analysis.direct_logit_attribution is direct_logit_attribution
    assert hasattr(analysis, "DirectLogitAttribution")
