"""Unit tests for the Direct Logit Attribution guards and argument validation.

These exercise the fast-failing checks (argument validation, the
compatibility-mode requirement, and the hybrid-architecture refusal) without
loading a real model, using a ``spec``-ed mock bridge so the checks are reached
before any forward pass.
"""

from unittest.mock import MagicMock

import pytest
import torch

from transformer_lens.model_bridge import TransformerBridge
from transformer_lens.tools.analysis import dla


def _mock_bridge(compatibility_mode=True, layer_types=("attn+mlp",)):
    """A mock TransformerBridge that satisfies isinstance/type checks."""
    bridge = MagicMock(spec=TransformerBridge)
    bridge.compatibility_mode = compatibility_mode
    bridge.layer_types.return_value = list(layer_types)
    return bridge


def test_prompt_answer_length_mismatch_raises():
    bridge = _mock_bridge()
    with pytest.raises(ValueError, match="matching row"):
        dla(bridge, ["a", "b"], torch.tensor([[1]]))


def test_invalid_answer_columns_raises():
    bridge = _mock_bridge()
    with pytest.raises(ValueError, match="columns"):
        dla(bridge, ["a"], torch.tensor([[1, 2, 3]]))


def test_requires_compatibility_mode():
    bridge = _mock_bridge(compatibility_mode=False)
    with pytest.raises(ValueError, match="compatibility mode"):
        dla(bridge, ["a"], torch.tensor([[1]]))


def test_rejects_hybrid_architecture():
    bridge = _mock_bridge(layer_types=("attn+mlp", "mamba+mlp"))
    with pytest.raises(NotImplementedError, match="hybrid"):
        dla(bridge, ["a"], torch.tensor([[1]]))


def test_dla_is_exported_from_analysis_package():
    from transformer_lens.tools import analysis

    assert analysis.dla is dla
