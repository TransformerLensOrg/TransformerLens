"""Tests for component setup utilities."""


import torch.nn as nn

from tests.mocks.architecture_adapter import mock_model_adapter
from transformer_lens.model_bridge.component_setup import replace_remote_component


class TestComponentSetup:
    """Test suite for component setup utilities."""

    def test_replace_remote_component(self, mock_model_adapter):
        """Test replacing a remote component."""
        new_ln = nn.LayerNorm(10)
        replace_remote_component(new_ln, "ln_final", mock_model_adapter)
        assert mock_model_adapter.ln_final is new_ln
