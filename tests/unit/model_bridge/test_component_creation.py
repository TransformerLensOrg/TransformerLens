"""Tests for component creation utilities."""


import torch.nn as nn

from tests.mocks.architecture_adapter import MockArchitectureAdapter, mock_model_adapter
from transformer_lens.model_bridge.component_creation import (
    create_bridged_component,
    replace_remote_component,
)
from transformer_lens.model_bridge.generalized_components import LayerNormBridge


class TestComponentCreation:
    """Test suite for component creation utilities."""

    def test_create_bridged_component(self, mock_model_adapter):
        """Test creating a bridged component."""
        adapter = MockArchitectureAdapter()
        component = create_bridged_component(
            mock_model_adapter,
            adapter,
            "ln_final",
            LayerNormBridge,
        )
        assert isinstance(component, LayerNormBridge)
        assert component.name == "ln_final"
        assert isinstance(component.original_component, nn.LayerNorm)

    def test_replace_remote_component(self, mock_model_adapter):
        """Test replacing a remote component."""
        new_ln = nn.LayerNorm(10)
        replace_remote_component(new_ln, "ln_final", mock_model_adapter)
        assert mock_model_adapter.ln_final is new_ln
