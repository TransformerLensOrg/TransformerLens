"""Tests for component creation utilities."""

import pytest
import torch
import torch.nn as nn

from transformer_lens.model_bridge.architecture_adapter import ArchitectureAdapter
from transformer_lens.model_bridge.component_creation import create_bridged_component
from transformer_lens.model_bridge.generalized_components import LayerNormBridge
from transformer_lens.model_bridge.types import RemoteImport


class MockArchitectureAdapter(ArchitectureAdapter):
    """Mock architecture adapter for testing."""

    def __init__(self, cfg=None):
        super().__init__(cfg)
        self.component_mapping = {}

    def get_remote_component(self, model: nn.Module, path: str) -> nn.Module:
        """Mock implementation that returns a component based on the path."""
        if path == "model.ln":
            return nn.LayerNorm(10)
        elif path == "blocks.0.ln":
            return nn.LayerNorm(10)
        else:
            raise AttributeError(f"Component not found at path: {path}")


class TestCreateBridgedComponent:
    """Test suite for create_bridged_component function."""

    @pytest.fixture
    def mock_model(self):
        """Create a mock model for testing."""
        model = nn.Module()
        model.model = nn.Module()
        model.model.ln = nn.LayerNorm(10)
        model.blocks = nn.ModuleList([nn.Module()])
        model.blocks[0].ln = nn.LayerNorm(10)
        return model

    @pytest.fixture
    def adapter(self):
        """Create a mock architecture adapter for testing."""
        return MockArchitectureAdapter()

    def test_create_basic_component(self, mock_model, adapter):
        """Test creating a basic component without prepend."""
        remote_import: RemoteImport = ("model.ln", LayerNormBridge)
        component = create_bridged_component(remote_import, mock_model, adapter)

        assert isinstance(component, LayerNormBridge)
        assert component.name == "model.ln"
        assert isinstance(component.original_component, nn.LayerNorm)
        assert component.architecture_adapter == adapter
        # Check that hook points are initialized
        assert hasattr(component, "hook_scale")
        assert hasattr(component, "hook_normalized")

    def test_create_component_with_prepend(self, mock_model, adapter):
        """Test creating a component with prepend path."""
        remote_import: RemoteImport = ("ln", LayerNormBridge)
        component = create_bridged_component(remote_import, mock_model, adapter, prepend="blocks.0")

        assert isinstance(component, LayerNormBridge)
        assert component.name == "blocks.0.ln"
        assert isinstance(component.original_component, nn.LayerNorm)
        assert component.architecture_adapter == adapter
        # Check that hook points are initialized
        assert hasattr(component, "hook_scale")
        assert hasattr(component, "hook_normalized")

    def test_invalid_remote_import(self, mock_model, adapter):
        """Test that invalid remote import raises ValueError."""
        with pytest.raises(ValueError, match="RemoteImport must be a tuple of \\(path, component_type\\)"):
            create_bridged_component(("model.ln",), mock_model, adapter)

    def test_component_not_found(self, mock_model, adapter):
        """Test that non-existent component raises AttributeError."""
        remote_import: RemoteImport = ("nonexistent.path", LayerNormBridge)
        with pytest.raises(AttributeError, match="Component not found at path"):
            create_bridged_component(remote_import, mock_model, adapter)

    def test_component_forward(self, mock_model, adapter):
        """Test that the created component can perform forward pass."""
        remote_import: RemoteImport = ("model.ln", LayerNormBridge)
        component = create_bridged_component(remote_import, mock_model, adapter)

        # Create some test input
        x = torch.randn(5, 10)
        
        # Test forward pass
        output = component(x)
        assert isinstance(output, torch.Tensor)
        assert output.shape == x.shape
        
        # Test that hook outputs are stored
        assert "output" in component.hook_outputs
        assert component.hook_outputs["output"].shape == x.shape 