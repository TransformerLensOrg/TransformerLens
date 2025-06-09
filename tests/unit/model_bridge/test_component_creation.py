"""Tests for component creation utilities."""

from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn

from tests.mocks.architecture_adapter import MockArchitectureAdapter, mock_model
from transformer_lens.model_bridge.component_creation import (
    create_and_replace_components_from_mapping,
    create_bridged_component,
    replace_remote_component,
)
from transformer_lens.model_bridge.generalized_components import (
    AttentionBridge,
    LayerNormBridge,
)
from transformer_lens.model_bridge.types import ComponentMapping, RemoteImport


class TestCreateBridgedComponent:
    """Test suite for create_bridged_component function."""

    @pytest.fixture
    def adapter(self):
        """Create a mock architecture adapter for testing."""
        return MockArchitectureAdapter()

    def test_create_basic_component(self, mock_model, adapter):
        """Test creating a basic component without prepend."""
        adapter.component_mapping = {"model.ln": ("model.ln", LayerNormBridge)}
        remote_import: RemoteImport = ("model.ln", LayerNormBridge)
        component = create_bridged_component(remote_import, mock_model, adapter, name="model.ln")

        assert isinstance(component, LayerNormBridge)
        assert component.name == "model.ln"
        assert isinstance(component.original_component, nn.LayerNorm)
        assert component.architecture_adapter == adapter
        # Check that hook points are initialized
        assert hasattr(component, "hook_scale")
        assert hasattr(component, "hook_normalized")

    def test_create_component_with_prepend(self, mock_model, adapter):
        """Test creating a component with prepend path."""
        adapter.component_mapping = {"blocks.0.ln": ("ln", LayerNormBridge)}
        remote_import: RemoteImport = ("ln", LayerNormBridge)
        component = create_bridged_component(
            remote_import, mock_model, adapter, name="blocks.0.ln", prepend="blocks.0"
        )

        assert isinstance(component, LayerNormBridge)
        assert component.name == "blocks.0.ln"
        assert isinstance(component.original_component, nn.LayerNorm)
        assert component.architecture_adapter == adapter
        # Check that hook points are initialized
        assert hasattr(component, "hook_scale")
        assert hasattr(component, "hook_normalized")

    def test_invalid_remote_import(self, mock_model, adapter):
        """Test that invalid remote import raises ValueError."""
        with pytest.raises(
            ValueError, match="RemoteImport must be a tuple of \\(path, component_type\\)"
        ):
            create_bridged_component(("model.ln",), mock_model, adapter, name="model.ln")

    def test_component_not_found(self, mock_model, adapter):
        """Test that non-existent component raises AttributeError."""
        remote_import: RemoteImport = ("nonexistent.path", LayerNormBridge)
        with pytest.raises(AttributeError, match="Component not found at path"):
            create_bridged_component(remote_import, mock_model, adapter, name="nonexistent.path")

    def test_component_forward(self, mock_model, adapter):
        """Test that the created component can perform forward pass."""
        remote_import: RemoteImport = ("model.ln", LayerNormBridge)
        component = create_bridged_component(remote_import, mock_model, adapter, name="model.ln")

        # Create some test input
        x = torch.randn(5, 10)

        # Test forward pass
        output = component(x)
        assert isinstance(output, torch.Tensor)
        assert output.shape == x.shape

        # Test that hook outputs are stored
        assert "output" in component.hook_outputs
        assert component.hook_outputs["output"].shape == x.shape

    def test_replace_remote_component(self, mock_model, adapter):
        """Test replacing a component on the remote model."""
        # 1. Test replacing a nested component
        remote_import: RemoteImport = ("model.ln", LayerNormBridge)
        bridged_component = create_bridged_component(
            remote_import, mock_model, adapter, name="model.ln"
        )

        # Replace the component
        replace_remote_component(bridged_component, "model.ln", mock_model)

        # Check that it was replaced
        assert isinstance(mock_model.model.ln, LayerNormBridge)
        assert mock_model.model.ln is bridged_component

        # 2. Test replacing a component in a ModuleList
        remote_import_block: RemoteImport = ("ln", LayerNormBridge)
        bridged_component_block = create_bridged_component(
            remote_import_block,
            mock_model,
            adapter,
            name="blocks.0.ln",
            prepend="blocks.0",
        )

        # Replace the component
        replace_remote_component(bridged_component_block, "blocks.0.ln", mock_model)

        # Check that it was replaced
        assert isinstance(mock_model.blocks[0].ln, LayerNormBridge)
        assert mock_model.blocks[0].ln is bridged_component_block


class TestCreateAndReplaceComponentsFromMapping:
    """Test suite for create_and_replace_components_from_mapping."""

    @pytest.fixture
    def mock_model(self):
        """Create a mock model for testing."""
        return MagicMock(spec=nn.Module)

    @pytest.fixture
    def adapter(self):
        """Create a mock architecture adapter for testing."""
        return MockArchitectureAdapter()

    @patch("transformer_lens.model_bridge.component_creation.create_bridged_component")
    @patch("transformer_lens.model_bridge.component_creation.replace_remote_component")
    def test_recursive_component_creation(
        self,
        mock_replace: MagicMock,
        mock_create: MagicMock,
        mock_model: MagicMock,
        adapter: MockArchitectureAdapter,
    ):
        """Test recursive creation and replacement of components."""
        # Create a mock bridged component to be returned by the mock_create function
        mock_create.return_value = MagicMock(spec=nn.Module)

        component_mapping: ComponentMapping = {
            "ln_final": ("ln_final", LayerNormBridge),
            "blocks.0": (
                "h.0",
                MagicMock,
                {
                    "ln1": ("ln_1", LayerNormBridge),
                    "attn": ("attn", AttentionBridge),
                },
            ),
        }

        create_and_replace_components_from_mapping(component_mapping, mock_model, adapter)

        # Check that create_bridged_component was called for all components
        assert mock_create.call_count == 4
        # Check that replace_remote_component was called for all components
        assert mock_replace.call_count == 4

        # Check calls for ln_final
        mock_create.assert_any_call(
            ("ln_final", LayerNormBridge),
            mock_model,
            adapter,
            name="ln_final",
            prepend=None,
        )
        mock_replace.assert_any_call(mock_create.return_value, "ln_final", mock_model)

        # Check calls for block's layer norm
        mock_create.assert_any_call(
            ("ln_1", LayerNormBridge),
            mock_model,
            adapter,
            name="blocks.0.ln1",
            prepend="h.0",
        )
        mock_replace.assert_any_call(mock_create.return_value, "h.0.ln_1", mock_model)

        # Check calls for block's attention
        mock_create.assert_any_call(
            ("attn", AttentionBridge),
            mock_model,
            adapter,
            name="blocks.0.attn",
            prepend="h.0",
        )
        mock_replace.assert_any_call(mock_create.return_value, "h.0.attn", mock_model)

        # Check that the block itself was bridged
        block_mapping = component_mapping["blocks.0"]
        mock_create.assert_any_call(
            (block_mapping[0], block_mapping[1]),
            mock_model,
            adapter,
            name="blocks.0",
            prepend=None,
        )
        mock_replace.assert_any_call(mock_create.return_value, "h.0", mock_model)
