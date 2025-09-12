"""Unit tests for the TransformerBridge class.

This module tests the bridge functionality, including component mapping formatting
and other bridge operations.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from tests.mocks.architecture_adapter import mock_adapter, mock_model_adapter
from transformer_lens.model_bridge.bridge import TransformerBridge
from transformer_lens.model_bridge.generalized_components import (
    AttentionBridge,
    BlockBridge,
    EmbeddingBridge,
    MLPBridge,
    NormalizationBridge,
)


class TestTransformerBridge:
    """Test cases for the TransformerBridge class."""

    @pytest.fixture(autouse=True)
    def setup_method(self, mock_adapter, mock_model_adapter):
        """Set up test fixtures."""

        # Mock the get_component method to return expected components for formatting tests
        def mock_get_component(model, path):
            # Return mock bridge components for testing
            if "embed" in path:
                comp = EmbeddingBridge(name="embed")
                comp.set_original_component(model.embed)
                return comp
            elif "ln_final" in path:
                comp = NormalizationBridge(name="ln_final", config={})
                comp.set_original_component(model.ln_final)
                return comp
            elif "unembed" in path:
                comp = EmbeddingBridge(name="unembed")
                comp.set_original_component(model.unembed)
                return comp
            elif "blocks" in path and "attn" in path:
                # Minimal config with n_heads for AttentionBridge
                attn_cfg = SimpleNamespace(n_heads=1)
                comp = AttentionBridge(name="attn", config=attn_cfg)
                comp.set_original_component(model.blocks[0].attn)
                return comp
            elif "blocks" in path and "mlp" in path:
                comp = MLPBridge(name="mlp")
                comp.set_original_component(model.blocks[0].mlp)
                return comp
            elif "blocks" in path and "ln1" in path:
                comp = NormalizationBridge(name="ln1", config={})
                comp.set_original_component(model.blocks[0].ln1)
                return comp
            elif "blocks" in path and "ln2" in path:
                comp = NormalizationBridge(name="ln2", config={})
                comp.set_original_component(model.blocks[0].ln2)
                return comp
            elif "blocks" in path:
                comp = BlockBridge(name="blocks")
                comp.set_original_component(model.blocks[0])
                return comp
            else:
                # Return a generic component for unknown paths
                comp = EmbeddingBridge(name="unknown")
                return comp

        mock_adapter.get_component = mock_get_component
        self.bridge = TransformerBridge(mock_model_adapter, mock_adapter, MagicMock())
        mock_adapter.cfg = MagicMock()
        self.bridge.cfg = mock_adapter.cfg

    def test_format_remote_import_tuple(self):
        """Test formatting of bridge instances (like embed, ln_final, unembed)."""
        # Updated to use actual bridge instances instead of tuples
        mapping = {
            "embed": EmbeddingBridge(name="embed"),
            "ln_final": NormalizationBridge(name="ln_final", config={}),
            "unembed": EmbeddingBridge(name="unembed"),
        }
        self.bridge.adapter.component_mapping = mapping

        result = self.bridge._format_component_mapping(mapping, indent=1)

        assert len(result) == 3
        assert "embed:" in result[0]
        assert "ln_final:" in result[1]
        assert "unembed:" in result[2]
        # Check indentation
        for line in result:
            assert line.startswith("  ")  # 1 level of indentation

    def test_format_block_mapping_tuple(self):
        """Test formatting of BlockBridge instances (like blocks)."""
        mapping = {
            "blocks": BlockBridge(
                name="blocks",
                submodules={
                    "ln1": NormalizationBridge(name="ln1", config={}),
                    "ln2": NormalizationBridge(name="ln2", config={}),
                    "attn": AttentionBridge(name="attn", config=SimpleNamespace(n_heads=1)),
                    "mlp": MLPBridge(name="mlp"),
                },
            )
        }
        self.bridge.adapter.component_mapping = mapping

        result = self.bridge._format_component_mapping(mapping, indent=1)

        # Should have 5 lines: 1 for blocks + 4 for subcomponents
        assert len(result) == 5
        assert "blocks:" in result[0]
        assert "  ln1:" in result[1]  # Extra indentation for subcomponents
        assert "  ln2:" in result[2]
        assert "  attn:" in result[3]
        assert "  mlp:" in result[4]

    def test_format_mixed_mapping(self):
        """Test formatting of a mapping with both simple and block bridge instances."""
        mapping = {
            "embed": EmbeddingBridge(name="embed"),
            "blocks": BlockBridge(
                name="blocks",
                submodules={
                    "ln1": NormalizationBridge(name="ln1", config={}),
                    "attn": AttentionBridge(name="attn", config=SimpleNamespace(n_heads=1)),
                },
            ),
            "ln_final": NormalizationBridge(name="ln_final", config={}),
        }
        self.bridge.adapter.component_mapping = mapping

        result = self.bridge._format_component_mapping(mapping, indent=0)

        # Should have 5 lines: embed + blocks + 2 subcomponents + ln_final
        assert len(result) == 5
        assert any("embed:" in line for line in result)
        assert any("blocks:" in line for line in result)
        assert any("ln1:" in line for line in result)
        assert any("attn:" in line for line in result)
        assert any("ln_final:" in line for line in result)

    def test_format_with_prepend_path(self):
        """Test formatting with prepend path parameter."""
        mapping = {
            "ln1": NormalizationBridge(name="ln1", config={}),
            "attn": AttentionBridge(name="attn", config=SimpleNamespace(n_heads=1)),
        }
        # To test prepending, we need a parent structure in the component mapping
        self.bridge.adapter.component_mapping = {
            "blocks": BlockBridge(
                name="blocks",
                submodules=mapping,
            )
        }

        result = self.bridge._format_component_mapping(mapping, indent=2, prepend="blocks.0")

        assert len(result) == 2
        # The _format_single_component should be called with the prepended path
        for line in result:
            assert line.startswith("    ")  # 2 levels of indentation

    def test_format_empty_mapping(self):
        """Test formatting of an empty mapping."""
        mapping = {}
        self.bridge.adapter.component_mapping = mapping

        result = self.bridge._format_component_mapping(mapping, indent=1)

        assert result == []

    def test_format_non_bridge_values(self):
        """Test formatting when mapping contains non-bridge values."""
        mapping = {
            "some_component": "simple_string_value",
        }
        self.bridge.adapter.component_mapping = mapping

        result = self.bridge._format_component_mapping(mapping, indent=1)

        assert len(result) == 1
        assert "some_component:" in result[0]

    def test_format_nested_block_mappings(self):
        """Test formatting of nested block mappings."""
        mapping = {
            "outer_blocks": BlockBridge(
                name="outer_blocks",
                submodules={
                    "inner_blocks": BlockBridge(
                        name="inner_blocks",
                        submodules={
                            "ln": NormalizationBridge(name="ln", config={}),
                        },
                    )
                },
            )
        }
        self.bridge.adapter.component_mapping = mapping

        result = self.bridge._format_component_mapping(mapping, indent=0)

        # Should handle nested structure correctly
        assert len(result) == 3  # outer_blocks + inner_blocks + ln
        assert any("outer_blocks:" in line for line in result)
        assert any("inner_blocks:" in line for line in result)
        assert any("ln:" in line for line in result)

    def test_format_component_mapping_error_handling(self):
        """Test that the method handles errors gracefully when components can't be found."""
        mapping = {
            "nonexistent_component": EmbeddingBridge(name="path.to.nowhere"),
        }
        self.bridge.adapter.component_mapping = mapping

        # This should not raise an exception, but should handle the error in _format_single_component
        result = self.bridge._format_component_mapping(mapping, indent=1)

        assert len(result) == 1
        assert "nonexistent_component:" in result[0]
        assert "<error:" in result[0]

    def test_indentation_levels(self):
        """Test that indentation is applied correctly at different levels."""
        mapping = {
            "level0": EmbeddingBridge(name="embed"),
        }
        self.bridge.adapter.component_mapping = mapping

        # Test different indentation levels
        result_0 = self.bridge._format_component_mapping(mapping, indent=0)
        result_1 = self.bridge._format_component_mapping(mapping, indent=1)
        result_2 = self.bridge._format_component_mapping(mapping, indent=2)

        assert not result_0[0].startswith(" ")  # No indentation
        assert result_1[0].startswith("  ")  # 1 level (2 spaces)
        assert result_2[0].startswith("    ")  # 2 levels (4 spaces)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
