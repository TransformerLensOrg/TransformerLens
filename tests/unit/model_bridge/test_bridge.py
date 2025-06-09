"""Unit tests for the TransformerBridge class.

This module tests the bridge functionality, including component mapping formatting
and other bridge operations.
"""

from unittest.mock import MagicMock

import pytest

from tests.mocks.architecture_adapter import mock_adapter, mock_model_adapter
from transformer_lens.model_bridge.bridge import TransformerBridge
from transformer_lens.model_bridge.generalized_components import (
    AttentionBridge,
    BlockBridge,
    EmbeddingBridge,
    LayerNormBridge,
    MLPBridge,
)


class TestTransformerBridge:
    """Test cases for the TransformerBridge class."""

    @pytest.fixture(autouse=True)
    def setup_method(self, mock_adapter, mock_model_adapter):
        """Set up test fixtures."""
        self.bridge = TransformerBridge.__new__(TransformerBridge)
        self.bridge.model = mock_model_adapter
        self.bridge.bridge = mock_adapter
        self.bridge.tokenizer = MagicMock()
        mock_adapter.user_cfg = MagicMock()
        self.bridge.cfg = mock_adapter.user_cfg

    def test_format_remote_import_tuple(self):
        """Test formatting of RemoteImport tuples (like embed, ln_final, unembed)."""
        # This is the case that was causing the original bug
        mapping = {
            "embed": ("embed", EmbeddingBridge),
            "ln_final": ("ln_final", LayerNormBridge),
            "unembed": ("unembed", EmbeddingBridge),
        }
        self.bridge.bridge.component_mapping = mapping

        result = self.bridge._format_component_mapping(mapping, indent=1)

        assert len(result) == 3
        assert "embed:" in result[0]
        assert "ln_final:" in result[1]
        assert "unembed:" in result[2]
        # Check indentation
        for line in result:
            assert line.startswith("  ")  # 1 level of indentation

    def test_format_block_mapping_tuple(self):
        """Test formatting of BlockMapping tuples (like blocks)."""
        mapping = {
            "blocks": (
                "blocks",
                BlockBridge,
                {
                    "ln1": ("ln1", LayerNormBridge),
                    "ln2": ("ln2", LayerNormBridge),
                    "attn": ("attn", AttentionBridge),
                    "mlp": ("mlp", MLPBridge),
                },
            )
        }
        self.bridge.bridge.component_mapping = mapping

        result = self.bridge._format_component_mapping(mapping, indent=1)

        # Should have 5 lines: 1 for blocks + 4 for subcomponents
        assert len(result) == 5
        assert "blocks:" in result[0]
        assert "  ln1:" in result[1]  # Extra indentation for subcomponents
        assert "  ln2:" in result[2]
        assert "  attn:" in result[3]
        assert "  mlp:" in result[4]

    def test_format_mixed_mapping(self):
        """Test formatting of a mapping with both RemoteImport and BlockMapping tuples."""
        mapping = {
            "embed": ("embed", EmbeddingBridge),
            "blocks": (
                "blocks",
                BlockBridge,
                {
                    "ln1": ("ln1", LayerNormBridge),
                    "attn": ("attn", AttentionBridge),
                },
            ),
            "ln_final": ("ln_final", LayerNormBridge),
        }
        self.bridge.bridge.component_mapping = mapping

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
            "ln1": ("ln1", LayerNormBridge),
            "attn": ("attn", AttentionBridge),
        }
        # To test prepending, we need a parent structure in the component mapping
        self.bridge.bridge.component_mapping = {
            "blocks": (
                "blocks",
                BlockBridge,
                mapping,
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
        self.bridge.bridge.component_mapping = mapping

        result = self.bridge._format_component_mapping(mapping, indent=1)

        assert result == []

    def test_format_non_tuple_values(self):
        """Test formatting when mapping contains non-tuple values."""
        mapping = {
            "some_component": "simple_string_value",
        }
        self.bridge.bridge.component_mapping = mapping

        result = self.bridge._format_component_mapping(mapping, indent=1)

        assert len(result) == 1
        assert "some_component:" in result[0]

    def test_format_nested_block_mappings(self):
        """Test formatting of nested block mappings."""
        mapping = {
            "outer_blocks": (
                "outer_blocks",
                BlockBridge,
                {
                    "inner_blocks": (
                        "inner_blocks",
                        BlockBridge,
                        {
                            "ln": ("ln", LayerNormBridge),
                        },
                    )
                },
            )
        }
        self.bridge.bridge.component_mapping = mapping

        result = self.bridge._format_component_mapping(mapping, indent=0)

        # Should handle nested structure correctly
        assert len(result) == 3  # outer_blocks + inner_blocks + ln
        assert any("outer_blocks:" in line for line in result)
        assert any("inner_blocks:" in line for line in result)
        assert any("ln:" in line for line in result)

    def test_format_component_mapping_error_handling(self):
        """Test that the method handles errors gracefully when components can't be found."""
        mapping = {
            "nonexistent_component": ("path.to.nowhere", EmbeddingBridge),
        }
        self.bridge.bridge.component_mapping = mapping

        # This should not raise an exception, but should handle the error in _format_single_component
        result = self.bridge._format_component_mapping(mapping, indent=1)

        assert len(result) == 1
        assert "nonexistent_component:" in result[0]
        assert "<error:" in result[0]

    def test_indentation_levels(self):
        """Test that indentation is applied correctly at different levels."""
        mapping = {
            "level0": ("embed", EmbeddingBridge),
        }
        self.bridge.bridge.component_mapping = mapping

        # Test different indentation levels
        result_0 = self.bridge._format_component_mapping(mapping, indent=0)
        result_1 = self.bridge._format_component_mapping(mapping, indent=1)
        result_2 = self.bridge._format_component_mapping(mapping, indent=2)

        assert not result_0[0].startswith(" ")  # No indentation
        assert result_1[0].startswith("  ")  # 1 level (2 spaces)
        assert result_2[0].startswith("    ")  # 2 levels (4 spaces)

    def test_regression_original_bug(self):
        """Regression test for the original bug where EmbeddingBridge was treated as a dict."""
        # This is the exact scenario that was causing the AttributeError
        mapping = {
            "embed": ("embed", EmbeddingBridge),
            "blocks": (
                "blocks",
                BlockBridge,
                {
                    "attn": ("attn", AttentionBridge),
                },
            ),
            "unembed": ("unembed", EmbeddingBridge),
        }
        self.bridge.bridge.component_mapping = mapping

        # This should not raise AttributeError: type object 'EmbeddingBridge' has no attribute 'items'
        try:
            result = self.bridge._format_component_mapping(mapping, indent=1)
            # If we get here, the bug is fixed
            assert len(result) == 4  # embed + blocks + attn + unembed
        except AttributeError as e:
            if "has no attribute 'items'" in str(e):
                pytest.fail(
                    "Original bug still present: RemoteImport tuples being treated as BlockMapping"
                )
            else:
                raise  # Re-raise if it's a different AttributeError


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
