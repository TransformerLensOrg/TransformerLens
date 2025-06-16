"""Unit tests for the TransformerBridge class.

This module tests the bridge functionality, including component mapping formatting
and other bridge operations.
"""

from unittest.mock import Mock

import pytest

from transformer_lens.model_bridge.bridge import TransformerBridge
from transformer_lens.model_bridge.generalized_components import (
    AttentionBridge,
    EmbeddingBridge,
    LayerNormBridge,
    MLPBridge,
)


class TestTransformerBridge:
    """Test cases for the TransformerBridge class."""

    def setup_method(self):
        """Set up test fixtures."""
        # Create mock model, adapter, and tokenizer
        self.mock_model = Mock()
        self.mock_adapter = Mock()
        self.mock_tokenizer = Mock()

        # Mock the adapter's component_mapping attribute
        self.mock_adapter.component_mapping = {}

        # Mock the get_component method to return mock components
        self.mock_components = {
            "embed": Mock(spec=EmbeddingBridge),
            "blocks.0": Mock(),
            "blocks.0.ln1": Mock(spec=LayerNormBridge),
            "blocks.0.ln2": Mock(spec=LayerNormBridge),
            "blocks.0.attn": Mock(spec=AttentionBridge),
            "blocks.0.mlp": Mock(spec=MLPBridge),
            "ln_final": Mock(spec=LayerNormBridge),
            "unembed": Mock(spec=EmbeddingBridge),
        }

        def mock_get_component(model, path):
            if path in self.mock_components:
                comp = self.mock_components[path]
                # Add original_component attribute for bridge components
                if hasattr(comp, "spec") and comp.spec in [
                    EmbeddingBridge,
                    LayerNormBridge,
                    AttentionBridge,
                    MLPBridge,
                ]:
                    comp.original_component = Mock()
                    comp.original_component.__class__.__name__ = (
                        f"Mock{comp.spec.__name__.replace('Bridge', '')}"
                    )
                return comp
            raise AttributeError(f"Component {path} not found")

        self.mock_adapter.get_component = mock_get_component

        # Mock the cfg to have required attributes
        self.mock_adapter.cfg = Mock()
        self.mock_adapter.cfg.num_hidden_layers = 1

        # Create a bridge instance (we'll mock the initialization to avoid complexity)
        self.bridge = TransformerBridge.__new__(TransformerBridge)
        self.bridge.model = self.mock_model
        self.bridge.bridge = self.mock_adapter
        self.bridge.cfg = self.mock_adapter.cfg
        self.bridge.tokenizer = self.mock_tokenizer

    def test_format_remote_import_tuple(self):
        """Test formatting of RemoteImport tuples (like embed, ln_final, unembed)."""
        # This is the case that was causing the original bug
        mapping = {
            "embed": ("model.embed_tokens", EmbeddingBridge),
            "ln_final": ("model.norm", LayerNormBridge),
            "unembed": ("model.embed_tokens", EmbeddingBridge),
        }

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
                "model.layers",
                {
                    "ln1": ("input_layernorm", LayerNormBridge),
                    "ln2": ("post_attention_layernorm", LayerNormBridge),
                    "attn": ("self_attn", AttentionBridge),
                    "mlp": ("mlp", MLPBridge),
                },
            )
        }

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
            "embed": ("model.embed_tokens", EmbeddingBridge),
            "blocks": (
                "model.layers",
                {
                    "ln1": ("input_layernorm", LayerNormBridge),
                    "attn": ("self_attn", AttentionBridge),
                },
            ),
            "ln_final": ("model.norm", LayerNormBridge),
        }

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
            "ln1": ("input_layernorm", LayerNormBridge),
            "attn": ("self_attn", AttentionBridge),
        }

        result = self.bridge._format_component_mapping(mapping, indent=2, prepend="blocks.0")

        assert len(result) == 2
        # The _format_single_component should be called with the prepended path
        for line in result:
            assert line.startswith("    ")  # 2 levels of indentation

    def test_format_empty_mapping(self):
        """Test formatting of an empty mapping."""
        mapping = {}

        result = self.bridge._format_component_mapping(mapping, indent=1)

        assert result == []

    def test_format_non_tuple_values(self):
        """Test formatting when mapping contains non-tuple values."""
        mapping = {
            "some_component": "simple_string_value",
        }

        result = self.bridge._format_component_mapping(mapping, indent=1)

        assert len(result) == 1
        assert "some_component:" in result[0]

    def test_format_nested_block_mappings(self):
        """Test formatting of nested block mappings."""
        mapping = {
            "outer_blocks": (
                "model.outer_layers",
                {
                    "inner_blocks": (
                        "inner_layers",
                        {
                            "ln": ("layernorm", LayerNormBridge),
                        },
                    )
                },
            )
        }

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

        # This should not raise an exception, but should handle the error in _format_single_component
        result = self.bridge._format_component_mapping(mapping, indent=1)

        assert len(result) == 1
        assert "nonexistent_component:" in result[0]

    def test_indentation_levels(self):
        """Test that indentation is applied correctly at different levels."""
        mapping = {
            "level0": ("path", EmbeddingBridge),
        }

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
            "embed": ("model.embed_tokens", EmbeddingBridge),
            "blocks": (
                "model.layers",
                {
                    "attn": ("self_attn", AttentionBridge),
                },
            ),
            "unembed": ("model.embed_tokens", EmbeddingBridge),
        }

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
