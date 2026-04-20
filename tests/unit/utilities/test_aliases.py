"""Unit tests for transformer_lens.utilities.aliases module."""

import warnings
from unittest.mock import Mock

import pytest

from transformer_lens.utilities.aliases import (
    _collect_aliases_from_module,
    collect_aliases_recursive,
    resolve_alias,
)


class TestResolveHookAlias:
    """Test cases for resolve_alias function."""

    def test_resolve_existing_alias(self):
        """Test resolving an alias that exists in the hook_aliases dictionary."""
        mock_target = Mock()
        mock_hook = Mock()
        mock_target.actual_hook = mock_hook
        mock_target.disable_warnings = False  # Ensure warnings are enabled

        hook_aliases = {"old_hook": "actual_hook"}

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = resolve_alias(mock_target, "old_hook", hook_aliases)

            # Check that the correct hook was returned
            assert result == mock_hook

            # Check that a deprecation warning was issued
            assert len(w) == 1
            assert issubclass(w[0].category, FutureWarning)
            assert "Hook 'old_hook' is deprecated" in str(w[0].message)
            assert "Use 'actual_hook' instead" in str(w[0].message)

    def test_resolve_nonexistent_alias(self):
        """Test resolving an alias that doesn't exist."""
        mock_target = Mock()
        hook_aliases = {"old_hook": "actual_hook"}

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = resolve_alias(mock_target, "nonexistent_hook", hook_aliases)

            # Should return None for non-existent aliases
            assert result is None

            # No warning should be issued
            assert len(w) == 0

    def test_resolve_alias_empty_dictionary(self):
        """Test resolving with an empty hook_aliases dictionary."""
        mock_target = Mock()
        hook_aliases = {}

        result = resolve_alias(mock_target, "any_hook", hook_aliases)
        assert result is None


class TestCollectAliasesFromModule:
    """Test cases for _collect_aliases_from_module helper function."""

    def test_collect_named_aliases(self):
        """Test collecting named hook aliases from a module."""
        mock_module = Mock()
        mock_module.hook_aliases = {"old_hook": "new_hook", "another_old": "another_new"}
        mock_module.named_children.return_value = []  # No children

        aliases = {}
        visited = set()

        _collect_aliases_from_module(mock_module, "prefix", aliases, visited)

        expected = {
            "prefix.old_hook": "prefix.new_hook",
            "prefix.another_old": "prefix.another_new",
        }
        assert aliases == expected

    def test_collect_cache_aliases(self):
        """Test collecting cache aliases (empty string keys) from a module."""
        mock_module = Mock()
        mock_module.hook_aliases = {"": "hook_out", "regular_alias": "target"}
        mock_module.named_children.return_value = []  # No children

        aliases = {}
        visited = set()

        _collect_aliases_from_module(mock_module, "embed", aliases, visited)

        expected = {
            "embed": "embed.hook_out",  # Cache alias from empty string
            "embed.regular_alias": "embed.target",  # Named alias
        }
        assert aliases == expected

    def test_collect_cache_aliases_no_prefix(self):
        """Test that cache aliases require a prefix."""
        mock_module = Mock()
        mock_module.hook_aliases = {"": "hook_out", "regular_alias": "target"}
        mock_module.named_children.return_value = []  # No children

        aliases = {}
        visited = set()

        _collect_aliases_from_module(mock_module, "", aliases, visited)

        # Should only include the named alias, not the cache alias
        expected = {"regular_alias": "target"}
        assert aliases == expected

    def test_collect_from_module_no_prefix(self):
        """Test collecting aliases from a module with no path prefix."""
        mock_module = Mock()
        mock_module.hook_aliases = {"old_hook": "new_hook"}
        mock_module.named_children.return_value = []  # No children

        aliases = {}
        visited = set()

        _collect_aliases_from_module(mock_module, "", aliases, visited)

        expected = {"old_hook": "new_hook"}
        assert aliases == expected

    def test_collect_prevents_infinite_recursion(self):
        """Test that visited modules are not processed again."""
        mock_module = Mock()
        mock_module.hook_aliases = {"old_hook": "new_hook"}

        aliases = {}
        visited = {id(mock_module)}  # Already visited

        _collect_aliases_from_module(mock_module, "prefix", aliases, visited)

        # Should be empty since module was already visited
        assert aliases == {}

    def test_collect_from_module_with_children(self):
        """Test collecting aliases recursively from child modules."""
        # Create parent module
        parent_module = Mock()
        parent_module.hook_aliases = {"parent_alias": "parent_target"}

        # Create child module
        child_module = Mock()
        child_module.hook_aliases = {"child_alias": "child_target", "": "hook_out"}

        # Set up named_children
        parent_module.named_children.return_value = [("child", child_module)]
        child_module.named_children.return_value = []  # No further children

        aliases = {}
        visited = set()

        _collect_aliases_from_module(parent_module, "", aliases, visited)

        expected = {
            "parent_alias": "parent_target",
            "child.child_alias": "child.child_target",
            "child": "child.hook_out",  # Cache alias from empty string
        }
        assert aliases == expected

    def test_collect_skips_original_model(self):
        """Test that original_model children are skipped."""
        parent_module = Mock()
        parent_module.hook_aliases = {"parent_alias": "parent_target"}

        # Create original_model child (should be skipped)
        original_model = Mock()
        original_model.hook_aliases = {"skip_this": "skip_target"}

        # Create regular child (should be included)
        regular_child = Mock()
        regular_child.hook_aliases = {"include_this": "include_target"}

        parent_module.named_children.return_value = [
            ("original_model", original_model),
            ("regular_child", regular_child),
        ]
        regular_child.named_children.return_value = []

        aliases = {}
        visited = set()

        _collect_aliases_from_module(parent_module, "", aliases, visited)

        expected = {
            "parent_alias": "parent_target",
            "regular_child.include_this": "regular_child.include_target",
        }
        assert aliases == expected
        assert "skip_this" not in str(aliases)  # Ensure original_model was skipped


class TestCollectAliasesRecursive:
    """Test cases for collect_aliases_recursive function."""

    def test_collect_aliases_integration(self):
        """Integration test for the unified aliases collection function."""
        # Create a mock module structure
        root_module = Mock()
        root_module.hook_aliases = {"root_alias": "root_target"}

        embed_module = Mock()
        embed_module.hook_aliases = {"": "hook_out", "hook_embed": "hook_out"}

        child_module = Mock()
        child_module.hook_aliases = {"child_alias": "child_target"}

        root_module.named_children.return_value = [("embed", embed_module), ("child", child_module)]
        embed_module.named_children.return_value = []
        child_module.named_children.return_value = []

        result = collect_aliases_recursive(root_module, "prefix")

        expected = {
            "prefix.root_alias": "prefix.root_target",
            "prefix.embed": "prefix.embed.hook_out",  # Cache alias
            "hook_embed": "prefix.embed.hook_out",  # Regular alias
            "prefix.child.child_alias": "prefix.child.child_target",
        }
        assert result == expected

    def test_collect_aliases_no_prefix(self):
        """Test collecting aliases without a prefix."""
        mock_module = Mock()
        mock_module.hook_aliases = {"alias": "target", "": "hook_out"}
        mock_module.named_children.return_value = []

        result = collect_aliases_recursive(mock_module)

        # Should only include named alias since no prefix for cache alias
        expected = {"alias": "target"}
        assert result == expected

    def test_collect_mixed_aliases(self):
        """Test collecting both regular and cache aliases together."""
        # Create a mock module structure that mimics real bridge components
        root_module = Mock()
        root_module.hook_aliases = {}

        embed_module = Mock()
        embed_module.hook_aliases = {"": "hook_out", "hook_embed": "hook_out"}

        pos_embed_module = Mock()
        pos_embed_module.hook_aliases = {"": "hook_out"}

        block_module = Mock()
        block_module.hook_aliases = {"hook_resid_pre": "hook_in", "hook_resid_post": "hook_out"}

        root_module.named_children.return_value = [
            ("embed", embed_module),
            ("pos_embed", pos_embed_module),
            ("blocks.0", block_module),
        ]
        embed_module.named_children.return_value = []
        pos_embed_module.named_children.return_value = []
        block_module.named_children.return_value = []

        result = collect_aliases_recursive(root_module)

        expected = {
            # Cache aliases (from empty string keys)
            "embed": "embed.hook_out",
            "pos_embed": "pos_embed.hook_out",
            # Named hook aliases
            "hook_embed": "embed.hook_out",
            "blocks.0.hook_resid_pre": "blocks.0.hook_in",
            "blocks.0.hook_resid_post": "blocks.0.hook_out",
        }
        assert result == expected


class TestModuleWithoutHookAliases:
    """Test cases for modules that don't have hook_aliases attribute."""

    def test_collect_from_module_without_hook_aliases(self):
        """Test collecting from a module that doesn't have hook_aliases."""
        mock_module = Mock()
        # Remove hook_aliases attribute
        del mock_module.hook_aliases
        mock_module.named_children.return_value = []  # No children

        aliases = {}
        visited = set()

        # Should not raise an exception
        _collect_aliases_from_module(mock_module, "prefix", aliases, visited)

        # Should result in empty aliases
        assert aliases == {}

    def test_collect_from_module_without_named_children(self):
        """Test collecting from a module that doesn't have named_children."""
        mock_module = Mock()
        mock_module.hook_aliases = {"alias": "target"}
        # Remove named_children attribute
        del mock_module.named_children

        aliases = {}
        visited = set()

        # Should not raise an exception
        _collect_aliases_from_module(mock_module, "prefix", aliases, visited)

        # Should still collect the hook aliases
        expected = {"prefix.alias": "prefix.target"}
        assert aliases == expected


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_resolve_alias_attribute_error(self):
        """Test resolve_alias when target attribute doesn't exist."""

        # Create a class that will actually raise AttributeError
        class MockWithoutAttribute:
            pass

        mock_target = MockWithoutAttribute()
        hook_aliases = {"old_hook": "nonexistent_hook"}

        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            with pytest.raises(AttributeError):
                resolve_alias(mock_target, "old_hook", hook_aliases)

    def test_collect_aliases_with_circular_references(self):
        """Test handling of circular references in module hierarchy."""
        # Create two modules that reference each other
        module_a = Mock()
        module_b = Mock()

        module_a.hook_aliases = {"alias_a": "target_a"}
        module_b.hook_aliases = {"alias_b": "target_b"}

        # Create circular reference
        module_a.named_children.return_value = [("child_b", module_b)]
        module_b.named_children.return_value = [("child_a", module_a)]

        aliases = {}
        visited = set()

        # Should handle circular references without infinite recursion
        _collect_aliases_from_module(module_a, "", aliases, visited)

        # Should collect from both modules without duplication
        expected = {
            "alias_a": "target_a",
            "child_b.alias_b": "child_b.target_b"
            # module_a should not be processed again when reached via module_b
        }
        assert aliases == expected
