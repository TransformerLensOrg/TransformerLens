"""Unit tests for TypedModuleList.

TypedModuleList is a generic, type-preserving wrapper around nn.ModuleList. These
tests pin down its runtime behaviour: it must stay a drop-in nn.ModuleList (same
iteration, indexing, slicing, and submodule registration). The static typing
guarantee (iteration/indexing yield the element type) is exercised separately by
mypy over the rest of the package.
"""

import pytest
import torch.nn as nn

from transformer_lens.utilities import TypedModuleList
from transformer_lens.utilities.typed_module_list import (
    TypedModuleList as TypedModuleListDirect,
)


def _linears(n: int) -> list[nn.Linear]:
    return [nn.Linear(3, 3) for _ in range(n)]


class TestTypedModuleList:
    """Runtime behaviour of TypedModuleList."""

    def test_exported_from_utilities_package(self):
        """It is re-exported from transformer_lens.utilities."""
        assert TypedModuleList is TypedModuleListDirect

    def test_is_an_nn_module_list(self):
        """It must remain a genuine nn.ModuleList subclass."""
        assert isinstance(TypedModuleList(_linears(2)), nn.ModuleList)

    @pytest.mark.parametrize("n", [0, 1, 3])
    def test_construction_and_len(self, n: int):
        """len() matches the number of modules passed in (including empty)."""
        assert len(TypedModuleList(_linears(n))) == n

    def test_construction_with_no_arguments(self):
        """It can be constructed empty, like nn.ModuleList()."""
        empty = TypedModuleList()
        assert len(empty) == 0
        assert list(empty) == []

    def test_iteration_preserves_order_and_identity(self):
        """Iterating yields the exact module objects, in order."""
        layers = _linears(3)
        assert [block for block in TypedModuleList(layers)] == layers

    def test_integer_indexing_returns_the_element(self):
        """Integer indexing (including negative) returns the stored module."""
        layers = _linears(3)
        tml = TypedModuleList(layers)
        assert tml[0] is layers[0]
        assert tml[-1] is layers[-1]

    def test_slicing_returns_a_typed_module_list(self):
        """Slicing returns a TypedModuleList (not a bare nn.ModuleList) with the right modules."""
        layers = _linears(4)
        sliced = TypedModuleList(layers)[1:3]
        assert isinstance(sliced, TypedModuleList)
        assert list(sliced) == layers[1:3]

    def test_modules_are_registered_as_submodules(self):
        """Child modules are registered, so parameters() / named_children() work."""
        tml = TypedModuleList(_linears(2))
        # Two Linear(3, 3) layers => 2 * (weight + bias) == 4 parameter tensors.
        assert len(list(tml.parameters())) == 4
        assert [name for name, _ in tml.named_children()] == ["0", "1"]

    def test_registers_correctly_when_nested_in_a_module(self):
        """Used as a submodule, its parameters surface on the parent under the attribute name."""

        class Net(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.blocks = TypedModuleList(_linears(2))

        net = Net()
        param_names = [name for name, _ in net.named_parameters()]
        assert len(param_names) == 4
        assert all(name.startswith("blocks.") for name in param_names)

    def test_append_mutates_and_returns_self(self):
        """append adds the module and returns the same list (for chaining), like nn.ModuleList."""
        tml: TypedModuleList[nn.Linear] = TypedModuleList()
        layer = nn.Linear(3, 3)
        returned = tml.append(layer)
        assert returned is tml
        assert isinstance(returned, TypedModuleList)
        assert len(tml) == 1
        assert tml[0] is layer

    def test_setitem_replaces_element(self):
        """__setitem__ replaces the module at an existing index and re-registers it."""
        layers = _linears(2)
        tml = TypedModuleList(layers)
        replacement = nn.Linear(3, 3)
        tml[0] = replacement
        assert tml[0] is replacement
        assert tml[1] is layers[1]
        assert len(tml) == 2
        # The replacement must be registered (old child gone, new child present).
        assert replacement in tml.children()
        assert layers[0] not in tml.children()
