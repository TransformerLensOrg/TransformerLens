"""Typed Module List.

A thin generic wrapper around :class:`torch.nn.ModuleList`, inspired by
https://github.com/pytorch/pytorch/issues/80821#issuecomment-3188314929.

``nn.ModuleList`` is not generic, so iterating or indexing it yields a bare
``nn.Module`` (or ``Any`` on torch < 2.8). ``TypedModuleList[T]`` preserves the
element type ``T`` through ` iteration and other methods while remaining a drop-in
``nn.ModuleList`` at runtime (children are still registered as submodules).

The element-type-preserving signatures live in a companion stub
``typed_module_list.pyi`` for technical reasons related to jaxtyping/beartype.
"""

from typing import Generic, TypeVar

from torch import nn

T = TypeVar("T", bound=nn.Module)


class TypedModuleList(nn.ModuleList, Generic[T]):
    """An ``nn.ModuleList`` that remembers the type ``T`` of the modules it holds.

    Runtime behaviour is exactly ``nn.ModuleList``'s; the typed API is declared in
    ``typed_module_list.pyi``.
    """
