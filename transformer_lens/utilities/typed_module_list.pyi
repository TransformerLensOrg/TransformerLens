"""Typed Module List type annotations

The element-type-preserving signatures live here rather than in
``typed_module_list.py`` for technical reasons related to jaxtyping/beartype.
"""

from typing import Generic, Iterable, Iterator, Optional, TypeVar, overload

from torch import nn

T = TypeVar("T", bound=nn.Module)

class TypedModuleList(nn.ModuleList, Generic[T]):
    def __init__(self, modules: Optional[Iterable[T]] = ...) -> None: ...
    def __iter__(self) -> Iterator[T]: ...
    @overload
    def __getitem__(self, idx: slice) -> TypedModuleList[T]: ...
    @overload
    def __getitem__(self, idx: int) -> T: ...
    def append(self, module: T) -> TypedModuleList[T]: ...  # type: ignore[override]
    def __setitem__(self, idx: int, module: T) -> None: ...  # type: ignore[override]
