"""Structural host contracts for statically specialized Warp operations.

These helpers preserve callable signatures while exposing Warp's runtime
Function markers to static checking. They return the actual Warp descriptors;
they never wrap a device callable.
"""
from typing import TYPE_CHECKING, Any, Callable, ParamSpec, Protocol, TypeAlias, TypeVar, cast, overload

import jax
import numpy as np
import warp as wp

P = ParamSpec("P")
R = TypeVar("R", covariant=True)
HostDataT = TypeVar("HostDataT")
DeviceDataT = TypeVar("DeviceDataT")
ItemT = TypeVar("ItemT")
PointT = TypeVar("PointT", covariant=True)
T_co = TypeVar("T_co", covariant=True)
StructT = TypeVar("StructT")
ElementT = TypeVar("ElementT")
ReferenceT = TypeVar("ReferenceT")

# Warp references describe argument passing, not a different Python value.
# Preserve the value's fields/signature for static checking while passing the
# actual reference descriptor to Warp's compiler at runtime.
if TYPE_CHECKING:
    WarpRef: TypeAlias = ReferenceT
else:
    WarpRef = wp.ref


class DeviceArray(Protocol[ElementT]):
    """Static view of a one-dimensional Warp array; runtime dtype stays explicit."""

    @property
    def shape(self) -> tuple[int, ...]: ...

    def __getitem__(self, index: int) -> ElementT: ...
    def __setitem__(self, index: int, value: ElementT) -> None: ...


class DeviceArray2D(Protocol[ElementT]):
    """Static view of a two-dimensional Warp output buffer."""

    @property
    def shape(self) -> tuple[int, ...]: ...

    def __getitem__(self, index: tuple[int, int]) -> ElementT: ...
    def __setitem__(self, index: tuple[int, int], value: ElementT) -> None: ...


class WarpFunction(Protocol[P, R]):
    """A genuine Warp Function with a statically known device signature."""

    @property
    def key(self) -> str: ...

    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> R: ...


class WarpType(Protocol[T_co]):
    """Runtime Warp value/struct constructor, separate from its instance type."""

    def __call__(self, *args: Any, **kwargs: Any) -> T_co: ...


@overload
def warp_func(function: Callable[P, R], *, name: str | None = None,
              module: str | None = None, inline: bool | None = None) -> WarpFunction[P, R]: ...


@overload
def warp_func(function: None = None, *, name: str | None = None,
              module: str | None = None, inline: bool | None = None) -> Callable[[Callable[P, R]], WarpFunction[P, R]]: ...


def warp_func(function: Callable[P, R] | None = None, *, name: str | None = None,
              module: str | None = None, inline: bool | None = None
              ) -> WarpFunction[P, R] | Callable[[Callable[P, R]], WarpFunction[P, R]]:
    """Typed ``wp.func``, including ``@warp_func(module="unique")``.

    Only the host decorator is bridged. The returned device function is exactly
    the object created by Warp, with no extra callable on its execution path.
    """
    if function is None:
        def decorate(value: Callable[P, R]) -> WarpFunction[P, R]:
            return warp_func(value, name=name, module=module, inline=inline)
        return decorate
    if name is None and module is None and inline is None:
        return cast(WarpFunction[P, R], wp.func(function))
    return cast(WarpFunction[P, R], wp.func(name=name, module=module, inline=inline)(function))


def warp_struct(value_class: type[StructT]) -> type[StructT]:
    """Retain declared field types while returning Warp's actual descriptor.

    The static class describes the constructed struct value. At runtime Warp
    replaces it with a Struct descriptor, not a Python type; do not use the
    result for Python inheritance or isinstance checks. This cast is confined
    to that constructor/annotation boundary and creates no runtime wrapper.
    """
    return cast(type[StructT], wp.struct(value_class))


def as_warp_function(function: Callable[P, R]) -> WarpFunction[P, R]:
    """Recover a known signature after an upstream Warp decorator.

    A signature annotation on the input must supply the information lost by
    upstream typing. This checks Warp identity, not argument types at runtime.
    """
    if not isinstance(function, wp.Function):
        raise TypeError("expected a decorated Warp Function")
    return cast(WarpFunction[P, R], function)


class PrimitiveSpec(Protocol[HostDataT, DeviceDataT, ItemT, PointT]):
    """Shared contract for built-in and external primitives; no registration."""

    @property
    def dimension(self) -> int: ...
    @property
    def point_type(self) -> WarpType[PointT]: ...
    @property
    def data_type(self) -> WarpType[DeviceDataT]: ...
    @property
    def load(self) -> WarpFunction[[DeviceDataT, int], ItemT]: ...
    @property
    def bounds_of_item(self) -> WarpFunction[[ItemT], tuple[wp.vec3, wp.vec3]]: ...

    def bounds(self, data: HostDataT) -> tuple[np.ndarray, np.ndarray]: ...
    def pack(self, data: HostDataT, ids: np.ndarray | jax.Array) -> HostDataT: ...


def make_struct(name: str, fields: dict[str, Any]) -> WarpType[Any]:
    """Construct a device state type from a factory's resolved field schema."""
    return cast(WarpType[Any], wp.struct(type(name, (), {
        "__module__": __name__, "__annotations__": fields,
    })))
