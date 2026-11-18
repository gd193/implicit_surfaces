"""Builtin item access and bounds, selected during Python construction."""
from dataclasses import dataclass
from functools import cache
from typing import Generic, TypeVar

import warp as wp

from ..contracts import WarpFunction, WarpType, as_warp_function
from ..io.arrays import load_array


DeviceDataT = TypeVar("DeviceDataT")
ItemT = TypeVar("ItemT")


@dataclass(frozen=True)
class ItemFunctions(Generic[DeviceDataT, ItemT]):
    load: WarpFunction[[DeviceDataT, int], ItemT]
    bounds: WarpFunction[[ItemT], tuple[wp.vec3, wp.vec3]]


@cache
def item_functions(kind: str, dimension: int, Data: WarpType[DeviceDataT],
                   Item: WarpType[ItemT]) -> ItemFunctions[DeviceDataT, ItemT]:
    """Return concrete operations; primitive kinds never enter device code."""
    Vec = wp.vec2 if dimension == 2 else wp.vec3

    @wp.func
    def to_vec3(v: Vec):
        if wp.static(dimension == 2):
            return wp.vec3(v[0], v[1], 0.0)
        else:
            return v

    if kind == "points":
        @wp.func
        def load(data: Data, index: int):
            value = Item()
            value.a = load_array(data.a, index)
            return value

        @wp.func
        def bounds(item: Item):
            point = to_vec3(item.a)
            return point, point

    elif kind in ("segments", "silhouette_edges"):
        if kind == "segments":
            @wp.func
            def load(data: Data, index: int):
                value = Item()
                value.a = load_array(data.a, index)
                value.b = load_array(data.b, index)
                return value
        else:
            @wp.func
            def load(data: Data, index: int):
                value = Item()
                value.a = load_array(data.a, index)
                value.b = load_array(data.b, index)
                value.n0 = load_array(data.n0, index)
                value.n1 = load_array(data.n1, index)
                return value

        @wp.func
        def bounds(item: Item):
            a, b = to_vec3(item.a), to_vec3(item.b)
            return wp.min(a, b), wp.max(a, b)

    elif kind == "triangles":
        @wp.func
        def load(data: Data, index: int):
            value = Item()
            value.a = load_array(data.a, index)
            value.e1 = load_array(data.e1, index)
            value.e2 = load_array(data.e2, index)
            value.b = load_array(data.b, index)
            value.c = load_array(data.c, index)
            return value

        @wp.func
        def bounds(item: Item):
            return (wp.min(item.a, wp.min(item.b, item.c)),
                    wp.max(item.a, wp.max(item.b, item.c)))

    else:
        raise ValueError(f"unknown builtin primitive {kind!r}")
    return ItemFunctions(as_warp_function(load), as_warp_function(bounds))
