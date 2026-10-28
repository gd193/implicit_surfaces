"""Static primitive specifications and array payloads."""
from collections import namedtuple
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Protocol, TypeVar

import jax.numpy as jnp
import numpy as np
import warp as wp

from ._warp_traversal import load_array

from ._warp_domain import Ray


DataT = TypeVar("DataT")


class PrimitiveSpec(Protocol[DataT]):
    dimension: int
    point_type: Any
    data_type: Any
    load: Any
    bounds_of_item: Any

    def bounds(self, data: DataT): ...
    def pack(self, data: DataT, ids) -> DataT: ...


def _struct(name, fields):
    return wp.struct(type(name, (), {"__module__": __name__, "__annotations__": fields}))


@dataclass(frozen=True)
class Primitive:
    name: str
    dimension: int
    point_type: Any
    data_type: Any
    payload_type: Any
    load: Any
    bounds_of_item: Any
    closest_euclidean: Any
    intersect: Any = None
    segment_endpoints: Any = None
    normal_sides: Any = None
    point_bounds: bool = False

    def data(self, *arrays):
        if self.name == "triangles":
            if len(arrays) != 2:
                raise ValueError("triangles expect vertices and faces")
            return self.indexed(*arrays)
        count = 2 if self.name == "segments" else len(self.payload_type._fields)
        if len(arrays) != count:
            raise ValueError(f"{self.name} expects {count} arrays")
        arrays = tuple(jnp.asarray(a, jnp.float32) for a in arrays)
        if any(a.ndim != 2 or a.shape != arrays[0].shape or a.shape[1] != self.dimension for a in arrays):
            raise ValueError(f"primitive arrays must have matching shape (N, {self.dimension})")
        if any(not np.isfinite(np.asarray(a)).all() for a in arrays):
            raise ValueError("primitive data must be finite float32")
        if self.name == "segments":
            arrays += (jnp.empty((0, self.dimension), jnp.float32), jnp.empty((0, 2), jnp.int32))
        return self.payload_type(*arrays)

    def indexed(self, vertices, indices):
        if self.name not in ("triangles", "segments"):
            raise TypeError("indexed geometry requires segments or triangles")
        vertices, indices = np.asarray(vertices, np.float32), np.asarray(indices)
        width = 3 if self.name == "triangles" else 2
        if vertices.ndim != 2 or vertices.shape[1:] != (self.dimension,):
            raise ValueError(f"vertices must have shape (V, {self.dimension})")
        if indices.ndim != 2 or indices.shape[1:] != (width,) or not np.issubdtype(indices.dtype, np.integer):
            raise ValueError(f"indices must contain integer tuples of length {width}")
        if indices.size and (np.any(indices < 0) or np.any(indices >= len(vertices))):
            raise ValueError("primitive index outside vertices")
        items = vertices[indices]
        arrays = ((items[:, 0], items[:, 1]-items[:, 0], items[:, 2]-items[:, 0])
                  if width == 3 else (items[:, 0], items[:, 1]))
        if any(not np.isfinite(a).all() for a in arrays):
            raise ValueError("primitive data must be finite float32")
        return self.payload_type(*(jnp.asarray(a) for a in arrays),
                                 jnp.asarray(vertices), jnp.asarray(indices, jnp.int32))

    def bounds(self, data):
        arrays = [np.asarray(a, np.float32) for a in data[:3 if self.name == "triangles" else 2]]
        if self.name == "triangles":
            arrays = [arrays[0], arrays[0]+arrays[1], arrays[0]+arrays[2]]
            arrays.extend(np.asarray(data.vertices)[np.asarray(data.indices)].transpose(1, 0, 2))
        return np.minimum.reduce(arrays), np.maximum.reduce(arrays)

    def pack(self, data, ids):
        if self.name == "triangles":
            return self.payload_type(*(a[ids] for a in data[:3]), data.vertices, data.indices[ids])
        if self.name == "segments":
            indices = data.indices[ids] if data.indices.shape[0] else data.indices
            return self.payload_type(data.a[ids], data.b[ids], data.vertices, indices)
        return self.payload_type(*(a[ids] for a in data))


@lru_cache(maxsize=None)
def _primitive(kind, dimension):
    if dimension not in (2, 3) or (kind in ("triangles", "silhouette_edges") and dimension != 3):
        raise ValueError("points/segments support 2D or 3D; triangles support 3D")
    Vec = wp.vec2 if dimension == 2 else wp.vec3
    fields = {"points": ("a",), "segments": ("a", "b"), "triangles": ("a", "e1", "e2"),
              "silhouette_edges": ("a", "b", "n0", "n1")}[kind]
    data_fields = {name: wp.array(dtype=Vec) for name in fields}
    if kind in ("triangles", "segments"):
        index_type = wp.vec3i if kind == "triangles" else wp.vec2i
        data_fields.update(vertices=wp.array(dtype=Vec), indices=wp.array(dtype=index_type))
    Data = _struct(f"{kind.title()}Data{dimension}", data_fields)
    Item = _struct(f"{kind.title()}Item{dimension}", {name: Vec for name in fields})
    Payload = namedtuple(f"{kind.title()}Arrays{dimension}", data_fields)

    @wp.func
    def to_vec3(v: Vec):
        if wp.static(dimension == 2):
            return wp.vec3(v[0], v[1], 0.0)
        else:
            return v


    @wp.func
    def load(data: Data, index: int):
        value = Item()
        value.a = load_array(data.a, index)
        if wp.static(kind in ("segments", "silhouette_edges")):
            value.b = load_array(data.b, index)
        if wp.static(kind == "silhouette_edges"):
            value.n0 = load_array(data.n0, index)
            value.n1 = load_array(data.n1, index)
        if wp.static(kind == "triangles"):
            value.e1 = load_array(data.e1, index)
            value.e2 = load_array(data.e2, index)
        return value

    @wp.func
    def item_bounds(item: Item):
        lo = to_vec3(item.a)
        hi = lo
        if wp.static(kind in ("segments", "silhouette_edges")):
            lo = wp.min(lo, to_vec3(item.b))
            hi = wp.max(hi, to_vec3(item.b))
        if wp.static(kind == "triangles"):
            b, c = item.a+item.e1, item.a+item.e2
            lo = wp.min(lo, wp.min(b, c))
            hi = wp.max(hi, wp.max(b, c))
        return lo, hi

    @wp.func
    def endpoints(item: Any):
        if wp.static(kind == "points"):
            return item.a, item.a
        else:
            return item.a, item.b

    @wp.func
    def normal_sides(item: Any):
        return item.a, item.n0, item.n1

    if kind == "triangles":
        @wp.func_native("return bvh::closest_triangle(query, item.a, item.e1, item.e2);")
        def closest(query: Vec, item: Item) -> Vec: ...
    elif kind == "points":
        @wp.func
        def closest(query: Vec, item: Any):
            return item.a
    else:
        @wp.func_native("return bvh::closest_segment(query, item.a, item.b);")
        def closest(query: Vec, item: Item) -> Vec: ...

    intersect = None
    if kind == "triangles":
        from ._warp_crossings import native_triangle_crossing

        @wp.func_native("return bvh::first_triangle_intersection(ray, minimum, maximum, data, slot);")
        def first_intersection(ray: Ray, minimum: float, maximum: float,
                               data: Data, slot: int) -> wp.vec2: ...

        @wp.func
        def intersect(ray: Ray, minimum: float, maximum: float,
                      data: Data, slot: int, half_open: bool, store_times: bool):
            if half_open:
                return native_triangle_crossing(ray, minimum, maximum, slot,
                                                 data.vertices, data.indices, half_open, store_times)
            return first_intersection(ray, minimum, maximum, data, slot)
    elif kind == "segments" and dimension == 2:
        @wp.func_native("return bvh::segment_intersection(ray, minimum, maximum, data, slot, half_open, store_times);")
        def intersect(ray: Ray, minimum: float, maximum: float,
                      data: Data, slot: int, half_open: bool, store_times: bool) -> wp.vec2: ...

    return Primitive(kind, dimension, Vec, Data, Payload, load, item_bounds, closest, intersect,
                     endpoints if kind != "triangles" else None,
                     normal_sides if kind == "silhouette_edges" else None, kind == "points")


def Points(dimension=3):
    return _primitive("points", dimension)


def Segments(dimension=3):
    return _primitive("segments", dimension)


def Triangles():
    return _primitive("triangles", 3)


def SilhouetteEdges():
    return _primitive("silhouette_edges", 3)
