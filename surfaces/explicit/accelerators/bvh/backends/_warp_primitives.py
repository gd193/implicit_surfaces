"""Static primitive specifications and array payloads."""
from collections import namedtuple
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Protocol, TypeVar

import jax.numpy as jnp
import numpy as np
import warp as wp

from ._warp_geometry import _NATIVE_SEGMENT_CLOSEST, _NATIVE_TRIANGLE_CLOSEST


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

    def data(self, *arrays):
        if self.name == "triangles":
            if len(arrays) != 2:
                raise ValueError("triangles expect vertices and faces")
            vertices, faces = np.asarray(arrays[0], np.float32), np.asarray(arrays[1])
            if vertices.ndim != 2 or vertices.shape[1:] != (3,):
                raise ValueError("vertices must have shape (V, 3)")
            if faces.ndim != 2 or faces.shape[1:] != (3,) or not np.issubdtype(faces.dtype, np.integer):
                raise ValueError("faces must contain integer index triplets")
            if faces.size and (np.any(faces < 0) or np.any(faces >= len(vertices))):
                raise ValueError("face index outside vertices")
            triangles = vertices[faces]
            arrays = triangles[:, 0], triangles[:, 1]-triangles[:, 0], triangles[:, 2]-triangles[:, 0]
            if any(not np.isfinite(a).all() for a in arrays):
                raise ValueError("triangle data must be finite float32")
            return self.payload_type(*(jnp.asarray(a) for a in arrays),
                                     jnp.asarray(vertices), jnp.asarray(faces, jnp.int32))
        if len(arrays) != len(self.payload_type._fields):
            raise ValueError(f"{self.name} expects {len(self.payload_type._fields)} arrays")
        arrays = tuple(jnp.asarray(a, jnp.float32) for a in arrays)
        if any(a.ndim != 2 or a.shape != arrays[0].shape or a.shape[1] != self.dimension for a in arrays):
            raise ValueError(f"primitive arrays must have matching shape (N, {self.dimension})")
        if any(not np.isfinite(np.asarray(a)).all() for a in arrays):
            raise ValueError("primitive data must be finite float32")
        return self.payload_type(*arrays)

    def bounds(self, data):
        arrays = [np.asarray(a, np.float32) for a in data[:3 if self.name == "triangles" else 2]]
        if self.name == "triangles":
            arrays = [arrays[0], arrays[0]+arrays[1], arrays[0]+arrays[2]]
        return np.minimum.reduce(arrays), np.maximum.reduce(arrays)

    def pack(self, data, ids):
        if self.name == "triangles":
            return self.payload_type(*(a[ids] for a in data[:3]), data.vertices, data.indices[ids])
        return self.payload_type(*(a[ids] for a in data))


@lru_cache(maxsize=None)
def _primitive(kind, dimension):
    if dimension not in (2, 3) or (kind in ("triangles", "silhouette_edges") and dimension != 3):
        raise ValueError("points/segments support 2D or 3D; triangles support 3D")
    Vec = wp.vec2 if dimension == 2 else wp.vec3
    fields = {"points": ("a",), "segments": ("a", "b"), "triangles": ("a", "e1", "e2"),
              "silhouette_edges": ("a", "b", "n0", "n1")}[kind]
    data_fields = {name: wp.array(dtype=Vec) for name in fields}
    if kind == "triangles":
        data_fields.update(vertices=wp.array(dtype=Vec), indices=wp.array(dtype=wp.vec3i))
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
        value.a = data.a[index]
        if wp.static(kind in ("segments", "silhouette_edges")):
            value.b = data.b[index]
        if wp.static(kind == "silhouette_edges"):
            value.n0 = data.n0[index]
            value.n1 = data.n1[index]
        if wp.static(kind == "triangles"):
            value.e1 = data.e1[index]
            value.e2 = data.e2[index]
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

    if kind == "triangles":
        @wp.func_native("const wp::vec3 p = query;\n" + _NATIVE_TRIANGLE_CLOSEST + r"""
            return closest_triangle(item.a, item.e1, item.e2);
        """)
        def closest(query: Vec, item: Item) -> Vec: ...
    elif kind == "points":
        @wp.func
        def closest(query: Vec, item: Any):
            return item.a
    else:
        @wp.func_native("const auto p = query;\n" + _NATIVE_SEGMENT_CLOSEST
                        + "return closest_segment(item.a, item.b);")
        def closest(query: Vec, item: Item) -> Vec: ...

    intersect = None
    if kind == "triangles":
        from ._warp_crossings import native_triangle_crossing

        @wp.func
        def intersect(origin: Vec, direction: Vec, minimum: float, maximum: float,
                      data: Data, slot: int, half_open: bool, store_times: bool):
            return native_triangle_crossing(origin, direction, minimum, maximum, slot,
                                             data.vertices, data.indices, half_open, store_times)
    elif kind == "segments" and dimension == 2:
        @wp.func_native(r"""
            const wp::vec2 a=data.a.data[slot], b=data.b.data[slot];
            const double ex=double(b[0])-double(a[0]), ey=double(b[1])-double(a[1]);
            const double dx=direction[0],dy=direction[1];
            const double ax=double(a[0])-double(origin[0]),ay=double(a[1])-double(origin[1]);
            const double denominator=dx*ey-dy*ex;
            if (denominator == 0) return wp::vec2(0.f,0.f);
            const double u=(ax*dy-ay*dx)/denominator;
            const double t=(ax*ey-ay*ex)/denominator;
            if (u < 0 || u > 1 || t < minimum || t > maximum) return wp::vec2(0.f,0.f);
            if (half_open) {
                const double sa=ax*dy-ay*dx;
                const double sb=(double(b[0])-double(origin[0]))*dy
                               -(double(b[1])-double(origin[1]))*dx;
                if (!((sa <= 0 && sb > 0) || (sb <= 0 && sa > 0))) return wp::vec2(0.f,0.f);
            }
            const float result=static_cast<float>(t);
            return wp::vec2(result, store_times && !wp::isfinite(result) ? -1.f : 1.f);
        """)
        def intersect(origin: Vec, direction: Vec, minimum: float, maximum: float,
                      data: Data, slot: int, half_open: bool, store_times: bool) -> wp.vec2: ...

    return Primitive(kind, dimension, Vec, Data, Payload, load, item_bounds, closest, intersect)


def Points(dimension=3):
    return _primitive("points", dimension)


def Segments(dimension=3):
    return _primitive("segments", dimension)


def Triangles():
    return _primitive("triangles", 3)


def SilhouetteEdges():
    return _primitive("silhouette_edges", 3)
