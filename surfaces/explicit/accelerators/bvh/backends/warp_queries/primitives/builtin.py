"""Static primitive specifications and array payloads."""
from dataclasses import dataclass
from functools import lru_cache
from typing import TYPE_CHECKING, Any, Callable, Generic, Literal, TypeVar, cast, overload

from ..._triangle_frames import pack_triangle_frames
from ..contracts import FusedNearestOps, WarpFunction, WarpType, as_warp_function
from ..contracts import PrimitiveSpec as PrimitiveSpec

import jax
import jax.numpy as jnp
import numpy as np
import warp as wp

from .items import PointItem2, PointItem3, SegmentItem2, SegmentItem3, TriangleItem3, SilhouetteItem3
from .data import PointsArrays2, PointsArrays3, PointsData2, PointsData3, SegmentsArrays2, SegmentsArrays3, SegmentsData2, SegmentsData3, TrianglesArrays3, TrianglesData3, SilhouetteEdgesArrays3, SilhouetteEdgesData3

from .device import item_functions

from ..domain import Ray

if TYPE_CHECKING:
    from ..._warp_boundary import BoundaryAdapter


HostDataT = TypeVar("HostDataT")
DeviceDataT = TypeVar("DeviceDataT")
ItemT = TypeVar("ItemT")
PointT = TypeVar("PointT")


@dataclass(frozen=True)
class Primitive(Generic[HostDataT, DeviceDataT, ItemT, PointT]):
    """Immutable structural primitive with explicit host and device operations.

    Dynamic Warp struct and host payload classes remain runtime objects; the
    four type parameters connect extensions' payloads and device callbacks.
    """
    dimension: int
    point_type: WarpType[PointT]
    data_type: WarpType[DeviceDataT]
    load: WarpFunction[[DeviceDataT, int], ItemT]
    bounds_of_item: WarpFunction[[ItemT], tuple[wp.vec3, wp.vec3]]
    host_bounds: Callable[[HostDataT], tuple[np.ndarray, np.ndarray]]
    host_pack: Callable[[HostDataT, np.ndarray | jax.Array], HostDataT]
    item_type: WarpType[ItemT] | None = None
    closest_euclidean: WarpFunction[[PointT, ItemT], PointT] | None = None
    intersect: WarpFunction[[Ray, float, float, DeviceDataT, int, bool, bool], wp.vec2] | None = None
    segment_endpoints: WarpFunction[[ItemT], tuple[PointT, PointT]] | None = None
    normal_sides: WarpFunction[[ItemT], tuple[PointT, wp.vec3, wp.vec3]] | None = None
    point_bounds: bool = False
    name: str = "primitive"
    payload_type: Any = None
    prepare_data: Callable[..., HostDataT] | None = None
    prepare_indexed: Callable[[Any, Any], HostDataT] | None = None
    boundary_adapter: "BoundaryAdapter[HostDataT, DeviceDataT, ItemT, PointT, Any, Any, Any, Any, Any] | None" = None

    score_euclidean_at: WarpFunction[[PointT, DeviceDataT, int], wp.float64] | None = None
    closest_euclidean_at: WarpFunction[[PointT, DeviceDataT, int], PointT] | None = None
    group_leaves: bool = False
    cache_nearest_bounds: bool = True
    bind_fused_euclidean: Callable[
        [PrimitiveSpec[HostDataT, DeviceDataT, ItemT, PointT]],
        FusedNearestOps[PointT, DeviceDataT, ItemT, PointT] | None,
    ] | None = None

    def data(self, *arrays: Any) -> HostDataT:
        if self.prepare_data is None:
            raise TypeError("primitive has no array data constructor")
        return self.prepare_data(*arrays)

    def indexed(self, vertices: Any, indices: Any) -> HostDataT:
        if self.prepare_indexed is None:
            raise TypeError("primitive has no indexed data constructor")
        return self.prepare_indexed(vertices, indices)

    def bounds(self, data: HostDataT) -> tuple[np.ndarray, np.ndarray]:
        return self.host_bounds(data)

    def pack(self, data: HostDataT, ids: np.ndarray | jax.Array) -> HostDataT:
        return self.host_pack(data, ids)


def _array_data(payload, dimension, count, name, *, segment=False):
    def prepare(*arrays):
        if len(arrays) != count:
            raise ValueError(f"{name} expects {count} arrays")
        arrays = tuple(jnp.asarray(a, jnp.float32) for a in arrays)
        if any(a.ndim != 2 or a.shape != arrays[0].shape or a.shape[1] != dimension for a in arrays):
            raise ValueError(f"primitive arrays must have matching shape (N, {dimension})")
        if any(not np.isfinite(np.asarray(a)).all() for a in arrays):
            raise ValueError("primitive data must be finite float32")
        if segment:
            arrays += (jnp.empty((0, dimension), jnp.float32), jnp.empty((0, 2), jnp.int32))
        return payload(*arrays)
    return prepare


def _indexed_data(payload, dimension, width):
    def prepare(vertices, indices):
        source_vertices = vertices
        vertices, indices = np.asarray(vertices, np.float32), np.asarray(indices)
        if vertices.ndim != 2 or vertices.shape[1:] != (dimension,):
            raise ValueError(f"vertices must have shape (V, {dimension})")
        if indices.ndim != 2 or indices.shape[1:] != (width,) or not np.issubdtype(indices.dtype, np.integer):
            raise ValueError(f"indices must contain integer tuples of length {width}")
        if indices.size and (np.any(indices < 0) or np.any(indices >= len(vertices))):
            raise ValueError("primitive index outside vertices")
        items = vertices[indices]
        arrays = ((items[:, 0], items[:, 1]-items[:, 0], items[:, 2]-items[:, 0])
                  if width == 3 else (items[:, 0], items[:, 1]))
        if any(not np.isfinite(a).all() for a in arrays):
            raise ValueError("primitive data must be finite float32")
        fields = (*arrays, jnp.asarray(source_vertices, dtype=jnp.float32), indices.astype(np.int32))
        if width == 3:
            fields += (items[:, 1], items[:, 2], pack_triangle_frames(items))
        return payload(*(jnp.asarray(field) for field in fields))
    return prepare


def _array_bounds(data):
    arrays = [np.asarray(a, np.float32) for a in data[:2]]
    return np.minimum.reduce(arrays), np.maximum.reduce(arrays)


def _triangle_bounds(data):
    facets = np.asarray(data.vertices)[np.asarray(data.indices)]
    return facets.min(axis=1), facets.max(axis=1)


def _array_pack(data, ids):
    return type(data)(*(a[ids] for a in data))


def _segment_pack(data, ids):
    indices = data.indices[ids] if data.indices.shape[0] else data.indices
    return type(data)(data.a[ids], data.b[ids], data.vertices, indices)


def _triangle_pack(data, ids):
    return type(data)(*(a[ids] for a in data[:3]), data.vertices, data.indices[ids],
                      data.b[ids], data.c[ids], data.frame[ids])


@lru_cache(maxsize=None)
def _primitive(kind, dimension):
    if dimension not in (2, 3) or (kind in ("triangles", "silhouette_edges") and dimension != 3):
        raise ValueError("points/segments support 2D or 3D; triangles support 3D")
    Vec = wp.vec2 if dimension == 2 else wp.vec3
    fields = {"points": ("a",), "segments": ("a", "b"), "triangles": ("a", "e1", "e2"),
              "silhouette_edges": ("a", "b", "n0", "n1")}[kind]
    Data, Payload = {
        ("points", 2): (PointsData2, PointsArrays2),
        ("points", 3): (PointsData3, PointsArrays3),
        ("segments", 2): (SegmentsData2, SegmentsArrays2),
        ("segments", 3): (SegmentsData3, SegmentsArrays3),
        ("triangles", 3): (TrianglesData3, TrianglesArrays3),
        ("silhouette_edges", 3): (SilhouetteEdgesData3, SilhouetteEdgesArrays3),
    }[kind, dimension]
    Item = {("points", 2): PointItem2, ("points", 3): PointItem3,
            ("segments", 2): SegmentItem2, ("segments", 3): SegmentItem3,
            ("triangles", 3): TriangleItem3,
            ("silhouette_edges", 3): SilhouetteItem3}[kind, dimension]

    functions = item_functions(kind, dimension, Data, Item)
    load, item_bounds = functions.load, functions.bounds

    if kind == "points":
        @wp.func
        def endpoints(item: Any):
            return item.a, item.a
    else:
        @wp.func
        def endpoints(item: Any):
            return item.a, item.b

    @wp.func
    def normal_sides(item: Any):
        return item.a, item.n0, item.n1

    score_at = closest_at = bind_fused_euclidean = None
    if kind == "triangles":
        @wp.func_native("return bvh::triangle_score(query, data, slot);")
        def score_at(query: Vec, data: Data, slot: int) -> wp.float64: ...

        @wp.func_native("return bvh::triangle_closest(query, data, slot);")
        def closest_at(query: Vec, data: Data, slot: int) -> Vec: ...

        @wp.func_native("return bvh::closest_triangle(query, item.a, item.b, item.c);")
        def closest(query: Vec, item: Item) -> Vec: ...

        @wp.func_native("return bvh::triangle_nearest_at(query, data, slot, limit);")
        def nearest_at(query: Vec, data: Data, slot: int, limit: float) -> wp.vec4: ...

        @wp.func_native("return bvh::triangle_nearest_candidate(query, item.a, item.b, item.c, limit);")
        def nearest_item(query: Vec, item: Item, limit: float) -> wp.vec4: ...

        @wp.func
        def candidate_at(query: Vec, data: Data, slot: int, limit: float):
            evaluated = nearest_at(query, data, slot, limit)
            return wp.vec3(evaluated[0], evaluated[1], evaluated[2]), evaluated[3]

        @wp.func
        def candidate_item(query: Vec, item: Item, limit: float):
            evaluated = nearest_item(query, item, limit)
            return wp.vec3(evaluated[0], evaluated[1], evaluated[2]), evaluated[3]

        @wp.func_native("return bvh::conservative_box_score<true>(query, lower, upper);")
        def nearest_bound(query: Vec, lower: wp.vec3, upper: wp.vec3) -> float: ...

        fused = FusedNearestOps(as_warp_function(candidate_at),
                                as_warp_function(candidate_item),
                                as_warp_function(nearest_bound))

        def bind_fused_euclidean(primitive):
            # The slot evaluator reads the original triangle layout directly.
            # Copied descriptors may supply a different geometry operation.
            if (primitive.dimension == dimension and primitive.point_type is Vec
                    and primitive.data_type is Data and primitive.item_type is Item
                    and primitive.load is load and primitive.bounds_of_item is item_bounds
                    and primitive.closest_euclidean is closest
                    and primitive.score_euclidean_at is score_at
                    and primitive.closest_euclidean_at is closest_at
                    and primitive.host_bounds is host_bounds
                    and primitive.host_pack is host_pack):
                return fused
            return None
    elif kind == "points":
        @wp.func
        def closest(query: Vec, item: Any):
            return item.a
    else:
        @wp.func_native("return bvh::closest_segment(query, item.a, item.b);")
        def closest(query: Vec, item: Item) -> Vec: ...

    intersect = None
    if kind == "triangles":
        from .crossings import native_triangle_crossing

        @wp.func
        def intersect(ray: Ray, minimum: float, maximum: float,
                      data: Data, slot: int, half_open: bool, store_times: bool):
            return native_triangle_crossing(ray, minimum, maximum, data.a[slot],
                                            data.b[slot], data.c[slot], half_open, store_times)
    elif kind == "segments" and dimension == 2:
        @wp.func_native("return bvh::segment_intersection(ray, minimum, maximum, data, slot, half_open, store_times);")
        def intersect(ray: Ray, minimum: float, maximum: float,
                      data: Data, slot: int, half_open: bool, store_times: bool) -> wp.vec2: ...

    prepare_indexed = None
    boundary_adapter = None
    if kind == "triangles":
        from ..._warp_boundary import _builtin_adapter
        prepare_indexed = _indexed_data(Payload, dimension, 3)
        def prepare_data(*arrays):
            if len(arrays) != 2:
                raise ValueError("triangles expect vertices and faces")
            return prepare_indexed(*arrays)
        host_bounds, host_pack = _triangle_bounds, _triangle_pack
        boundary_adapter = _builtin_adapter(3)
    elif kind == "segments":
        prepare_data = _array_data(Payload, dimension, 2, kind, segment=True)
        prepare_indexed = _indexed_data(Payload, dimension, 2)
        host_bounds, host_pack = _array_bounds, _segment_pack
        if dimension == 2:
            from ..._warp_boundary import _builtin_adapter
            boundary_adapter = _builtin_adapter(2)
    else:
        prepare_data = _array_data(Payload, dimension, len(fields), kind)
        host_bounds, host_pack = _array_bounds, _array_pack

    return Primitive(dimension=dimension, point_type=Vec, data_type=Data,
                     load=as_warp_function(load), bounds_of_item=as_warp_function(item_bounds),
                     host_bounds=host_bounds, host_pack=host_pack, item_type=Item,
                     closest_euclidean=as_warp_function(closest),
                     score_euclidean_at=as_warp_function(score_at) if score_at is not None else None,
                     closest_euclidean_at=as_warp_function(closest_at) if closest_at is not None else None,
                     intersect=as_warp_function(intersect) if intersect is not None else None,
                     segment_endpoints=as_warp_function(endpoints) if kind != "triangles" else None,
                     normal_sides=as_warp_function(normal_sides) if kind == "silhouette_edges" else None,
                     point_bounds=kind == "points", group_leaves=kind == "triangles",
                     cache_nearest_bounds=kind != "triangles", name=kind, payload_type=Payload,
                     prepare_data=prepare_data, prepare_indexed=prepare_indexed,
                     boundary_adapter=boundary_adapter, bind_fused_euclidean=bind_fused_euclidean)


@overload
def Points(dimension: Literal[2]) -> Primitive[PointsArrays2, PointsData2, PointItem2, wp.vec2]: ...
@overload
def Points(dimension: Literal[3] = 3) -> Primitive[PointsArrays3, PointsData3, PointItem3, wp.vec3]: ...
@overload
def Points(dimension: int) -> Primitive[Any, Any, Any, Any]: ...
def Points(dimension=3) -> Primitive[Any, Any, Any, Any]:
    return _primitive("points", dimension)


@overload
def Segments(dimension: Literal[2]) -> Primitive[SegmentsArrays2, SegmentsData2, SegmentItem2, wp.vec2]: ...
@overload
def Segments(dimension: Literal[3] = 3) -> Primitive[SegmentsArrays3, SegmentsData3, SegmentItem3, wp.vec3]: ...
@overload
def Segments(dimension: int) -> Primitive[Any, Any, Any, Any]: ...
def Segments(dimension=3) -> Primitive[Any, Any, Any, Any]:
    return _primitive("segments", dimension)


def Triangles() -> Primitive[TrianglesArrays3, TrianglesData3, TriangleItem3, wp.vec3]:
    return cast(Primitive[TrianglesArrays3, TrianglesData3, TriangleItem3, wp.vec3], _primitive("triangles", 3))


def SilhouetteEdges() -> Primitive[SilhouetteEdgesArrays3, SilhouetteEdgesData3, SilhouetteItem3, wp.vec3]:
    return cast(Primitive[SilhouetteEdgesArrays3, SilhouetteEdgesData3, SilhouetteItem3, wp.vec3], _primitive("silhouette_edges", 3))
