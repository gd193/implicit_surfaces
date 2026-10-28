"""Indexed boundary adapters and optional silhouette metadata."""
from collections import namedtuple
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, NamedTuple

import jax.numpy as jnp
import numpy as np
import warp as wp

from ._warp_traversal import load_array
from ._warp_tree import DeviceTree, build_tree


@wp.struct
class BoundaryCore:
    vertices: wp.array(dtype=wp.vec3)
    indices: wp.array2d(dtype=wp.int32)
    closed: wp.array(dtype=wp.int32)
    primitive_normal: wp.array(dtype=wp.vec3)
    primitive_measure: wp.array(dtype=wp.float32)


@wp.struct
class SilhouetteMeta:
    feature_a: wp.array(dtype=wp.vec3)
    feature_b: wp.array(dtype=wp.vec3)
    feature_n0: wp.array(dtype=wp.vec3)
    feature_n1: wp.array(dtype=wp.vec3)
    feature_refs: wp.array(dtype=wp.int32)
    node_feature_offset: wp.array(dtype=wp.int32)
    node_feature_count: wp.array(dtype=wp.int32)
    node_cone_axis: wp.array(dtype=wp.vec3)
    node_cone_sin_cos: wp.array(dtype=wp.vec2)
    node_cone_radius: wp.array(dtype=wp.float32)


BoundaryCoreData = namedtuple("BoundaryCoreData", BoundaryCore.vars)
SilhouetteData = namedtuple("SilhouetteData", SilhouetteMeta.vars)


class BoundaryData(NamedTuple):
    core: Any
    silhouette: Any = None


@dataclass(frozen=True)
class BoundaryAdapter:
    prepare: object
    feature_primitive: object = None
    build_silhouette: object = None
    visibility: object = None
    classification: object = None
    predicate: object = None


def _core(primitive, data, dimension):
    vertices, indices = np.asarray(data.vertices), np.asarray(data.indices)
    if len(data.a) and not len(indices):
        raise ValueError("boundary queries require indexed connectivity; use Segments(2).indexed")
    if vertices.shape[1:] != (dimension,) or indices.shape != (len(data.a), dimension):
        raise ValueError("indexed boundary arrays have incompatible shapes")
    facets = vertices.astype(np.float64)[indices]
    if dimension == 2:
        edge = facets[:, 1]-facets[:, 0]
        cross = np.column_stack((edge[:, 1], -edge[:, 0]))
        boundary_features = indices.reshape(-1, 1)
    else:
        cross = np.cross(np.asarray(data.e1, np.float32), np.asarray(data.e2, np.float32))
        original_cross = np.cross(facets[:, 1]-facets[:, 0], facets[:, 2]-facets[:, 0])
        cross[np.all(original_cross == 0.0, axis=1)] = 0.0
        boundary_features = np.sort(indices[:, [[0, 1], [1, 2], [2, 0]]].reshape(-1, 2), axis=1)
    lengths = np.linalg.norm(cross, axis=1)
    measure = lengths*(.5 if dimension == 3 else 1.)
    normal = cross/np.where(lengths[:, None] > 0, lengths[:, None], 1.)
    if (not np.isfinite(normal).all() or not np.isfinite(measure).all()
            or np.any(measure > np.finfo(np.float32).max)):
        raise ValueError("boundary normals and measures must be finite float32")
    _, counts = np.unique(boundary_features, axis=0, return_counts=True)
    closed = int(np.all(counts == 2) and np.all(lengths > 0))
    if dimension == 2:
        vertices, normal = (np.pad(a, ((0, 0), (0, 1))) for a in (vertices, normal))
    return BoundaryCoreData(data.vertices if dimension == 3 else vertices, data.indices,
                            np.array([closed], np.int32),
                            normal.astype(np.float32), measure.astype(np.float32))


def _segments_core(primitive, data):
    return _core(primitive, data, 2)


def _triangles_core(primitive, data):
    return _core(primitive, data, 3)


def _features(core, tree, geometry):
    vertices, indices, normals = (np.asarray(v) for v in
        (core.vertices, core.indices, core.primitive_normal))
    if np.any(np.asarray(core.primitive_measure) <= 0):
        raise ValueError("silhouettes require non-degenerate boundary elements")
    adjacent = {}
    for primitive, face in enumerate(indices):
        keys = [(v,) for v in face] if len(face) == 2 else (
            tuple(sorted((face[i], face[(i+1) % 3]))) for i in range(3))
        for key in keys:
            adjacent.setdefault(key, []).append(primitive)
    features, owners = [], {}
    for key, incident in adjacent.items():
        if len(incident) > 2:
            raise ValueError("silhouettes require manifold boundary connectivity")
        index = len(features)
        n0 = normals[incident[0]]
        n1 = normals[incident[1]] if len(incident) == 2 else -n0
        features.append((vertices[key[0]], vertices[key[-1]], n0, n1))
        for primitive in incident:
            owners.setdefault(primitive, []).append(index)
    values = np.asarray(features, np.float32).reshape(-1, 4, 3)
    left, right, offset, count, ids = map(np.asarray,
        (tree.node_left, tree.node_right, tree.node_leaf_offset, tree.node_leaf_count, tree.prim_ids))
    n = len(left)
    node_offset, node_count = np.zeros(n, np.int32), np.zeros(n, np.int32)
    axis, angles = np.zeros((n, 3), np.float32), np.tile(np.array([-1., 0.], np.float32), (n, 1))
    radii = np.zeros(n, np.float32)
    references, contained, post = [], [set() for _ in range(n)], []
    pending = [(0, False)] if n else []
    while pending:
        node, done = pending.pop()
        if done:
            post.append(node)
        else:
            pending.append((node, True))
            if left[node] >= 0:
                pending.extend(((int(left[node]), False), (int(right[node]), False)))
    lo, hi = np.asarray(tree.node_lower, np.float64), np.asarray(tree.node_upper, np.float64)
    for node in post:
        if left[node] < 0:
            contained[node] = {feature for primitive in ids[offset[node]:offset[node]+count[node]]
                               for feature in owners.get(int(primitive), ())}
            node_offset[node], node_count[node] = len(references), len(contained[node])
            references.extend(sorted(contained[node]))
        else:
            contained[node] = contained[left[node]] | contained[right[node]]
        if not contained[node]:
            continue
        ns = values[sorted(contained[node]), 2:].reshape(-1, 3).astype(np.float64)
        average = ns.sum(axis=0)
        length = np.linalg.norm(average)
        angles[node] = [1., 0.]
        if length > 1e-12:
            direction = (average/length).astype(np.float32)
            alpha = np.arctan2(np.linalg.norm(np.cross(ns, direction), axis=1), ns @ direction).max()
            alpha = np.nextafter(np.float32(alpha), np.float32(np.inf))
            axis[node] = direction
            if alpha < np.pi/2:
                angles[node] = (np.nextafter(np.float32(np.sin(alpha)), np.float32(np.inf)),
                                np.nextafter(np.float32(np.cos(alpha)), np.float32(-np.inf)))
        radii[node] = np.nextafter(np.float32(.5*np.linalg.norm(hi[node]-lo[node])), np.float32(np.inf))
    return SilhouetteData(*(values[:, i] for i in range(4)), np.asarray(references, np.int32),
                          node_offset, node_count, axis, angles, radii)


@wp.func_native("return bvh::cone_may_have_silhouette(viewpoint, node, tree.node_lower, tree.node_upper, meta.node_cone_sin_cos, meta.node_cone_axis, meta.node_cone_radius);")
def cone_visibility(viewpoint: wp.vec3, tree: DeviceTree, meta: SilhouetteMeta, node: int) -> bool: ...


@dataclass(frozen=True)
class _FeaturePrimitive:
    dimension: int
    point_type: object
    data_type: object
    load: object
    bounds_of_item: object
    closest_euclidean: object
    segment_endpoints: object
    normal_sides: object


@lru_cache(maxsize=None)
def _feature_primitive(dimension):
    from ._warp_primitives import _struct
    Vec = wp.vec2 if dimension == 2 else wp.vec3
    Feature = _struct(f"SilhouetteFeature{dimension}", {"index": wp.int32, "data": SilhouetteMeta})

    @wp.func
    def load(data: SilhouetteMeta, slot: int):
        value = Feature()
        value.index = load_array(data.feature_refs, slot)
        value.data = data
        return value

    @wp.func
    def endpoints(item: Feature):
        a = load_array(item.data.feature_a, item.index)
        if wp.static(dimension == 2):
            point = wp.vec2(a[0], a[1])
            return point, point
        else:
            return a, load_array(item.data.feature_b, item.index)

    @wp.func
    def bounds(item: Feature):
        a = load_array(item.data.feature_a, item.index)
        b = a
        if wp.static(dimension == 3):
            b = load_array(item.data.feature_b, item.index)
        return wp.min(a, b), wp.max(a, b)

    @wp.func_native("return bvh::feature_closest(query, item);")
    def closest(query: Vec, item: Feature) -> Vec: ...

    @wp.func
    def sides(item: Feature):
        return (load_array(item.data.feature_a, item.index), load_array(item.data.feature_n0, item.index),
                load_array(item.data.feature_n1, item.index))

    return _FeaturePrimitive(dimension, Vec, SilhouetteMeta, load, bounds, closest, endpoints, sides)


def _classification(primitive):
    from ._warp_domain import builtin_classification
    return builtin_classification(primitive, BoundaryCore)


@lru_cache(maxsize=None)
def _builtin_adapter(dimension):
    from ._warp_metrics import Silhouette
    return BoundaryAdapter(_segments_core if dimension == 2 else _triangles_core,
                           _feature_primitive(dimension), _features, cone_visibility,
                           _classification, Silhouette())


def resolve_boundary_adapter(primitive, adapter=None):
    if adapter is not None:
        return adapter
    from ._warp_primitives import Primitive, Segments, Triangles
    if type(primitive) is Primitive:
        if primitive == Triangles():
            return _builtin_adapter(3)
        if primitive == Segments(2):
            return _builtin_adapter(2)
    raise TypeError("boundary queries require an explicit BoundaryAdapter")


def build_boundary(primitive, data, config=None, *, adapter=None, silhouettes=True):
    from ._warp_primitives import Primitive, Segments, Triangles

    adapter = resolve_boundary_adapter(primitive, adapter)
    tree, geometry = build_tree(primitive, data, config)
    core = adapter.prepare(primitive, data)
    meta = None
    if silhouettes:
        if adapter.build_silhouette is None or adapter.feature_primitive is None:
            raise TypeError("boundary adapter has no silhouette construction")
        meta = adapter.build_silhouette(core, tree, geometry)
        meta = type(meta)(*(jnp.asarray(v) for v in meta))
    ids = tree.prim_ids
    shared = (type(primitive) is Primitive and primitive in (Segments(2), Triangles())
              and adapter.prepare in (_segments_core, _triangles_core))
    indices = (geometry.indices if shared
               else jnp.asarray(core.indices)[ids])
    core = BoundaryCoreData(jnp.asarray(core.vertices), indices, jnp.asarray(core.closed),
                            jnp.asarray(core.primitive_normal)[ids], jnp.asarray(core.primitive_measure)[ids])
    return tree, geometry, BoundaryData(core, meta)
