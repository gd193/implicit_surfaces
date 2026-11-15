"""Bounded closest silhouette with box-support cone pruning."""
from dataclasses import dataclass, replace
from functools import lru_cache
from typing import Generic, TypeVar

import jax.numpy as jnp
import warp as wp

from ..._warp_tree import DeviceTree, STACK_CAPACITY


H = TypeVar("H")
D = TypeVar("D")
ItemT = TypeVar("ItemT")
P = TypeVar("P")
Q = TypeVar("Q")
CH = TypeVar("CH")
CD = TypeVar("CD")
FH = TypeVar("FH")
FD = TypeVar("FD")
FI = TypeVar("FI")


@dataclass(frozen=True)
class _SilhouetteMetric:
    metric: object
    visibility: object

    @lru_cache(maxsize=None)
    def bind(self, primitive):
        from ..._warp_tree import DeviceTree
        ops = self.metric.bind(primitive)
        if self.visibility is not None and ops.viewpoint is None:
            raise TypeError("silhouette metrics must supply a viewpoint")
        Query, Meta = ops.query_type, primitive.data_type
        visibility = self.visibility

        @wp.func
        def visible(query: Query, tree: DeviceTree, meta: Meta, node: int):
            if wp.static(ops.node_filter is not None):
                if not wp.static(ops.node_filter)(query, tree, meta, node):
                    return False
            if wp.static(visibility is not None):
                return wp.static(visibility)(wp.static(ops.viewpoint)(query), tree, meta, node)
            return True

        return replace(ops, node_filter=visible,
                       cull_items=False)


@dataclass(frozen=True)
class SilhouetteQuery(Generic[H, D, ItemT, P, Q, CH, CD, FH, FD, FI]):
    query: object
    device: object

    @property
    def _device_kernel(self):
        return self.query.kernel

    @property
    def kernel(self):
        return self.query.kernel

    def __call__(self, tree, geometry, boundary, queries, max_distance=jnp.inf):
        meta = getattr(boundary, "silhouette", boundary)
        if meta is None:
            raise ValueError("build boundary silhouette metadata before querying")
        feature_tree = tree._replace(node_leaf_offset=meta.node_feature_offset,
            node_leaf_count=meta.node_feature_count, prim_ids=meta.feature_refs)
        return self.query(feature_tree, meta, queries, max_distance)


def make_silhouette(primitive, metric=None, *, adapter=None, stack_capacity=STACK_CAPACITY, block_dim=64):
    from ..._warp_boundary import resolve_boundary_adapter
    from ..metrics import Euclidean

    metric = Euclidean() if metric is None else metric
    adapter = resolve_boundary_adapter(primitive, adapter)
    return _make_silhouette(primitive, metric, adapter, stack_capacity, block_dim)


@lru_cache(maxsize=None)
def _make_silhouette(primitive, metric, adapter, stack_capacity, block_dim):
    from .nearest import make_nearest

    if adapter.visibility is not None and not isinstance(adapter.visibility, wp.Function):
        raise TypeError("silhouette visibility must be a decorated Warp Function")
    feature = adapter.feature_primitive
    if feature is None:
        raise TypeError("boundary adapter has no silhouette feature operations")
    query = make_nearest(feature, _SilhouetteMetric(metric, adapter.visibility), predicate=adapter.predicate,
                         stack_capacity=stack_capacity, block_dim=block_dim, cache_bounds=False,
                         packed_nodes=False, group_leaves=False)
    nearest = query.device
    Query, Meta = query.ops.query_type, feature.data_type

    @wp.func
    def device(point: Query, cap: float, tree: DeviceTree, meta: Meta):
        feature_tree = DeviceTree(tree.node_lower, tree.node_upper, tree.node_left, tree.node_right,
                                 meta.node_feature_offset, meta.node_feature_count, meta.feature_refs,
                                 tree.node_ray_lower, tree.node_ray_upper)
        return wp.static(nearest)(point, cap, feature_tree, meta)

    return SilhouetteQuery(query, device)


@dataclass(frozen=True)
class StarRadiusQuery(Generic[H, D, ItemT, P, CH, CD, FH, FD, FI]):
    silhouette: object
    device: object

    @property
    def _device_kernel(self):
        return self.silhouette.kernel

    def __call__(self, tree, geometry, boundary, points, cap):
        _, distance, _, status = self.silhouette(tree, geometry, boundary, points, cap)
        cap = jnp.asarray(cap, jnp.float32)
        valid = (status >= 0) & (cap >= 0.0) & ~jnp.isnan(cap)
        return (jnp.where(valid, jnp.minimum(cap, distance), jnp.nan),
                jnp.where(valid, 1, -1).astype(jnp.int32))


def make_star_radius(primitive, *, adapter=None, stack_capacity=STACK_CAPACITY, block_dim=64):
    return _make_star_radius(make_silhouette(primitive, adapter=adapter,
        stack_capacity=stack_capacity, block_dim=block_dim))


@lru_cache(maxsize=None)
def _make_star_radius(silhouette):
    closest = silhouette.device
    Point = silhouette.query.ops.query_type
    Meta = silhouette.query.primitive.data_type

    @wp.func
    def device(point: Point, cap: float, tree: DeviceTree, meta: Meta):
        closest_point, distance, index, status = wp.static(closest)(point, cap, tree, meta)
        if status < 0 or not (cap >= 0.0):
            return wp.nan, -1
        return wp.min(cap, distance), 1

    return StarRadiusQuery(silhouette, device)
