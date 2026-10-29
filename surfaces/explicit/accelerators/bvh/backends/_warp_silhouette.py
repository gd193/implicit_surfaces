"""Bounded closest silhouette with box-support cone pruning."""
from dataclasses import dataclass, replace
from functools import lru_cache

import jax.numpy as jnp
import warp as wp

from ._warp_tree import STACK_CAPACITY


@dataclass(frozen=True)
class _SilhouetteMetric:
    metric: object
    visibility: object

    @lru_cache(maxsize=None)
    def bind(self, primitive):
        from ._warp_tree import DeviceTree
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
class SilhouetteQuery:
    query: object

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


@lru_cache(maxsize=None)
def make_silhouette(primitive, metric=None, *, adapter=None, stack_capacity=STACK_CAPACITY, block_dim=64):
    from ._warp_boundary import resolve_boundary_adapter
    from ._warp_metrics import Euclidean
    from ._warp_nearest import make_nearest

    metric = Euclidean() if metric is None else metric
    adapter = resolve_boundary_adapter(primitive, adapter)
    feature = adapter.feature_primitive
    if feature is None:
        raise TypeError("boundary adapter has no silhouette feature operations")
    query = make_nearest(feature, _SilhouetteMetric(metric, adapter.visibility), predicate=adapter.predicate,
                         stack_capacity=stack_capacity, block_dim=block_dim, cache_bounds=False)
    return SilhouetteQuery(query)


@dataclass(frozen=True)
class StarRadiusQuery:
    silhouette: object

    def __call__(self, tree, geometry, boundary, points, cap):
        _, distance, _, status = self.silhouette(tree, geometry, boundary, points, cap)
        cap = jnp.asarray(cap, jnp.float32)
        valid = (status >= 0) & (cap >= 0.0) & ~jnp.isnan(cap)
        return (jnp.where(valid, jnp.minimum(cap, distance), jnp.nan),
                jnp.where(valid, 1, -1).astype(jnp.int32))


@lru_cache(maxsize=None)
def make_star_radius(primitive, *, adapter=None, stack_capacity=STACK_CAPACITY, block_dim=64):
    return StarRadiusQuery(make_silhouette(primitive, adapter=adapter,
        stack_capacity=stack_capacity, block_dim=block_dim))
