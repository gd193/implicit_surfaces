"""Nearest traversal specialized by static Python primitive and metric arguments."""
from dataclasses import dataclass
from functools import lru_cache

import jax.numpy as jnp
import warp as wp

from ._warp_factory import BoundQuery, bind_query, is_native_triangle
from ._warp_specs import STACK_CAPACITY
from ._warp_tree import DeviceTree


@dataclass(frozen=True)
class NearestQuery(BoundQuery):
    ops: object

    def __call__(self, tree, geometry, queries, max_distance=jnp.inf):
        queries = jnp.asarray(queries, jnp.float32)
        tail = self.ops.query_shape
        if tail and queries.shape[-len(tail):] != tail:
            raise ValueError(f"expected queries with trailing shape {tail}")
        shape = queries.shape[:-len(tail)] if tail else queries.shape
        cap = jnp.broadcast_to(jnp.asarray(max_distance, jnp.float32), shape).reshape((-1,))
        return self.launch(tree, geometry, shape, queries.reshape((-1,)+tail), cap)


@lru_cache(maxsize=None)
def make_nearest(primitive, metric, *, predicate=None, stack_capacity=STACK_CAPACITY, block_dim=64):
    if not 1 <= stack_capacity <= STACK_CAPACITY or not 1 <= block_dim <= 1024:
        raise ValueError("invalid stack capacity or block dimension")
    ops = metric.bind(primitive)
    accepts = predicate.bind(primitive, ops) if hasattr(predicate, "bind") else predicate
    Query, Point, Geometry = ops.query_type, primitive.point_type, primitive.data_type
    IndexStack = wp.types.vector(stack_capacity, wp.int32)
    ScoreStack = wp.types.vector(stack_capacity, wp.float32)

    @wp.func
    def node_bound(query: Query, tree: DeviceTree, geometry: Geometry, node: int):
        if wp.static(ops.node_bound_with_data is not None):
            return wp.static(ops.node_bound_with_data)(query, tree, geometry, node)
        else:
            return wp.static(ops.node_bound)(query, tree.node_lower[node], tree.node_upper[node])

    @wp.func
    def nearest(query: Query, cap: float, tree: DeviceTree, geometry: Geometry):
        point = Point(0.0)
        if not wp.static(ops.valid)(query) or wp.isnan(cap) or cap < 0.0:
            return point, wp.inf, -1, -1
        if tree.node_left.shape[0] == 0:
            return point, wp.inf, -1, 0
        initial = wp.static(ops.to_score)(cap)
        if wp.isnan(initial):
            return point, wp.inf, -1, -1
        best = wp.min(initial, 3.4028234e38)
        best_id = int(-1)
        nodes = IndexStack()
        scores = ScoreStack()
        size = int(0)
        node = int(0)
        root_score = node_bound(query, tree, geometry, 0)
        if wp.isnan(root_score):
            return point, wp.inf, -1, -1
        if root_score > best:
            if (wp.isinf(root_score) and initial > best
                    and wp.static(ops.node_bound_with_data is None)):
                return point, wp.inf, -1, -1
            return point, wp.inf, -1, 0
        while True:
            while True:
                count = tree.node_leaf_count[node]
                if count > 0:
                    offset = tree.node_leaf_offset[node]
                    for j in range(count):
                        slot = offset+j
                        item = wp.static(primitive.load)(geometry, slot)
                        if wp.static(accepts is not None):
                            if not wp.static(accepts)(query, item):
                                continue
                        lo, hi = wp.static(primitive.bounds_of_item)(item)
                        bound = wp.static(ops.node_bound)(query, lo, hi)
                        if wp.isnan(bound):
                            return Point(0.0), wp.inf, -1, -1
                        if bound <= wp.static(ops.cull_limit)(query, best):
                            candidate, score = wp.static(ops.closest)(query, item)
                            if wp.isnan(score):
                                return Point(0.0), wp.inf, -1, -1
                            if score <= best:
                                index = tree.prim_ids[slot]+0
                                if score < best or best_id < 0 or index < best_id:
                                    point, best, best_id = candidate, score, index
                    break
                left, right = tree.node_left[node]+0, tree.node_right[node]+0
                left_score = node_bound(query, tree, geometry, left)
                right_score = node_bound(query, tree, geometry, right)
                if wp.isnan(left_score) or wp.isnan(right_score):
                    return Point(0.0), wp.inf, -1, -1
                if not (left_score <= best or right_score <= best):
                    break
                if left_score <= best and right_score <= best:
                    if size == wp.static(stack_capacity):
                        return Point(0.0), wp.inf, -1, -1
                    near, far, far_score = left, right, right_score
                    if right_score < left_score:
                        near, far, far_score = right, left, left_score
                    nodes[size] = far
                    scores[size] = far_score
                    size += 1
                    node = near
                elif left_score <= best:
                    node = left
                else:
                    node = right
            found = bool(False)
            while size > 0:
                size -= 1
                if scores[size] <= best:
                    node = nodes[size]+0
                    found = True
                    break
            if not found:
                break
        if best_id < 0:
            return point, wp.inf, -1, 0
        distance = wp.static(ops.to_distance)(best)
        if not wp.isfinite(distance):
            return Point(0.0), wp.inf, -1, -1
        return point, distance, best_id, 1

    inputs = [("queries", wp.array(dtype=Query)), ("caps", wp.array(dtype=wp.float32))]
    outputs = [("out_point", wp.array(dtype=Point)), ("out_distance", wp.array(dtype=wp.float32)),
               ("out_id", wp.array(dtype=wp.int32)), ("out_status", wp.array(dtype=wp.int32))]
    body = ["p, d, index, status = wp.static(nearest)(queries[i], caps[i], tree, geometry)",
            "out_point[i] = p", "out_distance[i] = d", "out_id[i] = index", "out_status[i] = status"]
    from ._warp_metrics import Euclidean
    native = None
    if is_native_triangle(primitive, stack_capacity) and type(metric) is Euclidean and predicate is None:
        from ._warp_closest import native_closest_point as native
    return bind_query(NearestQuery, "nearest_kernel", primitive, inputs, outputs, body, {"nearest": nearest},
                       block_dim, native=native, ops=ops)
