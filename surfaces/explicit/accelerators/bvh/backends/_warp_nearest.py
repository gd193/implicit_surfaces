"""Nearest queries with static primitive and metric callbacks."""
from dataclasses import dataclass
from functools import lru_cache

import jax.numpy as jnp
import warp as wp

from ._warp_factory import BoundQuery, bind_query
from ._warp_traversal import load_array, make_walk
from ._warp_primitives import _struct
from ._warp_tree import DeviceTree, STACK_CAPACITY


def prepare_nearest(ops, queries, max_distance):
    queries = jnp.asarray(queries, jnp.float32)
    tail = ops.query_shape
    if tail and queries.shape[-len(tail):] != tail:
        raise ValueError(f"expected queries with trailing shape {tail}")
    shape = queries.shape[:-len(tail)] if tail else queries.shape
    cap = jnp.broadcast_to(jnp.asarray(max_distance, jnp.float32), shape).reshape((-1,))
    return queries.reshape((-1,)+tail), cap, shape


@dataclass(frozen=True)
class NearestQuery(BoundQuery):
    ops: object

    def __call__(self, tree, geometry, queries, max_distance=jnp.inf):
        queries, cap, shape = prepare_nearest(self.ops, queries, max_distance)
        return self.launch(tree, geometry, shape, queries, cap)


@lru_cache(maxsize=None)
def make_nearest(primitive, metric, *, predicate=None, stack_capacity=STACK_CAPACITY, block_dim=64, cache_bounds=True):
    if not 1 <= stack_capacity <= STACK_CAPACITY or not 1 <= block_dim <= 1024:
        raise ValueError("invalid stack capacity or block dimension")
    ops = metric.bind(primitive)
    accepts = predicate.bind(primitive, ops) if hasattr(predicate, "bind") else predicate
    Query, Point, Geometry = ops.query_type, primitive.point_type, primitive.data_type
    Context = _struct("NearestContext", {"query": Query, "geometry": Geometry, "initial": wp.float32})
    State = _struct("NearestState", {"limit": wp.float32, "failed": wp.bool, "stop": wp.bool,
                                     "point": Point, "index": wp.int32})

    @wp.func
    def bounds(ctx: Context, tree: DeviceTree, node: int, lower: wp.vec3, upper: wp.vec3, state: wp.ref[State]):
        score = wp.static(ops.node_bound)(ctx.query, lower, upper)
        eligible = score <= state.limit
        if wp.static(ops.node_filter is not None):
            if eligible:
                eligible = wp.static(ops.node_filter)(ctx.query, tree, ctx.geometry, node)
        if wp.static(ops.node_bound_with_data is not None):
            if eligible:
                score = wp.static(ops.node_bound_with_data)(ctx.query, tree, ctx.geometry, node)
                eligible = score <= state.limit
        if wp.static(ops.node_filter is None and ops.node_bound_with_data is None):
            if node == 0 and score == wp.inf and ctx.initial > state.limit:
                state.failed = True
        return score, eligible, False

    @wp.func
    def visit(ctx: Context, tree: DeviceTree, node: int, begin: int, end: int, tag: bool, state: wp.ref[State]):
        for slot in range(begin, end):
            item = wp.static(primitive.load)(ctx.geometry, slot)
            if wp.static(accepts is not None):
                if not wp.static(accepts)(ctx.query, item):
                    continue
            if wp.static(ops.cull_items):
                lo, hi = wp.static(primitive.bounds_of_item)(item)
                bound = wp.static(ops.node_bound)(ctx.query, lo, hi)
                state.failed = wp.where(bound != bound, True, state.failed)
                if not (bound <= wp.static(ops.cull_limit)(ctx.query, state.limit)):
                    continue
            point, score = wp.static(ops.closest)(ctx.query, item)
            state.failed = wp.where(score != score, True, state.failed)
            if score <= state.limit:
                index = load_array(tree.prim_ids, slot)
                if score < state.limit or state.index < 0 or index < state.index:
                    state.point = point
                    state.limit = score
                    state.index = index
        return

    @wp.func
    def resume(ctx: Context, tree: DeviceTree, node: int, lower: wp.vec3, upper: wp.vec3):
        return wp.static(ops.node_bound)(ctx.query, lower, upper)

    walk = make_walk(Context, State, bounds, visit, stack_capacity=stack_capacity,
                     packed_nodes=cache_bounds, resume=None if cache_bounds else resume)

    @wp.func
    def nearest(query: Query, cap: float, tree: DeviceTree, geometry: Geometry):
        if not wp.static(ops.valid)(query) or not (cap >= 0.0):
            return Point(0.0), wp.inf, -1, -1
        ctx = Context()
        ctx.query = query
        ctx.geometry = geometry
        ctx.initial = wp.static(ops.to_score)(cap)
        if ctx.initial != ctx.initial:
            return Point(0.0), wp.inf, -1, -1
        state = State()
        state.limit = wp.min(ctx.initial, 3.4028234e38)
        state.index = -1
        wp.static(walk)(tree, ctx, state)
        if state.failed:
            return Point(0.0), wp.inf, -1, -1
        if state.index < 0:
            return Point(0.0), wp.inf, -1, 0
        distance = wp.static(ops.to_distance)(state.limit)
        if not wp.isfinite(distance):
            return Point(0.0), wp.inf, -1, -1
        return state.point, distance, state.index, 1

    inputs = [("queries", wp.array(dtype=Query)), ("caps", wp.array(dtype=wp.float32))]
    outputs = [("out_point", wp.array(dtype=Point)), ("out_distance", wp.array(dtype=wp.float32)),
               ("out_id", wp.array(dtype=wp.int32)), ("out_status", wp.array(dtype=wp.int32))]
    body = ["p, d, index, status = wp.static(nearest)(load_array(queries, i), load_array(caps, i), tree, geometry)",
            "store_array(out_point, i, p)", "store_array(out_distance, i, d)", "store_array(out_id, i, index)", "store_array(out_status, i, status)"]
    return bind_query(NearestQuery, "nearest_kernel", primitive, inputs, outputs, body, {"nearest": nearest},
                       block_dim, ops=ops)
