"""Nearest queries with static primitive and metric callbacks."""
from dataclasses import dataclass, replace
from functools import lru_cache
from typing import Generic, TypeVar

import jax.numpy as jnp
import warp as wp

from ..launch import BoundQuery, bind_query
from ..io.arrays import load_array
from ..traversal.scheduling import make_walk
from ..contracts import make_struct
from ..metrics import Euclidean
from ..primitives.builtin import Triangles
from ..._warp_tree import DeviceTree, STACK_CAPACITY
from .nearest_ops import bind_nearest as _bind_nearest, nearest_functions, resolve_metric
from .results import resolve_result


def prepare_nearest(ops, queries, max_distance):
    queries = jnp.asarray(queries, jnp.float32)
    tail = ops.query_shape
    if tail and queries.shape[-len(tail):] != tail:
        raise ValueError(f"expected queries with trailing shape {tail}")
    shape = queries.shape[:-len(tail)] if tail else queries.shape
    cap = jnp.broadcast_to(jnp.asarray(max_distance, jnp.float32), shape).reshape((-1,))
    return queries.reshape((-1,)+tail), cap, shape


HostDataT = TypeVar("HostDataT")
DeviceDataT = TypeVar("DeviceDataT")
ItemT = TypeVar("ItemT")
PointT = TypeVar("PointT")
QueryT = TypeVar("QueryT")
ResultT = TypeVar("ResultT")


@dataclass(frozen=True)
class NearestQuery(BoundQuery, Generic[HostDataT, DeviceDataT, ItemT, PointT, QueryT]):
    ops: object

    def __call__(self, tree, geometry, queries, max_distance=jnp.inf):
        queries, cap, shape = prepare_nearest(self.ops, queries, max_distance)
        return self.launch(tree, geometry, shape, queries, cap)


@dataclass(frozen=True)
class SelectedNearestQuery(BoundQuery, Generic[HostDataT, DeviceDataT, ItemT, PointT, QueryT, ResultT]):
    """Device-only query with a statically typed caller-owned result."""
    ops: object
    result_policy: object


def _builtin_triangle(primitive):
    builtin = Triangles()
    return (primitive is builtin or
            (type(primitive) is type(builtin) and replace(primitive, name=builtin.name) == builtin))


def make_nearest(primitive, metric=None, *, evaluate=None, lower_bound=None, predicate=None,
                 stack_capacity=STACK_CAPACITY, block_dim=64, cache_bounds=None,
                 packed_nodes=True, group_leaves=None, result_policy=None):
    """Specialize a fused nearest query using a metric or two static Warp callbacks.

    Callback scores and ``max_distance`` caps share units without conversion.
    ``evaluate(query, item)`` returns a point and a finite nonnegative score;
    ``lower_bound(query, box_lower, box_upper)`` must conservatively bound that
    score, including float32 rounding. Box endpoints are always ``wp.vec3``.
    Queries use the primitive's point type and dimension.
    """
    metric = resolve_metric(metric, evaluate, lower_bound, "make_nearest")
    ops = _bind_nearest(primitive, metric)
    if cache_bounds is None:
        cache_bounds = ops.cache_nearest_bounds
    if group_leaves is None:
        group_leaves = ops.group_leaves
    if result_policy is not None:
        result_policy = resolve_result(result_policy, primitive, ops.query_type)
    return _make_nearest(primitive, metric, predicate=predicate,
                         stack_capacity=stack_capacity, block_dim=block_dim,
                         cache_bounds=cache_bounds, packed_nodes=packed_nodes,
                         group_leaves=group_leaves, result_policy=result_policy)


@lru_cache(maxsize=None)
def _make_nearest(primitive, metric, *, predicate=None,
                  stack_capacity=STACK_CAPACITY, block_dim=64, cache_bounds=True,
                  packed_nodes=True, group_leaves=False, result_policy=None):
    if not 1 <= stack_capacity <= STACK_CAPACITY or not 1 <= block_dim <= 1024:
        raise ValueError("invalid stack capacity or block dimension")
    ops = _bind_nearest(primitive, metric)
    score_at = getattr(ops, "score_at", None)
    point_at = getattr(ops, "point_at", None)
    fused_triangle = _builtin_triangle(primitive) and ops == Euclidean().bind(primitive)
    defer_point = not fused_triangle and score_at is not None and point_at is not None
    Query, Point, Geometry = ops.query_type, primitive.point_type, primitive.data_type
    Context = make_struct("NearestContext", {"query": Query, "geometry": Geometry, "initial": wp.float32})
    state_fields = {"limit": wp.float32, "failed": wp.bool, "stop": wp.bool,
                    "index": wp.int32}
    if not defer_point:
        state_fields["point"] = Point
    if defer_point or (result_policy is not None and not fused_triangle):
        state_fields["slot"] = wp.int32
    if fused_triangle and result_policy is not None:
        state_fields["item"] = primitive.item_type
    State = make_struct("NearestState", state_fields)

    functions = nearest_functions(primitive, metric, predicate, Context, State,
                                  fused_triangle=fused_triangle)
    accepts, bounds, resume = functions.accepts, functions.bounds, functions.resume

    @wp.func
    def visit_generic(ctx: Context, tree: DeviceTree, node: int, begin: int, end: int, tag: bool, state: wp.ref[State]):
        for slot in range(begin, end):
            if wp.static(accepts is not None or ops.cull_items or not defer_point):
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
            if wp.static(defer_point):
                score = wp.static(score_at)(ctx.query, ctx.geometry, slot)
            else:
                point, score = wp.static(ops.closest)(ctx.query, item)
            state.failed = wp.where(score != score, True, state.failed)
            if score <= state.limit:
                index = load_array(tree.prim_ids, slot)
                if score < state.limit or state.index < 0 or index < state.index:
                    if wp.static(not defer_point):
                        state.point = point
                    state.limit = score
                    state.index = index
                    if wp.static(defer_point or result_policy is not None):
                        state.slot = slot
        return

    visit = visit_generic
    if fused_triangle:
        @wp.func_native("return bvh::triangle_nearest_at(query, geometry, slot, limit);")
        def candidate(query: Query, geometry: Geometry, slot: int, limit: float) -> wp.vec4: ...

        @wp.func_native("return bvh::triangle_nearest_candidate(query, a, b, c, limit);")
        def candidate_vertices(query: Query, a: Point, b: Point, c: Point, limit: float) -> wp.vec4: ...

        @wp.func
        def visit(ctx: Context, tree: DeviceTree, node: int, begin: int, end: int, tag: bool,
                  state: wp.ref[State]):
            for slot in range(begin, end):
                if wp.static(accepts is not None or result_policy is not None):
                    item = wp.static(primitive.load)(ctx.geometry, slot)
                    if wp.static(accepts is not None):
                        if not wp.static(accepts)(ctx.query, item):
                            continue
                    evaluated = wp.static(candidate_vertices)(ctx.query, item.a, item.b, item.c, state.limit)
                else:
                    evaluated = wp.static(candidate)(ctx.query, ctx.geometry, slot, state.limit)
                score = evaluated[3]
                if score >= 0.0 and (score < state.limit or (state.index < 0 and score == state.limit)):
                    state.limit = score
                    state.point = wp.vec3(evaluated[0], evaluated[1], evaluated[2])
                    state.index = load_array(tree.prim_ids, slot)
                    if wp.static(result_policy is not None):
                        state.item = item
                elif not wp.isfinite(score):
                    state.failed = True

    walk = make_walk(Context, State, bounds, visit, stack_capacity=stack_capacity,
                     packed_nodes=packed_nodes, resume=None if cache_bounds else resume,
                     group_leaves=group_leaves)

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
        if wp.static(defer_point):
            point = wp.static(point_at)(query, geometry, state.slot)
        else:
            point = state.point
        distance = wp.static(ops.to_distance)(state.limit)
        if not wp.isfinite(point) or not wp.isfinite(distance):
            return Point(0.0), wp.inf, -1, -1
        return point, distance, state.index, 1

    inputs = [("queries", wp.array(dtype=Query)), ("caps", wp.array(dtype=wp.float32))]
    outputs = [("out_point", wp.array(dtype=Point)), ("out_distance", wp.array(dtype=wp.float32)),
               ("out_id", wp.array(dtype=wp.int32)), ("out_status", wp.array(dtype=wp.int32))]
    body = ["p, d, index, status = wp.static(nearest)(load_array(queries, i), load_array(caps, i), tree, geometry)",
            "store_array(out_point, i, p)", "store_array(out_distance, i, d)", "store_array(out_id, i, index)", "store_array(out_status, i, status)"]
    components = (("node_access", "packed" if packed_nodes else "indexed"),
                  ("scheduling", "grouped" if group_leaves else "flat"),
                  ("deferred", "cached_bound" if cache_bounds else "recomputed_bound"),
                  ("selection", "single"), ("output", "coordinates"))
    if result_policy is not None:
        Result = result_policy.result_type
        build = result_policy.build

        @wp.func
        def selected(query: Query, cap: float, tree: DeviceTree, geometry: Geometry):
            result = Result()
            if not wp.static(ops.valid)(query) or not (cap >= 0.0):
                return result, -1
            ctx = Context()
            ctx.query = query
            ctx.geometry = geometry
            ctx.initial = wp.static(ops.to_score)(cap)
            if ctx.initial != ctx.initial:
                return result, -1
            state = State()
            state.limit = wp.min(ctx.initial, 3.4028234e38)
            state.index = -1
            wp.static(walk)(tree, ctx, state)
            if state.failed:
                return result, -1
            if state.index < 0:
                return result, 0
            if wp.static(defer_point):
                point = wp.static(point_at)(query, geometry, state.slot)
            else:
                point = state.point
            distance = wp.static(ops.to_distance)(state.limit)
            if not wp.isfinite(point) or not wp.isfinite(distance) or distance < 0.0:
                return result, -1
            if wp.static(fused_triangle):
                item = state.item
            else:
                item = wp.static(primitive.load)(geometry, state.slot)
            result = wp.static(build)(query, item, state.index, point, distance)
            return result, 1

        outputs = [("out_result", wp.array(dtype=Result)), ("out_status", wp.array(dtype=wp.int32))]
        body = ["value, status = wp.static(selected)(load_array(queries, i), load_array(caps, i), tree, geometry)",
                "store_array(out_result, i, value)", "store_array(out_status, i, status)"]
        components = (*components[:-1], ("output", "custom_struct"))
        return bind_query(SelectedNearestQuery, "nearest_result_kernel", primitive, inputs, outputs, body,
                          {"selected": selected}, block_dim, device=selected, ops=ops,
                          result_policy=result_policy, components=components)
    return bind_query(NearestQuery, "nearest_kernel", primitive, inputs, outputs, body, {"nearest": nearest},
                       block_dim, device=nearest, ops=ops, components=components)
