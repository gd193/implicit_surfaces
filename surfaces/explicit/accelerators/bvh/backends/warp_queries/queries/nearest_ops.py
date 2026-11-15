"""Host-resolved nearest callbacks and common traversal operations."""
from dataclasses import dataclass
from functools import lru_cache

import warp as wp

from ..metrics import NearestCallbacks, NearestOps
from ..._warp_tree import DeviceTree
from ..contracts import WarpFunction


@dataclass(frozen=True)
class NearestFunctions:
    ops: "NearestOps"
    accepts: WarpFunction[..., bool] | None
    node_test: wp.Function
    bounds: wp.Function
    resume: wp.Function


def resolve_metric(metric, evaluate, lower_bound, query_name):
    if metric is not None:
        if evaluate is not None or lower_bound is not None:
            raise TypeError(f"{query_name} accepts either metric or evaluate/lower_bound callbacks, not both")
        return metric
    if evaluate is None or lower_bound is None:
        raise TypeError(f"{query_name} requires a metric or both evaluate and lower_bound callbacks")
    return NearestCallbacks(evaluate, lower_bound)


@lru_cache(maxsize=None)
def bind_nearest(primitive, metric):
    return metric.bind(primitive)


@lru_cache(maxsize=None)
def nearest_functions(primitive, metric, predicate, Context, State):
    ops = bind_nearest(primitive, metric)
    accepts = predicate.bind(primitive, ops) if hasattr(predicate, "bind") else predicate

    # Select the callback combination on the host. The resulting device
    # function contains only the calls needed by this particular policy.
    if ops.node_filter is None and ops.node_bound_with_data is None:
        @wp.func
        def node_test(ctx: Context, tree: DeviceTree, node: int, lower: wp.vec3,
                      upper: wp.vec3, state: wp.ref[State]):
            score = wp.static(ops.node_bound)(ctx.query, lower, upper)
            return score, score <= state.limit
    elif ops.node_filter is not None and ops.node_bound_with_data is None:
        @wp.func
        def node_test(ctx: Context, tree: DeviceTree, node: int, lower: wp.vec3,
                      upper: wp.vec3, state: wp.ref[State]):
            score = wp.static(ops.node_bound)(ctx.query, lower, upper)
            eligible = score <= state.limit
            if eligible:
                eligible = wp.static(ops.node_filter)(ctx.query, tree, ctx.geometry, node)
            return score, eligible
    elif ops.node_filter is None:
        @wp.func
        def node_test(ctx: Context, tree: DeviceTree, node: int, lower: wp.vec3,
                      upper: wp.vec3, state: wp.ref[State]):
            score = wp.static(ops.node_bound)(ctx.query, lower, upper)
            eligible = score <= state.limit
            if eligible:
                score = wp.static(ops.node_bound_with_data)(ctx.query, tree, ctx.geometry, node)
                eligible = score <= state.limit
            return score, eligible
    else:
        @wp.func
        def node_test(ctx: Context, tree: DeviceTree, node: int, lower: wp.vec3,
                      upper: wp.vec3, state: wp.ref[State]):
            score = wp.static(ops.node_bound)(ctx.query, lower, upper)
            eligible = score <= state.limit
            if eligible:
                eligible = wp.static(ops.node_filter)(ctx.query, tree, ctx.geometry, node)
            if eligible:
                score = wp.static(ops.node_bound_with_data)(ctx.query, tree, ctx.geometry, node)
                eligible = score <= state.limit
            return score, eligible

    @wp.func
    def bounds(ctx: Context, tree: DeviceTree, node: int, lower: wp.vec3,
               upper: wp.vec3, state: wp.ref[State]):
        score, eligible = wp.static(node_test)(ctx, tree, node, lower, upper, state)
        if wp.static(ops.node_filter is None and ops.node_bound_with_data is None):
            if node == 0 and score == wp.inf and ctx.initial > state.limit:
                state.failed = True
        return score, eligible, False

    @wp.func
    def resume(ctx: Context, tree: DeviceTree, node: int, lower: wp.vec3, upper: wp.vec3):
        return wp.static(ops.node_bound)(ctx.query, lower, upper)

    return NearestFunctions(ops, accepts, node_test, bounds, resume)
