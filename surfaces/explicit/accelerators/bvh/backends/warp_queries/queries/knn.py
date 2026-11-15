"""Fused generic k-nearest queries with fixed-capacity candidate storage."""
from dataclasses import dataclass
from functools import lru_cache
from typing import Generic, TypeVar

import jax.numpy as jnp
import warp as wp

from ..launch import BoundQuery, bind_query
from .nearest import prepare_nearest
from .nearest_ops import bind_nearest as _bind_nearest, nearest_functions, resolve_metric
from ..selection.candidates import REGISTER_LIMIT, candidate_functions, configure_candidate_module
from ..io.results import result_writer
from ..io.arrays import load_array
from ..traversal.scheduling import make_walk
from ..contracts import make_struct
from ..._warp_tree import DeviceTree, STACK_CAPACITY


HostDataT = TypeVar("HostDataT")
DeviceDataT = TypeVar("DeviceDataT")
ItemT = TypeVar("ItemT")
PointT = TypeVar("PointT")
QueryT = TypeVar("QueryT")

@dataclass(frozen=True)
class KnnQuery(BoundQuery, Generic[HostDataT, DeviceDataT, ItemT, PointT, QueryT]):
    ops: object
    k: int
    return_points: bool

    def __call__(self, tree, geometry, queries, max_distance=jnp.inf):
        queries, cap, shape = prepare_nearest(self.ops, queries, max_distance)
        return self.launch(tree, geometry, shape, queries, cap,
                           output_shapes=((self.k,), (self.k,), (), ()))


def make_knn(primitive, k, metric=None, *, evaluate=None, lower_bound=None, predicate=None,
             stack_capacity=STACK_CAPACITY, block_dim=64, cache_bounds=True,
             packed_nodes=True, group_leaves=None, return_points=False):
    """Specialize a fused k-nearest query using a metric or two static Warp callbacks.

    Callback scores and ``max_distance`` caps share units without conversion.
    ``evaluate(query, item)`` returns a point and a finite nonnegative score;
    ``lower_bound(query, box_lower, box_upper)`` must conservatively bound that
    score, including float32 rounding. Box endpoints are always ``wp.vec3``.
    Queries use the policy query shape. Results are (scores_or_distances, ids,
    count, status), ordered by internal score then original ID. Missing entries
    have +inf/-1 padding; failed rows are cleared. k is a positive Python int.
    With return_points=True, coordinates replace IDs and use zero padding.
    """
    if isinstance(k, bool) or not isinstance(k, int):
        raise TypeError("k must be a positive static Python int")
    if k < 1:
        raise ValueError("k must be positive")
    if not isinstance(return_points, bool):
        raise TypeError("return_points must be a static Python bool")
    metric = resolve_metric(metric, evaluate, lower_bound, "make_knn")
    if group_leaves is None:
        group_leaves = _bind_nearest(primitive, metric).group_leaves
    return _make_knn(primitive, k, metric, predicate=predicate,
                     stack_capacity=stack_capacity, block_dim=block_dim,
                     cache_bounds=cache_bounds, packed_nodes=packed_nodes,
                     group_leaves=group_leaves, return_points=return_points)


@lru_cache(maxsize=None)
def _make_knn(primitive, k, metric, *, predicate=None,
              stack_capacity=STACK_CAPACITY, block_dim=64, cache_bounds=True,
              packed_nodes=True, group_leaves=False, return_points=False):
    if not 1 <= stack_capacity <= STACK_CAPACITY or not 1 <= block_dim <= 1024:
        raise ValueError("invalid stack capacity or block dimension")
    candidates = candidate_functions(k, return_points)
    Candidates = candidates.storage
    add, limit, sort, get_count = candidates.add, candidates.limit, candidates.sort, candidates.count
    ops = _bind_nearest(primitive, metric)
    Query, Geometry = ops.query_type, primitive.data_type
    Point = primitive.point_type
    Output = Point if return_points else wp.int32
    Context = make_struct("KnnContext", {"query": Query, "geometry": Geometry, "initial": wp.float32})
    State = make_struct("KnnState", {"limit": wp.float32, "failed": wp.bool, "stop": wp.bool,
                                 "candidates": Candidates})

    functions = nearest_functions(primitive, metric, predicate, Context, State)
    accepts, bounds, resume = functions.accepts, functions.bounds, functions.resume

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
            if not wp.isfinite(score) or score < 0.0:
                state.failed = True
                return
            if score <= state.limit:
                index = load_array(tree.prim_ids, slot)
                wp.static(add)(state.candidates, score, index, slot)
                state.limit = wp.static(limit)(state.candidates, ctx.initial)
        return

    walk = make_walk(Context, State, bounds, visit, stack_capacity=stack_capacity,
                     packed_nodes=packed_nodes, resume=None if cache_bounds else resume,
                     group_leaves=group_leaves)

    emit = result_writer(primitive, ops, candidates, State, k, return_points)

    @wp.func
    def knn(query: Query, cap: float, tree: DeviceTree, geometry: Geometry,
            out_score: wp.array2d(dtype=wp.float32), out_id: wp.array2d(dtype=Output), row: int):
        ctx = Context()
        ctx.query = query
        ctx.geometry = geometry
        ctx.initial = wp.static(ops.to_score)(cap)
        state = State()
        state.limit = wp.min(ctx.initial, 3.4028234e38)
        if not wp.static(ops.valid)(query) or not (cap >= 0.0) or not (ctx.initial >= 0.0):
            state.failed = True
        else:
            wp.static(walk)(tree, ctx, state)
        count = wp.static(get_count)(state.candidates)
        if not state.failed:
            wp.static(sort)(state.candidates)
        if state.failed:
            count = 0
        if wp.static(return_points):
            return wp.static(emit)(query, geometry, state, count, out_score, out_id, row)
        else:
            return wp.static(emit)(state, count, out_score, out_id, row)

    inputs = [("queries", wp.array(dtype=Query)), ("caps", wp.array(dtype=wp.float32))]
    outputs = [("out_score", wp.array2d(dtype=wp.float32)), ("out_id", wp.array2d(dtype=Output)),
               ("out_count", wp.array(dtype=wp.int32)), ("out_status", wp.array(dtype=wp.int32))]
    body = ["count, status = wp.static(knn)(load_array(queries, i), load_array(caps, i), tree, geometry, out_score, out_id, i)",
            "store_array(out_count, i, count)", "store_array(out_status, i, status)"]
    components = (("node_access", "packed" if packed_nodes else "indexed"),
                  ("scheduling", "grouped" if group_leaves else "flat"),
                  ("deferred", "cached_bound" if cache_bounds else "recomputed_bound"),
                  ("selection", "unrolled" if k <= REGISTER_LIMIT else "heap"),
                  ("output", "coordinates" if return_points else "ids"))
    query = bind_query(KnnQuery, "knn_kernel", primitive, inputs, outputs, body, {"knn": knn},
                       block_dim, ops=ops, k=k, return_points=return_points, components=components)
    configure_candidate_module(query.kernel.module, candidates, k, return_points)
    return query
