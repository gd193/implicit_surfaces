"""Fused generic k-nearest queries with fixed-capacity candidate storage."""
import ctypes
from dataclasses import dataclass
from functools import lru_cache
from typing import Generic, TypeVar

import jax.numpy as jnp
import warp as wp

from ._warp_factory import BoundQuery, bind_query
from ._warp_metrics import NearestCallbacks
from ._warp_nearest import _bind_nearest, prepare_nearest
from ._warp_traversal import load_array, store_array, make_walk, _PREAMBLE, _NATIVE_DIR
from ._warp_primitives import _struct
from ._warp_tree import DeviceTree, STACK_CAPACITY


HostDataT = TypeVar("HostDataT")
DeviceDataT = TypeVar("DeviceDataT")
ItemT = TypeVar("ItemT")
PointT = TypeVar("PointT")
QueryT = TypeVar("QueryT")

# Larger unrolled lists exhaust registers; use an in-place heap above this size.
_REGISTER_LIMIT = 64


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
    if metric is not None:
        if evaluate is not None or lower_bound is not None:
            raise TypeError("make_knn accepts either metric or evaluate/lower_bound callbacks, not both")
    else:
        if evaluate is None or lower_bound is None:
            raise TypeError("make_knn requires a metric or both evaluate and lower_bound callbacks")
        metric = NearestCallbacks(evaluate, lower_bound)
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
    Candidates, insert, limit, sort, get_score, get_id, get_count, get_slot = candidate_functions(k, return_points)
    ops = _bind_nearest(primitive, metric)
    accepts = predicate.bind(primitive, ops) if hasattr(predicate, "bind") else predicate
    Query, Geometry = ops.query_type, primitive.data_type
    Point = primitive.point_type
    Output = Point if return_points else wp.int32
    Context = _struct("KnnContext", {"query": Query, "geometry": Geometry, "initial": wp.float32})
    State = _struct("KnnState", {"limit": wp.float32, "failed": wp.bool, "stop": wp.bool,
                                 "candidates": Candidates})

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
            if not wp.isfinite(score) or score < 0.0:
                state.failed = True
                return
            if score <= state.limit:
                index = load_array(tree.prim_ids, slot)
                if wp.static(return_points):
                    wp.static(insert)(state.candidates, score, index, slot)
                else:
                    wp.static(insert)(state.candidates, score, index)
                state.limit = wp.static(limit)(state.candidates, ctx.initial)
        return

    @wp.func
    def resume(ctx: Context, tree: DeviceTree, node: int, lower: wp.vec3, upper: wp.vec3):
        return wp.static(ops.node_bound)(ctx.query, lower, upper)

    walk = make_walk(Context, State, bounds, visit, stack_capacity=stack_capacity,
                     packed_nodes=packed_nodes, resume=None if cache_bounds else resume,
                     group_leaves=group_leaves)

    # Separate emission keeps the ID-only output loop and register allocation
    # unchanged; both paths use the same traversal and candidate operations.
    @wp.func
    def emit_points(query: Query, geometry: Geometry, state: wp.ref[State], count: int,
                    out_score: wp.array2d(dtype=wp.float32), out_point: wp.array2d(dtype=Point), row: int):
        for slot in range(wp.static(k)):
            distance, point = wp.inf, Point(0.0)
            if slot < count:
                distance = wp.static(ops.to_distance)(wp.static(get_score)(state.candidates, slot))
                item = wp.static(primitive.load)(geometry, wp.static(get_slot)(state.candidates, slot))
                point, score = wp.static(ops.closest)(query, item)
                if not wp.isfinite(point) or not wp.isfinite(score) or score < 0.0:
                    state.failed = True
                if not wp.isfinite(distance) or distance < 0.0:
                    state.failed = True
            store_array(out_score, row*wp.static(k)+slot, distance)
            store_array(out_point, row*wp.static(k)+slot, point)
        if state.failed:
            count = 0
            for slot in range(wp.static(k)):
                store_array(out_score, row*wp.static(k)+slot, wp.inf)
                store_array(out_point, row*wp.static(k)+slot, Point(0.0))
        status = wp.where(state.failed, -1, wp.where(count > 0, 1, 0))
        return count, status

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
            return emit_points(query, geometry, state, count, out_score, out_id, row)
        else:
            for slot in range(wp.static(k)):
                distance = wp.inf
                index = int(-1)
                if slot < count:
                    distance = wp.static(ops.to_distance)(wp.static(get_score)(state.candidates, slot))
                    index = wp.static(get_id)(state.candidates, slot)
                    if not wp.isfinite(distance) or distance < 0.0:
                        state.failed = True
                if wp.static(k <= 16):
                    out_score[row, slot] = distance
                    out_id[row, slot] = index
                else:
                    # FFI outputs are dense; static strides reduce address registers
                    # for larger lists without changing their public (batch, k) shape.
                    store_array(out_score, row*wp.static(k)+slot, distance)
                    store_array(out_id, row*wp.static(k)+slot, index)
            if state.failed:
                count = 0
                for slot in range(wp.static(k)):
                    store_array(out_score, row*wp.static(k)+slot, wp.inf)
                    store_array(out_id, row*wp.static(k)+slot, -1)
            status = wp.where(state.failed, -1, wp.where(count > 0, 1, 0))
            return count, status

    inputs = [("queries", wp.array(dtype=Query)), ("caps", wp.array(dtype=wp.float32))]
    outputs = [("out_score", wp.array2d(dtype=wp.float32)), ("out_id", wp.array2d(dtype=Output)),
               ("out_count", wp.array(dtype=wp.int32)), ("out_status", wp.array(dtype=wp.int32))]
    body = ["count, status = wp.static(knn)(load_array(queries, i), load_array(caps, i), tree, geometry, out_score, out_id, i)",
            "store_array(out_count, i, count)", "store_array(out_status, i, status)"]
    query = bind_query(KnnQuery, "knn_kernel", primitive, inputs, outputs, body, {"knn": knn},
                       block_dim, ops=ops, k=k, return_points=return_points)
    small = "true" if k <= _REGISTER_LIMIT else "false"
    tracked = "true" if return_points else "false"
    name = Candidates.__name__
    preamble = (_PREAMBLE + '\n#include "knn_candidates.h"\n'
                + f"namespace bvh {{ using {name} = KnnCandidates<{k}, {small}, {tracked}>; }}\n"
                + f"static_assert(sizeof(bvh::{name}) == {ctypes.sizeof(Candidates)});\n"
                + f"static_assert(alignof(bvh::{name}) == {ctypes.alignment(Candidates)});\n")
    options = wp.ModuleBuildOptions(
        extra_cpu_include_dirs=[_NATIVE_DIR], extra_cuda_include_dirs=[_NATIVE_DIR],
        extra_cpu_preamble=preamble, extra_cuda_preamble=preamble,
        extra_build_dependencies=sorted(_NATIVE_DIR.glob("*.h")))
    wp.set_module_options({"extra_build_options": options, "max_unroll": max(16, min(k, _REGISTER_LIMIT))}, module=query.kernel.module)
    return query


@lru_cache(maxsize=None)
def candidate_functions(k, track_slots=False):
    # Match the native register-list / heap representation exactly.
    fields = ([('scores', ctypes.c_float*k), ('ids', ctypes.c_int32*k)] if k <= _REGISTER_LIMIT
              else [('entries', ctypes.c_uint64*k)])
    if track_slots:
        fields = [('slots', ctypes.c_int32*k), *fields]
    name = f"KnnCandidates_{k}" + ("_points" if track_slots else "")
    Candidates = type(name, (ctypes.Structure,),
                      {"_fields_": [*fields, ("size", ctypes.c_int32)]})
    wp.build_experimental.add_native_type(Candidates, native_name=f"bvh::{Candidates.__name__}")

    if track_slots:
        @wp.func_native("bvh::knn_insert(candidates, score, index, slot);")
        def insert(candidates: wp.ref[Candidates], score: float, index: int, slot: int): ...
    else:
        @wp.func_native("bvh::knn_insert(candidates, score, index);")
        def insert(candidates: wp.ref[Candidates], score: float, index: int): ...

    @wp.func_native("return bvh::knn_limit(candidates, initial);")
    def limit(candidates: wp.ref[Candidates], initial: float) -> float: ...

    @wp.func_native("bvh::knn_sort(candidates);")
    def sort(candidates: wp.ref[Candidates]): ...

    @wp.func_native("return bvh::knn_score(candidates, index);")
    def get_score(candidates: wp.ref[Candidates], index: int) -> float: ...

    @wp.func_native("return bvh::knn_id(candidates, index);")
    def get_id(candidates: wp.ref[Candidates], index: int) -> int: ...

    @wp.func_native("return candidates.size;")
    def get_count(candidates: wp.ref[Candidates]) -> int: ...

    get_slot = None
    if track_slots:
        @wp.func_native("return candidates.slots[index];")
        def get_slot(candidates: wp.ref[Candidates], index: int) -> int: ...

    return Candidates, insert, limit, sort, get_score, get_id, get_count, get_slot
