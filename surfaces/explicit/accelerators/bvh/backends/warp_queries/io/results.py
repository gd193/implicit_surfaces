"""Static kNN result policies; output arrays are written inside the fused query."""
from functools import lru_cache

import warp as wp

from .arrays import store_array


@lru_cache(maxsize=None)
def result_writer(primitive, ops, candidates, State, k, return_points):
    Point = primitive.point_type
    Query, Geometry = ops.query_type, primitive.data_type
    get_score, get_id, get_slot = candidates.score, candidates.item_id, candidates.slot

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
    def emit_ids(state: wp.ref[State], count: int,
                 out_score: wp.array2d(dtype=wp.float32), out_id: wp.array2d(dtype=wp.int32), row: int):
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

    return emit_points if return_points else emit_ids


@lru_cache(maxsize=None)
def custom_result_writer(primitive, ops, candidates, State, k, policy):
    """Project final selected neighbors into a caller-defined Warp struct."""
    Query, Geometry = ops.query_type, primitive.data_type
    Result = policy.result_type
    build = policy.build
    get_score, get_id, get_slot = candidates.score, candidates.item_id, candidates.slot

    @wp.func
    def emit_custom(query: Query, geometry: Geometry, state: wp.ref[State], count: int,
                    out_score: wp.array2d(dtype=wp.float32),
                    out_result: wp.array2d(dtype=Result), row: int):
        for index in range(wp.static(k)):
            value = wp.inf
            result = Result()
            if index < count:
                value = wp.static(ops.to_distance)(wp.static(get_score)(state.candidates, index))
                item = wp.static(primitive.load)(geometry, wp.static(get_slot)(state.candidates, index))
                point, score = wp.static(ops.closest)(query, item)
                if not wp.isfinite(point) or not wp.isfinite(score) or score < 0.0:
                    state.failed = True
                if not wp.isfinite(value) or value < 0.0:
                    state.failed = True
                if not state.failed:
                    result = wp.static(build)(query, item, wp.static(get_id)(state.candidates, index), point, value)
            store_array(out_score, row*wp.static(k)+index, value)
            store_array(out_result, row*wp.static(k)+index, result)
        if state.failed:
            count = 0
            for index in range(wp.static(k)):
                store_array(out_score, row*wp.static(k)+index, wp.inf)
                store_array(out_result, row*wp.static(k)+index, Result())
        status = wp.where(state.failed, -1, wp.where(count > 0, 1, 0))
        return count, status

    return emit_custom
