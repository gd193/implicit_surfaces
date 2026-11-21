"""Typed traversal references and reductions use the same source on device."""
from typing import assert_type
import warp as wp
from sengi.bvh.warp import (
    DeviceTree, Euclidean, PointItem3, Points, PointsData3, RadiusAccumulator,
    WarpRef, make_radius, make_traversal, warp_func, warp_struct,
)


@warp_struct
class SumState:
    limit: float
    failed: bool
    stop: bool
    matched: bool
    total: float


@warp_struct
class Context:
    limit: float


@warp_func
def initialize(query: wp.vec3, limit: float) -> SumState:
    state = SumState()
    state.limit = limit
    return state


@warp_func
def update(query: wp.vec3, item: PointItem3, point: wp.vec3, score: float,
           index: int, state: WarpRef[SumState]) -> None:
    state.total += score


@warp_func
def finish(state: SumState) -> float:
    return state.total


@warp_func
def bounds(ctx: Context, tree: DeviceTree, node: int, lower: wp.vec3,
           upper: wp.vec3, state: WarpRef[SumState]) -> tuple[float, bool, bool]:
    return 0.0, True, False


@warp_func
def visit(ctx: Context, tree: DeviceTree, node: int, begin: int, end: int,
          tag: bool, state: WarpRef[SumState]) -> None:
    state.total += float(end - begin)


accumulator = RadiusAccumulator(SumState, initialize, update, finish)
reduction = make_radius(Points(), Euclidean(), mode="accumulate", accumulator=accumulator)
traversal = make_traversal(Context, SumState, bounds, visit)


def signatures(tree: DeviceTree, geometry: PointsData3, point: wp.vec3) -> None:
    assert_type(make_radius(Points(), Euclidean()).device(point, 1.0, tree, geometry), tuple[int, int])
    assert_type(make_radius(Points(), Euclidean(), mode="count").device(point, 1.0, tree, geometry), tuple[int, int])
    assert_type(reduction.device(point, 1.0, tree, geometry), tuple[float, int])
    state = SumState()
    traversal.device(tree, Context(), state)
