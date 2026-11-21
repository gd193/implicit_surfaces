"""Reject mode, context, state, primitive and geometry mismatches."""
import warp as wp
from sengi.bvh.warp import (
    DeviceTree, Euclidean, Points, PointsData3, Triangles, WarpRef, make_radius, warp_struct,
)
from tests.typing.warp_api.range_valid import Context, SumState, accumulator, reduction, traversal


@warp_struct
class WrongState:
    limit: float
    failed: bool
    stop: bool


def invalid(tree: DeviceTree, geometry: PointsData3, point: wp.vec3,
            state: WarpRef[SumState]) -> None:
    state.total = "wrong"  # expect: reportAttributeAccessIssue
    traversal.device(tree, point, state)  # expect: reportArgumentType
    traversal.device(tree, Context(), WrongState())  # expect: reportArgumentType
    reduction.device(wp.vec2(), 1.0, tree, geometry)  # expect: reportArgumentType
    reduction.device(point, 1.0, tree, point)  # expect: reportArgumentType
    make_radius(Points(), Euclidean(), mode="all")  # expect: reportArgumentType
    make_radius(Points(), Euclidean(), mode="accumulate")  # expect: reportArgumentType
    make_radius(Triangles(), Euclidean(), mode="accumulate", accumulator=accumulator)  # expect: reportCallIssue  # expect: reportArgumentType
