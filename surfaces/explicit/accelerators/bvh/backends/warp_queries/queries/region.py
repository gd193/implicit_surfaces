"""Exact primitive versus closed-box queries with static intersection policies."""

from dataclasses import dataclass
from functools import lru_cache
from typing import Generic, TypeVar

import warp as wp

from ..contracts import WarpFunction, make_struct
from ..ffi import _flatten_vec_queries
from ..launch import BoundQuery, bind_query
from ..primitives.geometry import boxes_overlap
from ..primitives.region import (point2_intersects, point3_intersects,
                                 segment2_intersects, segment3_intersects,
                                 triangle3_intersects)
from ..traversal.policy import make_traversal
from ..._warp_tree import DeviceTree, STACK_CAPACITY
from .classification import make_classify

HostDataT = TypeVar("HostDataT")
DeviceDataT = TypeVar("DeviceDataT")
PointT = TypeVar("PointT")
HostMetaT = TypeVar("HostMetaT")
MetaT = TypeVar("MetaT")


@dataclass(frozen=True)
class RegionIntersectionPolicy(Generic[DeviceDataT]):
    """Static callback intersects(geometry, packed_slot, box_lower, box_upper)."""
    intersects: WarpFunction[[DeviceDataT, int, wp.vec3, wp.vec3], bool]

    def __post_init__(self):
        if not isinstance(self.intersects, wp.Function):
            raise TypeError("intersection callback must be a decorated Warp function")


def point_region_policy(dimension: int) -> RegionIntersectionPolicy:
    if dimension == 2:
        return RegionIntersectionPolicy(point2_intersects)
    if dimension == 3:
        return RegionIntersectionPolicy(point3_intersects)
    raise ValueError("points support 2D or 3D")


def segment_region_policy(dimension: int) -> RegionIntersectionPolicy:
    if dimension == 2:
        return RegionIntersectionPolicy(segment2_intersects)
    if dimension == 3:
        return RegionIntersectionPolicy(segment3_intersects)
    raise ValueError("segments support 2D or 3D")


def triangle_region_policy() -> RegionIntersectionPolicy:
    return RegionIntersectionPolicy(triangle3_intersects)


@dataclass(frozen=True)
class RegionQuery(BoundQuery, Generic[HostDataT, DeviceDataT, PointT]):
    mode: str
    def __call__(self, tree, geometry, lower, upper):
        lower, shape = _flatten_vec_queries(lower, self.primitive.dimension)
        upper, other = _flatten_vec_queries(upper, self.primitive.dimension)
        if shape != other:
            raise ValueError("box lower and upper batch shapes must match")
        result, status = self.launch(tree, geometry, shape, lower, upper)
        if self.mode == "any":
            result = result.astype(bool)
        return result, status


@dataclass(frozen=True)
class RegionClassifyQuery(BoundQuery, Generic[HostDataT, DeviceDataT, PointT, HostMetaT, MetaT]):
    def __call__(self, tree, geometry, boundary, lower, upper):
        lower, shape = _flatten_vec_queries(lower, self.primitive.dimension)
        upper, other = _flatten_vec_queries(upper, self.primitive.dimension)
        if shape != other:
            raise ValueError("box lower and upper batch shapes must match")
        return self.launch(tree, geometry, shape, lower, upper,
                           metadata=(getattr(boundary, "core", boundary),))


def make_region(primitive, policy: RegionIntersectionPolicy, *, mode="any", count_limit=None,
                stack_capacity=STACK_CAPACITY, block_dim=256):
    """Count exact hits or test for any exact hit; a limit returns a saturated count."""
    if mode not in ("any", "count"):
        raise ValueError("mode must be 'any' or 'count'")
    if count_limit is not None and (mode != "count" or type(count_limit) is not int or not 1 <= count_limit <= 2147483647):
        raise ValueError("count_limit requires count mode and a positive static int32 integer")
    return _make_region(primitive, policy, mode, count_limit, stack_capacity, block_dim)


@lru_cache(maxsize=None)
def _make_region(primitive, policy, mode, count_limit, stack_capacity, block_dim):
    if not 1 <= stack_capacity <= STACK_CAPACITY or not 1 <= block_dim <= 1024:
        raise ValueError("invalid stack capacity or block dimension")
    Vec, Geometry = primitive.point_type, primitive.data_type
    Context = make_struct("RegionContext", {"lower": wp.vec3, "upper": wp.vec3, "geometry": Geometry})
    State = make_struct("RegionState", {"limit": float, "failed": bool, "stop": bool, "count": int})

    @wp.func
    def bounds(ctx: Context, tree: DeviceTree, node: int, lo: wp.vec3, hi: wp.vec3, state: wp.ref[State]):
        return 0.0, boxes_overlap(ctx.lower, ctx.upper, lo, hi), False

    @wp.func
    def visit(ctx: Context, tree: DeviceTree, node: int, begin: int, end: int, tag: bool, state: wp.ref[State]):
        for slot in range(begin, end):
            if wp.static(policy.intersects)(ctx.geometry, slot, ctx.lower, ctx.upper):
                if state.count == 2147483647:
                    state.failed = True
                    return
                state.count += 1
                if wp.static(mode == "any"):
                    state.stop = True
                    return
                if wp.static(count_limit is not None):
                    if state.count >= wp.static(count_limit if count_limit is not None else 0):
                        state.stop = True
                        return
        return

    walk = make_traversal(Context, State, bounds, visit, stack_capacity=stack_capacity,
                          near_first=False, packed_nodes=True)

    @wp.func
    def region(lower: Vec, upper: Vec, tree: DeviceTree, geometry: Geometry):
        if not wp.isfinite(lower) or not wp.isfinite(upper) or wp.max(lower - upper) > 0.0:
            return 0, -1
        lo = wp.vec3(lower[0], lower[1], 0.0)
        hi = wp.vec3(upper[0], upper[1], 0.0)
        if wp.static(primitive.dimension == 3):
            lo = wp.vec3(lower[0], lower[1], lower[2])
            hi = wp.vec3(upper[0], upper[1], upper[2])
        state = State()
        wp.static(walk.device)(tree, Context(lo, hi, geometry), state)
        if state.failed:
            return 0, -1
        return state.count, 1

    inputs = [("lower", wp.array(dtype=Vec)), ("upper", wp.array(dtype=Vec))]
    outputs = [("out_result", wp.array(dtype=wp.int32)), ("out_status", wp.array(dtype=wp.int32))]
    body = ["result, status = wp.static(region)(load_array(lower, i), load_array(upper, i), tree, geometry)",
            "store_array(out_result, i, result)", "store_array(out_status, i, status)"]
    return bind_query(RegionQuery, "region_kernel", primitive, inputs, outputs, body,
                      {"region": region}, block_dim, device=region, mode=mode,
                      components=(("intersection", "exact_policy"), ("result", mode)))


def make_region_classify(primitive, region_policy: RegionIntersectionPolicy, classify_policy,
                         *, metadata_host_type=None, stack_capacity=STACK_CAPACITY, block_dim=128):
    """Classify a closed box: -1 outside, 0 surface contact, +1 inside."""
    return _make_region_classify(primitive, region_policy, classify_policy,
                                 stack_capacity=stack_capacity, block_dim=block_dim)


@lru_cache(maxsize=None)
def _make_region_classify(primitive, region_policy, classify_policy, *,
                          stack_capacity, block_dim):
    exact = make_region(primitive, region_policy, mode="any", stack_capacity=stack_capacity,
                        block_dim=block_dim)
    point = make_classify(primitive, classify_policy, stack_capacity=stack_capacity,
                          block_dim=block_dim)
    Vec, Geometry, Meta = primitive.point_type, primitive.data_type, classify_policy.metadata_type

    @wp.func
    def classify_region(lower: Vec, upper: Vec, tree: DeviceTree, geometry: Geometry, meta: Meta):
        if not wp.static(classify_policy.valid)(meta):
            return 0, -1
        count, status = wp.static(exact.device)(lower, upper, tree, geometry)
        if status < 0:
            return 0, -1
        if count != 0:
            return 0, 1
        sign, point_status = wp.static(point.device)(lower, tree, geometry, meta)
        if point_status < 0:
            return 0, -1
        if sign == 0.0:
            return 0, 1
        return wp.where(sign < 0.0, 1, -1), 1

    inputs = [("lower", wp.array(dtype=Vec)), ("upper", wp.array(dtype=Vec))]
    outputs = [("out_class", wp.array(dtype=wp.int32)), ("out_status", wp.array(dtype=wp.int32))]
    body = ["result, status = wp.static(classify_region)(load_array(lower, i), load_array(upper, i), tree, geometry, meta)",
            "store_array(out_class, i, result)", "store_array(out_status, i, status)"]
    return bind_query(RegionClassifyQuery, "region_classify_kernel", primitive, inputs,
                      outputs, body, {"classify_region": classify_region}, block_dim,
                      device=classify_region, groups=(("meta", Meta),),
                      components=(("intersection", "exact_policy"), ("classification", "policy")))
