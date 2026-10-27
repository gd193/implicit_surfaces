"""Primitive-AABB overlap queries."""
from functools import lru_cache

import warp as wp

from ._warp_factory import BoundQuery, bind_query
from ._warp_geometry import boxes_overlap
from ._warp_jax import _flatten_vec_queries
from ._warp_primitives import _struct
from ._warp_traversal import make_walk
from ._warp_tree import DeviceTree, STACK_CAPACITY


class OverlapQuery(BoundQuery):
    def __call__(self, tree, geometry, lower, upper):
        lower, shape = _flatten_vec_queries(lower, self.primitive.dimension)
        upper, upper_shape = _flatten_vec_queries(upper, self.primitive.dimension)
        if shape != upper_shape:
            raise ValueError("box lower and upper batch shapes must match")
        hit, status = self.launch(tree, geometry, shape, lower, upper)
        return hit.astype(bool), status


@lru_cache(maxsize=None)
def make_overlap(primitive, *, stack_capacity=STACK_CAPACITY, block_dim=256):
    if not 1 <= stack_capacity <= STACK_CAPACITY or not 1 <= block_dim <= 1024:
        raise ValueError("invalid stack capacity or block dimension")
    Vec, Geometry = primitive.point_type, primitive.data_type
    Context = _struct("OverlapContext", {"lower": wp.vec3, "upper": wp.vec3, "geometry": Geometry})
    State = _struct("OverlapState", {"limit": float, "failed": bool, "stop": bool, "hit": int})

    @wp.func
    def bounds(ctx: Context, lo: wp.vec3, hi: wp.vec3, state: wp.ref[State]):
        return 0.0, boxes_overlap(ctx.lower, ctx.upper, lo, hi), False

    @wp.func
    def visit(ctx: Context, tree: DeviceTree, begin: int, end: int, state: wp.ref[State]):
        for slot in range(begin, end):
            item = wp.static(primitive.load)(ctx.geometry, slot)
            lo, hi = wp.static(primitive.bounds_of_item)(item)
            if boxes_overlap(ctx.lower, ctx.upper, lo, hi):
                state.hit = 1
                state.stop = True
                return
        return

    walk = make_walk(Context, State, bounds, visit, stack_capacity=stack_capacity, near_first=False, packed=True)

    @wp.func
    def overlaps(lower: Vec, upper: Vec, tree: DeviceTree, geometry: Geometry):
        if not wp.isfinite(lower) or not wp.isfinite(upper) or wp.max(lower-upper) > 0.0:
            return 0, -1
        lo, hi = wp.vec3(lower[0], lower[1], 0.0), wp.vec3(upper[0], upper[1], 0.0)
        if wp.static(primitive.dimension == 3):
            lo = wp.vec3(lower[0], lower[1], lower[2])
            hi = wp.vec3(upper[0], upper[1], upper[2])
        state = State()
        wp.static(walk)(tree, Context(lo, hi, geometry), state)
        if state.failed:
            return 0, -1
        return state.hit, state.hit

    inputs = [("lower", wp.array(dtype=Vec)), ("upper", wp.array(dtype=Vec))]
    outputs = [("out_overlap", wp.array(dtype=wp.int32)), ("out_status", wp.array(dtype=wp.int32))]
    body = ["hit, status = wp.static(overlaps)(load_array(lower, i), load_array(upper, i), tree, geometry)",
            "store_array(out_overlap, i, hit)", "store_array(out_status, i, status)"]
    return bind_query(OverlapQuery, "overlap_kernel", primitive, inputs, outputs, body,
                      {"overlaps": overlaps}, block_dim)
