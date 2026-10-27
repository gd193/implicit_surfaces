"""Ray queries using the shared BVH walker."""
from dataclasses import dataclass
from functools import lru_cache
from operator import index
from typing import Any

import jax.numpy as jnp
import warp as wp

from ._warp_domain import Ray, RayOps, prepare_ray
from ._warp_factory import BoundQuery, bind_query
from ._warp_geometry import _NATIVE_RAY_BOX_TEST
from ._warp_jax import _crossing_args
from ._warp_primitives import _struct
from ._warp_traversal import load_array, store_array, make_walk
from ._warp_tree import DeviceTree, STACK_CAPACITY


@dataclass(frozen=True)
class RayQuery(BoundQuery):
    mode: str
    exclusion: bool

    def __call__(self, tree, geometry, origin, direction, max_t=jnp.inf, *,
                 min_t=0.0, exclude_face=None, max_hits=None):
        if exclude_face is not None and not self.exclusion:
            raise ValueError("construct this ray query with exclude=True")
        args, shape = _crossing_args(origin, direction, min_t, max_t,
            -1 if exclude_face is None else exclude_face, self.primitive.dimension)
        if not self.exclusion:
            args = args[:4]
        shapes = None
        if self.mode == "all":
            max_hits = index(max_hits)
            if not 1 <= max_hits < 2**31:
                raise ValueError("max_hits must be a positive static int32 capacity")
            shapes = ((max_hits,), (max_hits,), (), ())
        elif max_hits is not None:
            raise ValueError("max_hits requires an all-hit query")
        result = self.launch(tree, geometry, shape, *args, output_shapes=shapes)
        if self.mode == "first":
            return result[0], result[1], result[2].astype(jnp.bool_), result[3]
        if self.mode == "any":
            return result[0].astype(jnp.bool_), result[1]
        return result


@wp.func_native(r"""
    const auto o=ray.origin, d=ray.direction, inv_dir=ray.inverse;
    const bool fast_aabb=ray.fast;
""" + _NATIVE_RAY_BOX_TEST + r"""
    float enter;
    const bool hit=box_interval(lower,upper,minimum,maximum,enter,conservative);
    return wp::vec2(enter,hit ? 1.f : 0.f);
""")
def ray_box(ray: Ray, lower: wp.vec3, upper: wp.vec3,
            minimum: float, maximum: float, conservative: bool) -> wp.vec2: ...


@wp.func_native(r"""
    for (float v : origin.c) if (!(v == v && wp::abs(v) < 3.4028234e38f)) return false;
    bool nonzero = false;
    for (float v : direction.c) {
        if (!(v == v && wp::abs(v) < 3.4028234e38f)) return false;
        nonzero |= v != 0.f;
    }
    if (!(minimum <= maximum) || minimum == INFINITY || maximum == -INFINITY) return false;
    if (exclusion && (excluded < -1 || excluded >= count)) return false;
    return line ? nonzero : minimum >= 0.f && wp::dot(direction, direction) > 1.e-30f;
""")
def valid_ray(origin: Any, direction: Any, minimum: float, maximum: float,
              excluded: int, count: int, line: bool, exclusion: bool):
    return bool(False)  # Generic native return-type inference.


@lru_cache(maxsize=None)
def make_ray_query(primitive, mode="first", *, ops=None, exclude=False,
                   stack_capacity=STACK_CAPACITY, block_dim=128):
    if mode not in ("first", "any", "count", "all"):
        raise ValueError("ray mode must be first, any, count, or all")
    if not 1 <= stack_capacity <= STACK_CAPACITY or not 1 <= block_dim <= 1024:
        raise ValueError("invalid stack capacity or block dimension")
    ops = ops or getattr(primitive, "ray_ops", None) or RayOps(intersect=getattr(primitive, "intersect", None))
    if ops.intersect is None and ops.next_hit is None:
        raise TypeError("primitive has no ray intersection operation")
    multiple = ops.next_hit is not None and (mode in ("count", "all") or ops.intersect is None)
    Iterator = ops.state_type
    exclusion, half_open = exclude or mode in ("count", "all"), mode in ("count", "all")
    store_times = mode in ("first", "all")
    Vec, Geometry = primitive.point_type, primitive.data_type
    fields = {"ray": Ray, "minimum": wp.float32, "excluded": wp.int32, "geometry": Geometry, "qi": wp.int32}
    if mode == "all":
        fields.update(times=wp.array2d(dtype=wp.float32), faces=wp.array2d(dtype=wp.int32))
    Context = _struct("RayContext", fields)
    State = _struct("RayState", {"limit": wp.float32, "failed": wp.bool, "stop": wp.bool,
                                 "count": wp.int32, "index": wp.int32})

    @wp.func
    def bounds(ctx: Context, lower: wp.vec3, upper: wp.vec3, state: wp.ref[State]):
        hit = ray_box(ctx.ray, lower, upper, ctx.minimum, state.limit, wp.static(half_open))
        return hit[0], hit[1] > 0.0, False

    @wp.func
    def visit(ctx: Context, tree: DeviceTree, begin: int, end: int, state: wp.ref[State]):
        for slot in range(begin, end):
            face = load_array(tree.prim_ids, slot)
            if wp.static(exclusion):
                if face == ctx.excluded:
                    continue
            if wp.static(multiple):
                if wp.static(wp.types.type_is_scalar(Iterator)):
                    iterator = Iterator(0)
                else:
                    iterator = Iterator()
            while True:
                hit = wp.vec2(0.0)
                if wp.static(multiple):
                    iterator, t, flag = wp.static(ops.next_hit)(ctx.ray, ctx.minimum, state.limit,
                        ctx.geometry, slot, iterator, wp.static(half_open), wp.static(store_times))
                    if flag == 0:
                        break
                    hit = wp.vec2(t, wp.float32(flag))
                else:
                    hit = wp.static(ops.intersect)(ctx.ray, ctx.minimum, state.limit, ctx.geometry,
                                                   slot, wp.static(half_open), wp.static(store_times))
                if hit[1] < 0.0:
                    state.failed = True
                    return
                if hit[1] > 0.0:
                    if wp.static(mode == "any"):
                        state.count = 1
                        state.stop = True
                        return
                    elif wp.static(mode == "first"):
                        if hit[0] < state.limit or state.index < 0 or (hit[0] == state.limit and face < state.index):
                            state.limit = hit[0]
                            state.index = face
                    else:
                        if wp.static(mode == "all"):
                            if state.count < ctx.times.shape[1]:
                                store_array(ctx.times, wp.int64(ctx.qi)*wp.int64(ctx.times.shape[1])+wp.int64(state.count), hit[0])
                                store_array(ctx.faces, wp.int64(ctx.qi)*wp.int64(ctx.faces.shape[1])+wp.int64(state.count), face)
                        state.count += 1
                if wp.static(not multiple):
                    break
        return

    walk = make_walk(Context, State, bounds, visit, stack_capacity=stack_capacity,
                     near_first=mode in ("first", "any"), packed=True, test_root=half_open)

    @wp.func
    def ray_query(qi: int, origin: Vec, direction: Vec, minimum: float, maximum: float,
                  excluded: int, tree: DeviceTree, geometry: Geometry,
                  times: Any, faces: Any, out_count: Any, out_status: Any):
        store_array(out_count, qi, 0)
        store_array(out_status, qi, 0)
        if wp.static(mode == "first"):
            store_array(times, qi, maximum)
            store_array(faces, qi, -1)
        if wp.static(mode == "all"):
            for j in range(times.shape[1]):
                store_array(times, wp.int64(qi)*wp.int64(times.shape[1])+wp.int64(j), wp.inf)
                store_array(faces, wp.int64(qi)*wp.int64(faces.shape[1])+wp.int64(j), -1)
        valid = valid_ray(origin, direction, minimum, maximum, excluded,
                          tree.prim_ids.shape[0], wp.static(half_open), wp.static(exclusion))
        state = State()
        state.limit = maximum
        state.index = -1
        if valid:
            ctx = Context()
            o, d = wp.vec3(origin[0], origin[1], 0.0), wp.vec3(direction[0], direction[1], 0.0)
            if wp.static(primitive.dimension == 3):
                o[2] = origin[2]
                d[2] = direction[2]
            ctx.ray = prepare_ray(o, d)
            ctx.minimum = minimum
            ctx.excluded = excluded
            ctx.geometry = geometry
            ctx.qi = qi
            if wp.static(mode == "all"):
                ctx.times = times
                ctx.faces = faces
            wp.static(walk)(tree, ctx, state)
        else:
            state.failed = True
        if state.failed:
            store_array(out_status, qi, -1)
            if wp.static(half_open):
                store_array(out_count, qi, -1)
            if wp.static(mode == "all"):
                for j in range(times.shape[1]):
                    store_array(times, wp.int64(qi)*wp.int64(times.shape[1])+wp.int64(j), wp.inf)
                    store_array(faces, wp.int64(qi)*wp.int64(faces.shape[1])+wp.int64(j), -1)
        else:
            if wp.static(mode == "first"):
                if state.index >= 0:
                    store_array(times, qi, state.limit)
                    store_array(faces, qi, state.index)
                    state.count = 1
            store_array(out_count, qi, state.count)
            store_array(out_status, qi, wp.int32(state.count > 0))
            if wp.static(mode == "all"):
                if state.count > times.shape[1]:
                    store_array(out_status, qi, -2)

    outputs = [("out_count", wp.array(dtype=wp.int32)), ("out_status", wp.array(dtype=wp.int32))]
    times, faces = "out_count", "out_status"
    if store_times:
        array = wp.array2d if mode == "all" else wp.array
        outputs = [("out_t", array(dtype=wp.float32)), ("out_face", array(dtype=wp.int32)), *outputs]
        times, faces = "out_t", "out_face"
    inputs = [("origins", wp.array(dtype=Vec)), ("directions", wp.array(dtype=Vec)),
              ("minimum", wp.array(dtype=wp.float32)), ("maximum", wp.array(dtype=wp.float32))]
    if exclusion:
        inputs.append(("excluded", wp.array(dtype=wp.int32)))
    excluded = "load_array(excluded, i)" if exclusion else "-1"
    body = [f"wp.static(ray_query)(i, load_array(origins, i), load_array(directions, i), load_array(minimum, i), load_array(maximum, i), {excluded}, "
            f"tree, geometry, {times}, {faces}, out_count, out_status)"]
    return bind_query(RayQuery, "ray_kernel", primitive, inputs, outputs, body,
                      {"ray_query": ray_query}, block_dim, mode=mode, exclusion=exclusion)
