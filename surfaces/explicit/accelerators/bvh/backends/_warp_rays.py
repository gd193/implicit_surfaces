"""Ray traversal specialized by primitive and result accumulation."""
from dataclasses import dataclass
from functools import lru_cache
from operator import index
from typing import Any

import warp as wp
import jax.numpy as jnp

from ._warp_factory import BoundQuery, bind_query, is_native_triangle
from ._warp_geometry import _NATIVE_RAY_BOX
from ._warp_specs import STACK_CAPACITY
from ._warp_tree import DeviceTree
from ._warp_jax import _crossing_args


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


@lru_cache(maxsize=None)
def make_ray_query(primitive, mode="first", *, exclude=False,
                   stack_capacity=STACK_CAPACITY, block_dim=128):
    if mode not in ("first", "any", "count", "all"):
        raise ValueError("ray mode must be first, any, count, or all")
    if not 1 <= stack_capacity <= STACK_CAPACITY or not 1 <= block_dim <= 1024:
        raise ValueError("invalid stack capacity or block dimension")
    intersect = getattr(primitive, "intersect", None)
    if intersect is None:
        raise TypeError("primitive has no ray intersection operation")
    exclusion = exclude or mode in ("count", "all")
    Vec, Geometry, dimension = primitive.point_type, primitive.data_type, primitive.dimension
    Stack = wp.types.vector(stack_capacity, wp.int32)
    half_open, store_times = mode in ("count", "all"), mode in ("first", "all")

    @wp.func_native(r"""
        wp::vec3 o(origin[0],origin[1],0.f),d(direction[0],direction[1],0.f);
    """ + ("o[2]=origin[2]; d[2]=direction[2];\n" if dimension == 3 else "")
        + _NATIVE_RAY_BOX + r"""
        float enter;
        const bool hit=box_interval(lower,upper,minimum,maximum,enter,true);
        return wp::vec2(enter,hit ? 1.f : 0.f);
    """)
    def ray_box(origin: Vec, direction: Vec, lower: wp.vec3, upper: wp.vec3,
                minimum: float, maximum: float) -> wp.vec2: ...

    @wp.func
    def ray(qi: int, origin: Vec, direction: Vec, minimum: float, maximum: float,
            excluded: int, tree: DeviceTree, geometry: Geometry,
            times: Any, faces: Any, out_count: Any, out_status: Any):
        out_count[qi] = 0
        out_status[qi] = 0
        if wp.static(mode == "first"):
            times[qi] = maximum
            faces[qi] = -1
        if wp.static(mode == "all"):
            for j in range(times.shape[1]):
                times[qi, j] = wp.inf
                faces[qi, j] = -1
        valid = (wp.isfinite(origin) and wp.isfinite(direction) and wp.max(wp.abs(direction)) > 0.0
                 and not wp.isnan(minimum) and not wp.isnan(maximum)
                 and minimum <= maximum and minimum != wp.inf and maximum != -wp.inf
                 and excluded >= -1 and excluded < tree.prim_ids.shape[0])
        if wp.static(mode in ("first", "any")):
            valid = valid and minimum >= 0.0 and wp.dot(direction, direction) > 1.e-30
        if not valid:
            out_status[qi] = -1
            if wp.static(half_open):
                out_count[qi] = -1
            return
        if tree.node_left.shape[0] == 0:
            return
        root = ray_box(origin, direction, tree.node_lower[0], tree.node_upper[0], minimum, maximum)
        if root[1] == 0.0:
            return
        node, size, count, best_id = int(0), int(0), int(0), int(-1)
        best = float(maximum)
        failed = bool(False)
        stack = Stack()
        while True:
            if tree.node_leaf_count[node] > 0:
                offset = tree.node_leaf_offset[node]
                for j in range(tree.node_leaf_count[node]):
                    slot = offset+j
                    face = tree.prim_ids[slot]+0
                    if wp.static(exclusion):
                        if face == excluded:
                            continue
                    hit = wp.static(intersect)(origin, direction, minimum, best, geometry,
                                                slot, wp.static(half_open), wp.static(store_times))
                    if hit[1] < 0.0:
                        failed = True
                        break
                    if hit[1] > 0.0:
                        if wp.static(mode == "any"):
                            out_count[qi] = 1
                            out_status[qi] = 1
                            return
                        elif wp.static(mode == "first"):
                            if hit[0] < best or best_id < 0 or (hit[0] == best and face < best_id):
                                best, best_id = hit[0]+0.0, face
                        else:
                            if wp.static(mode == "all"):
                                if count < times.shape[1]:
                                    times[qi, count] = hit[0]
                                    faces[qi, count] = face
                            count += 1
                if failed:
                    break
            else:
                left, right = tree.node_left[node]+0, tree.node_right[node]+0
                a = ray_box(origin, direction, tree.node_lower[left], tree.node_upper[left], minimum, best)
                b = ray_box(origin, direction, tree.node_lower[right], tree.node_upper[right], minimum, best)
                if a[1] > 0.0 and b[1] > 0.0:
                    if size == wp.static(stack_capacity):
                        failed = True
                        break
                    near, far = left, right
                    if wp.static(mode in ("first", "any")):
                        if b[0] < a[0]:
                            near, far = right, left
                    stack[size] = far
                    size += 1
                    node = near
                    continue
                if a[1] > 0.0 or b[1] > 0.0:
                    node = wp.where(a[1] > 0.0, left, right)
                    continue
            if size == 0:
                break
            size -= 1
            node = stack[size]+0
        if failed:
            out_status[qi] = -1
            if wp.static(half_open):
                out_count[qi] = -1
            if wp.static(mode == "all"):
                for j in range(times.shape[1]):
                    times[qi, j] = wp.inf
                    faces[qi, j] = -1
        elif wp.static(mode == "first"):
            if best_id >= 0:
                times[qi] = best
                faces[qi] = best_id
                out_count[qi] = 1
                out_status[qi] = 1
        else:
            out_count[qi] = count
            out_status[qi] = wp.int32(count > 0)
            if wp.static(mode == "all"):
                if count > times.shape[1]:
                    out_status[qi] = -2

    outputs = [("out_count", wp.array(dtype=wp.int32)), ("out_status", wp.array(dtype=wp.int32))]
    times, faces = "out_count", "out_status"
    if mode in ("first", "all"):
        array = wp.array2d if mode == "all" else wp.array
        outputs = [("out_t", array(dtype=wp.float32)), ("out_face", array(dtype=wp.int32)), *outputs]
        times, faces = "out_t", "out_face"
    inputs = [("origins", wp.array(dtype=Vec)), ("directions", wp.array(dtype=Vec)),
              ("minimum", wp.array(dtype=wp.float32)), ("maximum", wp.array(dtype=wp.float32))]
    if exclusion:
        inputs.append(("excluded", wp.array(dtype=wp.int32)))
    excluded = "excluded[i]" if exclusion else "-1"
    body = [f"wp.static(ray)(i, origins[i], directions[i], minimum[i], maximum[i], {excluded}, "
            f"tree, geometry, {times}, {faces}, out_count, out_status)"]
    native = None
    if is_native_triangle(primitive, stack_capacity) and (half_open or not exclusion):
        if half_open:
            from ._warp_crossings import native_all_hits, native_count_hits
            native = native_all_hits if mode == "all" else native_count_hits
        else:
            from ._warp_ray import native_any_hit, native_first_hit
            native = native_first_hit if mode == "first" else native_any_hit
    return bind_query(RayQuery, "ray_kernel", primitive, inputs, outputs, body, {"ray": ray}, block_dim,
                       native=native, id_field="out_face", excluded=excluded, mode=mode, exclusion=exclusion)
