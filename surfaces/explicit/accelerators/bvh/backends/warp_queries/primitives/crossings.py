"""Robust triangle crossing predicate with closed or half-open edge ownership."""
import warp as wp

from ..domain import Ray


@wp.func_native("return bvh::triangle_crossing(ray, min_t, max_t, a, b, c, half_open, store_times);")
def native_triangle_crossing(ray: Ray, min_t: float, max_t: float, a: wp.vec3, b: wp.vec3, c: wp.vec3,
    half_open: bool, store_times: bool) -> wp.vec2: ...
