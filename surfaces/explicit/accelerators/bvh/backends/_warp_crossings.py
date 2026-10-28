"""Robust half-open triangle crossing predicate."""
import warp as wp

from ._warp_domain import Ray


@wp.func_native("return bvh::triangle_crossing(ray, min_t, max_t, slot, mesh_vertices, tri_indices, half_open, store_times);")
def native_triangle_crossing(ray: Ray, min_t: float, max_t: float, slot: int,
    mesh_vertices: wp.array(dtype=wp.vec3), tri_indices: wp.array(dtype=wp.vec3i),
    half_open: bool, store_times: bool) -> wp.vec2: ...
