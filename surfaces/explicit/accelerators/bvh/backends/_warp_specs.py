"""Query bindings; safe to import without Warp."""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from importlib import import_module
from typing import Any


STACK_CAPACITY = 64


@dataclass(frozen=True)
class WarpQuerySpec:
    module: str
    num_query_args: int
    num_outputs: int
    block_dim: int = 256
    payload_fields: tuple[str, ...] = ()

    def payload_arguments(self, payload: Any) -> tuple[Any, ...]:
        return tuple(getattr(payload, name) for name in self.payload_fields)


_SPATIAL_FIELDS = (
    "node_lower", "node_upper", "node_left", "node_right",
    "node_leaf_offset", "node_leaf_count", "prim_ids",
    "tri_v0", "tri_e1", "tri_e2",
)
_RAY_FIELDS = (
    "node_ray_lower", "node_ray_upper", "prim_ids", "tri_v0", "tri_e1", "tri_e2",
)

_CROSSING_FIELDS = (
    "node_ray_lower", "node_ray_upper", "prim_ids", "mesh_vertices", "tri_indices",
)

QUERY_SPECS = {
    "count_hits_kernel": WarpQuerySpec("_warp_crossings", 5, 2, 128, _CROSSING_FIELDS),
    "all_hits_kernel": WarpQuerySpec("_warp_crossings", 5, 4, 128, _CROSSING_FIELDS),
    "classify_point_kernel": WarpQuerySpec(
        "_warp_inside", 1, 2, 128, payload_fields=(
            "node_ray_lower", "node_ray_upper", "mesh_vertices",
            "tri_indices", "mesh_closed")),
    "closest_point_kernel": WarpQuerySpec(
        "_warp_closest", 1, 4, 64, _SPATIAL_FIELDS),
    "first_hit_kernel": WarpQuerySpec(
        "_warp_ray", 4, 4, 128, _RAY_FIELDS),
    "any_hit_kernel": WarpQuerySpec(
        "_warp_ray", 4, 2, 128, _RAY_FIELDS),
    "aabb_overlap_kernel": WarpQuerySpec(
        "_warp_overlap", 2, 2, payload_fields=(
            "node_lower", "node_upper", "node_left", "node_right",
            "node_leaf_offset", "node_leaf_count", "tri_v0", "tri_e1", "tri_e2")),
    "closest_silhouette_kernel": WarpQuerySpec(
        "_warp_silhouette", 2, 4, 64,
        payload_fields=(
            "node_lower", "node_upper", "node_left", "node_right",
            "node_silhouette_offset", "node_silhouette_count",
            "silhouette_refs", "silhouette_v0", "silhouette_v1", "silhouette_n0",
            "silhouette_n1", "node_cone_axis", "node_cone_sin_cos", "node_cone_radius")),
    "sample_boundary_kernel": WarpQuerySpec(
        "_warp_sampling", 3, 5,
        payload_fields=(*_SPATIAL_FIELDS, "tri_normal", "tri_area", "node_area")),
    "robin_radius_kernel": WarpQuerySpec(
        "_warp_robin", 2, 2, 64,
        payload_fields=(
            "node_lower", "node_upper", "node_left", "node_right",
            "node_leaf_offset", "node_leaf_count", "node_robin_bounds",
            "node_silhouette_offset", "node_silhouette_count", "silhouette_refs",
            "silhouette_v0", "silhouette_v1", "silhouette_n0", "silhouette_n1",
            "node_cone_axis", "node_cone_sin_cos", "node_cone_radius",
            "tri_v0", "tri_e1", "tri_e2", "tri_normal", "tri_robin_bounds")),
}


def query_spec(kernel_name: str) -> WarpQuerySpec:
    try:
        return QUERY_SPECS[kernel_name]
    except KeyError:
        raise ValueError(f"unknown Warp BVH query kernel {kernel_name!r}") from None


@lru_cache(maxsize=None)
def query_kernel(kernel_name: str):
    """Load lazily and check payload argument order."""
    spec = query_spec(kernel_name)
    module = import_module(f".{spec.module}", __package__)
    kernel = getattr(module, kernel_name, None)
    if kernel is None:
        from ._warp_factory import native_entrypoint

        native = getattr(module, "native_" + kernel_name.removesuffix("_kernel"))
        kernel = native_entrypoint(kernel_name, native, spec)
    args = kernel.adj.args
    payload_end = len(args) - spec.num_outputs
    actual_fields = tuple(arg.label for arg in args[spec.num_query_args:payload_end])
    if actual_fields != spec.payload_fields:
        raise RuntimeError(f"{kernel_name} payload arguments disagree with its query specification")
    return kernel
