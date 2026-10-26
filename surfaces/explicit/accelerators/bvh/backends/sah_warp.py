"""Warp SAH backend."""
from __future__ import annotations

from typing import Tuple

import jax.numpy as jnp

from surfaces.explicit.accelerators.protocol import Backend, Config
from surfaces.explicit.accelerators.registry import register_backend
from surfaces.explicit.primitives import Primitives

from ._warp_build import finite_primitive_bounds, validate_warp_mesh_and_config
from ._warp_jax import query as _warp_query, _flatten_vec_queries, _restore
from ._warp_layout import (WarpBVHPayload, empty_warp_payload,
                           make_warp_payload)
from .sah_jax import _build_payload


def _factory_data(payload):
    from ._warp_primitives import Triangles
    from ._warp_tree import TreeData

    primitive = Triangles()
    geometry = primitive.payload_type(payload.tri_v0, payload.tri_e1, payload.tri_e2,
                                       payload.mesh_vertices, payload.tri_indices)
    return primitive, TreeData.from_payload(payload), geometry


def _ray_query(payload, mode, *args, **kwargs):
    from ._warp_rays import make_ray_query

    primitive, tree, geometry = _factory_data(payload)
    return make_ray_query(primitive, mode)(tree, geometry, *args, **kwargs)


@register_backend
class BVHSahWarpBackend(Backend[WarpBVHPayload]):

    name = "bvh.sah.warp"
    flavor = "bvh"
    algorithm = "sah"
    runtime = "warp"
    payload_type = WarpBVHPayload
    is_differentiable = False
    has_distance_bounds = False
    has_silhouette_query = False
    has_ray_interval = True
    has_boundary_sampling = False
    has_point_classification = True
    has_all_hits = True

    def build(self, primitives: Primitives, config: Config) -> WarpBVHPayload:
        vertices, faces = validate_warp_mesh_and_config(
            primitives, config, "bvh.sah.warp")
        if faces.shape[0] == 0:
            return empty_warp_payload()
        lo_np, hi_np = finite_primitive_bounds(primitives)
        tree = _build_payload(lo_np, hi_np, int(config.leaf_size_threshold))
        payload, depth = make_warp_payload(
            tree, vertices, faces, mesh_vertices=primitives.mesh.vertices)
        if depth > config.max_stack_depth:
            raise ValueError(
                f"built SAH tree depth {depth} exceeds configured "
                f"max_stack_depth={config.max_stack_depth}"
            )
        return payload

    def closest_point(self, payload: WarpBVHPayload, x: jnp.ndarray
                      ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        return self.closest_point_with_status(payload, x)[:3]

    def closest_point_with_status(self, payload: WarpBVHPayload, x: jnp.ndarray):
        from ._warp_metrics import Euclidean
        from ._warp_nearest import make_nearest
        primitive, tree, geometry = _factory_data(payload)
        return make_nearest(primitive, Euclidean())(tree, geometry, x)

    def classify_point(self, payload: WarpBVHPayload, x: jnp.ndarray):
        return self.classify_point_with_status(payload, x)[0]

    def classify_point_with_status(self, payload: WarpBVHPayload, x: jnp.ndarray):
        q, shape = _flatten_vec_queries(x)
        sign, status = _warp_query("classify_point_kernel", payload, q)
        return _restore(sign, shape), _restore(status, shape)

    def first_hit(self, payload: WarpBVHPayload, ray_origin: jnp.ndarray,
                  ray_direction: jnp.ndarray, max_t: jnp.ndarray,
                  min_t: jnp.ndarray | float = 0.0
                  ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        return self.first_hit_with_status(payload, ray_origin, ray_direction,
                                          max_t, min_t)[:3]

    def first_hit_with_status(self, payload: WarpBVHPayload,
                              ray_origin: jnp.ndarray,
                              ray_direction: jnp.ndarray, max_t: jnp.ndarray,
                              min_t: jnp.ndarray | float = 0.0):
        return _ray_query(payload, "first", ray_origin, ray_direction, max_t, min_t=min_t)

    def any_hit(self, payload: WarpBVHPayload, ray_origin: jnp.ndarray,
                ray_direction: jnp.ndarray, max_t: jnp.ndarray,
                min_t: jnp.ndarray | float = 0.0) -> jnp.ndarray:
        return self.any_hit_with_status(payload, ray_origin, ray_direction,
                                        max_t, min_t)[0]

    def any_hit_with_status(self, payload: WarpBVHPayload,
                            ray_origin: jnp.ndarray,
                            ray_direction: jnp.ndarray, max_t: jnp.ndarray,
                            min_t: jnp.ndarray | float = 0.0):
        return _ray_query(payload, "any", ray_origin, ray_direction, max_t, min_t=min_t)

    def count_hits(self, payload, ray_origin, ray_direction, max_t=jnp.inf,
                   *, min_t=0.0, exclude_face=-1):
        return self.count_hits_with_status(payload, ray_origin, ray_direction,
                                           max_t, min_t=min_t, exclude_face=exclude_face)[0]

    def count_hits_with_status(self, payload, ray_origin, ray_direction, max_t=jnp.inf,
                               *, min_t=0.0, exclude_face=-1):
        return _ray_query(payload, "count", ray_origin, ray_direction, max_t,
                           min_t=min_t, exclude_face=exclude_face)

    def all_hits(self, payload, ray_origin, ray_direction, max_t=jnp.inf,
                 *, max_hits, min_t=0.0, exclude_face=-1):
        return self.all_hits_with_status(payload, ray_origin, ray_direction, max_t,
            max_hits=max_hits, min_t=min_t, exclude_face=exclude_face)[:3]

    def all_hits_with_status(self, payload, ray_origin, ray_direction, max_t=jnp.inf,
                             *, max_hits, min_t=0.0, exclude_face=-1):
        return _ray_query(payload, "all", ray_origin, ray_direction, max_t,
                           max_hits=max_hits, min_t=min_t, exclude_face=exclude_face)

    def aabb_overlap(self, payload: WarpBVHPayload,
                     lower: jnp.ndarray, upper: jnp.ndarray) -> jnp.ndarray:
        return self.aabb_overlap_with_status(payload, lower, upper)[0]

    def aabb_overlap_with_status(self, payload: WarpBVHPayload,
                                 lower: jnp.ndarray, upper: jnp.ndarray):
        lo, shape = _flatten_vec_queries(lower)
        hi, hshape = _flatten_vec_queries(upper)
        if shape != hshape:
            raise ValueError("lower and upper batch shapes must match")
        result, status = _warp_query(
            "aabb_overlap_kernel", payload, lo, hi)
        return (_restore(result.astype(jnp.bool_), shape),
                _restore(status, shape))
