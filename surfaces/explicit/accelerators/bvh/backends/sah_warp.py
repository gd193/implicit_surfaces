"""Warp SAH backend."""
from __future__ import annotations

from typing import Tuple

import jax.numpy as jnp

from surfaces.explicit.accelerators.protocol import Backend, Config
from surfaces.explicit.accelerators.registry import register_backend
from surfaces.explicit.primitives import Primitives

from ._warp_build import finite_primitive_bounds, validate_warp_mesh_and_config
from ._warp_jax import query as _warp_query
from ._warp_layout import (WarpBVHPayload, empty_warp_payload,
                           make_warp_payload)
from .sah_jax import _build_payload


def _flatten_vec_queries(x):
    x = jnp.asarray(x, dtype=jnp.float32)
    if x.shape[-1:] != (3,):
        raise ValueError(f"expected query shape (..., 3), got {x.shape}")
    batch_shape = x.shape[:-1]
    return x.reshape((-1, 3)), batch_shape


def _restore(value, batch_shape):
    return value.reshape(batch_shape + value.shape[1:])


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

    def build(self, primitives: Primitives, config: Config) -> WarpBVHPayload:
        vertices, faces = validate_warp_mesh_and_config(
            primitives, config, "bvh.sah.warp")
        if faces.shape[0] == 0:
            return empty_warp_payload()
        lo_np, hi_np = finite_primitive_bounds(primitives)
        tree = _build_payload(lo_np, hi_np, int(config.leaf_size_threshold))
        payload, depth = make_warp_payload(tree, vertices, faces)
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
        q, shape = _flatten_vec_queries(x)
        cp, dist, prim, _status = _warp_query(
            "closest_point_kernel", payload, q)
        dist = jnp.where(prim >= 0, dist, jnp.inf)
        return (_restore(cp, shape), _restore(dist, shape),
                _restore(prim, shape), _restore(_status, shape))

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
        origins, shape = _flatten_vec_queries(ray_origin)
        directions, dshape = _flatten_vec_queries(ray_direction)
        if shape != dshape:
            raise ValueError("ray_origin and ray_direction batch shapes must match")
        max_t = jnp.asarray(max_t, dtype=jnp.float32)
        max_t = jnp.broadcast_to(max_t, shape).reshape((-1,))
        min_t = jnp.broadcast_to(jnp.asarray(min_t, dtype=jnp.float32),
                                 shape).reshape((-1,))
        t, prim, hit, _status = _warp_query(
            "first_hit_kernel", payload, origins, directions, min_t, max_t)
        return (_restore(t, shape), _restore(prim, shape),
                _restore(hit.astype(jnp.bool_), shape),
                _restore(_status, shape))

    def any_hit(self, payload: WarpBVHPayload, ray_origin: jnp.ndarray,
                ray_direction: jnp.ndarray, max_t: jnp.ndarray,
                min_t: jnp.ndarray | float = 0.0) -> jnp.ndarray:
        return self.any_hit_with_status(payload, ray_origin, ray_direction,
                                        max_t, min_t)[0]

    def any_hit_with_status(self, payload: WarpBVHPayload,
                            ray_origin: jnp.ndarray,
                            ray_direction: jnp.ndarray, max_t: jnp.ndarray,
                            min_t: jnp.ndarray | float = 0.0):
        origins, shape = _flatten_vec_queries(ray_origin)
        directions, dshape = _flatten_vec_queries(ray_direction)
        if shape != dshape:
            raise ValueError("ray_origin and ray_direction batch shapes must match")
        mt = jnp.broadcast_to(jnp.asarray(max_t, jnp.float32), shape).reshape((-1,))
        nt = jnp.broadcast_to(jnp.asarray(min_t, jnp.float32), shape).reshape((-1,))
        hit, status = _warp_query(
            "any_hit_kernel", payload, origins, directions, nt, mt)
        return (_restore(hit.astype(jnp.bool_), shape),
                _restore(status, shape))

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
