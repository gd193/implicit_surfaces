"""Warp SAH backend."""
from __future__ import annotations

from operator import index
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


def _ray_args(origin, direction, minimum, maximum):
    origins, shape = _flatten_vec_queries(origin)
    directions, dshape = _flatten_vec_queries(direction)
    if shape != dshape:
        raise ValueError("ray_origin and ray_direction batch shapes must match")
    bounds = tuple(jnp.broadcast_to(jnp.asarray(v, jnp.float32), shape).reshape((-1,))
                   for v in (minimum, maximum))
    return (origins, directions, *bounds), shape


def _crossing_args(origin, direction, minimum, maximum, exclude_face):
    args, shape = _ray_args(origin, direction, minimum, maximum)
    face = jnp.asarray(exclude_face)
    if not jnp.issubdtype(face.dtype, jnp.integer):
        raise TypeError("exclude_face must contain integer face IDs")
    face32 = face.astype(jnp.int32)
    min_face = 0 if jnp.issubdtype(face.dtype, jnp.unsignedinteger) else -1
    face = jnp.where((face32 >= min_face) & (face32.astype(face.dtype) == face), face32, -2)
    return (*args, jnp.broadcast_to(face, shape).reshape((-1,))), shape


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
        q, shape = _flatten_vec_queries(x)
        cp, dist, prim, _status = _warp_query(
            "closest_point_kernel", payload, q)
        dist = jnp.where(prim >= 0, dist, jnp.inf)
        return (_restore(cp, shape), _restore(dist, shape),
                _restore(prim, shape), _restore(_status, shape))

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
        args, shape = _ray_args(ray_origin, ray_direction, min_t, max_t)
        t, prim, hit, _status = _warp_query("first_hit_kernel", payload, *args)
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
        args, shape = _ray_args(ray_origin, ray_direction, min_t, max_t)
        hit, status = _warp_query("any_hit_kernel", payload, *args)
        return (_restore(hit.astype(jnp.bool_), shape),
                _restore(status, shape))

    def count_hits(self, payload, ray_origin, ray_direction, max_t=jnp.inf,
                   *, min_t=0.0, exclude_face=-1):
        return self.count_hits_with_status(payload, ray_origin, ray_direction,
                                           max_t, min_t=min_t, exclude_face=exclude_face)[0]

    def count_hits_with_status(self, payload, ray_origin, ray_direction, max_t=jnp.inf,
                               *, min_t=0.0, exclude_face=-1):
        args, shape = _crossing_args(ray_origin, ray_direction, min_t, max_t, exclude_face)
        return tuple(_restore(v, shape) for v in _warp_query("count_hits_kernel", payload, *args))

    def all_hits(self, payload, ray_origin, ray_direction, max_t=jnp.inf,
                 *, max_hits, min_t=0.0, exclude_face=-1):
        return self.all_hits_with_status(payload, ray_origin, ray_direction, max_t,
            max_hits=max_hits, min_t=min_t, exclude_face=exclude_face)[:3]

    def all_hits_with_status(self, payload, ray_origin, ray_direction, max_t=jnp.inf,
                             *, max_hits, min_t=0.0, exclude_face=-1):
        max_hits = index(max_hits)
        if max_hits < 1 or max_hits > 2**31-1:
            raise ValueError("max_hits must be a positive static int32 capacity")
        args, shape = _crossing_args(ray_origin, ray_direction, min_t, max_t, exclude_face)
        result = _warp_query("all_hits_kernel", payload, *args,
                             output_shapes=((max_hits,), (max_hits,), (), ()))
        return tuple(_restore(v, shape) for v in result)

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
