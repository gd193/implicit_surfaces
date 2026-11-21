"""Warp SAH backend."""
from __future__ import annotations

from typing import Tuple

import jax.numpy as jnp

from surfaces.explicit.accelerators.protocol import Backend, Config
from surfaces.explicit.accelerators.registry import register_backend
from surfaces.explicit.primitives import Primitives

from ._warp_build import build_warp_mesh
from ._warp_layout import WarpBVHPayload
from .warp_queries.primitives.builtin import Triangles


def _ray_query(payload, mode, *args, **kwargs):
    from .warp_queries.queries.rays import make_ray_query

    return make_ray_query(Triangles(), mode)(payload.tree, payload.geometry, *args, **kwargs)


@register_backend
class BVHSahWarpBackend(Backend[WarpBVHPayload]):

    name = "bvh.sah.warp"
    flavor = "bvh"
    algorithm = "sah"
    runtime = "warp"
    payload_type = WarpBVHPayload
    is_differentiable = False
    has_silhouette_query = False
    has_ray_interval = True
    has_boundary_sampling = False
    has_point_classification = True
    has_all_hits = True

    def build(self, primitives: Primitives, config: Config) -> WarpBVHPayload:
        return build_warp_mesh(primitives, config)

    def closest_point(self, payload: WarpBVHPayload, x: jnp.ndarray
                      ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        return self.closest_point_with_status(payload, x)[:3]

    def closest_point_with_status(self, payload: WarpBVHPayload, x: jnp.ndarray):
        from .warp_queries.metrics import Euclidean
        from .warp_queries.queries.nearest import make_nearest
        return make_nearest(Triangles(), Euclidean())(payload.tree, payload.geometry, x)

    def classify_point(self, payload: WarpBVHPayload, x: jnp.ndarray):
        return self.classify_point_with_status(payload, x)[0]

    def classify_point_with_status(self, payload: WarpBVHPayload, x: jnp.ndarray):
        from .warp_queries.queries.classification import make_classify

        return make_classify(Triangles())(payload.tree, payload.geometry, payload.boundary, x)

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
        from .warp_queries.queries.overlap import make_overlap

        return make_overlap(Triangles())(payload.tree, payload.geometry, lower, upper)
