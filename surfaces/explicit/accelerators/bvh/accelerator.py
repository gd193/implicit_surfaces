"""Mesh queries through the optimized Warp BVH backends."""
from __future__ import annotations

from typing import TYPE_CHECKING, Tuple, cast

import jax.numpy as jnp

from surfaces.explicit.accelerators.base import MeshAccelerator
from surfaces.explicit.accelerators.protocol import Config
from surfaces.explicit.accelerators.registry import get_backend
from surfaces.explicit.primitives import Primitives

from .backends._warp_layout import WarpBVHPayload

if TYPE_CHECKING:
    from .backends.sah_warp import BVHSahWarpBackend
    from .backends.snch_warp import BVHSnchWarpBackend


class BVHAccelerator(MeshAccelerator):
    def __init__(self, primitives: Primitives, payload: WarpBVHPayload,
                 backend: BVHSahWarpBackend | BVHSnchWarpBackend):
        if backend.flavor != "bvh" or backend.runtime != "warp":
            raise ValueError(
                f"BVHAccelerator requires a Warp BVH backend; "
                f"got {backend.name!r}"
            )
        self.primitives = primitives
        self.payload = payload
        self.backend = backend
        self.has_distance_bounds = bool(backend.has_distance_bounds)
        self.has_silhouette_query = bool(backend.has_silhouette_query)
        self.has_boundary_sampling = bool(backend.has_boundary_sampling)
        self.has_ray_interval = bool(backend.has_ray_interval)
        self.has_robin_query = bool(backend.has_robin_query)
        self.has_point_classification = bool(backend.has_point_classification)
        self.has_all_hits = bool(backend.has_all_hits)

    @classmethod
    def build(cls, primitives: Primitives, backend_name: str,
              config: Config | None = None) -> "BVHAccelerator":
        backend_cls = get_backend(backend_name)
        if backend_cls.flavor != "bvh" or backend_cls.runtime != "warp":
            raise ValueError(
                f"backend {backend_name!r} is not a Warp BVH backend"
            )
        backend = cast("BVHSahWarpBackend | BVHSnchWarpBackend", backend_cls())
        cfg = config if config is not None else Config()
        payload = backend.build(primitives, cfg)
        return cls(primitives, payload, backend)

    def closest_point(self, x: jnp.ndarray
                      ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        return self.backend.closest_point(self.payload, x)

    def classify_point(self, x: jnp.ndarray) -> jnp.ndarray:
        return self.backend.classify_point(self.payload, x)

    def classify_point_with_status(self, x: jnp.ndarray):
        return self.backend.classify_point_with_status(self.payload, x)

    def first_hit(self, ray_origin: jnp.ndarray, ray_direction: jnp.ndarray,
                  max_t: jnp.ndarray, *, min_t: jnp.ndarray | float = 0.0,
                  ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        return self.backend.first_hit(self.payload, ray_origin,
                                      ray_direction, max_t, min_t=min_t)

    def any_hit(self, ray_origin: jnp.ndarray, ray_direction: jnp.ndarray,
                max_t: jnp.ndarray, *, min_t: jnp.ndarray | float = 0.0) -> jnp.ndarray:
        return self.backend.any_hit(self.payload, ray_origin, ray_direction,
                                    max_t, min_t=min_t)

    def count_hits(self, ray_origin, ray_direction, max_t=jnp.inf,
                   *, min_t=0.0, exclude_face=-1):
        return self.backend.count_hits(self.payload, ray_origin, ray_direction,
                                       max_t, min_t=min_t, exclude_face=exclude_face)

    def count_hits_with_status(self, ray_origin, ray_direction, max_t=jnp.inf,
                               *, min_t=0.0, exclude_face=-1):
        return self.backend.count_hits_with_status(
            self.payload, ray_origin, ray_direction, max_t,
            min_t=min_t, exclude_face=exclude_face)

    def all_hits(self, ray_origin, ray_direction, max_t=jnp.inf,
                 *, max_hits, min_t=0.0, exclude_face=-1):
        return self.backend.all_hits(self.payload, ray_origin, ray_direction, max_t,
                                     max_hits=max_hits, min_t=min_t, exclude_face=exclude_face)

    def all_hits_with_status(self, ray_origin, ray_direction, max_t=jnp.inf,
                             *, max_hits, min_t=0.0, exclude_face=-1):
        """(unsorted t, face, total count, status); -2 means output overflow."""
        return self.backend.all_hits_with_status(
            self.payload, ray_origin, ray_direction, max_t,
            max_hits=max_hits, min_t=min_t, exclude_face=exclude_face)

    def star_radius(self, x: jnp.ndarray, radius_cap: jnp.ndarray) -> jnp.ndarray:
        """min(cap, silhouette distance)."""
        if not self.has_silhouette_query:
            raise NotImplementedError("star radii require silhouette queries")
        return self.backend.star_radius(self.payload, x, radius_cap)

    def closest_point_with_status(self, x: jnp.ndarray):
        """(point, distance, face, status); status: 1 hit, 0 miss, -1 invalid."""
        return self.backend.closest_point_with_status(self.payload, x)

    def first_hit_with_status(self, ray_origin: jnp.ndarray,
                              ray_direction: jnp.ndarray, max_t: jnp.ndarray,
                              *, min_t: jnp.ndarray | float = 0.0):
        """(t, face, hit, status); closed [min_t, max_t]."""
        return self.backend.first_hit_with_status(self.payload, ray_origin,
                                                 ray_direction, max_t, min_t=min_t)

    def any_hit_with_status(self, ray_origin: jnp.ndarray,
                            ray_direction: jnp.ndarray, max_t: jnp.ndarray,
                            *, min_t: jnp.ndarray | float = 0.0):
        return self.backend.any_hit_with_status(self.payload, ray_origin,
                                               ray_direction, max_t, min_t=min_t)

    def aabb_overlap_with_status(self, lower: jnp.ndarray, upper: jnp.ndarray):
        return self.backend.aabb_overlap_with_status(self.payload, lower, upper)

    def closest_silhouette_with_status(self, x: jnp.ndarray,
                                        *, max_distance: jnp.ndarray | float = jnp.inf):
        """(point, distance, edge, status)."""
        if not self.has_silhouette_query:
            raise NotImplementedError("silhouette queries require an SNCH backend")
        return self.backend.closest_silhouette_with_status(
            self.payload, x, max_distance=max_distance)

    def star_radius_with_status(self, x: jnp.ndarray, radius_cap: jnp.ndarray):
        """(capped silhouette distance, validity status)."""
        if not self.has_silhouette_query:
            raise NotImplementedError("star radii require silhouette queries")
        return self.backend.star_radius_with_status(self.payload, x, radius_cap)

    def sample_boundary(self, x: jnp.ndarray, radius: jnp.ndarray,
                        randoms: jnp.ndarray):
        """(point, normal, area_pdf, face, status); retain null trials (status 0)."""
        if not self.has_boundary_sampling:
            raise NotImplementedError("this backend has no boundary sampler")
        return self.backend.sample_boundary(self.payload, x, radius, randoms)

    def with_robin_coefficients(self, minimum, maximum=None):
        """Return a tree with coefficient bounds in original face order."""
        if not self.has_robin_query:
            raise NotImplementedError(f"{self.backend.name} has no Robin radius query")
        payload = self.backend.with_robin_coefficients(self.payload, minimum, maximum)
        return type(self)(self.primitives, payload, self.backend)

    def robin_radius(self, x: jnp.ndarray, radius_cap: jnp.ndarray):
        if not self.has_robin_query:
            raise NotImplementedError(f"{self.backend.name} has no Robin radius query")
        return self.backend.robin_radius(self.payload, x, radius_cap)

    def robin_radius_with_status(self, x: jnp.ndarray, radius_cap: jnp.ndarray):
        """(Robin star radius, validity status)."""
        if not self.has_robin_query:
            raise NotImplementedError(f"{self.backend.name} has no Robin radius query")
        return self.backend.robin_radius_with_status(self.payload, x, radius_cap)

    def aabb_overlap(self, lower: jnp.ndarray, upper: jnp.ndarray
                     ) -> jnp.ndarray:
        return self.backend.aabb_overlap(self.payload, lower, upper)

    def distance_bounds(self, lower: jnp.ndarray, upper: jnp.ndarray
                        ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        return self.backend.distance_bounds(self.payload, lower, upper)

    def closest_silhouette(self, x: jnp.ndarray,
                           *, max_distance: jnp.ndarray | float = jnp.inf,
                           ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        if not self.has_silhouette_query:
            raise NotImplementedError("silhouette queries require an SNCH backend")
        return self.backend.closest_silhouette(self.payload, x,
                                               max_distance=max_distance)
