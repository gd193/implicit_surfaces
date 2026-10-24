"""Mesh queries through a selected BVH backend."""
from __future__ import annotations

from typing import Tuple, cast

import jax.numpy as jnp

from surfaces.explicit.accelerators.base import MeshAccelerator
from surfaces.explicit.accelerators.protocol import Backend, Config
from surfaces.explicit.accelerators.registry import get_backend
from surfaces.explicit.primitives import Primitives

from .payload import BVHPayload


class BVHAccelerator(MeshAccelerator):
    def __init__(self, primitives: Primitives, payload: BVHPayload,
                 backend: Backend):
        if backend.flavor != "bvh":
            raise ValueError(
                f"BVHAccelerator requires a bvh-flavor backend; "
                f"got flavor={backend.flavor!r} from {backend.name!r}"
            )
        self.primitives = primitives
        self.payload = payload
        self.backend = backend
        self.has_distance_bounds = bool(backend.has_distance_bounds)
        self.has_silhouette_query = bool(
            getattr(backend, "has_silhouette_query", False))
        self.has_boundary_sampling = bool(backend.has_boundary_sampling)
        self.has_ray_interval = bool(backend.has_ray_interval)
        self.has_robin_query = bool(backend.has_robin_query)
        self.has_point_classification = bool(backend.has_point_classification)

    @classmethod
    def build(cls, primitives: Primitives, backend_name: str,
              config: Config | None = None) -> "BVHAccelerator":
        backend_cls = get_backend(backend_name)
        backend = backend_cls()
        if backend.flavor != "bvh":
            raise ValueError(
                f"backend {backend_name!r} has flavor {backend.flavor!r}; "
                f"BVHAccelerator requires 'bvh'"
            )
        cfg = config if config is not None else Config()
        payload = cast(BVHPayload, backend.build(primitives, cfg))
        return cls(primitives, payload, backend)

    def closest_point(self, x: jnp.ndarray
                      ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        return self.backend.closest_point(self.payload, x)

    def classify_point(self, x: jnp.ndarray) -> jnp.ndarray:
        return self.backend.classify_point(self.payload, x)

    def classify_point_with_status(self, x: jnp.ndarray):
        return self._query_with_status("classify_point", x)

    def first_hit(self, ray_origin: jnp.ndarray, ray_direction: jnp.ndarray,
                  max_t: jnp.ndarray, *, min_t: jnp.ndarray | float = 0.0,
                  ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        if self.backend.has_ray_interval:
            return self.backend.first_hit(self.payload, ray_origin,
                                          ray_direction, max_t, min_t=min_t)
        if not isinstance(min_t, (int, float)) or min_t != 0.0:
            raise NotImplementedError("this backend does not support a ray lower bound")
        return self.backend.first_hit(self.payload, ray_origin,
                                      ray_direction, max_t)

    def any_hit(self, ray_origin: jnp.ndarray, ray_direction: jnp.ndarray,
                max_t: jnp.ndarray, *, min_t: jnp.ndarray | float = 0.0) -> jnp.ndarray:
        if hasattr(self.backend, "any_hit"):
            return self.backend.any_hit(self.payload, ray_origin, ray_direction,
                                        max_t, min_t=min_t)
        return self.first_hit(ray_origin, ray_direction, max_t, min_t=min_t)[2]

    def star_radius(self, x: jnp.ndarray, radius_cap: jnp.ndarray) -> jnp.ndarray:
        """min(cap, silhouette distance)."""
        if not self.has_silhouette_query:
            raise NotImplementedError("star radii require silhouette queries")
        if hasattr(self.backend, "star_radius"):
            return self.backend.star_radius(self.payload, x, radius_cap)
        return jnp.minimum(self.closest_silhouette(x)[1], radius_cap)

    def _query_with_status(self, name, *args, **kwargs):
        method = getattr(self.backend, name + "_with_status", None)
        if method is None:
            raise NotImplementedError(f"{self.backend.name} has no checked {name} query")
        return method(self.payload, *args, **kwargs)

    def closest_point_with_status(self, x: jnp.ndarray):
        """(point, distance, face, status); status: 1 hit, 0 miss, -1 invalid."""
        return self._query_with_status("closest_point", x)

    def first_hit_with_status(self, ray_origin: jnp.ndarray,
                              ray_direction: jnp.ndarray, max_t: jnp.ndarray,
                              *, min_t: jnp.ndarray | float = 0.0):
        """(t, face, hit, status); closed [min_t, max_t]."""
        return self._query_with_status("first_hit", ray_origin, ray_direction,
                                        max_t, min_t=min_t)

    def any_hit_with_status(self, ray_origin: jnp.ndarray,
                            ray_direction: jnp.ndarray, max_t: jnp.ndarray,
                            *, min_t: jnp.ndarray | float = 0.0):
        return self._query_with_status("any_hit", ray_origin, ray_direction,
                                        max_t, min_t=min_t)

    def aabb_overlap_with_status(self, lower: jnp.ndarray, upper: jnp.ndarray):
        return self._query_with_status("aabb_overlap", lower, upper)

    def closest_silhouette_with_status(self, x: jnp.ndarray,
                                        *, max_distance: jnp.ndarray | float = jnp.inf):
        """(point, distance, edge, status)."""
        return self._query_with_status("closest_silhouette", x,
                                        max_distance=max_distance)

    def star_radius_with_status(self, x: jnp.ndarray, radius_cap: jnp.ndarray):
        """(capped silhouette distance, validity status)."""
        if hasattr(self.backend, "star_radius_with_status"):
            return self.backend.star_radius_with_status(self.payload, x, radius_cap)
        _, distance, _, status = self.closest_silhouette_with_status(x)
        cap = jnp.asarray(radius_cap, dtype=distance.dtype)
        valid = (status >= 0) & (cap >= 0) & ~jnp.isnan(cap)
        radius = jnp.minimum(distance, cap)
        return jnp.where(valid, radius, jnp.nan), jnp.where(valid, 1, -1).astype(jnp.int32)

    def sample_boundary(self, x: jnp.ndarray, radius: jnp.ndarray,
                        randoms: jnp.ndarray):
        """(point, normal, area_pdf, face, status); retain null trials (status 0)."""
        if not hasattr(self.backend, "sample_boundary"):
            raise NotImplementedError("this backend has no boundary sampler")
        return self.backend.sample_boundary(self.payload, x, radius, randoms)

    def with_robin_coefficients(self, minimum, maximum=None):
        """Return a tree with coefficient bounds in original face order."""
        if not self.has_robin_query:
            raise NotImplementedError(f"{self.backend.name} has no Robin radius query")
        payload = self.backend.with_robin_coefficients(self.payload, minimum, maximum)
        return type(self)(self.primitives, payload, self.backend)

    def robin_radius(self, x: jnp.ndarray, radius_cap: jnp.ndarray):
        return self.robin_radius_with_status(x, radius_cap)[0]

    def robin_radius_with_status(self, x: jnp.ndarray, radius_cap: jnp.ndarray):
        """(Robin star radius, validity status)."""
        return self._query_with_status("robin_radius", x, radius_cap)

    def aabb_overlap(self, lower: jnp.ndarray, upper: jnp.ndarray
                     ) -> jnp.ndarray:
        return self.backend.aabb_overlap(self.payload, lower, upper)

    def distance_bounds(self, lower: jnp.ndarray, upper: jnp.ndarray
                        ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        return self.backend.distance_bounds(self.payload, lower, upper)

    def closest_silhouette(self, x: jnp.ndarray,
                           *, max_distance: jnp.ndarray | float = jnp.inf,
                           ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        if hasattr(self.backend, "star_radius"):
            return self.backend.closest_silhouette(self.payload, x,
                                                     max_distance=max_distance)
        if not isinstance(max_distance, (int, float)) or max_distance != float("inf"):
            raise NotImplementedError("this backend has no bounded silhouette query")
        return self.backend.closest_silhouette(self.payload, x)
