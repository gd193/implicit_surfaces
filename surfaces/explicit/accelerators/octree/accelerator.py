"""Octree-flavored MeshAccelerator shell."""
from __future__ import annotations

from typing import Tuple, cast

import jax.numpy as jnp

from surfaces.explicit.accelerators.base import MeshAccelerator
from surfaces.explicit.accelerators.protocol import Backend, Config
from surfaces.explicit.accelerators.registry import get_backend
from surfaces.explicit.primitives import Primitives

from .payload import OctreePayload


class OctreeAccelerator(MeshAccelerator):
    """Mesh accelerator backed by an octree."""

    def __init__(self, primitives: Primitives, payload: OctreePayload,
                 backend: Backend):
        if backend.flavor != "octree":
            raise ValueError(
                f"OctreeAccelerator requires an octree-flavor backend; "
                f"got flavor={backend.flavor!r} from {backend.name!r}"
            )
        self.primitives = primitives
        self.payload = payload
        self.backend = backend
        self.has_distance_bounds = bool(backend.has_distance_bounds)

    @classmethod
    def build(cls, primitives: Primitives, backend_name: str,
              config: Config | None = None) -> "OctreeAccelerator":
        backend_cls = get_backend(backend_name)
        backend = backend_cls()
        if backend.flavor != "octree":
            raise ValueError(
                f"backend {backend_name!r} has flavor {backend.flavor!r}; "
                f"OctreeAccelerator requires 'octree'"
            )
        cfg = config if config is not None else Config()
        payload = cast(OctreePayload, backend.build(primitives, cfg))
        return cls(primitives, payload, backend)

    @classmethod
    def from_payload(cls, primitives: Primitives, payload: OctreePayload,
                     backend_name: str) -> "OctreeAccelerator":
        """Wrap an externally-built payload (e.g. from the implicit-tree
        bridge) with a traversal-only backend.
        """
        backend = get_backend(backend_name)()
        return cls(primitives, payload, backend)

    # --- MeshAccelerator interface ---

    def closest_point(self, x: jnp.ndarray
                      ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        return self.backend.closest_point(self.payload, x)

    def first_hit(self, ray_origin: jnp.ndarray, ray_direction: jnp.ndarray,
                  max_t: jnp.ndarray
                  ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        return self.backend.first_hit(self.payload, ray_origin,
                                      ray_direction, max_t)

    def aabb_overlap(self, lower: jnp.ndarray, upper: jnp.ndarray
                     ) -> jnp.ndarray:
        return self.backend.aabb_overlap(self.payload, lower, upper)

    def distance_bounds(self, lower: jnp.ndarray, upper: jnp.ndarray
                        ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        return self.backend.distance_bounds(self.payload, lower, upper)
