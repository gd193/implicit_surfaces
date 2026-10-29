"""Kd-flavored MeshAccelerator shell."""
from __future__ import annotations

from typing import Tuple, cast

import jax.numpy as jnp

from surfaces.explicit.accelerators.base import MeshAccelerator
from surfaces.explicit.accelerators.protocol import Backend, Config
from surfaces.explicit.accelerators.registry import get_backend
from surfaces.explicit.primitives import Primitives

from .payload import KdPayload


class KdAccelerator(MeshAccelerator):
    """Mesh accelerator backed by a kd-tree."""

    def __init__(self, primitives: Primitives, payload: KdPayload,
                 backend: Backend):
        if backend.flavor != "kd":
            raise ValueError(
                f"KdAccelerator requires a kd-flavor backend; "
                f"got flavor={backend.flavor!r} from {backend.name!r}"
            )
        self.primitives = primitives
        self.payload = payload
        self.backend = backend
        self.has_distance_bounds = bool(backend.has_distance_bounds)

    @classmethod
    def build(cls, primitives: Primitives, backend_name: str,
              config: Config | None = None) -> "KdAccelerator":
        backend_cls = get_backend(backend_name)
        backend = backend_cls()
        if backend.flavor != "kd":
            raise ValueError(
                f"backend {backend_name!r} has flavor {backend.flavor!r}; "
                f"KdAccelerator requires 'kd'"
            )
        cfg = config if config is not None else Config()
        payload = cast(KdPayload, backend.build(primitives, cfg))
        return cls(primitives, payload, backend)

    @classmethod
    def from_payload(cls, primitives: Primitives, payload: KdPayload,
                     backend_name: str) -> "KdAccelerator":
        """Wrap an externally-built payload (e.g. from the implicit-tree
        bridge) with a traversal-only backend. Skips `backend.build`.
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
