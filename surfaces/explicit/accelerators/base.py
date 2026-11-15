"""Mesh-accelerator protocol.

A `MeshAccelerator` combines a static spatial index and primitive data.
Warp query factories specialize traversal and geometric operations for
each supported primitive type. `Primitives` supplies construction bounds
and surface normals.

The accelerator answers two query primitives consumed by
`ExplicitSurface` (`closest_point`, `first_hit`) plus an AABB-overlap
test used by `classify_box`. Capability flags advertise optional support; consumers
branch on these, not on accelerator type.
"""
from __future__ import annotations

import abc
from typing import Tuple

import jax.numpy as jnp

from surfaces.explicit.primitives import Primitives


class MeshAccelerator(abc.ABC):
    """Spatial accelerator over a `Primitives` collection.

    The BVH implementation builds from primitive AABBs and delegates
    queries to its Warp backend.
    """

    # Capability flags — surface-level flags propagate from these.
    has_silhouette_query: bool = False    # closest-silhouette-point query
    has_all_hits: bool = False
    has_point_classification: bool = False
    has_robin_query: bool = False
    has_aabb_overlap: bool = True         # mandatory in practice; flag kept
                                          #   so future layouts can advertise
                                          #   weaker support.

    # Construction bounds and surface normals, separate from Warp query data.
    primitives: Primitives

    # =====================================================================
    # Mandatory query primitives.
    # =====================================================================

    @abc.abstractmethod
    def closest_point(self, x: jnp.ndarray
                      ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Closest surface point to `x`. Returns `(cp, dist, primitive_idx)`."""

    @abc.abstractmethod
    def first_hit(self, ray_origin: jnp.ndarray, ray_direction: jnp.ndarray,
                  max_t: jnp.ndarray
                  ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """First ray-surface intersection in `[0, max_t]`.
        Returns `(t, primitive_idx, hit_flag)`.
        """

    @abc.abstractmethod
    def aabb_overlap(self, lower: jnp.ndarray, upper: jnp.ndarray
                     ) -> jnp.ndarray:
        """bool — does any primitive's AABB intersect `[lower, upper]`?
        Used to detect surface-crossing cells in `classify_box`.
        """

    # =====================================================================
    # Optional queries — gated by capability flags.
    # =====================================================================

    def classify_point(self, x: jnp.ndarray) -> jnp.ndarray:
        """Sign: -1 inside, 0 on boundary, +1 outside, NaN on failure."""
        raise NotImplementedError("point classification is unavailable")

    def closest_silhouette(self, x: jnp.ndarray
                           ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Closest silhouette point to `x`. Returns `(cp, dist, primitive_idx)`.
        Default raises; flip `has_silhouette_query = True` when implementing.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not implement closest_silhouette"
        )
