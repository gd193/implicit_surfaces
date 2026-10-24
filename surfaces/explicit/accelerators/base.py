"""Mesh-accelerator protocol.

A `MeshAccelerator` is a static spatial index plus a `Primitives`
backend. The spatial-partition logic (`SpatialTree` topology, traversal
kernels) is shared across primitive types; the leaf-level math —
closest-point, ray intersection, AABB — comes from the injected
`Primitives` object. New primitive types (point cloud, splines, ...)
plug in without touching accelerator code.

The accelerator answers two query primitives consumed by
`ExplicitSurface` (`closest_point`, `first_hit`) plus an AABB-overlap
test used by `classify_box`. Optional `distance_bounds` unlocks tighter
`box_bounds`. Capability flags advertise optional support; consumers
branch on these, not on accelerator type.
"""
from __future__ import annotations

import abc
from typing import Tuple

import jax.numpy as jnp

from surfaces.explicit.primitives import Primitives


class MeshAccelerator(abc.ABC):
    """Spatial accelerator over a `Primitives` collection.

    Concrete subclasses (BVH, kd, octree, uniform grid) build their own
    `SpatialTree` payload from the primitives' AABBs and traverse it,
    delegating leaf-level primitive math to `self.primitives`.
    """

    # Capability flags — surface-level flags propagate from these.
    has_distance_bounds: bool = False     # tight (d_min, d_max) over an AABB
    has_silhouette_query: bool = False    # closest-silhouette-point query
    has_aabb_overlap: bool = True         # mandatory in practice; flag kept
                                          #   so future layouts can advertise
                                          #   weaker support.

    # Each concrete accelerator stores its primitive backend here.
    # The `Primitives` object owns per-primitive data + leaf-level callbacks;
    # the accelerator owns spatial-partition data + traversal.
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

    def distance_bounds(self, lower: jnp.ndarray, upper: jnp.ndarray
                        ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Tight `(d_min, d_max)` of unsigned distance over the AABB.
        Default raises; flip `has_distance_bounds = True` when implementing.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not implement distance_bounds"
        )

    def closest_silhouette(self, x: jnp.ndarray
                           ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Closest silhouette point to `x`. Returns `(cp, dist, primitive_idx)`.
        Default raises; flip `has_silhouette_query = True` when implementing.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not implement closest_silhouette"
        )
