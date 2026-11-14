"""Primitive data, build bounds and normals for explicit surfaces.

Host tree builders consume `primitive_aabb`; surface normal queries
consume `primitive_normal`. Warp query factories own the geometric
intersection and closest-point operations.
"""
from __future__ import annotations

import abc
from typing import Tuple

import jax.numpy as jnp

from surfaces.base import TriMesh


class Primitives(abc.ABC):
    """Primitive metadata and bounds used by host tree construction."""

    # Static metadata — number of primitives in this collection.
    n_primitives: int

    @abc.abstractmethod
    def primitive_aabb(self, idx: jnp.ndarray
                       ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """`(lower, upper)` axis-aligned bounds of primitive `idx`.
        Used at build time.
        """

    @abc.abstractmethod
    def primitive_normal(self, idx: jnp.ndarray) -> jnp.ndarray:
        """Unit normal of primitive `idx` (used as the default surface
        normal at the closest-point face).
        """


# =====================================================================
# TriMeshPrimitives — triangle build data and surface normals.
# =====================================================================


class TriMeshPrimitives(Primitives):
    """Triangle build data over a `TriMesh`.

    Stores precomputed per-face quantities (vertex positions, edge
    vectors, normals, AABBs) for construction and surface normals.
    """

    def __init__(self, mesh: TriMesh):
        self.mesh = mesh
        v = mesh.vertices[mesh.faces]              # (F, 3, 3)
        self._v0 = v[:, 0]
        self._v1 = v[:, 1]
        self._v2 = v[:, 2]
        e1 = self._v1 - self._v0
        e2 = self._v2 - self._v0
        self._e1 = e1
        self._e2 = e2
        n = jnp.cross(e1, e2)
        n_norm = jnp.linalg.norm(n, axis=-1, keepdims=True) + 1e-20
        self._face_normals = n / n_norm

        self._aabb_lo = jnp.minimum(jnp.minimum(self._v0, self._v1), self._v2)
        self._aabb_hi = jnp.maximum(jnp.maximum(self._v0, self._v1), self._v2)

        self.n_primitives = int(mesh.faces.shape[0])

    def primitive_aabb(self, idx: jnp.ndarray
                       ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        return self._aabb_lo[idx], self._aabb_hi[idx]

    def primitive_normal(self, idx: jnp.ndarray) -> jnp.ndarray:
        return self._face_normals[idx]


# =====================================================================
# PointCloudPrimitives — point build data and optional normals.
# =====================================================================


class PointCloudPrimitives(Primitives):
    """Point-cloud build data.

    Each primitive is a single 3D point with zero extent. The natural
    domain of left-balanced kd-trees (Wald 2022 family).

    - `primitive_aabb(idx)` → degenerate `(point, point)`. Build paths
      tolerate this (lbvh has the degenerate-axis Morton fix; kd.sah and
      others handle empty / tight nodes).
    - `primitive_normal(idx)` → returns stored normal if the cloud is
      oriented (constructor arg supplied); raises otherwise.

    Query operations over points are supplied by the Warp `Points`
    primitive specification.
    """

    def __init__(self, points: jnp.ndarray,
                 normals: jnp.ndarray | None = None):
        if points.ndim != 2 or points.shape[-1] != 3:
            raise ValueError(
                f"points must have shape (N, 3); got {tuple(points.shape)}"
            )
        if normals is not None:
            if normals.shape != points.shape:
                raise ValueError(
                    "normals shape must match points; "
                    f"got points={tuple(points.shape)} "
                    f"normals={tuple(normals.shape)}"
                )
        self.points = points
        self.normals = normals
        self.n_primitives = int(points.shape[0])

    def primitive_aabb(self, idx: jnp.ndarray
                       ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        p = self.points[idx]
        return p, p

    def primitive_normal(self, idx: jnp.ndarray) -> jnp.ndarray:
        if self.normals is None:
            raise ValueError(
                "PointCloudPrimitives constructed without `normals`; "
                "primitive_normal is undefined for unoriented point clouds. "
                "Pass `normals=...` at construction to support this query."
            )
        return self.normals[idx]
