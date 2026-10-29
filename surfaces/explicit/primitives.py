"""Primitive backends for explicit surfaces.

A `Primitives` object encapsulates everything that depends on the
geometric primitive type (triangle, point, spline patch, ...). It
exposes four leaf-level callbacks plus per-primitive metadata. A
`MeshAccelerator` stores one and delegates all primitive-specific math
to it; the spatial-partition logic (`SpatialTree` topology, traversal
kernels) stays primitive-agnostic and is shared across all concrete
accelerators.

This is the explicit-side analog of the implicit-side bound machinery:
swappable, capability-flagged, hidden behind a uniform protocol so
consumers (`TriMeshSurface`, `MeshAccelerator`) never see the type.

Implementations supply per-primitive data as flat jnp arrays stored as
attributes; callbacks are pure JAX so they JIT and vmap cleanly. Build-
time work (precomputing normals, edge vectors, ...) happens once in
`__init__`.
"""
from __future__ import annotations

import abc
from typing import Tuple

import jax.numpy as jnp

from surfaces.base import TriMesh


class Primitives(abc.ABC):
    """Leaf-level geometric primitive backend.

    Implementations: `TriMeshPrimitives` (triangles), later
    `PointCloudPrimitives` (oriented points), `SplinePatchPrimitives`,
    etc. Each new type adds ~100 lines of leaf math and plugs into any
    existing accelerator.
    """

    # Static metadata — number of primitives in this collection.
    n_primitives: int

    @abc.abstractmethod
    def primitive_aabb(self, idx: jnp.ndarray
                       ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """`(lower, upper)` axis-aligned bounds of primitive `idx`.
        Used at build time and for `aabb_overlap` leaf tests.
        """

    @abc.abstractmethod
    def closest_point_on_primitive(self, x: jnp.ndarray, idx: jnp.ndarray
                                   ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Closest point on primitive `idx` to `x`, plus unsigned distance.
        Returns `(cp, dist)`. Sign is added by the surface via inside-test.
        """

    @abc.abstractmethod
    def ray_primitive_intersect(self, ray_origin: jnp.ndarray,
                                ray_direction: jnp.ndarray,
                                max_t: jnp.ndarray, idx: jnp.ndarray
                                ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Ray-primitive intersection in `[0, max_t]`.
        Returns `(t, hit_flag)`. When `hit_flag = False`, `t` is unspecified.
        """

    @abc.abstractmethod
    def primitive_normal(self, idx: jnp.ndarray) -> jnp.ndarray:
        """Unit normal of primitive `idx` (used as the default surface
        normal at the closest-point face).
        """


# =====================================================================
# TriMeshPrimitives — triangle leaf-test backend.
# =====================================================================


class TriMeshPrimitives(Primitives):
    """Triangle leaf-test backend over a `TriMesh`.

    Stores precomputed per-face quantities (vertex positions, edge
    vectors, normals, AABBs) so each leaf-level callback is a small
    arithmetic kernel without indexing into the raw mesh.
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

    def closest_point_on_primitive(self, x: jnp.ndarray, idx: jnp.ndarray
                                   ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        # Closest-point-on-triangle, Ericson "Real-Time Collision Detection"
        # ch.5: barycentric region classification (3 vertex regions, 3 edge
        # regions, 1 face region). All branches are evaluated and selected
        # via jnp.where so the kernel is jit/vmap clean. Defensive +eps on
        # every divisor; on a degenerate (zero-area) triangle the algorithm
        # collapses to the closest of the three vertices.
        eps = jnp.float32(1e-12)

        a = self._v0[idx]
        b = self._v1[idx]
        c = self._v2[idx]
        ab = b - a
        ac = c - a

        ap = x - a
        d1 = jnp.dot(ab, ap)
        d2 = jnp.dot(ac, ap)
        # Region A: vertex a (d1 <= 0 and d2 <= 0).
        in_A = (d1 <= 0.0) & (d2 <= 0.0)

        bp = x - b
        d3 = jnp.dot(ab, bp)
        d4 = jnp.dot(ac, bp)
        # Region B: vertex b (d3 >= 0 and d4 <= d3).
        in_B = (d3 >= 0.0) & (d4 <= d3)

        cp_x = x - c
        d5 = jnp.dot(ab, cp_x)
        d6 = jnp.dot(ac, cp_x)
        # Region C: vertex c (d6 >= 0 and d5 <= d6).
        in_C = (d6 >= 0.0) & (d5 <= d6)

        # Region AB: edge ab (vc <= 0 and d1 >= 0 and d3 <= 0).
        vc = d1 * d4 - d3 * d2
        in_AB = (vc <= 0.0) & (d1 >= 0.0) & (d3 <= 0.0)
        v_ab = d1 / (d1 - d3 + eps)
        cp_AB = a + v_ab * ab

        # Region AC: edge ac (vb <= 0 and d2 >= 0 and d6 <= 0).
        vb = d5 * d2 - d1 * d6
        in_AC = (vb <= 0.0) & (d2 >= 0.0) & (d6 <= 0.0)
        v_ac = d2 / (d2 - d6 + eps)
        cp_AC = a + v_ac * ac

        # Region BC: edge bc (va <= 0 and (d4 - d3) >= 0 and (d5 - d6) >= 0).
        va = d3 * d6 - d5 * d4
        in_BC = (va <= 0.0) & ((d4 - d3) >= 0.0) & ((d5 - d6) >= 0.0)
        v_bc = (d4 - d3) / ((d4 - d3) + (d5 - d6) + eps)
        cp_BC = b + v_bc * (c - b)

        # Region face: barycentric inside the triangle.
        denom = va + vb + vc
        denom_safe = denom + eps
        v_face = vb / denom_safe
        w_face = vc / denom_safe
        cp_face = a + v_face * ab + w_face * ac

        # Select region. Order matches Ericson's early-out structure.
        cp = cp_face
        cp = jnp.where(in_BC, cp_BC, cp)
        cp = jnp.where(in_AC, cp_AC, cp)
        cp = jnp.where(in_AB, cp_AB, cp)
        cp = jnp.where(in_C, c, cp)
        cp = jnp.where(in_B, b, cp)
        cp = jnp.where(in_A, a, cp)

        dist = jnp.linalg.norm(x - cp)
        return cp, dist

    def ray_primitive_intersect(self, ray_origin: jnp.ndarray,
                                ray_direction: jnp.ndarray,
                                max_t: jnp.ndarray, idx: jnp.ndarray
                                ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        # Möller–Trumbore (JGT 2(1) 1997), non-culled formulation. Returns
        # (t, hit). Small barycentric slack accepts hits exactly on shared
        # edges of well-formed meshes; parallel-ray rejection threshold is
        # at float32-epsilon scale to filter phantom hits where det ~ 0.
        eps = jnp.float32(1e-6)
        eps_div = jnp.float32(1e-7)

        v0 = self._v0[idx]
        e1 = self._e1[idx]
        e2 = self._e2[idx]

        pvec = jnp.cross(ray_direction, e2)
        det = jnp.dot(e1, pvec)
        inv_det = 1.0 / jnp.where(jnp.abs(det) < eps_div,
                                  jnp.float32(eps_div), det)

        tvec = ray_origin - v0
        u = jnp.dot(tvec, pvec) * inv_det

        qvec = jnp.cross(tvec, e1)
        v = jnp.dot(ray_direction, qvec) * inv_det

        t = jnp.dot(e2, qvec) * inv_det

        hit = (
            (jnp.abs(det) >= eps_div)
            & (u >= -eps)
            & (v >= -eps)
            & ((u + v) <= 1.0 + eps)
            & (t >= 0.0)
            & (t <= max_t)
        )
        # Replace NaN/inf t on miss with a finite placeholder.
        t_safe = jnp.where(jnp.isfinite(t), t, max_t)
        t_out = jnp.where(hit, t_safe, max_t)
        return t_out, hit

    def primitive_normal(self, idx: jnp.ndarray) -> jnp.ndarray:
        return self._face_normals[idx]


# =====================================================================
# PointCloudPrimitives — point leaf-test backend.
# =====================================================================


class PointCloudPrimitives(Primitives):
    """Point-cloud leaf-test backend.

    Each primitive is a single 3D point with zero extent. The natural
    domain of left-balanced kd-trees (Wald 2022 family). Triangle-style
    queries are handled as follows:

    - `primitive_aabb(idx)` → degenerate `(point, point)`. Build paths
      tolerate this (lbvh has the degenerate-axis Morton fix; kd.sah and
      others handle empty / tight nodes).
    - `closest_point_on_primitive(x, idx)` → the point itself plus
      Euclidean distance. The natural query.
    - `ray_primitive_intersect(...)` → always miss (zero-extent points
      have probability-zero ray hits). Returns `(max_t, False)` so the
      caller's reduction is a no-op rather than a NaN.
    - `primitive_normal(idx)` → returns stored normal if the cloud is
      oriented (constructor arg supplied); raises otherwise.

    For ray-traversal use cases over a point cloud, wrap the points as
    oriented disks or small spheres in a separate primitives backend
    (out of scope here).
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

    def closest_point_on_primitive(self, x: jnp.ndarray, idx: jnp.ndarray
                                   ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        p = self.points[idx]
        return p, jnp.linalg.norm(x - p)

    def ray_primitive_intersect(self, ray_origin: jnp.ndarray,
                                ray_direction: jnp.ndarray,
                                max_t: jnp.ndarray, idx: jnp.ndarray
                                ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        # Zero-extent: rays never hit a point with probability 1. Return
        # finite placeholder t so downstream reductions stay numerically
        # stable.
        return max_t, jnp.bool_(False)

    def primitive_normal(self, idx: jnp.ndarray) -> jnp.ndarray:
        if self.normals is None:
            raise ValueError(
                "PointCloudPrimitives constructed without `normals`; "
                "primitive_normal is undefined for unoriented point clouds. "
                "Pass `normals=...` at construction to support this query."
            )
        return self.normals[idx]
