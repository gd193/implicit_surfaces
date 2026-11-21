"""Triangle surfaces with optional vertex-velocity differentiation."""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, Optional, Tuple

import jax
import jax.numpy as jnp

from surfaces.base import TriMesh
from surfaces.explicit.base import ExplicitSurface

if TYPE_CHECKING:
    from sengi.boundary.data import BoundaryCoreData, BoundaryData, SilhouetteData
    from sengi.bvh import Index
    from sengi.primitives.data import TrianglesArrays3


class TriMeshSurface(ExplicitSurface):
    """Triangle surface with even-odd signs on closed meshes.

    Build the index and prepare its boundary from the same source mesh.
    """

    dim: int = 3

    def __init__(self,
                 mesh: TriMesh,
                 index: Index[Any, TrianglesArrays3],
                 boundary: BoundaryData[BoundaryCoreData, SilhouetteData],
                 *,
                 vertex_velocity_fn: Optional[Callable[..., Any]] = None):
        from sengi import queries
        from sengi.metrics.nearest import Euclidean
        from sengi.primitives.triangles import Triangles

        if index.spec.primitive is not Triangles():
            raise TypeError("TriMeshSurface requires an index over Triangles().")

        self.mesh = mesh
        self.index = index
        self._vertex_velocity_fn = vertex_velocity_fn
        self._boundary = boundary
        self._nearest = queries.nearest.bind(index.spec, Euclidean())
        self._ray = queries.rays.bind(index.spec, "first")
        self._overlap = queries.overlap.bind(index.spec)
        self._classify = queries.classification.bind(index.spec)

        # Boundary normals follow leaf order. Surface methods use source face IDs.
        normals = boundary.core.primitive_normal
        self._face_normals = jnp.zeros_like(normals).at[index.data.tree.prim_ids].set(normals)

        self.is_differentiable_wrt_params = vertex_velocity_fn is not None

    def closest_point(self, params: Any, x: jnp.ndarray
                      ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        return self._nearest(self.index.data, x)[:3]

    def first_hit(self, params: Any, ray_origin: jnp.ndarray,
                  ray_direction: jnp.ndarray, max_t: jnp.ndarray
                  ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        return self._ray(self.index.data, ray_origin, ray_direction, max_t)[:3]

    def _primitive_normal(self, prim_idx: jnp.ndarray) -> jnp.ndarray:
        return self._face_normals[prim_idx]

    def classify_box(self, params: Any,
                     lower: jnp.ndarray, upper: jnp.ndarray) -> jnp.ndarray:
        """Conservative box sign; 0 means unknown or intersecting."""
        overlaps, status = self._overlap(self.index.data, lower, upper)
        center = 0.5 * (lower + upper)
        sign = self._inside_sign(params, center)
        sign = jnp.where(jnp.isfinite(sign), sign, 0).astype(jnp.int8)
        return jnp.where(overlaps | (status < 0), jnp.int8(0), sign)

    def _inside_sign(self, params: Any, x: jnp.ndarray) -> jnp.ndarray:
        return self._classify(self.index.data, self._boundary, x)[0]

    def surface_velocity_normal(self, params: Any, x: jnp.ndarray,
                                normal: jnp.ndarray) -> Any:
        """Project barycentrically blended vertex velocities onto the normal."""
        if self._vertex_velocity_fn is None:
            raise NotImplementedError(
                "surface_velocity_normal requires vertex_velocity_fn"
            )
        cp, _dist, face_idx = self.closest_point(params, x)
        v_idx = self.mesh.faces[face_idx]
        bary = _barycentric(cp, self.mesh.vertices[v_idx])
        dV = self._vertex_velocity_fn(params, v_idx)

        def _blend(g):
            return jnp.einsum('i,ij,j->', bary, g, normal)

        return jax.tree.map(_blend, dV)


def _barycentric(p: jnp.ndarray, tri: jnp.ndarray) -> jnp.ndarray:
    """Barycentric coordinates for a coplanar point."""
    v0 = tri[1] - tri[0]
    v1 = tri[2] - tri[0]
    v2 = p - tri[0]
    d00 = jnp.dot(v0, v0)
    d01 = jnp.dot(v0, v1)
    d11 = jnp.dot(v1, v1)
    d20 = jnp.dot(v2, v0)
    d21 = jnp.dot(v2, v1)
    denom = d00 * d11 - d01 * d01 + 1e-30
    v = (d11 * d20 - d01 * d21) / denom
    w = (d00 * d21 - d01 * d20) / denom
    u = 1.0 - v - w
    return jnp.array([u, v, w])
