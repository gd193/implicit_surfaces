"""Triangle surfaces with optional vertex-velocity differentiation."""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np

from surfaces.meshes import TriMesh
from surfaces.explicit.base import ExplicitSurface

if TYPE_CHECKING:
    from sengi.boundary.data import BoundaryCoreData, BoundaryData, SilhouetteData
    from sengi.primitives.data import TrianglesArrays3


class TriMeshSurface(ExplicitSurface):
    """Triangle surface with even-odd signs on closed meshes.

    Build the index and prepare its boundary from the same source mesh.
    """

    dim: int = 3

    def __init__(self,
                 mesh: TriMesh,
                 boundary: BoundaryData[TrianglesArrays3, BoundaryCoreData, SilhouetteData],
                 *,
                 vertex_velocity_fn: Optional[Callable[..., Any]] = None):
        from sengi import queries
        from sengi.metrics.nearest import Euclidean
        from sengi.primitives.triangles import Triangles

        if not isinstance(mesh, TriMesh) or mesh.dim != 3:
            raise ValueError("TriMeshSurface requires a 3D TriMesh")

        spec = boundary.data.spec
        if spec.primitive is not Triangles():
            raise TypeError("TriMeshSurface requires an index over Triangles().")

        tree, geometry = boundary.data
        faces = np.asarray(mesh.elements)
        if (faces.shape != geometry.indices.shape
                or not np.array_equal(mesh.vertices, geometry.vertices)
                or not np.array_equal(faces[np.asarray(tree.prim_ids)], geometry.indices)):
            raise ValueError("surface mesh does not match the packed triangle index")

        self.mesh = jax.tree.map(jnp.asarray, mesh)
        self._vertex_velocity_fn = vertex_velocity_fn
        self._boundary = boundary
        self._nearest = queries.nearest.bind(spec, Euclidean())
        self._ray = queries.rays.bind(spec, "first")
        self._overlap = queries.overlap.bind(spec)
        self._classify = queries.classification.bind(spec)

        # Boundary normals follow leaf order. Surface methods use source face IDs.
        normals = boundary.core.primitive_normal
        self._face_normals = jnp.zeros_like(normals).at[tree.prim_ids].set(normals)

        if vertex_velocity_fn is not None:
            # Normalize each length before taking the area product. The square
            # root keeps the cached FP32 factor finite even for subnormal areas.
            area = np.asarray(boundary.core.primitive_measure)
            scale = np.zeros_like(area)
            np.divide(np.sqrt(np.float32(0.5)), np.sqrt(area), out=scale, where=area > 0)
            self._barycentric_scale = jnp.zeros_like(normals[:, 0]).at[tree.prim_ids].set(scale)

    def closest_point(self, params: Any, x: jnp.ndarray
                      ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        return self._nearest(self._boundary.data, x)[:3]

    def first_hit(self, params: Any, ray_origin: jnp.ndarray,
                  ray_direction: jnp.ndarray, max_t: jnp.ndarray
                  ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        return self._ray(self._boundary.data, ray_origin, ray_direction, max_t)[:3]

    def _primitive_normal(self, prim_idx: jnp.ndarray) -> jnp.ndarray:
        return self._face_normals[prim_idx]

    def classify_box(self, params: Any,
                     lower: jnp.ndarray, upper: jnp.ndarray) -> jnp.ndarray:
        """Conservative box sign; 0 means unknown or intersecting."""
        overlaps, status = self._overlap(self._boundary.data, lower, upper)
        center = 0.5 * (lower + upper)
        sign = self._inside_sign(params, center)
        sign = jnp.where(jnp.isfinite(sign), sign, 0).astype(jnp.int8)
        return jnp.where(overlaps | (status < 0), jnp.int8(0), sign)

    def _inside_sign(self, params: Any, x: jnp.ndarray) -> jnp.ndarray:
        return self._classify(self._boundary, x)[0]

    def surface_velocity_normal(self, params: Any, x: jnp.ndarray,
                                normal: jnp.ndarray) -> Any:
        """Project barycentrically blended vertex velocities onto the normal."""
        if self._vertex_velocity_fn is None:
            raise NotImplementedError(
                "surface_velocity_normal requires vertex_velocity_fn"
            )
        cp, _dist, face_idx = self.closest_point(params, x)
        v_idx = self.mesh.elements[face_idx]
        bary = _barycentric(cp, self.mesh.vertices[v_idx], self._face_normals[face_idx],
                            self._barycentric_scale[face_idx])
        dV = self._vertex_velocity_fn(params, v_idx)

        def _blend(g):
            return jnp.einsum('i,ij,j->', bary, g, normal)

        return jax.tree.map(_blend, dV)


def _barycentric(p, tri, face_normal, scale):
    """Signed subtriangle areas divided by the face's double area."""
    a, b, c = tri
    transverse = jnp.cross(face_normal, (p - a) * scale)
    v = jnp.dot(transverse, (c - a) * scale)
    w = -jnp.dot(transverse, (b - a) * scale)
    return jnp.stack((1 - v - w, v, w))
