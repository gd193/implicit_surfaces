"""Triangle surfaces with optional vertex-velocity differentiation."""
from __future__ import annotations

from typing import Any, Callable, Optional, Tuple

import jax
import jax.numpy as jnp

from surfaces.base import TriMesh
from surfaces.explicit.base import ExplicitSurface
from surfaces.explicit.accelerators.base import MeshAccelerator
from surfaces.explicit.primitives import TriMeshPrimitives


class TriMeshSurface(ExplicitSurface):
    """Triangle surface with even-odd signs on closed meshes."""

    dim: int = 3

    def __init__(self,
                 mesh: TriMesh,
                 accelerator: MeshAccelerator,
                 *,
                 vertex_velocity_fn: Optional[Callable[..., Any]] = None):
        self.mesh = mesh
        self.accelerator = accelerator
        self._vertex_velocity_fn = vertex_velocity_fn

        if not isinstance(accelerator.primitives, TriMeshPrimitives):
            raise TypeError(
                f"TriMeshSurface requires a triangle-primitive accelerator; "
                f"got primitives of type {type(accelerator.primitives).__name__}."
            )

        self.is_differentiable_wrt_params = vertex_velocity_fn is not None
        self.has_box_bounds = accelerator.has_distance_bounds

    def closest_point(self, params: Any, x: jnp.ndarray
                      ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        return self.accelerator.closest_point(x)

    def first_hit(self, params: Any, ray_origin: jnp.ndarray,
                  ray_direction: jnp.ndarray, max_t: jnp.ndarray
                  ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        return self.accelerator.first_hit(ray_origin, ray_direction, max_t)

    def _primitive_normal(self, prim_idx: jnp.ndarray) -> jnp.ndarray:
        return self.accelerator.primitives.primitive_normal(prim_idx)

    def classify_box(self, params: Any,
                     lower: jnp.ndarray, upper: jnp.ndarray) -> jnp.ndarray:
        """Conservative box sign; 0 means unknown or intersecting."""
        overlaps = self.accelerator.aabb_overlap(lower, upper)
        center = 0.5 * (lower + upper)
        sign = self._inside_sign(params, center)
        sign = jnp.where(jnp.isfinite(sign), sign, 0).astype(jnp.int8)
        return jnp.where(overlaps, jnp.int8(0), sign)

    def box_bounds(self, params: Any,
                   lower: jnp.ndarray, upper: jnp.ndarray
                   ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Use native distance bounds, or the Lipschitz enclosure."""
        if self.accelerator.has_distance_bounds:
            d_min, d_max = self.accelerator.distance_bounds(lower, upper)
            # Sign-extend by querying the center; tightness depends on
            # whether the surface crosses the AABB.
            center = 0.5 * (lower + upper)
            sign = jnp.sign(self.signed_value(params, center))
            return sign * d_min, sign * d_max
        return super().box_bounds(params, lower, upper)

    def _inside_sign(self, params: Any, x: jnp.ndarray) -> jnp.ndarray:
        return self.accelerator.classify_point(x)

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
