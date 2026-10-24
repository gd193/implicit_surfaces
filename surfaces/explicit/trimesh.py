"""Triangle-mesh-backed explicit surface.

`TriMeshSurface` is a concrete `ExplicitSurface`. It owns a `TriMesh`
plus a `MeshAccelerator`; the accelerator in turn owns a `Primitives`
backend (`TriMeshPrimitives`) that provides triangle-specific leaf
math. The surface itself adds mesh-level concerns: vertex-velocity
chain rule for shape opt, optional smoother normal schemes, etc.

Diff support is opt-in via `vertex_velocity_fn`. When `None`, the
surface advertises `is_differentiable_wrt_params = False` and
`surface_velocity_normal` raises. When provided, the callback receives
`(params, vertex_indices)` and returns vertex velocities; the surface
chains them through barycentric blending at the closest-point face to
produce `V_n`. The walk itself never touches the callback.
"""
from __future__ import annotations

from typing import Any, Callable, Optional, Tuple

import jax
import jax.numpy as jnp

from surfaces.base import TriMesh
from surfaces.explicit.base import ExplicitSurface
from surfaces.explicit.accelerators.base import MeshAccelerator
from surfaces.explicit.primitives import TriMeshPrimitives


class TriMeshSurface(ExplicitSurface):
    """Static triangle mesh + accelerator; optional differentiability.

    Args:
        mesh:                 TriMesh (vertices, faces).
        accelerator:          MeshAccelerator instance (BVH / kd / ...).
        face_normals:         (F, 3) optional precomputed unit face normals.
                              Computed from `mesh` if omitted.
        vertex_velocity_fn:   Optional `(params, vertex_indices) -> dV` hook
                              for differentiable shape opt. dV must be a
                              pytree shaped like `params` whose leaves
                              have a leading axis matching `vertex_indices`.
                              Only invoked at terminated walkers in dWoS;
                              never on the hot path.
        inside_test:          "winding" | "raycast" | "normal_proj" — sign
                              choice for `signed_value`. Default "winding"
                              (continuous, robust to non-watertight meshes).
    """

    dim: int = 3

    def __init__(self,
                 mesh: TriMesh,
                 accelerator: MeshAccelerator,
                 *,
                 vertex_velocity_fn: Optional[Callable[..., Any]] = None,
                 inside_test: str = "winding"):
        self.mesh = mesh
        self.accelerator = accelerator
        self._vertex_velocity_fn = vertex_velocity_fn
        self._inside_test = inside_test

        # Sanity: accelerator's primitive backend must agree with this mesh.
        # We don't require object identity — callers may build the
        # accelerator from a `TriMeshPrimitives(mesh)` themselves and pass
        # both in — but the primitives backend must be triangle-shaped.
        if not isinstance(accelerator.primitives, TriMeshPrimitives):
            raise TypeError(
                f"TriMeshSurface requires a triangle-primitive accelerator; "
                f"got primitives of type {type(accelerator.primitives).__name__}."
            )

        # Capability flag derives from the optional hook.
        self.is_differentiable_wrt_params = vertex_velocity_fn is not None
        # Inherit accelerator-level capabilities upward.
        self.has_box_bounds = bool(getattr(accelerator, "has_distance_bounds",
                                            False))

    # =====================================================================
    # Foundation primitives — delegate to accelerator.
    # =====================================================================

    def closest_point(self, params: Any, x: jnp.ndarray
                      ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        # `params` unused — explicit mesh has no learnable θ on the hot path.
        return self.accelerator.closest_point(x)

    def first_hit(self, params: Any, ray_origin: jnp.ndarray,
                  ray_direction: jnp.ndarray, max_t: jnp.ndarray
                  ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        return self.accelerator.first_hit(ray_origin, ray_direction, max_t)

    def _primitive_normal(self, prim_idx: jnp.ndarray) -> jnp.ndarray:
        # Defer to the primitives backend so swapping the leaf-test
        # implementation (e.g. a higher-order smoothing scheme) updates
        # normals consistently.
        return self.accelerator.primitives.primitive_normal(prim_idx)

    # =====================================================================
    # Tighter overrides: classify_box uses accelerator's AABB tests.
    # =====================================================================

    def classify_box(self, params: Any,
                     lower: jnp.ndarray, upper: jnp.ndarray) -> jnp.ndarray:
        """If any face overlaps the AABB, the box straddles the surface
        (UNKNOWN). Otherwise the surface is fully on one side; use the
        center's signed value to pick which.
        """
        overlaps = self.accelerator.aabb_overlap(lower, upper)
        center = 0.5 * (lower + upper)
        sign = jnp.sign(self.signed_value(params, center)).astype(jnp.int8)
        return jnp.where(overlaps, jnp.int8(0), sign)

    def box_bounds(self, params: Any,
                   lower: jnp.ndarray, upper: jnp.ndarray
                   ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Use the accelerator's distance bounds when available; fall
        back to the conservative Lipschitz argument otherwise.
        """
        if self.accelerator.has_distance_bounds:
            d_min, d_max = self.accelerator.distance_bounds(lower, upper)
            # Sign-extend by querying the center; tightness depends on
            # whether the surface crosses the AABB.
            center = 0.5 * (lower + upper)
            sign = jnp.sign(self.signed_value(params, center))
            return sign * d_min, sign * d_max
        return super().box_bounds(params, lower, upper)

    # =====================================================================
    # Inside test selection — driven by `inside_test` flag.
    # =====================================================================

    def _inside_sign(self, params: Any, x: jnp.ndarray) -> jnp.ndarray:
        # Concrete inside-tests are deferred to a follow-up — they need
        # the BVH/accelerator implementations to be standing up. For now,
        # raise loudly so consumers cannot silently get a wrong sign.
        raise NotImplementedError(
            f"inside_test={self._inside_test!r} not yet wired; "
            f"land BVH-backed accelerator first."
        )

    # =====================================================================
    # Differentiability bridge to dWoS.
    # =====================================================================

    def surface_velocity_normal(self, params: Any, x: jnp.ndarray,
                                normal: jnp.ndarray) -> Any:
        """V_n at boundary hit point `x` for shape-opt gradient flow.

        Locates the closest face, fetches its three vertices' velocities
        from the registered `vertex_velocity_fn`, and projects a
        barycentric blend onto `normal`. Caller is responsible for
        having `is_differentiable_wrt_params = True` (which only holds
        when a velocity callback was registered).
        """
        if self._vertex_velocity_fn is None:
            raise NotImplementedError(
                "TriMeshSurface constructed without vertex_velocity_fn — "
                "shape-opt gradient flow is unavailable. Provide one to "
                "enable surface_velocity_normal."
            )
        cp, _dist, face_idx = self.closest_point(params, x)
        v_idx = self.mesh.faces[face_idx]                     # (3,) int
        bary = _barycentric(cp, self.mesh.vertices[v_idx])    # (3,)
        dV = self._vertex_velocity_fn(params, v_idx)          # pytree, leaves (3, 3)

        def _blend(g):
            # g: (3, 3) — three vertex velocities, last axis is space.
            return jnp.einsum('i,ij,j->', bary, g, normal)

        return jax.tree.map(_blend, dV)


# =====================================================================
# Helpers.
# =====================================================================


def _barycentric(p: jnp.ndarray, tri: jnp.ndarray) -> jnp.ndarray:
    """Barycentric coords of `p` (assumed coplanar with `tri`).
    Standard 3D formulation; numerically benign for non-degenerate tris.
    """
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
