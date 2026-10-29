"""Explicit-surface protocol layer.

`ExplicitSurface` adds two abstract foundation primitives —
`closest_point` and `first_hit` — and provides default reductions of all
universal `Surface` methods (`signed_value`, `empty_sphere`, `ray_step`,
`classify_box`, `box_bounds`, `normal_at`) on top of them. Concrete reps
(triangle mesh, point cloud, ...) only implement the two primitives plus
any per-rep specifics.

The reductions are correct but generic. Subclasses with a tighter native
implementation are free to override.
"""
from __future__ import annotations

import abc
from typing import Any, Optional, Tuple

import jax.numpy as jnp

from surfaces.base import Surface


class ExplicitSurface(Surface):
    # Capability flags — explicit reps have explicit normals (no autograd
    # for shading) and tight box bounds via the accelerator.
    needs_grad_for_normals: bool = False
    has_box_bounds: bool = True
    has_classify_subcubes: bool = False
    is_differentiable_wrt_params: bool = False  # subclasses flip when
                                                #   vertex_velocity_fn is provided

    # =====================================================================
    # Foundation primitives — every explicit rep must implement these.
    # All other Surface methods reduce to them by default.
    # =====================================================================

    @abc.abstractmethod
    def closest_point(self, params: Any, x: jnp.ndarray
                      ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Closest surface point to `x`.

        Returns `(closest_point, distance, primitive_idx)`. `distance` is
        the unsigned Euclidean distance; sign is added by `signed_value`
        via an inside-test. `primitive_idx` (e.g. face id) lets callers
        recover normals or barycentrics.
        """

    @abc.abstractmethod
    def first_hit(self, params: Any, ray_origin: jnp.ndarray,
                  ray_direction: jnp.ndarray, max_t: jnp.ndarray
                  ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """First ray-surface intersection in `[0, max_t]`.

        Returns `(t, primitive_idx, hit_flag)`. When `hit_flag = False`,
        `t` is unspecified.
        """

    @abc.abstractmethod
    def _inside_sign(self, params: Any, x: jnp.ndarray) -> jnp.ndarray:
        """+1 outside, -1 inside, applied to closest-point distance to
        produce `signed_value`. Implementations: ray-cast parity,
        generalized winding number, normal-projection at closest point.
        Choice typically cached as an attribute set in __init__.
        """

    # =====================================================================
    # Default Surface protocol implementations — derived from primitives.
    # =====================================================================

    def signed_value(self, params: Any, x: jnp.ndarray) -> jnp.ndarray:
        _, dist, _ = self.closest_point(params, x)
        return dist * self._inside_sign(params, x)

    def empty_sphere(self, params: Any, x: jnp.ndarray,
                     trust_region_size: jnp.ndarray,
                     upper_lower_flag: Optional[jnp.ndarray] = None
                     ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Closest-point distance is the largest provably-empty radius.
        Clamp to half the trust region (matching the implicit convention
        that `empty_sphere` consumes `tr/2` internally) and recommend the
        same value as the next TR — explicit queries are exact, so the
        radius is the actual frontier and need not shrink.
        """
        fn_val = self.signed_value(params, x)
        radius = jnp.minimum(jnp.abs(fn_val), trust_region_size * 0.5)
        next_tr = trust_region_size  # exact distance — no need to adapt
        return fn_val, radius, next_tr

    def ray_step(self, params: Any, ray_origin: jnp.ndarray,
                 ray_direction: jnp.ndarray, trust_region_size: jnp.ndarray,
                 upper_lower_flag: Optional[jnp.ndarray] = None
                 ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """First-hit intersection clipped to the trust region. When no
        hit lies in `[0, tr]`, step the full TR and report
        `in_tr_complement = True` (safe step taken, integrator may grow
        TR next iter). When a hit is found, stop at the hit and report
        `False`.
        """
        fn_val = self.signed_value(params, ray_origin)
        t_hit, _face, hit = self.first_hit(params, ray_origin, ray_direction,
                                            trust_region_size)
        t_step = jnp.where(hit, t_hit, trust_region_size)
        in_tr_complement = jnp.logical_not(hit)
        return fn_val, t_step, in_tr_complement

    def classify_box(self, params: Any,
                     lower: jnp.ndarray, upper: jnp.ndarray) -> jnp.ndarray:
        """Conservative reduction via circumscribed-sphere closest-point.
        Subclasses with a native AABB-overlap test (BVH-backed) should
        override for tightness.
        """
        center = 0.5 * (lower + upper)
        half_diag = 0.5 * jnp.linalg.norm(upper - lower)
        fn_val = self.signed_value(params, center)
        sign = jnp.sign(fn_val).astype(jnp.int8)
        classified = jnp.abs(fn_val) >= half_diag
        return jnp.where(classified, sign, jnp.int8(0))

    def box_bounds(self, params: Any,
                   lower: jnp.ndarray, upper: jnp.ndarray
                   ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Conservative interval via Lipschitz argument: any explicit
        surface is 1-Lipschitz in the signed-distance sense, so
        `[f(c) - half_diag, f(c) + half_diag]` is a valid enclosure.
        Subclasses with an accelerator distance-bound query should override.
        """
        center = 0.5 * (lower + upper)
        half_diag = 0.5 * jnp.linalg.norm(upper - lower)
        fn_c = self.signed_value(params, center)
        return fn_c - half_diag, fn_c + half_diag

    def normal_at(self, params: Any, x: jnp.ndarray) -> jnp.ndarray:
        """Default = primitive normal at the closest point. Subclasses
        with smoother schemes (barycentric Phong on a mesh) should override.
        """
        _, _, prim = self.closest_point(params, x)
        return self._primitive_normal(prim)

    @abc.abstractmethod
    def _primitive_normal(self, prim_idx: jnp.ndarray) -> jnp.ndarray:
        """Per-primitive unit normal — face / patch / point normal. Used
        by the default `normal_at`.
        """
