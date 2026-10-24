"""Universal surface protocol.

A `Surface` is what `rayMarcher`, `sphereWalker`, and `meshExtractor` code
against. Two subhierarchies satisfy it:

  * `surfaces.implicit.ImplicitSurface` — level-set defined by a scalar
    function `fn(params, x) - level_set`. Bound machinery
    (interval / Lipschitz / autobound / NIQ) provides `empty_sphere`,
    `ray_step`, `classify_box`.
  * `surfaces.explicit.ExplicitSurface` — geometry given as primitives
    (mesh triangles, point cloud, NURBS, ...). A `MeshAccelerator` (BVH /
    kd / octree / grid) backs the same query methods via closest-point
    and ray-triangle intersection.

Both types are interchangeable from a consumer's view. Capability flags
(class attributes) advertise which optional methods are available;
consumers check at construction and refuse mismatches with a clear error
rather than failing deep inside a JIT trace.

Convention for `signed_value`: returns `f(x) - level_set` (already
shifted). Comparisons use 0, never `level_set`. Implicit subclasses have
a back-compat `__call__` that returns the raw `f(x)`.
"""
from __future__ import annotations

import abc
from typing import Any, NamedTuple, Optional, Tuple

import jax
import jax.numpy as jnp


class TriMesh(NamedTuple):
    """Plain mesh container shared by extractors, explicit surfaces, and writers.

    `vertices`: (V, 3) float32 — vertex positions.
    `faces`:    (F, 3) int32   — vertex indices per triangle (CCW).
    """
    vertices: jnp.ndarray
    faces: jnp.ndarray


class Surface(abc.ABC):
    """Universal contract for implicit and explicit surfaces.

    Subclasses MUST implement the abstract methods below. Optional
    methods raise NotImplementedError by default; subclasses opt in and
    flip the corresponding capability flag.
    """

    # ---- capability flags (single source of truth) ----
    has_box_bounds: bool = False
    has_classify_subcubes: bool = False
    is_differentiable_wrt_params: bool = False
    needs_grad_for_normals: bool = True
    dim: int = 3

    # =====================================================================
    # Universal hot-path protocol — every consumer codes against these.
    # =====================================================================

    @abc.abstractmethod
    def signed_value(self, params: Any, x: jnp.ndarray) -> jnp.ndarray:
        """Return f(x) − level_set at point `x` (already shifted by level set).

        Compare to 0 in consumer code, not to the surface's level set.
        """

    @abc.abstractmethod
    def empty_sphere(self,
                     params: Any,
                     x: jnp.ndarray,
                     trust_region_size: jnp.ndarray,
                     upper_lower_flag: Optional[jnp.ndarray] = None
                     ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Largest provably-empty sphere around `x` within the trust region.

        Returns `(fn_val, radius, next_trust_region_size)`. `fn_val` is the
        signed value at `x` (= signed_value(params, x)). `next_trust_region_size`
        is the surface's recommended TR size for the next step (grow on
        boundary-touch, shrink otherwise; clamped to surface-specific range).
        """

    @abc.abstractmethod
    def ray_step(self,
                 params: Any,
                 ray_origin: jnp.ndarray,
                 ray_direction: jnp.ndarray,
                 trust_region_size: jnp.ndarray,
                 upper_lower_flag: Optional[jnp.ndarray] = None
                 ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Forward step along a ray within the trust region.

        Returns `(fn_val, t_step, in_tr_complement)`. `in_tr_complement`
        is True iff the bound machinery proved no level-set crossing
        inside `tr` (= safe step taken; tr should grow next iter).
        """

    @abc.abstractmethod
    def classify_box(self, params: Any,
                     lower: jnp.ndarray, upper: jnp.ndarray) -> jnp.ndarray:
        """Three-valued sign over the AABB `[lower, upper]`.

        int8 in {-1, 0, +1}: -1 strictly below, +1 strictly above, 0 unknown.
        """

    @abc.abstractmethod
    def normal_at(self, params: Any, x: jnp.ndarray) -> jnp.ndarray:
        """Unit outward normal at `x` (assumed to lie on the surface)."""

    # =====================================================================
    # Optional / capability-gated methods.
    # Default behavior is conservative reductions or NotImplementedError.
    # =====================================================================

    def box_bounds(self, params: Any,
                   lower: jnp.ndarray, upper: jnp.ndarray
                   ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Provably-valid (lo, hi) interval of the signed value over the AABB.

        Strictly more informative than `classify_box`. Default raises;
        surfaces with native interval bounds (autobound, BVH-based explicit)
        override and flip `has_box_bounds = True`.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not implement box_bounds"
        )

    def classify_subcubes(self,
                          params: Any,
                          parent_lower: jnp.ndarray, parent_upper: jnp.ndarray,
                          sub_lowers: jnp.ndarray, sub_uppers: jnp.ndarray
                          ) -> jnp.ndarray:
        """Classify N sub-boxes inside a parent. Default = N independent
        `classify_box` calls. Surfaces with a parent-level model
        (autobound polynomial) override and flip
        `has_classify_subcubes = True`.
        """
        return jax.vmap(lambda lo, hi: self.classify_box(params, lo, hi))(
            sub_lowers, sub_uppers
        )

    def ray_step_with_grad(self,
                           params: Any,
                           ray_origin: jnp.ndarray,
                           ray_direction: jnp.ndarray,
                           trust_region_size: jnp.ndarray,
                           upper_lower_flag: Optional[jnp.ndarray] = None
                           ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Same as `ray_step` but also returns ∇f at the step endpoint.

        Default: `ray_step` + `spatial_derivative`. Subclasses with a
        cheap grad path (autobound polynomial) override.
        """
        fn_val, t_step, in_tr = self.ray_step(
            params, ray_origin, ray_direction, trust_region_size, upper_lower_flag
        )
        grad = self.spatial_derivative(params, ray_origin + t_step * ray_direction)
        return fn_val, t_step, in_tr, grad

    def spatial_derivative(self, params: Any, x: jnp.ndarray) -> jnp.ndarray:
        """∇_x f(x). Implicit surfaces use autograd; explicit surfaces use
        face/closest-point normals. Default raises."""
        raise NotImplementedError(
            f"{type(self).__name__} does not implement spatial_derivative"
        )

    def parameter_derivative(self, params: Any, x: jnp.ndarray) -> Any:
        """∂f/∂params at `x`. Pytree shaped like `params`. Default raises;
        flip `is_differentiable_wrt_params = True` when implementing."""
        raise NotImplementedError(
            f"{type(self).__name__} does not implement parameter_derivative"
        )

    # =====================================================================
    # Bridge from any surface to dWoS — single contract, both worlds satisfy.
    # =====================================================================

    def surface_velocity_normal(self, params: Any, x: jnp.ndarray,
                                normal: jnp.ndarray) -> Any:
        """V_n(x) — normal-direction velocity of the surface as θ changes.

        Pytree shaped like `params`. Implicit subclasses use the standard
        −(∂γ/∂θ)/‖∇_x γ‖ form; explicit subclasses chain through a
        `vertex_velocity_fn` provided at construction. Default raises;
        consumers check `is_differentiable_wrt_params` first.
        """
        raise NotImplementedError(
            f"{type(self).__name__} is not differentiable wrt params"
        )
