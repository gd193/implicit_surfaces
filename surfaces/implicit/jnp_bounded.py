from typing import Optional, Tuple, cast

import jax
# Force fp32 matmuls. JAX/XLA's default precision lets large fp32 matmuls
# (batch ≥ 2) be promoted to TF32 (19-bit mantissa, ~1e-3 relative error)
# on Ampere+ GPUs. The autobound polynomial *constant* term `a000` is the
# MLP forward at the Taylor centre; TF32 corrupts it by ~5e-3 absolute,
# which crosses the surface sign boundary in `ray_step`'s flag selection
# and causes wrong-envelope root-finding → ray-march tunneling. Set here
# at import time so anyone instantiating jnpboundedImplicit gets the fix.
jax.config.update("jax_default_matmul_precision", "highest")
import jax.numpy as jnp
import jnp_bounds.autobounds as jb

from .base import implicitSurface
from utils.jnpbounds_roots import (
    roots_2nd_order_polynomial,
    roots_2nd_order_tensor,
    transform_2nd_order_3d,
)
from jnp_bounds.mixed_polynomials_arithmetic import eval as mp_eval
from jnp_bounds.types import Interval
from utils.jnpbound_empty_sphere import empty_sphere_2nd_order_tensor
from utils.jnpbound_box_bounds import (
    box_grad_bounds as _poly_box_grad_bounds,
    box_hessian_bounds as _poly_box_hessian_bounds,
    box_value_bounds as _poly_box_value_bounds,
    polynomial_subboxes_bounds as _poly_subboxes_bounds,
)
from rayMarcher.utils import ray_aabb_intersection_inside


class jnpboundedImplicit(implicitSurface):
    """
    A class for implicit surfaces that can bounded by the jnp autobound library.
    Current limitations of autobound:
        - Only JAX-traceable functions can be automatically bounded.
        - Many JAX library functions are not yet supported. 
          What is supported is bounding MLPs or CNNs that uses the jax.nn.sigmoid, jax.nn.softplus, or jax.nn.swish activations.
        - To compute accurate bounds for deeper neural networks, you may need to use float64 rather than float32.
        - Tracing times can be very long because the library uses tons of explicit for loops --> jax.jit unrolls them,
          which can lead to long compilation times. (1-10minutes)
    For details see https://github.com/google/autobound/
    """
    def __init__(self,
                autobounds_max_degree: int,
                use_classify_subcubes: bool = True,
                use_warp: bool = True,
                grow_factor: float = 1.5,
                shrink_factor: float = 0.5,
                max_trust_region_size: float = float('inf'),
                min_trust_region_size: float = 1e-6,
                **kwargs
                ):
        """
        Initialize the autobounded implicit surface.

        Args:
            fn (Callable): A jnp autobound-able JAX function that defines the implicit surface.
            params (Optional[dict]): Additional parameters for the jnp autobound library.
            level_set (float): The level set value for the implicit surface.
            grow_factor / shrink_factor / max_trust_region_size / min_trust_region_size:
                Trust-region update for `empty_sphere` (used by `WoSIntegrator`).
                Mirrors NIQ's interval_grow_fac / interval_shrink_fac /  hit_eps clip.
        """
        super().__init__(**kwargs)
        self.autobounds_max_degree = autobounds_max_degree
        self.use_classify_subcubes = use_classify_subcubes
        self.use_warp = use_warp
        self._grow = float(grow_factor)
        self._shrink = float(shrink_factor)
        self._max_tr = float(max_trust_region_size)
        self._min_tr = float(min_trust_region_size)
        # use_warp routes polynomial composition through fused NVIDIA-Warp kernels for the
        # supported activations (softplus / swish / sigmoid). GPU-only fast path; harmless
        # but unused on CPU. Default True since neural-surface workloads are the common case.
        composition_implementation = "warp_fused" if self.use_warp else "jax"
        # Box ranges of the Taylor polynomial (step guard, box/subcube classification)
        # use the Warp range kernel on GPU; elsewhere, and for unsupported layouts, JAX eval.
        self.range_implementation = (
            "warp" if self.use_warp and jax.default_backend() == "gpu" else "jax"
        )
        self.taylor_bounds = jb.taylor_bounds(
            self.fn,
            max_degree=self.autobounds_max_degree,
            composition_algorithm="softplus_cheap",
            composition_implementation=composition_implementation,
        )


    # Polynomial linear-coef indices in the autobound 3D-degree-2 ordering
    # (multi_index_3d_2nd_order = [[0,0,0],[0,0,1],[0,1,0],[1,0,0],...]):
    #   poly[3] = ∂f/∂x, poly[2] = ∂f/∂y, poly[1] = ∂f/∂z   at the Taylor centre.
    # Constants here so we don't recompute the lookup inside the jit'd ray_step.
    _GRAD_INDICES = (3, 2, 1)

    def _ray_step_internals(self, params, ray_origin, ray_direction, trust_region_size, upper_lower_flag=None):
        """Single Taylor pass; returns everything `ray_step` and `ray_step_with_grad`
        need. The grad output is a slice of the polynomial (free byproduct of the
        Taylor pass). When called via `ray_step` the slice is unused → JIT
        dead-code-eliminates it; no runtime cost."""
        SAFETY_FACTOR = 0.98  # mirror NIQ's cast_rays_iter
        half_tr = 0.5 * trust_region_size

        trust_region_vec = jnp.broadcast_to(trust_region_size, ray_origin.shape)
        auto_bound = self.taylor_bounds(ray_origin, x_trust_region=None,
                                        trust_region_size=trust_region_vec)
        poly = auto_bound.polynomial_enclosure

        # f at the Taylor centre = f at ray_origin, exact.
        fn_val_at_origin = jnp.squeeze(poly[0]) - self.level_set

        # ∇f at the Taylor centre (= ray_origin), exact. Free byproduct of the
        # Taylor pass — same polynomial we're already inspecting.
        idx_dx, idx_dy, idx_dz = self._GRAD_INDICES
        grad = jnp.array([jnp.squeeze(poly[idx_dx]),
                          jnp.squeeze(poly[idx_dy]),
                          jnp.squeeze(poly[idx_dz])])

        # Classify-first guard: bound the polynomial over the full TR. If decided
        # (hi < level OR lo > level), the surface cannot lie inside → take the full
        # forward exit step without root-finding, no shrink.
        half_extent_vec = half_tr * jnp.ones_like(ray_origin)
        lo_box, hi_box = _poly_box_value_bounds(poly, half_extent_vec, self.range_implementation)
        decided = jnp.logical_or(hi_box - self.level_set < 0,
                                 lo_box - self.level_set > 0)

        if upper_lower_flag is None:
            upper_lower_flag = jnp.where(
                fn_val_at_origin > 0, jnp.array(-1), jnp.array(1),
            )

        # 1-D polynomial along the ray, in local frame from ray_origin.
        coeffs = transform_2nd_order_3d(poly, ray_direction, upper_lower_flag)
        c_abs = coeffs[0] - self.level_set
        b_abs = coeffs[1]
        a_abs = coeffs[2]

        hit, s_root = roots_2nd_order_polynomial(jnp.stack([c_abs, b_abs, a_abs]))

        max_abs_dir = jnp.max(jnp.abs(ray_direction))
        s_exit = half_tr / (max_abs_dir + 1e-12)

        in_box = jnp.logical_and(jnp.logical_not(decided),
                                 jnp.logical_and(hit, s_root < s_exit))
        s_step = jnp.where(in_box, s_root, s_exit)
        s_step = SAFETY_FACTOR * s_step

        s_step = cast(jnp.ndarray, jnp.squeeze(s_step))
        in_box = jnp.squeeze(in_box)

        return fn_val_at_origin, s_step, jnp.logical_not(in_box), grad


    def ray_step(self, params, ray_origin, ray_direction, trust_region_size, upper_lower_flag=None):
        """Sphere-trace step with the Taylor centre at ray_origin. Returns
        (fn_val, t_step, in_tr) — the gradient byproduct of the Taylor pass is
        dropped here; JIT dead-code-eliminates the unused slice. See class
        docstring for full step-decision semantics."""
        fn_val, t_step, in_tr, _grad = self._ray_step_internals(
            params, ray_origin, ray_direction, trust_region_size, upper_lower_flag)
        return fn_val, t_step, in_tr


    def ray_step_with_grad(self, params, ray_origin, ray_direction, trust_region_size, upper_lower_flag=None):
        """Same step decision as `ray_step` but also returns ∇f(ray_origin).
        Override of base default: extracts the gradient from the polynomial
        linear coefficients (free) instead of doing a separate `jax.grad` pass."""
        return self._ray_step_internals(
            params, ray_origin, ray_direction, trust_region_size, upper_lower_flag)
    

    def _box_polynomial(self, lower: jnp.ndarray, upper: jnp.ndarray):
        """One Taylor pass over the AABB; reused by classify_box / box_bounds / box_*_bounds."""
        center = 0.5 * (lower + upper)
        half_extent = 0.5 * (upper - lower)
        # taylor_bounds expects trust_region_size such that the box is [center - tr/2, center + tr/2],
        # so tr = 2 * half_extent (per-axis). half_extent returned for downstream use.
        auto_bound = self.taylor_bounds(center, x_trust_region=None,
                                        trust_region_size=2.0 * half_extent)
        return auto_bound.polynomial_enclosure, half_extent


    def classify_box(self, params, lower, upper):
        """Native AABB classification via the autobound polynomial enclosure.

        Tighter than the default empty-sphere fallback: the trust region is per-axis,
        matching the box exactly, instead of the circumscribed sphere.
        """
        poly, half_extent = self._box_polynomial(lower, upper)
        lo, hi = _poly_box_value_bounds(poly, half_extent, self.range_implementation)
        lo = lo - self.level_set
        hi = hi - self.level_set
        return jnp.where(hi < 0, jnp.int8(-1),
                jnp.where(lo > 0, jnp.int8(1), jnp.int8(0)))


    def box_bounds(self, params, lower, upper):
        poly, half_extent = self._box_polynomial(lower, upper)
        return _poly_box_value_bounds(poly, half_extent, self.range_implementation)


    def box_grad_bounds(self, params, lower, upper):
        """Per-axis (lo[d], hi[d]) bound on ∇f over the box. Requires autobounds_max_degree >= 2."""
        if self.autobounds_max_degree < 2:
            raise ValueError(
                f"box_grad_bounds requires autobounds_max_degree >= 2 "
                f"(got {self.autobounds_max_degree})"
            )
        poly, half_extent = self._box_polynomial(lower, upper)
        return _poly_box_grad_bounds(poly, dim=self.dim, half_extent=half_extent,
                                     implementation=self.range_implementation)


    def box_hessian_bounds(self, params, lower, upper):
        """(lo[d, d], hi[d, d]) bound on Hessian over the box. Requires autobounds_max_degree >= 3."""
        if self.autobounds_max_degree < 3:
            raise ValueError(
                f"box_hessian_bounds requires autobounds_max_degree >= 3 "
                f"(got {self.autobounds_max_degree})"
            )
        poly, half_extent = self._box_polynomial(lower, upper)
        return _poly_box_hessian_bounds(poly, dim=self.dim, half_extent=half_extent,
                                        implementation=self.range_implementation)


    def classify_subcubes(self, params, parent_lower, parent_upper, sub_lowers, sub_uppers):
        """One Taylor pass on the parent classifies all N sub-boxes via polynomial range.

        N MLP forward passes → 1. The parent polynomial is provably valid over the parent
        box and any subset thereof, so each sub-box's (lo, hi) is computed by polynomial
        interval evaluation, no extra autobound trace.
        """
        poly, _half_extent = self._box_polynomial(parent_lower, parent_upper)
        parent_center = 0.5 * (parent_lower + parent_upper)
        delta_lowers = sub_lowers - parent_center[None, :]
        delta_uppers = sub_uppers - parent_center[None, :]

        lo, hi = _poly_subboxes_bounds(poly, delta_lowers, delta_uppers,
                                       self.range_implementation)
        lo = lo - self.level_set
        hi = hi - self.level_set
        return jnp.where(hi < 0, jnp.int8(-1),
                jnp.where(lo > 0, jnp.int8(1), jnp.int8(0)))


    def empty_sphere(self, params: jnp.ndarray, x: jnp.ndarray, trust_region_size: jnp.ndarray, upper_lower_flag: Optional[jnp.ndarray]=None) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """
        Largest provably-empty sphere around `x` within the trust region.

        Returns (fn_val, safe_radius, next_trust_region_size) matching NIQ's
        `empty_sphere` and `WoSIntegrator.step`'s contract:
          * fn_val: f(x) − level_set
          * safe_radius: largest empty-sphere radius, clamped to TR/2
          * next_tr: TR · grow_factor when the polynomial-derived radius fills
            the TR (surface provably outside TR), else TR · shrink_factor.
            Clipped to [min_tr, max_tr].
        """
        auto_bound = self.taylor_bounds(x, x_trust_region=None, trust_region_size=trust_region_size)
        taylor_enclosure = auto_bound.polynomial_enclosure
        fn_val = taylor_enclosure[0].squeeze()

        if upper_lower_flag is None:
            upper_lower_flag = jnp.where(fn_val.squeeze() > self.level_set, jnp.array(-1), jnp.array(1)) # stay on same side of level set
        radius = empty_sphere_2nd_order_tensor(taylor_enclosure, upper_lower_flag, dim=x.shape[0], isolevel=self.level_set).squeeze()
        # Sanitize: empty_sphere_2nd_order_tensor can return NaN/negative for
        # degenerate polynomial cases (root-find on near-flat coefs). Treat those
        # as "no usable radius" → 0; downstream logic shrinks tr next step.
        radius = jnp.where(jnp.isfinite(radius), radius, jnp.asarray(0.0, dtype=radius.dtype))
        radius = jnp.maximum(radius, jnp.asarray(0.0, dtype=radius.dtype))
        # Surface provably outside TR ⇔ radius reaches the TR boundary (= TR/2).
        outside_trust_region = radius > trust_region_size * 0.5
        safe_radius = jnp.where(outside_trust_region, trust_region_size * 0.5, radius)
        # NIQ-style adaptive TR update.
        next_tr = jnp.where(outside_trust_region,
                            trust_region_size * self._grow,
                            trust_region_size * self._shrink)
        next_tr = jnp.clip(next_tr, self._min_tr, self._max_tr)
        return fn_val, safe_radius, next_tr
