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
import jnp_bounds.autobounds as autobounds

from .base import implicitSurface
from jnp_bounds import interval_arithmetic as interval_arithmetic
from jnp_bounds import mixed_polynomials_arithmetic as polynomial_arithmetic
from jnp_bounds.polynomial_roots import RootStatus, envelope_roots, first_root
from utils.jnpbound_empty_sphere import empty_sphere_2nd_order_tensor


def _scalar(value):
    """Normalize only the scalar field output, preserving query batch axes elsewhere."""
    value = jnp.asarray(value)
    if value.size != 1:
        raise ValueError(f"Expected one scalar field value, got {value.shape}")
    return value.reshape(())


def _scalar_pair(value):
    return tuple(_scalar(x) for x in interval_arithmetic.endpoints(value))


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

    def __init__(
        self,
        autobounds_max_degree: int,
        use_classify_subcubes: bool = True,
        use_warp: bool = True,
        grow_factor: float = 1.5,
        shrink_factor: float = 0.5,
        max_trust_region_size: float = float("inf"),
        min_trust_region_size: float = 1e-6,
        **kwargs,
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
        self._derivative_builders = {}
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
        self.taylor_bounds = autobounds.taylor_bounds(
            self.fn,
            max_degree=self.autobounds_max_degree,
            composition_algorithm="softplus_cheap",
            composition_implementation=composition_implementation,
        )

    def _derivative_builder(self, derivative_axes):
        # Scalar forward derivatives avoid the bound walk's unsupported
        # Jacobian batch axis. Builders are cached independently of parameters.
        if derivative_axes not in self._derivative_builders:
            differentiated_field = lambda params, x: jnp.asarray(self.fn(params, x)).reshape(())

            def make_directional_derivative(previous_derivative, derivative_axis):
                def directional_derivative(params, x):
                    coordinate_direction = jnp.asarray(
                        [int(input_axis == derivative_axis) for input_axis in range(self.dim)],
                        x.dtype,
                    )
                    return jax.jvp(
                        lambda y: previous_derivative(params, y), (x,), (coordinate_direction,)
                    )[1]

                return directional_derivative

            for derivative_axis in derivative_axes:
                differentiated_field = make_directional_derivative(
                    differentiated_field, derivative_axis
                )
            directional_derivative = differentiated_field

            def derivative_function(params, x):
                return directional_derivative(params, x)

            self._derivative_builders[derivative_axes] = autobounds.taylor_bounds(
                derivative_function,
                max_degree=self.autobounds_max_degree,
                composition_implementation="jax",
            )
        return self._derivative_builders[derivative_axes]

    def _function_derivative_bounds(self, params, lower, upper, order):
        box_center = (lower + upper) * 0.5
        box_half_extents = (upper - lower) * 0.5
        derivative_axis_combinations = (
            [(derivative_axis,) for derivative_axis in range(self.dim)]
            if order == 1
            else [
                (first_derivative_axis, second_derivative_axis)
                for first_derivative_axis in range(self.dim)
                for second_derivative_axis in range(first_derivative_axis, self.dim)
            ]
        )
        derivative_ranges = {}
        for derivative_axes in derivative_axis_combinations:
            try:
                derivative_enclosure = self._derivative_builder(derivative_axes)(
                    params, box_center, trust_region_size=upper - lower
                )
            except (NotImplementedError, TypeError) as exc:
                raise NotImplementedError(
                    "Function derivative bounds require supported derivative primitives; "
                    "differentiating the value enclosure is not a valid fallback"
                ) from exc
            derivative_ranges[derivative_axes] = _scalar_pair(
                polynomial_arithmetic.range_over_box(
                    derivative_enclosure.polynomial_enclosure,
                    -box_half_extents,
                    box_half_extents,
                    implementation=self.range_implementation,
                )
            )
        if order == 1:
            return tuple(
                jnp.stack(
                    [
                        derivative_ranges[(first_derivative_axis,)][endpoint_index]
                        for first_derivative_axis in range(self.dim)
                    ]
                )
                for endpoint_index in (0, 1)
            )
        return tuple(
            jnp.stack(
                [
                    jnp.stack(
                        [
                            derivative_ranges[
                                tuple(sorted((first_derivative_axis, second_derivative_axis)))
                            ][endpoint_index]
                            for second_derivative_axis in range(self.dim)
                        ]
                    )
                    for first_derivative_axis in range(self.dim)
                ]
            )
            for endpoint_index in (0, 1)
        )

    def _ray_step_internals(
        self, params, ray_origin, ray_direction, trust_region_size, upper_lower_flag=None
    ):
        """Single Taylor pass; returns everything `ray_step` and `ray_step_with_grad`
        need. The grad output is a slice of the polynomial (free byproduct of the
        Taylor pass). When called via `ray_step` the slice is unused → JIT
        dead-code-eliminates it; no runtime cost."""
        SAFETY_FACTOR = 0.98  # mirror NIQ's cast_rays_iter
        trust_region_half_width = 0.5 * trust_region_size

        trust_region_widths = jnp.broadcast_to(trust_region_size, ray_origin.shape)
        taylor_bounds_result = self.taylor_bounds(
            params, ray_origin, x_trust_region=None, trust_region_size=trust_region_widths
        )
        polynomial_enclosure = taylor_bounds_result.polynomial_enclosure

        # Fixed constant coefficient at the Taylor center.
        field_residual_at_origin = (
            _scalar(polynomial_arithmetic.coefficient(polynomial_enclosure, (0,) * self.dim))
            - self.level_set
        )

        # Fixed linear Taylor coefficients give the center gradient for degree >=2.
        if self.autobounds_max_degree < 2:
            raise ValueError("ray queries with center gradients require Taylor degree >=2")
        field_gradient = jnp.stack(
            [
                _scalar(
                    polynomial_arithmetic.coefficient(
                        polynomial_enclosure,
                        tuple(int(input_axis == derivative_axis) for input_axis in range(self.dim)),
                    )
                )
                for derivative_axis in range(self.dim)
            ]
        )

        # Classify-first guard: bound the polynomial over the full TR. If decided
        # (hi < level OR lo > level), the surface cannot lie inside → take the full
        # forward exit step without root-finding, no shrink.
        trust_region_half_extents = trust_region_half_width * jnp.ones_like(ray_origin)
        trust_region_value_lower_bound, trust_region_value_upper_bound = _scalar_pair(
            polynomial_arithmetic.range_over_box(
                polynomial_enclosure,
                -trust_region_half_extents,
                trust_region_half_extents,
                implementation=self.range_implementation,
            )
        )
        trust_region_excludes_surface = jnp.logical_or(
            trust_region_value_upper_bound - self.level_set < 0,
            trust_region_value_lower_bound - self.level_set > 0,
        )

        if upper_lower_flag is None:
            upper_lower_flag = jnp.where(
                field_residual_at_origin > 0,
                jnp.array(-1),
                jnp.array(1),
            )

        # The library owns interval-envelope construction, solving and root filtering.
        maximum_direction_component = jnp.max(jnp.abs(ray_direction))
        ray_exit_parameter = trust_region_half_width / (maximum_direction_component + 1e-12)
        envelope_root_result = envelope_roots(
            polynomial_arithmetic.restrict_to_line(polynomial_enclosure, ray_direction),
            side=upper_lower_flag,
            domain=(0.0, ray_exit_parameter),
            level=self.level_set,
        )
        has_envelope_root, envelope_root_parameter = first_root(envelope_root_result)
        has_envelope_root, envelope_root_parameter = (
            _scalar(has_envelope_root),
            _scalar(envelope_root_parameter),
        )
        root_solving_failed = _scalar(
            (envelope_root_result.status == RootStatus.INVALID_INPUT)
            | (envelope_root_result.status == RootStatus.NUMERICAL_FAILURE)
        )

        step_requires_smaller_region = jnp.logical_and(
            jnp.logical_not(trust_region_excludes_surface),
            jnp.logical_and(has_envelope_root, envelope_root_parameter < ray_exit_parameter),
        )
        ray_step_parameter = jnp.where(
            step_requires_smaller_region, envelope_root_parameter, ray_exit_parameter
        )
        ray_step_parameter = jnp.where(root_solving_failed, 0.0, SAFETY_FACTOR * ray_step_parameter)
        step_requires_smaller_region = step_requires_smaller_region | root_solving_failed

        ray_step_parameter = cast(jnp.ndarray, jnp.squeeze(ray_step_parameter))
        step_requires_smaller_region = jnp.squeeze(step_requires_smaller_region)

        return (
            field_residual_at_origin,
            ray_step_parameter,
            jnp.logical_not(step_requires_smaller_region),
            field_gradient,
        )

    def ray_step(self, params, ray_origin, ray_direction, trust_region_size, upper_lower_flag=None):
        """Sphere-trace step with the Taylor centre at ray_origin. Returns
        (field_residual, ray_step_parameter, uses_trust_region_exit) — the gradient byproduct of the Taylor pass is
        dropped here; JIT dead-code-eliminates the unused slice. See class
        docstring for full step-decision semantics."""
        field_residual, ray_step_parameter, uses_trust_region_exit, _field_gradient = (
            self._ray_step_internals(
                params, ray_origin, ray_direction, trust_region_size, upper_lower_flag
            )
        )
        return field_residual, ray_step_parameter, uses_trust_region_exit

    def ray_step_with_grad(
        self, params, ray_origin, ray_direction, trust_region_size, upper_lower_flag=None
    ):
        """Same step decision as `ray_step` but also returns ∇f(ray_origin).
        Override of base default: extracts the gradient from the polynomial
        linear coefficients (free) instead of doing a separate `jax.grad` pass."""
        return self._ray_step_internals(
            params, ray_origin, ray_direction, trust_region_size, upper_lower_flag
        )

    def _box_polynomial(self, params, lower: jnp.ndarray, upper: jnp.ndarray):
        """One Taylor pass over the AABB; reused by classify_box / box_bounds / box_*_bounds."""
        box_center = 0.5 * (lower + upper)
        box_half_extents = 0.5 * (upper - lower)
        # taylor_bounds expects trust_region_size such that the box is [center - tr/2, center + tr/2],
        # so tr = 2 * half_extent (per-axis). half_extent returned for downstream use.
        taylor_bounds_result = self.taylor_bounds(
            params, box_center, x_trust_region=None, trust_region_size=2.0 * box_half_extents
        )
        return taylor_bounds_result.polynomial_enclosure, box_half_extents

    def classify_box(self, params, lower, upper):
        """Native AABB classification via the autobound polynomial enclosure.

        Tighter than the default empty-sphere fallback: the trust region is per-axis,
        matching the box exactly, instead of the circumscribed sphere.
        """
        polynomial_enclosure, box_half_extents = self._box_polynomial(params, lower, upper)
        field_lower_bound, field_upper_bound = _scalar_pair(
            polynomial_arithmetic.range_over_box(
                polynomial_enclosure,
                -box_half_extents,
                box_half_extents,
                implementation=self.range_implementation,
            )
        )
        field_lower_bound = field_lower_bound - self.level_set
        field_upper_bound = field_upper_bound - self.level_set
        return jnp.where(
            field_upper_bound < 0,
            jnp.int8(-1),
            jnp.where(field_lower_bound > 0, jnp.int8(1), jnp.int8(0)),
        )

    def box_bounds(self, params, lower, upper):
        polynomial_enclosure, box_half_extents = self._box_polynomial(params, lower, upper)
        return _scalar_pair(
            polynomial_arithmetic.range_over_box(
                polynomial_enclosure,
                -box_half_extents,
                box_half_extents,
                implementation=self.range_implementation,
            )
        )

    def box_grad_bounds(self, params, lower, upper):
        """Bound the gradient function independently over the box."""
        return self._function_derivative_bounds(params, lower, upper, order=1)

    def box_hessian_bounds(self, params, lower, upper):
        """Bound the Hessian function independently over the box."""
        return self._function_derivative_bounds(params, lower, upper, order=2)

    def classify_subcubes(self, params, parent_lower, parent_upper, sub_lowers, sub_uppers):
        """One Taylor pass on the parent classifies all N sub-boxes via polynomial range.

        N MLP forward passes → 1. The parent polynomial is provably valid over the parent
        box and any subset thereof, so each sub-box's (lo, hi) is computed by polynomial
        interval evaluation, no extra autobound trace.
        """
        polynomial_enclosure, _parent_half_extents = self._box_polynomial(
            params, parent_lower, parent_upper
        )
        parent_box_center = 0.5 * (parent_lower + parent_upper)
        subbox_lower_offsets = sub_lowers - parent_box_center[None, :]
        subbox_upper_offsets = sub_uppers - parent_box_center[None, :]

        value = polynomial_arithmetic.range_over_box(
            polynomial_enclosure,
            subbox_lower_offsets,
            subbox_upper_offsets,
            implementation=self.range_implementation,
            batched=True,
        )
        field_lower_bound, field_upper_bound = (
            x.reshape(subbox_lower_offsets.shape[0]) for x in interval_arithmetic.endpoints(value)
        )
        field_lower_bound = field_lower_bound - self.level_set
        field_upper_bound = field_upper_bound - self.level_set
        return jnp.where(
            field_upper_bound < 0,
            jnp.int8(-1),
            jnp.where(field_lower_bound > 0, jnp.int8(1), jnp.int8(0)),
        )

    def empty_sphere(
        self,
        params: jnp.ndarray,
        x: jnp.ndarray,
        trust_region_size: jnp.ndarray,
        upper_lower_flag: Optional[jnp.ndarray] = None,
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """
        Largest provably-empty sphere around `x` within the trust region.

        Returns (field_residual, safe_radius, next_trust_region_size) matching NIQ's
        `empty_sphere` and `WoSIntegrator.step`'s contract:
          * field_residual: f(x) − level_set
          * safe_radius: largest empty-sphere radius, clamped to TR/2
          * next_trust_region_size: TR · grow_factor when the polynomial-derived radius fills
            the TR (surface provably outside TR), else TR · shrink_factor.
            Clipped to [min_tr, max_tr].
        """
        if self.autobounds_max_degree != 2:
            raise NotImplementedError("empty_sphere currently requires a degree-2 value enclosure")
        taylor_bounds_result = self.taylor_bounds(
            params, x, x_trust_region=None, trust_region_size=trust_region_size
        )
        polynomial_enclosure = taylor_bounds_result.polynomial_enclosure
        field_value_at_center = _scalar(
            polynomial_arithmetic.coefficient(polynomial_enclosure, (0,) * self.dim)
        )

        if upper_lower_flag is None:
            upper_lower_flag = jnp.where(
                field_value_at_center.squeeze() > self.level_set, jnp.array(-1), jnp.array(1)
            )  # stay on same side of level set
        radius = empty_sphere_2nd_order_tensor(
            polynomial_enclosure, upper_lower_flag, dim=x.shape[0], isolevel=self.level_set
        ).squeeze()
        # Sanitize: empty_sphere_2nd_order_tensor can return NaN/negative for
        # degenerate polynomial cases (root-find on near-flat coefs). Treat those
        # as "no usable radius" → 0; downstream logic shrinks tr next step.
        radius = jnp.where(jnp.isfinite(radius), radius, jnp.asarray(0.0, dtype=radius.dtype))
        radius = jnp.maximum(radius, jnp.asarray(0.0, dtype=radius.dtype))
        # Surface provably outside TR ⇔ radius reaches the TR boundary (= TR/2).
        outside_trust_region = radius > trust_region_size * 0.5
        safe_radius = jnp.where(outside_trust_region, trust_region_size * 0.5, radius)
        # NIQ-style adaptive TR update.
        next_trust_region_size = jnp.where(
            outside_trust_region, trust_region_size * self._grow, trust_region_size * self._shrink
        )
        next_trust_region_size = jnp.clip(next_trust_region_size, self._min_tr, self._max_tr)
        return field_value_at_center - self.level_set, safe_radius, next_trust_region_size
