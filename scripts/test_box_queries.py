"""Autobounds migration integration: real builders, runtime params and surface queries."""

import os
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
import numpy as np

from surfaces.implicit.jnp_bounded import jnpboundedImplicit


def quadratic(params, position):
    quadratic_weights, linear_weights, field_offset = params
    return (
        jnp.dot(quadratic_weights, position**2) + jnp.dot(linear_weights, position) + field_offset
    )


class BoxQueryTests(unittest.TestCase):
    def setUp(self):
        self.params = (jnp.array([1.0, 2.0, 3.0]), jnp.array([4.0, 5.0, 6.0]), jnp.array(7.0))
        self.surface = jnpboundedImplicit(
            fn=quadratic, dim=3, level_set=0.0, autobounds_max_degree=2, use_warp=False
        )
        self.lower = jnp.array([0.4, -0.6, -0.1])
        self.upper = jnp.array([0.6, -0.4, 0.1])

    def test_box_and_parameter_updates(self):
        box_bounds_query = jax.jit(self.surface.box_bounds)
        field_lower_bound, field_upper_bound = box_bounds_query(self.params, self.lower, self.upper)
        points = jnp.stack(
            jnp.meshgrid(
                *[
                    jnp.linspace(axis_lower_bound, axis_upper_bound, 5)
                    for axis_lower_bound, axis_upper_bound in zip(self.lower, self.upper)
                ],
                indexing="ij",
            ),
            axis=-1,
        ).reshape(-1, 3)
        reference_field_values = jax.vmap(lambda position: quadratic(self.params, position))(points)
        self.assertLessEqual(float(field_lower_bound), float(reference_field_values.min()))
        self.assertGreaterEqual(float(field_upper_bound), float(reference_field_values.max()))
        shifted_parameters = (*self.params[:2], self.params[2] + 2.0)
        shifted_field_lower_bound, shifted_field_upper_bound = box_bounds_query(
            shifted_parameters, self.lower, self.upper
        )
        np.testing.assert_allclose(
            [
                shifted_field_lower_bound - field_lower_bound,
                shifted_field_upper_bound - field_upper_bound,
            ],
            [2.0, 2.0],
            atol=2e-6,
        )
        self.assertEqual(int(self.surface.classify_box(self.params, self.lower, self.upper)), 1)
        self.assertEqual(
            int(self.surface.classify_box(self.params, -jnp.ones(3) * 3, jnp.ones(3) * 3)), 0
        )
        if hasattr(box_bounds_query, "_cache_size"):
            self.assertEqual(box_bounds_query._cache_size(), 1)

    def test_subboxes_share_one_parent_bound(self):
        subbox_lower_corners = jnp.stack([self.lower, (self.lower + self.upper) / 2])
        subbox_upper_corners = jnp.stack([(self.lower + self.upper) / 2, self.upper])
        enclosure_build_calls = []
        build_taylor_enclosure = self.surface.taylor_bounds

        def record_enclosure_build(*args, **kwargs):
            enclosure_build_calls.append(1)
            return build_taylor_enclosure(*args, **kwargs)

        self.surface.taylor_bounds = record_enclosure_build
        np.testing.assert_array_equal(
            self.surface.classify_subcubes(
                self.params, self.lower, self.upper, subbox_lower_corners, subbox_upper_corners
            ),
            [1, 1],
        )
        self.assertEqual(len(enclosure_build_calls), 1)

    def test_actual_function_derivative_enclosures(self):
        for power in (3, 4):
            with self.subTest(power=power):
                surface = jnpboundedImplicit(
                    fn=lambda field_parameters, position: jnp.sum(
                        field_parameters * position**power
                    ),
                    dim=1,
                    level_set=0.0,
                    autobounds_max_degree=2,
                    use_warp=False,
                )
                for derivative_order, derivative_bounds_query in [
                    (1, surface.box_grad_bounds),
                    (2, surface.box_hessian_bounds),
                ]:
                    derivative_lower_bound, derivative_upper_bound = jax.jit(
                        derivative_bounds_query
                    )(jnp.ones(1), -jnp.ones(1), jnp.ones(1))
                    position = jnp.linspace(-1.0, 1.0, 17)
                    reference_derivative_values = (
                        power * position ** (power - 1)
                        if derivative_order == 1
                        else power * (power - 1) * position ** (power - 2)
                    )
                    self.assertLessEqual(
                        float(derivative_lower_bound.reshape(())),
                        float(reference_derivative_values.min()),
                    )
                    self.assertGreaterEqual(
                        float(derivative_upper_bound.reshape(())),
                        float(reference_derivative_values.max()),
                    )

    def test_ray_and_empty_sphere_nonzero_level(self):
        surface = jnpboundedImplicit(
            fn=lambda radius, position: jnp.sum(position**2) - radius,
            dim=3,
            level_set=0.25,
            autobounds_max_degree=2,
            use_warp=False,
        )
        params, ray_origin, ray_direction, trust_region_size = (
            jnp.array(0.75),
            jnp.array([2.0, 0.0, 0.0]),
            jnp.array([-1.0, 0.0, 0.0]),
            jnp.array(4.0),
        )
        field_residual, ray_step_parameter, uses_trust_region_exit, field_gradient = jax.jit(
            surface.ray_step_with_grad
        )(params, ray_origin, ray_direction, trust_region_size)
        np.testing.assert_allclose(field_residual, 3.0)
        np.testing.assert_allclose(field_gradient, [4.0, 0.0, 0.0])
        np.testing.assert_allclose(ray_step_parameter, 0.98, atol=2e-6)
        self.assertFalse(bool(uses_trust_region_exit))
        empty_sphere_field_residual, radius, next_trust_region_size = jax.jit(surface.empty_sphere)(
            params, ray_origin, trust_region_size
        )
        np.testing.assert_allclose(empty_sphere_field_residual, field_residual)
        self.assertTrue(np.isfinite(radius) and 0 <= radius <= trust_region_size / 2)
        self.assertTrue(np.isfinite(next_trust_region_size) and next_trust_region_size > 0)
        np.testing.assert_allclose(
            jax.jit(jax.vmap(surface.ray_step, in_axes=(None, 0, 0, None)))(
                params,
                jnp.stack([ray_origin, ray_origin]),
                jnp.stack([ray_direction, ray_direction]),
                trust_region_size,
            )[1],
            [0.98, 0.98],
            atol=2e-6,
        )

    def test_unsupported_derivatives_do_not_use_value_polynomial(self):
        surface = jnpboundedImplicit(
            fn=lambda field_parameters, position: jnp.sum(field_parameters * position * position),
            dim=1,
            level_set=0.0,
            autobounds_max_degree=2,
            use_warp=False,
        )
        with self.assertRaisesRegex(NotImplementedError, "not a valid fallback"):
            surface.box_hessian_bounds(jnp.ones(1), -jnp.ones(1), jnp.ones(1))


if __name__ == "__main__":
    unittest.main()
