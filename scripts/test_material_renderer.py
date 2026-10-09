"""CPU integration tests for material rendering through implicit surface queries."""
import os
import sys
import unittest
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import jax
import jax.numpy as jnp
import numpy as np

from rayMarcher.camera import Camera
from rayMarcher.integrator import MaterialIntegrator
from rayMarcher.materials import Material
from surfaces.implicit.lipschitz import sdfImplicit, lipschitzImplicit
from utils.helpers import geometric_termination_criterion


def camera():
    return Camera.setup_Camera(4, 4, 40., jnp.array([0., 0., 3.]),
                               jnp.zeros(3), jnp.array([0., 1., 0.]))


def sphere():
    return sdfImplicit(fn=lambda p, x: jnp.linalg.norm(x - p["center"]) - p["radius"], dim=3)


def params(center=(0., 0., 0.), radius=1.):
    return {"center": jnp.array(center), "radius": jnp.array(radius)}


def materials(types):
    count = len(types)
    return Material(jnp.array(types), jnp.tile(jnp.array([0.8, 0.6, 0.4]), (count, 1)),
                    jnp.zeros(count), jnp.ones(count))


class MaterialRendererTests(unittest.TestCase):
    def test_nearest_hit_matches_simple_queries_and_updates_params(self):
        surfaces = [sphere(), sphere()]
        ps = [params((0., 0., -2.), 0.5), params()]
        origins = jnp.array([[0., 0., 3.], [3., 0., 3.], [0., 0., 3.]])
        directions = jnp.array([[0., 0., -2.], [0., 0., -1.], [0., 0., -1.]])
        active = jnp.array([True, True, False])
        for compact in (0, 4):
            renderer = MaterialIntegrator(camera(), materials([0, 1]), jnp.ones(3), surfaces,
                                          epsilon=1e-5, n_compactions=compact)
            hit = renderer.intersect(ps, origins, directions, active_mask=active)
            np.testing.assert_array_equal(hit.hit_flag, [True, False, False])
            self.assertEqual(int(hit.material_index[0]), 1)
            self.assertAlmostEqual(float(hit.distance[0]), 2., places=5)
            np.testing.assert_allclose(hit.ray_position[0], [0., 0., 1.], atol=1e-5)
            for index, marcher in enumerate(renderer.marchers):
                direct = marcher.trace_rays(ps[index], origins, directions, active_mask=active)
                if index == 1:
                    np.testing.assert_allclose(hit.ray_position[0], direct.ray_position[0], atol=1e-6)
            changed = renderer.intersect([ps[0], params(radius=0.5)], origins[:1], directions[:1])
            np.testing.assert_allclose(changed.ray_position[0], [0., 0., 0.5], atol=1e-5)

    def test_non_sdf_nonzero_level_and_gradient_termination(self):
        surface = lipschitzImplicit(fn=lambda p, x: jnp.dot(x, x),
                                    lipschitz_bound=14., level_set=1., dim=3)
        renderer = MaterialIntegrator(
            camera(), materials([0]), jnp.ones(3), [surface], epsilon=1e-4,
            termination_crit_fn=geometric_termination_criterion,
        )
        hit = renderer.intersect([None], jnp.array([[0., 0., 3.]]), jnp.array([[0., 0., -1.]]))
        self.assertTrue(bool(hit.hit_flag[0]))
        np.testing.assert_allclose(hit.ray_position[0], [0., 0., 1.], atol=2e-4)

    def test_diffuse_metal_and_glass_bounces(self):
        origin, direction = jnp.array([[0., 0., 3.]]), jnp.array([[0., 0., -1.]])
        for kind, interactions in ((0, 1), (1, 1), (2, 2)):
            with self.subTest(material=kind):
                renderer = MaterialIntegrator(camera(), materials([kind]), jnp.ones(3), [sphere()],
                                              epsilon=1e-5, max_depth=4, n_compactions=0)
                color = renderer.trace_paths([params()], origin, direction, jax.random.PRNGKey(5))
                # Index-matched glass passes through both sphere interfaces.
                expected = jnp.array([[0.8, 0.6, 0.4]]) ** interactions
                np.testing.assert_allclose(color, expected, atol=1e-5)

    def test_adaptive_zero_steps_and_iteration_limit(self):
        class AdaptivePlane:
            def ray_step(self, p, x, direction, tr):
                residual = 1. - x[2]
                safe = tr <= 0.25
                return residual, jnp.where(safe, jnp.abs(residual), 0.), safe

            def normal_at(self, p, x):
                return jnp.array([0., 0., -1.])

        for budget, expected_hit in ((1, False), (8, True)):
            renderer = MaterialIntegrator(camera(), materials([0]), jnp.ones(3), [AdaptivePlane()],
                                          max_ray_march_steps=budget)
            result = renderer.intersect([None], jnp.zeros((1, 3)), jnp.array([[0., 0., 1.]]), 1.)
            self.assertEqual(bool(result.hit_flag[0]), expected_hit)

    def test_render_and_camera_update(self):
        renderer = MaterialIntegrator(camera(), materials([0, 1, 2]), jnp.ones(3),
                                      [sphere(), sphere(), sphere()], samples_per_pixel=2, max_depth=3)
        ps = [params((-0.8, 0., 0.), 0.4), params((0., 0., 0.), 0.4), params((0.8, 0., 0.), 0.4)]
        image = renderer.render(ps, jax.random.PRNGKey(2))
        self.assertEqual(image.shape, (4, 4, 3))
        self.assertTrue(bool(jnp.all(jnp.isfinite(image))))
        self.assertTrue(bool(jnp.any(image < 0.95)))
        renderer.camera = Camera.setup_Camera(4, 4, 40., jnp.array([0., 0., 3.]),
                                              jnp.array([0., 0., 4.]), jnp.array([0., 1., 0.]))
        renderer.background_color = jnp.array([0.5, 0.7, 1.])
        missed = renderer.render(ps, jax.random.PRNGKey(2))
        self.assertTrue(bool(jnp.all(jnp.isfinite(missed))))
        np.testing.assert_allclose(missed[:, :, 2], 1., atol=1e-6)
        self.assertTrue(bool(jnp.all(missed[:, :, 0] > 0.5)))


if __name__ == "__main__":
    unittest.main()
