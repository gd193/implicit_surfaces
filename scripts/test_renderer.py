"""CPU renderer regressions: run directly or through unittest discovery."""
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
from rayMarcher.integrator import Integrator
from rayMarcher.materials import Material
from rayMarcher.marcher import SimpleIntegrator
from rayMarcher.sampler import generate_antialiased_ray
from rayMarcher.utils import validate_scene_bounds
from utils.helpers import geometric_termination_criterion, level_set_termination_criterion
from utils.types import RayMarchState


def make_camera(focus_distance: float = 1.0) -> Camera:
    return Camera.setup_Camera(
        16, 8, 60.0, jnp.zeros(3), jnp.array([0., 0., 1.]),
        jnp.array([0., 1., 0.]), focus_distance=focus_distance,
    )


class RendererTests(unittest.TestCase):
    def test_background_is_independent_of_batch_size(self) -> None:
        def miss_batch(origins, directions):
            count = origins.shape[0]
            return (jnp.zeros(count, dtype=bool), origins + directions,
                    jnp.zeros(count, dtype=jnp.int32))

        integrator = Integrator(
            make_camera(), None, jnp.array([0.5, 0.7, 1.]),
            None, miss_batch, [], [lambda x: jnp.array([0., 0., 1.])], 1, 1,
        )
        pixel_i = jnp.array([0, 7, 10, 15])
        pixel_j = jnp.array([0, 3, 5, 7])
        keys = jax.random.split(jax.random.PRNGKey(4), 4)
        batched, _ = integrator.render_pixels_simple(pixel_i, pixel_j, keys)
        for index in range(4):
            single, _ = integrator.render_pixels_simple(
                pixel_i[index:index+1], pixel_j[index:index+1], keys[index:index+1])
            np.testing.assert_allclose(batched[index], single[0], atol=1e-6)
            _, direction = generate_antialiased_ray(
                integrator.camera, pixel_i[index], pixel_j[index], keys[index])
            expected = integrator.compute_background_contribution(
                jnp.asarray(False), jnp.asarray(False), direction)
            np.testing.assert_allclose(batched[index], expected, atol=1e-6)

    def test_sphere_trace_direction_scale(self) -> None:
        origin = jnp.array([0., 0., 3.])
        directions = jnp.array([[0., 0., -scale] for scale in (0.25, 1., 2., 4.)])
        for bounds in (None, ((-2., -2., -2.), (2., 2., 2.))):
            with self.subTest(bounds=bounds):
                integrator = Integrator(
                    make_camera(), None, jnp.ones(3), None, None,
                    [lambda x: jnp.linalg.norm(x) - 1.],
                    [lambda x: x / jnp.linalg.norm(x)],
                    1, 1, scene_bounds=bounds,
                )
                trace = jax.jit(jax.vmap(integrator.sphere_trace, in_axes=(None, 0)))
                hits, points, _, normals = trace(origin, directions)
                np.testing.assert_array_equal(hits, jnp.ones(4, dtype=bool))
                expected = jnp.tile(jnp.array([0., 0., 1.]), (4, 1))
                np.testing.assert_allclose(points, expected, atol=1e-6)
                np.testing.assert_allclose(normals, expected, atol=1e-6)
                hit, point, _, _ = integrator.sphere_trace(origin, jnp.zeros(3))
                self.assertFalse(bool(hit))
                np.testing.assert_allclose(point, origin)

    def test_scene_bounds(self) -> None:
        bounds = ((-1., -1., -1.), (1., 1., 1.))
        # Outside entry, away, parallel miss, parallel on face, inside exit,
        # a hit at the exit face, custom translated bounds, and clipping off.
        cases = [
            ((0., 0., 3.), (0., 0., -1.), 0., bounds, True),
            ((0., 0., 3.), (0., 0., 1.), 4., bounds, False),
            ((2., 0., 0.), (0., 0., 1.), 0.5, bounds, False),
            ((1., 0., 0.), (0., 0., 1.), 0.5, bounds, True),
            ((0., 0., 0.), (0., 0., 1.), 3., bounds, False),
            ((0., 0., 0.), (0., 0., 1.), 1., bounds, True),
            ((0., 0., 0.), (0., 0., 1.), 5., ((-1., -1., 4.), (1., 1., 6.)), True),
            ((0., 0., 0.), (0., 0., 1.), 3., None, True),
        ]
        for origin, direction, plane_z, box, expected_hit in cases:
            origin, direction = jnp.array(origin), jnp.array(direction)

            def distance(x):
                return jnp.abs(x[2] - plane_z)

            def normal(x):
                return jnp.array([0., 0., 1.])

            with self.subTest(origin=origin, direction=direction, box=box):
                fallback = Integrator(
                    make_camera(), None, jnp.ones(3), None, None,
                    [distance], [normal], 1, 1, scene_bounds=box,
                )
                hit, point, _, _ = fallback.sphere_trace(origin, direction)
                self.assertEqual(bool(hit), expected_hit)
                if expected_hit:
                    self.assertAlmostEqual(float(point[2]), plane_z)
                for compactions in (0, 4):
                    marcher = SimpleIntegrator(
                        make_camera(), jnp.ones(3),
                        lambda p, x, d, tr: (distance(x), distance(x), jnp.asarray(True)),
                        lambda p, x: normal(x),
                        image_height=2, image_width=2, max_ray_march_steps=8,
                        max_ray_distance=20., level_set=0., epsilon=1e-5,
                        n_compactions=compactions, scene_bounds=box,
                    )
                    marcher.sample_camera_rays = lambda *args: (
                        jnp.tile(origin, (4, 1)), jnp.tile(direction, (4, 1)))
                    image, _ = marcher.render(None, jax.random.PRNGKey(0))
                    expected = [0.5, 0.5, 1.] if expected_hit else [1., 1., 1.]
                    np.testing.assert_allclose(image, jnp.broadcast_to(jnp.array(expected), image.shape))

    def test_invalid_scene_bounds(self) -> None:
        for bounds in (((0., 0., 0.), (0., 1., 1.)),
                       ((0., 0.), (1., 1.)),
                       ((0., 0., 0.), (1., 1., float('inf')))):
            with self.subTest(bounds=bounds), self.assertRaises(ValueError):
                validate_scene_bounds(bounds)

    def test_fallback_sphere_trace_step_cap(self) -> None:
        integrator = Integrator(
            make_camera(),
            Material(jnp.array([0]), jnp.ones((1, 3)), jnp.zeros(1), jnp.ones(1)),
            jnp.array([0.5, 0.7, 1.0]), None, None,
            [lambda x: x[1]], [lambda x: jnp.array([0., 1., 0.])],
            samples_per_pixel=1, max_depth=1, max_sphere_trace_steps=3,
            scene_bounds=None,
        )
        origin = jnp.array([0., 1., 0.])
        direction = jnp.array([1., 0., 0.])
        hit, point, _, _ = integrator.sphere_trace(origin, direction)
        self.assertFalse(bool(hit))
        np.testing.assert_allclose(point, [3., 1., 0.])
        color = integrator.trace_ray(origin, direction, jax.random.PRNGKey(0))
        np.testing.assert_allclose(color, [0.75, 0.85, 1.0])
        # Ordinary hits and the existing SDF-distance escape still work.
        hit, point, _, _ = integrator.sphere_trace(origin, jnp.array([0., -1., 0.]))
        self.assertTrue(bool(hit))
        np.testing.assert_allclose(point, jnp.zeros(3))
        hit, _, _, _ = integrator.sphere_trace(jnp.array([0., 101., 0.]), direction)
        self.assertFalse(bool(hit))

    def test_plane_hits_and_distance_limit(self) -> None:
        # f(z) = 2-z; each callback follows the surface protocol: f-level_set.
        for criterion in (level_set_termination_criterion, geometric_termination_criterion):
            for level in (0.0, 1.0, -1.0):
                distance = 2.0 - level

                def ray_step(params, position, direction, trust_region):
                    residual = 2.0 - position[2] - level
                    return residual, jnp.abs(residual), jnp.asarray(True)

                def normal(params, position):
                    return jnp.array([0., 0., -1.])

                def ray_step_with_grad(*args):
                    return (*ray_step(*args), normal(None, None))

                for limit in (distance / 2, distance, distance + 1):
                    with self.subTest(criterion=criterion.__name__, level=level, limit=limit):
                        integrator = SimpleIntegrator(
                            make_camera(), jnp.ones(3), ray_step, normal,
                            image_height=1, image_width=1, max_ray_march_steps=4,
                            max_ray_distance=limit, level_set=level, epsilon=1e-5,
                            n_compactions=0, termination_crit_fn=criterion,
                            ray_march_with_grad_fn=ray_step_with_grad,
                            scene_bounds=None,
                        )
                        state = RayMarchState(
                            niter=jnp.zeros(1, dtype=jnp.int32), t=jnp.zeros(1),
                            ray_origin=jnp.zeros((1, 3)),
                            ray_direction=jnp.array([[0., 0., 1.]]),
                            trust_region_size=jnp.ones(1),
                            done_flag=jnp.zeros(1, dtype=bool),
                            hit_flag=jnp.zeros(1, dtype=bool),
                        )
                        for _ in range(6):
                            state = integrator._step_states_fn(None, state)
                            self.assertLessEqual(float(state.t[0]), limit)
                            if bool(state.done_flag[0]):
                                break
                        self.assertTrue(bool(state.done_flag[0]))
                        self.assertEqual(bool(state.hit_flag[0]), distance <= limit)
                        self.assertAlmostEqual(float(state.t[0]), min(distance, limit))
                        # Compaction can leave finished rays in a batch.
                        repeated = integrator._step_states_fn(None, state)
                        for before, after in zip(state, repeated):
                            np.testing.assert_array_equal(before, after)

    def test_focus_distance_preserves_pinhole_rays(self) -> None:
        key = jax.random.PRNGKey(3)
        reference = make_camera()
        for focus in (0.5, 2.0, 10.0):
            camera = make_camera(focus)
            for pixel in ((0, 0), (8, 4), (15, 7)):
                with self.subTest(focus=focus, pixel=pixel):
                    origin, direction = generate_antialiased_ray(camera, *pixel, key)
                    expected_origin, expected_direction = generate_antialiased_ray(reference, *pixel, key)
                    np.testing.assert_allclose(origin, expected_origin, atol=1e-6)
                    np.testing.assert_allclose(direction, expected_direction, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
