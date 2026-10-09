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
from rayMarcher.integrator import MaterialIntegrator
from surfaces.implicit.lipschitz import sdfImplicit
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
        integrator = MaterialIntegrator(
            make_camera(),
            Material(jnp.array([0]), jnp.ones((1, 3)), jnp.zeros(1), jnp.ones(1)),
            jnp.array([0.5, 0.7, 1.]),
            [sdfImplicit(fn=lambda p, x: jnp.linalg.norm(x) - 1., dim=3)],
        )
        pixel_i = jnp.array([0, 7, 10, 15])
        pixel_j = jnp.array([0, 3, 5, 7])
        keys = jax.random.split(jax.random.PRNGKey(4), 4)
        origins, directions = jax.vmap(generate_antialiased_ray, in_axes=(None, 0, 0, 0))(
            integrator.camera, pixel_i, pixel_j, keys)
        origins = origins + jnp.array([0., 0., 3.])
        batched = integrator.trace_paths([None], origins, directions, keys[0])
        for index in range(4):
            single = integrator.trace_paths(
                [None], origins[index:index+1], directions[index:index+1], keys[0])
            np.testing.assert_allclose(batched[index], single[0], atol=1e-6)
            blend = 0.5 * (directions[index, 1] + 1.)
            expected = (1. - blend) + blend * integrator.background_color
            np.testing.assert_allclose(batched[index], expected, atol=1e-6)

    def test_ray_direction_scale(self) -> None:
        origin = jnp.array([0., 0., 3.])
        directions = jnp.array([[0., 0., -scale] for scale in (0.25, 1., 2., 4., 0.)])
        surface = sdfImplicit(fn=lambda p, x: jnp.linalg.norm(x) - 1., dim=3)
        for bounds in (None, ((-2., -2., -2.), (2., 2., 2.))):
            with self.subTest(bounds=bounds):
                integrator = MaterialIntegrator(
                    make_camera(),
                    Material(jnp.array([0]), jnp.ones((1, 3)), jnp.zeros(1), jnp.ones(1)),
                    jnp.ones(3), [surface], scene_bounds=bounds,
                )
                result = integrator.intersect([None], jnp.tile(origin, (5, 1)), directions)
                np.testing.assert_array_equal(result.hit_flag, [True, True, True, True, False])
                expected = jnp.tile(jnp.array([0., 0., 1.]), (4, 1))
                np.testing.assert_allclose(result.ray_position[:4], expected, atol=1e-6)
                np.testing.assert_allclose(result.normal[:4], expected, atol=1e-6)
                np.testing.assert_allclose(result.ray_position[4], origin)

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

    def test_march_step_cap_returns_background(self) -> None:
        integrator = MaterialIntegrator(
            make_camera(),
            Material(jnp.array([0]), jnp.ones((1, 3)), jnp.zeros(1), jnp.ones(1)),
            jnp.array([0.5, 0.7, 1.0]),
            [sdfImplicit(fn=lambda p, x: x[1], dim=3)],
            max_depth=1, max_ray_march_steps=3, scene_bounds=None,
        )
        origins = jnp.array([[0., 1., 0.]])
        directions = jnp.array([[1., 0., 0.]])
        result = integrator.intersect([None], origins, directions)
        self.assertFalse(bool(result.hit_flag[0]))
        self.assertEqual(int(result.niter[0]), 3)
        color = integrator.trace_paths([None], origins, directions, jax.random.PRNGKey(0))
        np.testing.assert_allclose(color, [[0.75, 0.85, 1.0]])
        hit = integrator.intersect([None], origins, jnp.array([[0., -1., 0.]]))
        self.assertTrue(bool(hit.hit_flag[0]))
        np.testing.assert_allclose(hit.ray_position, jnp.zeros((1, 3)))

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
