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
from rayMarcher.marcher import SimpleIntegrator
from rayMarcher.sampler import generate_antialiased_ray
from utils.helpers import geometric_termination_criterion, level_set_termination_criterion
from utils.types import RayMarchState


def make_camera(focus_distance: float = 1.0) -> Camera:
    return Camera.setup_Camera(
        16, 8, 60.0, jnp.zeros(3), jnp.array([0., 0., 1.]),
        jnp.array([0., 1., 0.]), focus_distance=focus_distance,
    )


class RendererTests(unittest.TestCase):
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
