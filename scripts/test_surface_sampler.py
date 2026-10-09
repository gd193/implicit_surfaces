"""CPU regression checks for ray-casting surface sampling."""
import os
import sys
import unittest
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import jax
import jax.numpy as jnp
import numpy as np

from surfaceSampler.sampler import SurfaceSampler
from utils.types import MultiHitRayState, RayBuffer


def plane_step(params, position, direction, trust_region):
    residual = 1.0 - position[0]
    return residual, jnp.abs(residual), jnp.asarray(True)


def make_sampler(compactions: int = 1, dim: int = 2) -> SurfaceSampler:
    return SurfaceSampler(
        plane_step, level_set=0., dim=dim, max_step=40, epsilon=1e-4,
        max_hits=4, init_trust_region_size=1., n_compactions=compactions,
        ray_sample_batch_size=8, ray_sample_max_iter=100,
    )


class SurfaceSamplerTests(unittest.TestCase):
    def test_lipschitz_scaled_hit_escape(self) -> None:
        class TwoPlanes:
            def __init__(self, bound):
                self.lipschitz_bound = bound

            def ray_step(self, params, position, direction, trust_region):
                distance = jnp.minimum(jnp.abs(position[0] - 0.0005),
                                       jnp.abs(position[0] - 0.0012))
                return self.lipschitz_bound * distance, distance, jnp.asarray(True)

        rays = RayBuffer(jnp.zeros((1, 2)), jnp.array([[1., 0.]]), jnp.array([0.002]))
        for bound in (1., 1000., 10000.):
            for inferred in (False, True):
                with self.subTest(bound=bound, inferred=inferred):
                    surface = TwoPlanes(bound)
                    callback = surface.ray_step if inferred else lambda *args: surface.ray_step(*args)
                    kwargs = {} if inferred else {"lipschitz_bound": bound}
                    sampler = SurfaceSampler(
                        callback, level_set=0., dim=2, max_step=128,
                        epsilon=bound * 1e-7, max_hits=4, init_trust_region_size=1.,
                        n_compactions=1, **kwargs,
                    )
                    self.assertEqual(sampler.lipschitz_bound, bound)
                    sampler.sample_rays_batched = lambda *args, **kw: (rays, jnp.asarray(True))
                    points = sampler.sample_surface_points(
                        None, jax.random.PRNGKey(0), jnp.array([0., -1.]),
                        jnp.array([0.002, 1.]), 1)
                    np.testing.assert_allclose(points, [[0.0005, 0.], [0.0012, 0.]], atol=1e-7, rtol=0.)

    def test_unknown_safe_step_does_not_force_advance(self) -> None:
        sampler = SurfaceSampler(
            lambda *args: (jnp.asarray(1.), jnp.asarray(0.), jnp.asarray(False)),
            level_set=0., dim=2, max_step=10, epsilon=1e-3,
            max_hits=2, init_trust_region_size=1.,
        )
        state = MultiHitRayState(
            niter=jnp.asarray(0), t=jnp.asarray(0.), t_hits=jnp.zeros(2),
            t_max=jnp.asarray(1.), hit_counts=jnp.asarray(0),
            inside_surface=jnp.asarray(False), ray_origin=jnp.zeros(2),
            ray_direction=jnp.array([1., 0.]), trust_region_size=jnp.asarray(1.),
            done_flag=jnp.asarray(False),
        )
        next_state = jax.jit(sampler.step_fn)(None, state)
        self.assertEqual(float(next_state.t), 0.)
        self.assertEqual(int(next_state.hit_counts), 0)
        self.assertLess(float(next_state.trust_region_size), float(state.trust_region_size))

    def test_invalid_lipschitz_bound(self) -> None:
        for bound in (0., -1., float('inf'), float('nan')):
            with self.subTest(bound=bound), self.assertRaises(ValueError):
                SurfaceSampler(
                    plane_step, level_set=0., dim=2, max_step=10, epsilon=1e-3,
                    max_hits=2, init_trust_region_size=1., lipschitz_bound=bound,
                )

    def test_nonzero_level_sets(self) -> None:
        rays = RayBuffer(jnp.zeros((1, 2)), jnp.array([[1., 0.]]), jnp.array([4.]))
        for level in (-1., 0., 1.):
            for compactions in (0, 2):
                with self.subTest(level=level, compactions=compactions):
                    # Surface protocol for f(x)=2-x: return f-level, not raw f.
                    def ray_step(params, position, direction, trust_region):
                        residual = 2.0 - position[0] - level
                        return residual, jnp.abs(residual), jnp.asarray(True)

                    sampler = SurfaceSampler(
                        ray_step, level_set=level, dim=2, max_step=40, epsilon=1e-4,
                        max_hits=4, init_trust_region_size=1., n_compactions=compactions,
                    )
                    sampler.sample_rays_batched = lambda *args, **kwargs: (rays, jnp.asarray(True))
                    points = sampler.sample_surface_points(
                        None, jax.random.PRNGKey(0), jnp.array([0., -1.]),
                        jnp.array([4., 1.]), 1)
                    np.testing.assert_allclose(points, [[2.0 - level, 0.]], atol=1e-6)

    def test_compaction_preserves_hits_including_exit(self) -> None:
        rays = RayBuffer(
            ray_origin=jnp.array([[0., 0.], [0., 1.], [0., 2.], [0., 3.]]),
            ray_direction=jnp.tile(jnp.array([1., 0.]), (4, 1)),
            max_ts=jnp.array([1., 1.5, 0.5, 2.]),
        )
        for compactions in (0, 1, 3, 8):
            with self.subTest(compactions=compactions):
                sampler = make_sampler(compactions)
                sampler.sample_rays_batched = lambda *args, **kwargs: (rays, jnp.asarray(True))
                points = sampler.sample_surface_points(
                    None, jax.random.PRNGKey(0), jnp.array([0., 0.]),
                    jnp.array([2., 3.]), 4)
                np.testing.assert_allclose(points, [[1., 0.], [1., 1.], [1., 3.]], atol=1e-6)

    def test_exit_is_evaluated_once_without_overshoot(self) -> None:
        sampler = make_sampler()
        step = jax.jit(sampler.step_fn)
        for exit_t, expected_hits in ((0.5, 0), (1., 1)):
            with self.subTest(exit_t=exit_t):
                state = MultiHitRayState(
                    niter=jnp.asarray(0), t=jnp.asarray(0.), t_hits=jnp.zeros(4),
                    t_max=jnp.asarray(exit_t), hit_counts=jnp.asarray(0),
                    inside_surface=jnp.asarray(False), ray_origin=jnp.zeros(2),
                    ray_direction=jnp.array([1., 0.]), trust_region_size=jnp.asarray(1.),
                    done_flag=jnp.asarray(False),
                )
                state = step(None, state)
                self.assertEqual(float(state.t), exit_t)
                self.assertFalse(bool(state.done_flag))
                state = step(None, state)
                self.assertTrue(bool(state.done_flag))
                self.assertEqual(int(state.hit_counts), expected_hits)
                self.assertEqual(float(state.t), exit_t)
                repeated = step(None, state)
                for before, after in zip(state, repeated):
                    np.testing.assert_array_equal(before, after)

    def test_ray_generation_with_x64_enabled(self) -> None:
        previous_x64 = jax.config.x64_enabled
        jax.config.update("jax_enable_x64", True)
        self.addCleanup(jax.config.update, "jax_enable_x64", previous_x64)
        for dim in (2, 3):
            for dtype in (jnp.float32, jnp.float64):
                with self.subTest(dim=dim, dtype=dtype):
                    sampler = make_sampler(dim=dim)
                    lower, upper = -jnp.ones(dim, dtype), jnp.ones(dim, dtype)
                    rays, complete = sampler.sample_rays_batched(
                        jax.random.PRNGKey(0), lower, upper, 16, 8, 100, dtype)
                    self.assertTrue(bool(complete))
                    self.assertEqual(rays.ray_origin.dtype, np.dtype(dtype))
                    self.assertTrue(bool(jnp.all(jnp.isfinite(rays.ray_origin))))
                    self.assertTrue(bool(jnp.all(rays.max_ts > 0)))
                    np.testing.assert_allclose(jnp.linalg.norm(rays.ray_direction, axis=-1), 1., atol=1e-6)
                    endpoints = rays.ray_origin + rays.max_ts[:, None] * rays.ray_direction
                    for points in (rays.ray_origin, endpoints):
                        self.assertTrue(bool(jnp.all((points >= lower - 1e-5) & (points <= upper + 1e-5))))


if __name__ == "__main__":
    unittest.main()
