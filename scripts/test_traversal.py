"""Three core contracts: first hit, adaptive termination, and multi-hit sampling."""
import os
import sys
import unittest
from pathlib import Path

os.environ.setdefault('JAX_PLATFORMS', 'cpu')
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import jax
import jax.numpy as jnp
import numpy as np

from rayMarcher.scene import ImplicitScene
from rayMarcher.traversal import RayMarcher
from surfaces.implicit.dense import denseImplicit
from surfaces.implicit.lipschitz import sdfImplicit, lipschitzImplicit
from surfaceSampler.sampler import SurfaceSampler
from utils.helpers import geometric_termination_criterion, level_set_termination_criterion
from utils.types import RayBuffer, Termination as T


class TraversalTests(unittest.TestCase):
    def test_first_hit_contract(self):
        surface = sdfImplicit(fn=lambda p, x: jnp.linalg.norm(x) - 1., dim=3)
        origins = jnp.array([[0., 0., 3.], [0., 0., 0.], [3., 0., 3.],
                             [0., 0., 3.], [0., 0., 3.]])
        directions = jnp.array([[0., 0., -2.], [0., 0., 1.], [0., 0., -1.],
                                [0., 0., 0.], [0., 0., -1.]])
        for compact, cap in ((0, 1), (4, 2), (4, 16)):
            with self.subTest(compaction=compact):
                marcher = RayMarcher.from_surface(surface, n_compactions=compact, max_active_states=cap)
                observed = []
                step = marcher.step_batch
                marcher.step_batch = lambda p, state: (observed.append(state.t.size), step(p, state))[1]
                hit = marcher.first_hit(None, origins, directions,
                                        active_mask=jnp.array([True]*4 + [False]))
                np.testing.assert_array_equal(hit.status, [T.HIT, T.HIT, T.EXIT, T.INVALID, T.INACTIVE])
                np.testing.assert_allclose(hit.ray_position[:2], [[0., 0., 1.]]*2, atol=1e-5)
                self.assertLessEqual(max(observed), cap)
                empty = jnp.zeros((0, 3))
                self.assertEqual(marcher.first_hit(None, empty, empty).status.size, 0)
        # Shifted residuals and clipping at/before a surface, with both criteria.
        plane = lipschitzImplicit(fn=lambda p, x: 2. - x[0], level_set=1., lipschitz_bound=1., dim=2)
        for criterion in (level_set_termination_criterion, geometric_termination_criterion):
            for distance, status in ((0.5, T.EXIT), (1., T.HIT)):
                with self.subTest(criterion=criterion.__name__, distance=distance):
                    hit = RayMarcher.from_surface(plane, max_distance=distance, termination=criterion).first_hit(
                        None, jnp.zeros((1, 2)), jnp.array([[1., 0.]]))
                    self.assertEqual(int(hit.status[0]), status)
                    np.testing.assert_allclose(hit.ray_position, [[distance, 0.]], atol=1e-6)

        for invalid in (0, -1, 1.5, True):
            with self.assertRaisesRegex(ValueError, 'max_active_states'):
                RayMarcher.from_surface(surface, max_active_states=invalid)

    def test_adaptive_queries_and_limits(self):
        class AdaptivePlane:
            def ray_step(self, p, x, d, tr):
                safe = tr <= 0.25
                return 1.-x[0], jnp.where(safe, jnp.abs(1.-x[0]), 0.), safe

            def normal_at(self, p, x):
                return jnp.array([-1., 0., 0.])

        surface = AdaptivePlane()
        origin, direction = jnp.zeros((1, 3)), jnp.array([[1., 0., 0.]])
        for budget, status, point in ((1, T.MAX_STEPS, 0.), (8, T.HIT, 1.)):
            with self.subTest(budget=budget):
                hit = RayMarcher.from_surface(surface, max_steps=budget).first_hit(None, origin, direction, 1.)
                self.assertEqual(int(hit.status[0]), status)
                np.testing.assert_allclose(hit.ray_position, [[point, 0., 0.]])
                self.assertLessEqual(int(hit.niter[0]), budget)
        sphere = sdfImplicit(fn=lambda p, x: jnp.linalg.norm(x) - 1., dim=3)
        hit = ImplicitScene([sphere, surface], max_steps=1).closest_hit([None, None], -direction, direction, 1.)
        self.assertTrue(bool(hit.hit_flag[0]))
        self.assertFalse(bool(hit.complete[0]))
        # Fixed-step reference traversal must not shrink its unused TR to zero.
        dense = denseImplicit(fn=lambda p, x: x[0] - 0.2, level_set=0., dim=3, gt_step=0.001)
        hit = RayMarcher.from_surface(dense, max_steps=256).first_hit(None, origin, direction)
        self.assertEqual(int(hit.status[0]), T.HIT)
        np.testing.assert_allclose(hit.ray_position, [[0.2, 0., 0.]], atol=1e-5)

        # A finished lane is refilled while another ray is still marching.
        refill = RayMarcher.from_surface(dense, max_active_states=2, max_steps=256)
        batches = []
        step = refill.step_batch
        def observe(p, state):
            batches.append((np.asarray(state.ray_origin[:, 0]), np.asarray(state.niter)))
            self.assertLessEqual(state.t.size, 2)
            return step(p, state)
        refill.step_batch = observe
        starts = jnp.array([[0., 0., 0.], [0.2, 0., 0.], [0.1, 0., 0.]])
        actual = refill.first_hit(None, starts, jnp.tile(direction, (3, 1)))
        expected = RayMarcher.from_surface(dense, max_steps=256).first_hit(None, starts, jnp.tile(direction, (3, 1)))
        for a, b in zip(actual, expected):
            np.testing.assert_allclose(a, b)
        self.assertTrue(any(np.any(np.isclose(x, .1)) and np.any((x == 0) & (steps > 0))
                            for x, steps in batches))

    def test_multi_hit_sampling_contract(self):
        # A thin slab exercises entering, interior traversal, exit-face hits,
        # misses, and epsilon/L scaling without a large parameter cross-product.
        rays = RayBuffer(jnp.array([[0., 0.], [0.001, 0.], [0., 1.]]),
                         jnp.tile(jnp.array([1., 0.]), (3, 1)), jnp.array([0.00135, 0.00035, 0.0003]))
        for scale, level, compact in ((1., 0., 0), (10000., 2., 4)):
            with self.subTest(scale=scale, compaction=compact):
                surface = lipschitzImplicit(
                    fn=lambda p, x: level + scale*(jnp.abs(x[0]-0.001)-0.00035),
                    level_set=level, lipschitz_bound=scale, dim=2)
                sampler = SurfaceSampler.from_surface(surface, epsilon=scale*1e-7, n_compactions=compact, max_active_states=2)
                sizes = []
                step = sampler.step_states_fn
                sampler.step_states_fn = lambda p, state: (sizes.append(state.t.size), step(p, state))[1]
                hits = sampler.trace_rays(None, rays)
                self.assertLessEqual(max(sizes), 2)
                np.testing.assert_array_equal(hits.hit_counts, [2, 1, 0])
                np.testing.assert_array_equal(hits.status, [T.EXIT]*3)
                np.testing.assert_allclose(hits.t_hits[0, :2], [0.00065, 0.00135], atol=2e-7, rtol=0)
        # Invalid front-of-queue rays and empty input must drain without loss.
        sampler = SurfaceSampler.from_surface(surface, epsilon=scale*1e-7, max_active_states=1)
        mixed = RayBuffer(rays.ray_origin, rays.ray_direction.at[0].set(0.), rays.max_ts)
        output = sampler.trace_rays(None, mixed)
        np.testing.assert_array_equal(output.status, [T.INVALID, T.EXIT, T.EXIT])
        np.testing.assert_array_equal(output.hit_counts, [0, 1, 0])
        empty = RayBuffer(jnp.zeros((0, 2)), jnp.zeros((0, 2)), jnp.zeros(0))
        self.assertEqual(sampler.trace_rays(None, empty).status.size, 0)
        sphere = sdfImplicit(fn=lambda p, x: jnp.linalg.norm(x)-1., dim=2)
        args = (None, jax.random.PRNGKey(2), -2*jnp.ones(2), 2*jnp.ones(2), 16)
        sampler = SurfaceSampler.from_surface(sphere, ray_sample_batch_size=8, max_active_states=3)
        samples = sampler.sample_surface_points(*args, return_details=True)
        self.assertTrue(bool(jnp.all(samples.rays.complete)))
        self.assertGreater(len(samples.points), 0)
        np.testing.assert_allclose(jnp.linalg.norm(samples.points, axis=1), 1., atol=1e-4)
        for options, reason in (({'max_hits': 1}, T.HIT_CAPACITY), ({'max_step': 1}, T.MAX_STEPS)):
            with self.subTest(limit=reason):
                sampler = SurfaceSampler.from_surface(sphere, **options)
                sampler.sample_rays_batched = lambda *a, **kw: (
                    RayBuffer(jnp.array([[-2., 0.]]), jnp.array([[1., 0.]]), jnp.array([4.])), True)
                self.assertEqual(int(sampler.sample_surface_points(*args, return_details=True).rays.status[0]), reason)
                with self.assertRaisesRegex(RuntimeError, 'incomplete'):
                    sampler.sample_surface_points(*args)


if __name__ == '__main__':
    unittest.main()
