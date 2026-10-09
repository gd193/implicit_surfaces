"""Three integration checks: material paths, rendered images, and sphere walking."""
import os
import sys
import unittest
from tempfile import TemporaryDirectory
from pathlib import Path

os.environ.setdefault('JAX_PLATFORMS', 'cpu')
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import jax
import jax.numpy as jnp
import numpy as np

from rayMarcher.camera import Camera
from rayMarcher.integrator import MaterialIntegrator
from rayMarcher.marcher import SimpleIntegrator
from rayMarcher.materials import Material
from rayMarcher.sampler import generate_antialiased_ray
from sphereWalker.integrator import WoSIntegrator, DifferentialWoSIntegrator
from sphereWalker.types import WalkerTermination as WT
from surfaces.implicit.lipschitz import sdfImplicit


def camera(away=False, focus=1.):
    return Camera.setup_Camera(4, 4, 40., jnp.array([0., 0., 3.]),
                               jnp.array([0., 0., 4. if away else 0.]),
                               jnp.array([0., 1., 0.]), focus_distance=focus)


def materials(kinds):
    n = len(kinds)
    return Material(jnp.array(kinds), jnp.tile(jnp.array([0.8, 0.6, 0.4]), (n, 1)),
                    jnp.zeros(n), jnp.ones(n))


def sphere():
    return sdfImplicit(fn=lambda radius, x: jnp.linalg.norm(x) - radius, dim=3)


class RendererIntegrationTests(unittest.TestCase):
    def test_material_paths_and_nearest_surface(self):
        origins, directions = jnp.array([[0., 0., 3.]]), jnp.array([[0., 0., -1.]])
        for kind, interactions in ((0, 1), (1, 1), (2, 2)):
            with self.subTest(material=kind):
                renderer = MaterialIntegrator(camera(), materials([kind]), jnp.ones(3), [sphere()],
                                              epsilon=1e-5, max_depth=4, max_active_states=1)
                colors = renderer.trace_paths([1.], origins, directions, jax.random.PRNGKey(5))
                # Index-matched glass crosses both interfaces; diffuse/metal bounce once.
                np.testing.assert_allclose(colors, jnp.array([[0.8, 0.6, 0.4]])**interactions, atol=1e-5)
        # Path RNG and bounce budgets belong to tasks, independent of refills.
        origins = jnp.array([[0., 0., 3.], [0., 0., 0.], [3., 0., 3.], [0.2, 0., 3.], [0., 0., 3.]])
        directions = jnp.tile(jnp.array([0., 0., -1.]), (5, 1))
        reference = MaterialIntegrator(camera(), materials([0]), jnp.array([.5, .7, 1.]), [sphere()], max_active_states=16)
        bounded = MaterialIntegrator(camera(), materials([0]), jnp.array([.5, .7, 1.]), [sphere()], max_active_states=2)
        sizes = []
        step = bounded._step_paths
        bounded._step_paths = lambda p, state: (sizes.append(state.depth.size), step(p, state))[1]
        for renderer in (reference, bounded):
            colors = renderer.trace_paths([1.], origins, directions, jax.random.PRNGKey(23))
            if renderer is reference:
                expected = colors
            else:
                np.testing.assert_allclose(colors, expected, atol=1e-6)
        self.assertLessEqual(max(sizes), 2)
        origins, directions = jnp.array([[0., 0., 3.]]), jnp.array([[0., 0., -1.]])
        renderer = MaterialIntegrator(camera(), materials([0, 1]), jnp.ones(3), [sphere(), sphere()])
        for radii, winner in (([0.5, 1.], 1), ([1., 0.5], 0)):
            with self.subTest(radii=radii):
                hit = renderer.intersect(radii, origins, directions)
                self.assertEqual(int(hit.material_index[0]), winner)
                self.assertTrue(bool(hit.complete[0]))
                np.testing.assert_allclose(hit.ray_position, [[0., 0., 1.]], atol=1e-5)

    def test_renderers_and_camera(self):
        key = jax.random.PRNGKey(2)
        renderers = [SimpleIntegrator.from_surface(camera(), jnp.ones(3), sphere(), max_active_states=3),
                     MaterialIntegrator(camera(), materials([1]), jnp.ones(3), [sphere()], samples_per_pixel=2, max_active_states=3)]
        for renderer, params in zip(renderers, (1., [1.])):
            with self.subTest(renderer=type(renderer).__name__):
                image = renderer.render(params, key)
                image = image[0] if isinstance(image, tuple) else image
                self.assertEqual(image.shape, (4, 4, 3))
                self.assertTrue(bool(jnp.all(jnp.isfinite(image))))
                self.assertTrue(bool(jnp.any(image < 0.95)))
                renderer.camera = camera(away=True)
                renderer.background_color = jnp.array([0.5, 0.7, 1.])
                missed = renderer.render(params, key)
                missed = missed[0] if isinstance(missed, tuple) else missed
                self.assertTrue(bool(jnp.all(jnp.isfinite(missed))))
                np.testing.assert_allclose(missed[:, :, 2], 1., atol=1e-6)
                self.assertTrue(bool(jnp.all(missed[:, :, 0] > 0.5)))
        _, reference = generate_antialiased_ray(camera(), 0, 0, key)
        _, changed = generate_antialiased_ray(camera(focus=3.), 0, 0, key)
        np.testing.assert_allclose(changed, reference, atol=1e-6)
        # Exercise the orbit API and encoded output within the renderer integration test.
        from render_orbit import OrbitConfig, render_orbit
        from PIL import Image
        with TemporaryDirectory() as directory:
            path = render_orbit(sphere(), 1., Path(directory)/'orbit.gif',
                                OrbitConfig(width=8, height=8, frames=2, fps=10., max_active_states=3), progress=False)
            with Image.open(path) as animation:
                self.assertEqual((animation.size, animation.n_frames), ((8, 8), 2))
                self.assertEqual(animation.info['duration'], 100)

    def test_sphere_walking_constant_boundary(self):
        surface = sdfImplicit(fn=lambda p, x: jnp.linalg.norm(x)-1., dim=2)
        walker = WoSIntegrator(surface.empty_sphere, lambda x: jnp.array([1.]),
                               lambda x: jnp.array([0.]), dim=2, max_steps=64)
        result = walker.integrate_batch(None, jnp.zeros((2, 2)), jax.random.PRNGKey(2),
                                        init_trust_region_size=0.5, n_samples=4,
                                        steps_between_refill=4, max_active_states=4,
                                        jit_buck_size=1, n_compactions=2)
        np.testing.assert_allclose(result.w, jnp.ones_like(result.w), atol=1e-6)

        # One successful, one stalled, and one invalid starting point. Failed
        # samples must free slots without entering the estimate or its divisor.
        def query(p, x, tr):
            residual = jnp.where(x[0] == 2., jnp.nan, x[0]-1.)
            return residual, jnp.asarray(0., x.dtype), tr
        walker = WoSIntegrator(query, lambda x: jnp.asarray(7.), lambda x: jnp.asarray(0.),
                               dim=2, max_steps=2)
        positions = jnp.array([[1., 0.], [0., 0.], [2., 0.]])
        options = dict(init_trust_region_size=1., n_samples=3, steps_between_refill=4,
                       max_active_states=2, jit_buck_size=1, n_compactions=2)
        details = walker.integrate_batch(None, positions, key := jax.random.PRNGKey(7),
                                         return_details=True, **options)
        np.testing.assert_array_equal(details.termination_counts, [[0,3,0,0], [0,0,3,0], [0,0,0,3]])
        np.testing.assert_array_equal(details.successful_samples, [3,0,0])
        np.testing.assert_allclose(details.contribution_sum[:, 0], [21.,0.,0.])
        with self.assertRaisesRegex(RuntimeError, 'incomplete WoS'):
            walker.integrate_batch(None, positions, key, **options)

        differential = DifferentialWoSIntegrator(
            empty_sphere_fn=query, boundary_condition_fn=lambda x: jnp.asarray(7.),
            source_term_fn=lambda x: jnp.asarray(0.), dim=2, max_steps=2,
            normal_fn=lambda p,x: jnp.array([1.,0.]),
            Df=lambda p,x,g: {'a': jnp.asarray(0.)}, V_n=lambda p,x: {'a': jnp.asarray(1.)},
            offset_l=.25)
        # x=0 fails in phase zero; x=1 reaches it then fails in phase one.
        details = differential.integrate_batch(None, positions, key, return_details=True, **options)
        np.testing.assert_array_equal(details.termination_counts, [[0,0,3,0], [0,0,3,0], [0,0,0,3]])
        np.testing.assert_allclose(details.contribution_sum['a'], 0.)
        with self.assertRaisesRegex(RuntimeError, 'incomplete WoS'):
            differential.integrate_batch(None, positions, key, **options)
        # A boundary hit can return a tiny next TR. Completed state must stay
        # frozen, and phase one must restart with the caller's initial scale.
        def bounded_plane(p, x, tr):
            return x[0], jnp.minimum(jnp.abs(x[0]), tr), jnp.asarray(1e-12, x.dtype)
        differential = DifferentialWoSIntegrator(
            empty_sphere_fn=bounded_plane, boundary_condition_fn=lambda x: jnp.asarray(7.),
            source_term_fn=lambda x: jnp.asarray(0.), dim=2, max_steps=1,
            normal_fn=lambda p,x: jnp.array([1.,0.]),
            Df=lambda p,x,g: {'a': jnp.asarray(0.)}, V_n=lambda p,x: {'a': jnp.asarray(1.)},
            offset_l=.25)
        differential.sphere_sampling_fn = lambda key,dtype: jnp.array([1.,0.], dtype=dtype)
        state = differential._get_initial_state(jnp.zeros((1,2)), key[None,:], 1., 1)
        step = jax.vmap(differential.step, in_axes=(None, 0))
        hit = step(None, state)
        waited = jax.lax.fori_loop(0, 49, lambda i,s: step(None,s), hit)
        for a, b in zip(hit, waited):
            np.testing.assert_array_equal(a, b)
        restarted = differential._transition_state_phases(None, waited, 1.)
        np.testing.assert_allclose(restarted.trust_region_size, 1.)
        # Confirm the backend recommendation is honored, not replaced by the
        # old unconditional halving rule when its radius is small.
        tiny = state._replace(position=jnp.array([[-.001,0.]]))
        np.testing.assert_allclose(step(None, tiny).trust_region_size, 1e-12, atol=0.)
        for block in (1, 50):
            opts = {**options, 'steps_between_refill': block}
            result = differential.integrate_batch(None, jnp.zeros((1,2)), key, **opts)
            np.testing.assert_allclose(result.w['a'], 0.)



if __name__ == '__main__':
    unittest.main()
