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
from sphereWalker.integrator import WoSIntegrator
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


if __name__ == '__main__':
    unittest.main()
