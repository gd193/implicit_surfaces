"""Camera sampling and normal shading, backed by the independent RayMarcher."""
from typing import Any, Callable, Optional, Tuple

import jax
import jax.numpy as jnp
import jax.random as jr

from rayMarcher.camera import Camera
from rayMarcher.sampler import generate_antialiased_ray
from rayMarcher.traversal import RayMarcher
from rayMarcher.utils import DEFAULT_SCENE_BOUNDS
from surfaces.queries import RayQuery
from utils.helpers import level_set_termination_criterion
from utils.types import RayMarchOutput
from utils.batch_driver import DEFAULT_MAX_ACTIVE_STATES, bounded_map


class SimpleIntegrator:
    """Render surface normals. Camera and background can change between renders.

    Prefer from_surface(). The callback constructor remains supported; its
    level_set argument is ignored because callbacks already return f-level_set.
    """
    def __init__(self, camera: Camera, background_color: jax.Array,
                 ray_march_fn: Callable, surface_normal_fn: Callable, *,
                 image_height: int, image_width: int, max_ray_march_steps: int,
                 max_ray_distance: float, epsilon: float, n_compactions: int,
                 max_active_states: int = DEFAULT_MAX_ACTIVE_STATES,
                 level_set: float | None = None,
                 termination_crit_fn: Callable = level_set_termination_criterion,
                 update_tr_fn: Callable | None = None,
                 ray_march_with_grad_fn: Optional[Callable] = None,
                 scene_bounds=DEFAULT_SCENE_BOUNDS):
        self.camera = camera
        self.background_color = background_color
        self.surface_normal_fn = surface_normal_fn
        self.image_height = image_height
        self.image_width = image_width
        query = RayQuery(ray_march_fn, update_tr=update_tr_fn,
                         step_with_grad=ray_march_with_grad_fn,
                         needs_grad=getattr(termination_crit_fn, 'needs_grad', False),
                         certified=getattr(getattr(ray_march_fn, '__self__', None),
                                           'certified_ray_steps', True))
        self.marcher = RayMarcher(query, max_steps=max_ray_march_steps,
                                  max_distance=max_ray_distance, epsilon=epsilon,
                                  n_compactions=n_compactions, max_active_states=max_active_states, scene_bounds=scene_bounds,
                                  termination=termination_crit_fn)
        self.max_active_states = self.marcher.max_active_states
        self._post_process_jit = jax.jit(self._post_process_impl)

    @classmethod
    def from_surface(cls, camera, background_color, surface, *,
                     max_ray_march_steps=128, max_ray_distance=20., epsilon=1e-4,
                     n_compactions=4, **kwargs):
        return cls(camera, background_color, surface.ray_step, surface.normal_at,
                   image_height=camera.image_height, image_width=camera.image_width,
                   max_ray_march_steps=max_ray_march_steps, max_ray_distance=max_ray_distance,
                   epsilon=epsilon, n_compactions=n_compactions,
                   ray_march_with_grad_fn=getattr(surface, 'ray_step_with_grad', None), **kwargs)

    def trace_rays(self, params, ray_origins, ray_directions,
                   init_trust_region_size=0.5, active_mask=None):
        return self.marcher.first_hit(params, ray_origins, ray_directions,
                                      init_trust_region_size, active_mask)

    def sample_camera_rays(self, key: jax.Array, image_height: int, image_width: int):
        i_coords, j_coords = jnp.meshgrid(jnp.arange(image_width), jnp.arange(image_height))
        i_coords = i_coords.flatten()
        j_coords = j_coords.flatten()
        pixel_keys = jr.split(key, image_height * image_width)
        ray_origins, ray_directions = jax.vmap(generate_antialiased_ray, in_axes=(None, 0, 0, 0))(self.camera, i_coords, j_coords, pixel_keys)
        return ray_origins, ray_directions


    def _post_process_impl(self,
                           params: Any,
                           output_states: RayMarchOutput,
                           ray_directions: jnp.ndarray,
                           background_color: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        # Misses may terminate at t=0; use the actual sampled directions.
        view_dirs = ray_directions
        t = 0.5 * (view_dirs[:, 1] + 1.0)  # shape (N,)
        background_colors = (1.0 - t[:, None]) * jnp.array([1.0, 1.0, 1.0]) + t[:, None] * background_color
        hit_normals = jax.vmap(self.surface_normal_fn, in_axes=(None, 0))(params, output_states.ray_position)
        test_colors = jnp.clip(0.5 * hit_normals + 0.5, 0.0, 1.0)
        colors = jnp.where(output_states.hit_flag[:, None], test_colors, background_colors)
        return colors, output_states.niter

    def render(self,
               params: Any,
               key: jax.Array,
               init_trust_region_size: float = 0.5):
        ray_origins, ray_directions = self.sample_camera_rays(key, self.image_height, self.image_width)
        output_states = self.trace_rays(params, ray_origins, ray_directions, init_trust_region_size)

        ### Postprocessing to get the actual image
        image, niter = bounded_map(
            lambda states, directions: self._post_process_jit(
                params, states, directions, self.background_color),
            output_states, ray_directions, max_active_states=self.marcher.max_active_states)
        return (image.reshape(self.image_height, self.image_width, 3),
                niter.reshape(self.image_height, self.image_width))
