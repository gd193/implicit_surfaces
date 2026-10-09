"""Material path tracing using the shared RayMarcher."""
from typing import Any, Callable, Sequence, TYPE_CHECKING, NamedTuple

import jax
import jax.numpy as jnp
import jax.random as jr

from rayMarcher.camera import Camera
from rayMarcher.scene import ImplicitScene, SceneHit
from rayMarcher.materials import Material, scatter
from rayMarcher.sampler import generate_antialiased_ray
from rayMarcher.utils import DEFAULT_SCENE_BOUNDS, SceneBounds
from utils.helpers import level_set_termination_criterion
from utils.batch_driver import (DEFAULT_MAX_ACTIVE_STATES, validate_max_active_states,
                                run_bounded_batch, finished_rays, mask_finished_padding)

if TYPE_CHECKING:
    from surfaces.implicit.base import ImplicitSurface


class PathState(NamedTuple):
    origins: jax.Array
    directions: jax.Array
    throughput: jax.Array
    colors: jax.Array
    keys: jax.Array
    depth: jax.Array
    done_flag: jax.Array


def write_paths(state, output, indices):
    return output.at[indices].set(state.colors, mode='drop')


class MaterialIntegrator:
    """One material per implicit surface, with explicit per-surface parameters.

    surfaces[i] supplies ray_step(params[i], ...), normal_at(params[i], ...),
    and, for gradient-based termination, ray_step_with_grad(params[i], ...).
    Each bounce uses the same adaptive marcher and compaction as the simple renderer.
    The nearest hit selects row i of Material. Cameras/backgrounds can change
    between renders; surfaces and marching options are fixed at construction.

    render(params, key) returns linear RGB of shape (height, width, 3).
    ray_offset is a world-space separation for secondary rays; it should exceed
    the hit tolerance in spatial units for the chosen field/termination criterion.
    Like RayMarcher, the host compaction loop must not be wrapped in jit.

    Example:
        renderer = MaterialIntegrator(camera, materials, background, [surface_a, surface_b])
        image = renderer.render([params_a, params_b], key)
    """

    def __init__(self, camera: Camera, materials: Material,
                 background_color: jax.Array, surfaces: Sequence["ImplicitSurface"], *,
                 samples_per_pixel: int = 1, max_depth: int = 8,
                 max_ray_march_steps: int = 128, max_ray_distance: float = 20.,
                 epsilon: float = 1e-4, n_compactions: int = 4,
                 ray_offset: float = 1e-3,
                 max_active_states: int = DEFAULT_MAX_ACTIVE_STATES,
                 scene_bounds: SceneBounds | None = DEFAULT_SCENE_BOUNDS,
                 termination_crit_fn: Callable = level_set_termination_criterion,
                 update_tr_fn: Callable | None = None):
        self.max_active_states = validate_max_active_states(max_active_states)
        self.n_compactions = n_compactions
        self.initialize_paths = jax.jit(self._initialize_paths)
        self.surfaces = tuple(surfaces)
        if not self.surfaces:
            raise ValueError("at least one implicit surface is required")
        if samples_per_pixel < 1 or max_depth < 1 or max_ray_march_steps < 1:
            raise ValueError("sample, bounce, and marching counts must be positive")
        if epsilon <= 0 or ray_offset <= 0 or max_ray_distance <= 0:
            raise ValueError("epsilon, ray_offset, and max_ray_distance must be positive")
        count = len(self.surfaces)
        for name, expected in (("type", (count,)), ("albedo", (count, 3)),
                               ("fuzz", (count,)), ("ref_idx", (count,))):
            if getattr(materials, name).shape != expected:
                raise ValueError(f"materials.{name} must have shape {expected}")
        self.camera = camera
        self.materials = materials
        self.background_color = background_color
        self.samples_per_pixel = samples_per_pixel
        self.max_depth = max_depth
        self.ray_offset = ray_offset
        self.image_height = camera.image_height
        self.image_width = camera.image_width
        self.scene = ImplicitScene(
            self.surfaces, max_steps=max_ray_march_steps, max_distance=max_ray_distance,
            epsilon=epsilon, n_compactions=n_compactions, scene_bounds=scene_bounds,
            max_active_states=self.max_active_states,
            termination=termination_crit_fn, update_tr=update_tr_fn,
        )
        self.marchers = self.scene.marchers
        self._scatter = jax.jit(jax.vmap(scatter, in_axes=(None, 0, 0, 0, 0, 0)))

    def intersect(self, params, ray_origins, ray_directions,
                  init_trust_region_size=0.5, active_mask=None) -> SceneHit:
        return self.scene.closest_hit(params, ray_origins, ray_directions,
                                      init_trust_region_size, active_mask)

    def trace_paths(self, params: Sequence[Any], ray_origins: jax.Array,
                    ray_directions: jax.Array, key: jax.Array,
                    init_trust_region_size: float = 0.5) -> jax.Array:
        """Trace a batch of paths, accumulating background radiance at misses."""
        if ray_origins.ndim != 2 or ray_origins.shape != ray_directions.shape or ray_origins.shape[1] != 3:
            raise ValueError('path origins/directions must have matching (N, 3) shapes')
        dtype = jnp.result_type(ray_origins, ray_directions, jnp.float32)
        ray_origins, ray_directions = jnp.asarray(ray_origins, dtype), jnp.asarray(ray_directions, dtype)
        _, colors, _ = run_bounded_batch(
            (params, init_trust_region_size), (ray_origins, ray_directions, key),
            jnp.zeros_like(ray_origins), initialize=self.initialize_paths, step_batch=self._step_paths,
            finished_mask=finished_rays, write_finished=write_paths, mask_padding=mask_finished_padding,
            batch_size=ray_origins.shape[0], max_active_states=self.max_active_states,
            n_compactions=self.n_compactions)
        return colors

    def _initialize_paths(self, inputs, indices, valid_slots):
        ray_origins, ray_directions, key = inputs
        origins, directions = ray_origins[indices], ray_directions[indices]
        lengths = jnp.linalg.norm(directions, axis=-1)
        valid = valid_slots & jnp.isfinite(lengths) & (lengths > 0) & jnp.all(jnp.isfinite(origins), axis=-1)
        directions = jnp.where(valid[:, None], directions / jnp.where(valid, lengths, 1)[:, None], 0)
        # Task identity, not slot/refill order, determines the random stream.
        keys = jax.vmap(lambda i: jr.fold_in(key, i))(indices)
        return PathState(origins, directions, jnp.ones_like(origins), jnp.zeros_like(origins),
                         keys, jnp.zeros(indices.size, jnp.int32), ~valid)

    def _step_paths(self, payload, state):
        params, init_trust_region_size = payload
        origins, directions, throughput, colors, keys, depth, done = state
        active = ~done
        hit = self.intersect(params, origins, directions, init_trust_region_size, active)
        blend = 0.5 * (directions[:, 1] + 1.)
        background = (1. - blend[:, None]) + blend[:, None] * self.background_color
        escaped = active & ~hit.hit_flag
        colors = colors + jnp.where(escaped[:, None], throughput * background, 0.)
        # Diffuse/metal use the incident-facing normal. Glass needs the
        # geometric orientation to distinguish entering from leaving.
        back_face = jnp.sum(directions * hit.normal, axis=-1) > 0
        flip = back_face & (self.materials.type[hit.material_index] != 2)
        shading_normal = jnp.where(flip[:, None], -hit.normal, hit.normal)
        scattered, facing_normal, albedo, absorbed, reflected, keys = self._scatter(
            self.materials, hit.material_index, directions, shading_normal, hit.ray_position, keys)
        active = active & hit.hit_flag & ~absorbed
        # Transmission starts on the other side; reflection stays on this side.
        transmitted = (self.materials.type[hit.material_index] == 2) & ~reflected
        offset_normal = jnp.where(transmitted[:, None], -facing_normal, facing_normal)
        new_origins = hit.ray_position + self.ray_offset * offset_normal
        lengths = jnp.linalg.norm(scattered, axis=-1)
        valid = jnp.isfinite(lengths) & (lengths > 0)
        active = active & valid
        unit_scattered = scattered / jnp.where(valid, lengths, 1)[:, None]
        origins = jnp.where(active[:, None], new_origins, origins)
        directions = jnp.where(active[:, None], unit_scattered, directions)
        throughput = jnp.where(active[:, None], throughput * albedo, 0.)
        depth = depth + 1
        return PathState(origins, directions, throughput, colors, keys, depth,
                         ~active | (depth >= self.max_depth))

    def render(self, params: Sequence[Any], key: jax.Array,
               init_trust_region_size: float = 0.5) -> jax.Array:
        if (self.camera.image_height, self.camera.image_width) != (self.image_height, self.image_width):
            raise ValueError("camera resolution cannot change after construction")
        camera_key, path_key = jr.split(key)
        pixels = jnp.repeat(jnp.arange(self.image_height * self.image_width), self.samples_per_pixel)
        keys = jr.split(camera_key, pixels.shape[0])
        origins, directions = jax.vmap(generate_antialiased_ray, in_axes=(None, 0, 0, 0))(
            self.camera, pixels % self.image_width, pixels // self.image_width, keys)
        colors = self.trace_paths(params, origins, directions, path_key, init_trust_region_size)
        return colors.reshape(self.image_height, self.image_width, self.samples_per_pixel, 3).mean(axis=2)
