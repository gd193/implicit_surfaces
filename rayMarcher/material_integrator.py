"""Material path tracing using SimpleIntegrator's implicit-surface ray queries."""
from typing import Any, Callable, NamedTuple, Sequence, TYPE_CHECKING

import jax
import jax.numpy as jnp
import jax.random as jr

from rayMarcher.camera import Camera
from rayMarcher.marcher import SimpleIntegrator
from rayMarcher.materials import Material, scatter
from rayMarcher.sampler import generate_antialiased_ray
from rayMarcher.utils import DEFAULT_SCENE_BOUNDS, SceneBounds
from utils.helpers import level_set_termination_criterion, update_trust_region_size

if TYPE_CHECKING:
    from surfaces.implicit.base import ImplicitSurface


class SceneHit(NamedTuple):
    hit_flag: jax.Array
    ray_position: jax.Array
    normal: jax.Array
    material_index: jax.Array
    distance: jax.Array
    niter: jax.Array


class MaterialIntegrator:
    """One material per implicit surface, with explicit per-surface parameters.

    surfaces[i] supplies ray_step(params[i], ...), normal_at(params[i], ...),
    and, for gradient-based termination, ray_step_with_grad(params[i], ...).
    Each bounce uses the same adaptive marcher and compaction as SimpleIntegrator.
    The nearest hit selects row i of Material. Cameras/backgrounds can change
    between renders; surfaces and marching options are fixed at construction.

    render(params, key) returns linear RGB of shape (height, width, 3).
    ray_offset is a world-space separation for secondary rays; it should exceed
    the hit tolerance in spatial units for the chosen field/termination criterion.
    Like SimpleIntegrator, the host compaction loop must not be wrapped in jit.

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
                 scene_bounds: SceneBounds | None = DEFAULT_SCENE_BOUNDS,
                 termination_crit_fn: Callable = level_set_termination_criterion,
                 update_tr_fn: Callable = update_trust_region_size):
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
        self.marchers = tuple(SimpleIntegrator(
            camera, background_color, surface.ray_step, surface.normal_at,
            image_height=self.image_height, image_width=self.image_width,
            max_ray_march_steps=max_ray_march_steps, max_ray_distance=max_ray_distance,
            # The surface already returns its level-set-shifted residual.
            level_set=0., epsilon=epsilon, n_compactions=n_compactions,
            termination_crit_fn=termination_crit_fn, update_tr_fn=update_tr_fn,
            ray_march_with_grad_fn=getattr(surface, "ray_step_with_grad", None),
            scene_bounds=scene_bounds,
        ) for surface in self.surfaces)
        self._normal_fns = tuple(jax.jit(jax.vmap(surface.normal_at, in_axes=(None, 0)))
                                 for surface in self.surfaces)
        self._scatter = jax.jit(jax.vmap(scatter, in_axes=(None, 0, 0, 0, 0, 0)))

    def intersect(self, params: Sequence[Any], ray_origins: jax.Array,
                  ray_directions: jax.Array, init_trust_region_size: float = 0.5,
                  active_mask: jax.Array | None = None) -> SceneHit:
        """Return the nearest implicit-surface hit for each input ray."""
        if len(params) != len(self.surfaces):
            raise ValueError("params must contain one entry per surface (use None if unused)")
        count = ray_origins.shape[0]
        nearest = jnp.full(count, jnp.inf, dtype=ray_origins.dtype)
        positions = ray_origins
        normals = jnp.broadcast_to(jnp.array([0., 1., 0.], dtype=ray_origins.dtype), (count, 3))
        material_ids = jnp.zeros(count, dtype=jnp.int32)
        niters = jnp.zeros(count, dtype=jnp.int32)
        for index, (marcher, normal_fn, surface_params) in enumerate(zip(self.marchers, self._normal_fns, params)):
            result = marcher.trace_rays(surface_params, ray_origins, ray_directions,
                                       init_trust_region_size, active_mask)
            distance = jnp.linalg.norm(result.ray_position - ray_origins, axis=-1)
            closer = result.hit_flag & (distance < nearest)
            candidate_normals = normal_fn(surface_params, result.ray_position)
            nearest = jnp.where(closer, distance, nearest)
            positions = jnp.where(closer[:, None], result.ray_position, positions)
            normals = jnp.where(closer[:, None], candidate_normals, normals)
            material_ids = jnp.where(closer, index, material_ids)
            niters = niters + result.niter
        return SceneHit(jnp.isfinite(nearest), positions, normals, material_ids, nearest, niters)

    def trace_paths(self, params: Sequence[Any], ray_origins: jax.Array,
                    ray_directions: jax.Array, key: jax.Array,
                    init_trust_region_size: float = 0.5) -> jax.Array:
        """Trace a batch of paths, accumulating background radiance at misses."""
        lengths = jnp.linalg.norm(ray_directions, axis=-1)
        active = jnp.isfinite(lengths) & (lengths > 0)
        directions = jnp.where(active[:, None], ray_directions / jnp.where(active, lengths, 1)[:, None], 0)
        origins = ray_origins
        throughput = jnp.ones_like(origins)
        colors = jnp.zeros_like(origins)
        keys = jr.split(key, origins.shape[0])
        for _ in range(self.max_depth):
            if not bool(jnp.any(active)):
                break
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
        return colors

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
