"""Nearest known intersection across independently parameterized surfaces."""
from typing import Any, NamedTuple, Sequence
import jax
import jax.numpy as jnp

from rayMarcher.traversal import RayMarcher
from utils.types import Termination
from utils.batch_driver import bounded_map


class SceneHit(NamedTuple):
    hit_flag: jax.Array
    ray_position: jax.Array
    normal: jax.Array
    material_index: jax.Array
    distance: jax.Array
    niter: jax.Array
    surface_status: jax.Array  # (N, surfaces): preserves incomplete-query reasons

    @property
    def complete(self):
        return jnp.all((self.surface_status == Termination.HIT)
                       | (self.surface_status == Termination.EXIT)
                       | (self.surface_status == Termination.INACTIVE), axis=-1)


class ImplicitScene:
    """Surface i is geometry i; callers may assign material i independently.

    A closest_hit with complete=False is only the nearest *known* hit: an
    exhausted/invalid surface query could have hidden a closer intersection.
    """
    def __init__(self, surfaces, **march_options):
        self.surfaces = tuple(surfaces)
        if not self.surfaces:
            raise ValueError('at least one implicit surface is required')
        self.marchers = tuple(RayMarcher.from_surface(s, **march_options) for s in self.surfaces)
        self._normal_fns = tuple(jax.jit(jax.vmap(s.normal_at, in_axes=(None, 0))) for s in self.surfaces)

    def closest_hit(self, params: Sequence[Any], ray_origins: jax.Array,
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
        statuses = []
        for index, (marcher, surface_params) in enumerate(zip(self.marchers, params)):
            result = marcher.first_hit(surface_params, ray_origins, ray_directions,
                                       init_trust_region_size, active_mask)
            statuses.append(result.status)
            distance = jnp.linalg.norm(result.ray_position - ray_origins, axis=-1)
            closer = result.hit_flag & (distance < nearest)
            nearest = jnp.where(closer, distance, nearest)
            positions = jnp.where(closer[:, None], result.ray_position, positions)
            material_ids = jnp.where(closer, index, material_ids)
            niters = niters + result.niter

        hit_flag = jnp.isfinite(nearest)
        for index, (marcher, normal_fn, surface_params) in enumerate(
                zip(self.marchers, self._normal_fns, params)):
            won = hit_flag & (material_ids == index)
            winners = int(jnp.count_nonzero(won))
            if winners == 0:
                # Preserve normal dtype promotion without evaluating a losing surface.
                normal_shape = jax.eval_shape(
                    normal_fn, surface_params, jax.ShapeDtypeStruct((0, positions.shape[1]), positions.dtype))
                normals = normals.astype(jnp.result_type(normals, normal_shape.dtype))
                continue
            indices, = jnp.nonzero(won, size=winners)
            hit_normals = bounded_map(
                lambda points: normal_fn(surface_params, points), positions[indices],
                max_active_states=marcher.max_active_states)
            normals = normals.astype(jnp.result_type(normals, hit_normals))
            normals = normals.at[indices].set(hit_normals.astype(normals.dtype))
        return SceneHit(hit_flag, positions, normals, material_ids, nearest, niters, jnp.stack(statuses, axis=-1))
