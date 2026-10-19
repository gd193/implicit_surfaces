"""Walk on Stars with an implicit Dirichlet query and a Neumann enclosure.

Solves -Delta u = f, u = g on the Dirichlet surface, du/dn = 0 on
Neumann surfaces supporting star_radius and intersect_ray. Nonzero Neumann
and geometric derivatives are deliberately unsupported.
"""

import math
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import jax.random as jr

from sphereWalker.integrator import (
    BaseIntegrator,
    finish_step,
    get_scatter_fn,
    walker_status,
)
from sphereWalker.types import StarWalkerState
from sphereWalker.types import WalkerTermination as WT
from surfaces.implicit.base import ImplicitSurface
from utils.sampling_primitives import unit_interval
from utils.types import Termination as T


class WoStParameters(NamedTuple):
    """Internal dynamic context; public callbacks still receive their original params."""

    problem: Any
    neumann: Any


def sample_green_radius(key, dim, dtype):
    """Normalized radial density proportional to the ball Green function.

    2D: 4t log(1/t), sampled as sqrt(U V).
    3D: 6t(1-t), sampled as the median of three uniforms (Beta(2,2)).
    Its integral over the ball is R^2/(2*dim).
    """
    keys = jr.split(key, 3)
    u = jax.vmap(lambda k: unit_interval(k, dtype))(keys)
    if dim == 2:
        return jnp.sqrt(u[0] * u[1])
    if dim == 3:
        return jnp.sort(u)[1]
    raise ValueError("WoSt supports dimensions 2 and 3")


class WoStIntegrator(BaseIntegrator[StarWalkerState]):
    """Same batch driver and Dirichlet/source callbacks as WoS.

    Pass neumann_params explicitly to integrate_batch/pre_warm_compaction_buckets.
    Initial points may be interior or on supported Neumann contacts (including
    box edges/corners). Unresolved queries fail explicitly. eps retains the
    current WoS field-residual termination convention, not a distance certificate.
    """

    def __init__(
        self,
        empty_sphere_fn,
        boundary_condition_fn,
        source_term_fn,
        dim,
        *,
        neumann_surface,
        max_radius=math.inf,
        neumann_condition_fn=None,
        **kwargs,
    ):
        if dim not in (2, 3):
            raise ValueError("WoSt supports dimensions 2 and 3")
        if neumann_surface.dim != dim:
            raise ValueError("Neumann surface dimension must match the integrator")
        for name in ("star_radius", "intersect_ray"):
            if getattr(type(neumann_surface), name, None) is getattr(ImplicitSurface, name):
                raise NotImplementedError(
                    f"{type(neumann_surface).__name__} does not implement {name}"
                )
        if neumann_condition_fn is not None:
            raise NotImplementedError(
                "nonzero Neumann data requires boundary sampling; only homogeneous Neumann is supported"
            )
        if math.isnan(max_radius) or max_radius <= 0:
            raise ValueError("max_radius must be positive")
        super().__init__(
            empty_sphere_fn, boundary_condition_fn, source_term_fn, dim, **kwargs
        )
        self.neumann_surface = neumann_surface
        self.max_radius = max_radius

    def integrate_batch(
        self,
        params,
        initial_position_batch,
        rng_key,
        init_trust_region_size,
        n_samples,
        *args,
        neumann_params,
        **kwargs,
    ):
        return super().integrate_batch(
            WoStParameters(params, neumann_params),
            initial_position_batch,
            rng_key,
            init_trust_region_size,
            n_samples,
            *args,
            **kwargs,
        )

    def pre_warm_compaction_buckets(
        self,
        params,
        initial_position_batch,
        rng_key,
        init_trust_region_size,
        n_samples,
        *args,
        neumann_params,
        **kwargs,
    ):
        return super().pre_warm_compaction_buckets(
            WoStParameters(params, neumann_params),
            initial_position_batch,
            rng_key,
            init_trust_region_size,
            n_samples,
            *args,
            **kwargs,
        )

    def step(self, params: WoStParameters, state: StarWalkerState) -> StarWalkerState:
        residual, clearance, next_tr = self.empty_sphere(
            params.problem, state.position, state.trust_region_size
        )
        status = walker_status(
            state, residual, clearance, next_tr, self.eps, self.max_steps
        )
        active = status == WT.ACTIVE
        # Recognize exact initial boundary points; later contacts carry their normal.
        value = self.neumann_surface.signed_value(params.neumann, state.position)
        on_boundary = state.on_neumann | (value == 0)
        initial_normal = self.neumann_surface.normal_at(params.neumann, state.position)
        normal = jnp.where(state.on_neumann, state.normal, initial_normal)
        normal_valid = jnp.all(jnp.isfinite(normal)) & (
            jnp.abs(jnp.linalg.norm(normal) - 1) < 1e-4
        )
        cap = jnp.minimum(clearance, self.max_radius)
        star = self.neumann_surface.star_radius(
            params.neumann,
            state.position,
            jnp.where(active, cap, 1.0),
            on_boundary=on_boundary,
            normal=normal,
        )
        geometry_valid = (
            (star.status == T.EXIT)
            & jnp.isfinite(star.radius)
            & (star.radius > 0)
            & (star.radius <= jnp.where(active, cap, 1.0))
            & (~on_boundary | normal_valid)
        )
        # Containment applies even to terminal Dirichlet points.
        status = jnp.where(~geometry_valid, WT.INVALID, status)
        active = status == WT.ACTIVE
        radius = jnp.where(active, star.radius, 1.0)
        direction_key, source_key, next_key = jr.split(state.rng_key, 3)
        direction = self.sphere_sampling_fn(direction_key, dtype=state.position.dtype)
        inward = self.neumann_surface.inward_direction(
            params.neumann, state.position, direction, normal
        )
        direction = jnp.where(on_boundary, inward, direction)
        hit = self.neumann_surface.intersect_ray(
            params.neumann, state.position, direction, radius, on_boundary=on_boundary
        )
        hit_valid = (hit.status == T.HIT) | (hit.status == T.EXIT)
        hit_valid &= (
            jnp.isfinite(hit.distance) & (hit.distance > 0) & (hit.distance <= radius)
        )
        hit_valid &= jnp.all(jnp.isfinite(hit.position))
        hit_valid &= (hit.status != T.HIT) | (
            jnp.all(jnp.isfinite(hit.normal))
            & (jnp.abs(jnp.linalg.norm(hit.normal) - 1) < 1e-4)
        )
        status = jnp.where(active & ~hit_valid, WT.INVALID, status)
        advance = status == WT.ACTIVE
        source_distance = radius * sample_green_radius(
            source_key, self.dim, state.position.dtype
        )
        source_position = state.position + source_distance * direction
        # Don't evaluate f outside its domain, even when its weight would be zero.
        inside_star = source_distance < hit.distance
        source_position = jnp.where(
            inside_star & advance, source_position, state.position
        )
        source_value = jnp.asarray(self._source(params.problem, source_position))
        if source_value.shape != ():
            raise ValueError("WoSt currently requires a scalar source term")
        contribution = jnp.where(
            inside_star, source_value * radius**2 / (2 * self.dim), 0.0
        )
        updated = state._replace(
            position=jnp.where(advance, hit.position, state.position),
            nSteps=state.nSteps + advance.astype(jnp.int32),
            w=jnp.where(advance, state.w + contribution, state.w),
            trust_region_size=jnp.where(advance, next_tr, state.trust_region_size),
            rng_key=next_key,
            on_neumann=jnp.where(advance, hit.status == T.HIT, state.on_neumann),
            normal=jnp.where(advance, hit.normal, state.normal),
            status=jnp.asarray(status, jnp.int32),
            done_flag=status != WT.ACTIVE,
        )
        return finish_step(state, updated)

    def _get_initial_state(self, init_position_batch, rng_keys, init_tr, buffer_size):
        dtype = init_position_batch.dtype
        return StarWalkerState(
            position=init_position_batch,
            nSteps=jnp.zeros(buffer_size, jnp.int32),
            w=jnp.zeros((buffer_size, 1), dtype),
            trust_region_size=jnp.full((buffer_size,), init_tr, dtype),
            rng_key=rng_keys,
            done_flag=jnp.zeros(buffer_size, bool),
            status=jnp.zeros(buffer_size, jnp.int32),
            on_neumann=jnp.zeros(buffer_size, bool),
            normal=jnp.zeros_like(init_position_batch),
        )

    def _transition_state_phases(self, params, states, init_trust_region_size):
        return states

    def _compute_accumulator_update(self, params, state):
        return state.w + self.boundary_condition_fn(state.position)

    def _check_if_done(self, state, parents):
        return state.done_flag & (parents != -1)

    def _refill_state(
        self,
        current_states,
        is_valid_mask,
        free_indices,
        new_positions,
        new_keys,
        init_trust_region_size,
        N_total_samples,
    ):
        scatter = get_scatter_fn(free_indices, is_valid_mask, N_total_samples)
        fresh = self._get_initial_state(
            new_positions, new_keys, init_trust_region_size, new_positions.shape[0]
        )
        return jax.tree.map(
            lambda old, new: scatter(old, new, jnp.zeros(old.shape[1:], old.dtype)),
            current_states,
            fresh,
        )
