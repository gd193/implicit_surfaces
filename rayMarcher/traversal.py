"""Camera-independent first-hit ray marching over implicit-surface queries."""
import math
from typing import Any, Callable

import jax
import jax.numpy as jnp

from rayMarcher.utils import DEFAULT_SCENE_BOUNDS, ray_scene_interval, validate_scene_bounds
from surfaces.queries import RayQuery
from utils.batch_driver import (run_bounded_batch, finished_rays, mask_finished_padding,
                                DEFAULT_MAX_ACTIVE_STATES, validate_max_active_states)
from utils.helpers import level_set_termination_criterion
from utils.types import RayMarchState, RayMarchOutput, Termination


def write_ray_results(state, output, indices):
    return RayMarchOutput(
        output.niter.at[indices].set(state.niter, mode='drop'),
        output.ray_position.at[indices].set(
            state.ray_origin + state.t[:, None] * state.ray_direction, mode='drop'),
        output.hit_flag.at[indices].set(state.hit_flag, mode='drop'),
        output.status.at[indices].set(state.status, mode='drop'),
    )


class RayMarcher:
    """First accepted hit, from either side of a surface, on a bounded ray.

    Query results contain residuals; this class has no level-set parameter.
    max_steps counts query evaluations (including the accepted hit/exit).
    An undecided zero step may refine the TR but is not implicitly a hit.
    """
    def __init__(self, query: RayQuery, *, max_steps: int = 128,
                 max_distance: float = 20., epsilon: float = 1e-4,
                 n_compactions: int = 4, max_active_states: int = DEFAULT_MAX_ACTIVE_STATES,
                 scene_bounds=DEFAULT_SCENE_BOUNDS,
                 termination: Callable = level_set_termination_criterion):
        if max_steps < 1 or not math.isfinite(epsilon) or epsilon <= 0:
            raise ValueError('max_steps and epsilon must be positive and finite')
        if math.isnan(max_distance) or max_distance <= 0:
            raise ValueError('max_distance must be positive')
        if getattr(termination, 'needs_grad', False) and not query.needs_grad:
            raise ValueError('termination requires a gradient query')
        self.max_active_states = validate_max_active_states(max_active_states)
        self.initialize = jax.jit(self._initialize)
        self.query = query
        self.max_steps = max_steps
        self.max_distance = max_distance
        self.epsilon = epsilon
        self.n_compactions = n_compactions
        self.scene_bounds = validate_scene_bounds(scene_bounds)
        self.termination = termination
        self.step_batch = jax.jit(jax.vmap(self.step, in_axes=(None, 0)))

    @classmethod
    def from_surface(cls, surface, *, termination=level_set_termination_criterion,
                     update_tr=None, **kwargs):
        if 'scene_bounds' not in kwargs and getattr(surface, 'dim', 3) == 2:
            kwargs['scene_bounds'] = ((-2., -2.), (2., 2.))
        options = {} if update_tr is None else {'update_tr': update_tr}
        query = RayQuery.from_surface(surface, needs_grad=getattr(termination, 'needs_grad', False), **options)
        return cls(query, termination=termination, **kwargs)

    def step(self, params: Any, state: RayMarchState) -> RayMarchState:
        position = state.ray_origin + state.t * state.ray_direction
        query = self.query(params, position, state.ray_direction, state.trust_region_size)
        options = {'grad_norm': jnp.linalg.norm(query.gradient)} if getattr(self.termination, 'needs_grad', False) else {}
        hit = self.termination(query.distance, query.residual, state.trust_region_size,
                               0., self.epsilon, **options)
        valid = (jnp.isfinite(query.residual) & jnp.isfinite(query.distance)
                 & (query.distance >= 0))
        # TR is only needed if this evaluation will advance/refine.
        at_exit = state.t >= state.t_max
        next_tr_valid = jnp.isfinite(query.next_trust_region) & (query.next_trust_region > 0)
        valid = valid & (hit | at_exit | next_tr_valid)
        niter = state.niter + 1
        status = jnp.where(~valid, Termination.INVALID,
                 jnp.where(hit, Termination.HIT,
                 jnp.where(at_exit, Termination.EXIT,
                 jnp.where(niter >= self.max_steps, Termination.MAX_STEPS, Termination.ACTIVE))))
        done = status != Termination.ACTIVE
        updated = state._replace(
            niter=niter,
            t=jnp.where(done, state.t, jnp.minimum(state.t + query.distance, state.t_max)),
            trust_region_size=jnp.where(done, state.trust_region_size, query.next_trust_region),
            done_flag=done, hit_flag=status == Termination.HIT, status=status,
        )
        return jax.tree.map(lambda old, new: jnp.where(state.done_flag, old, new), state, updated)

    def first_hit(self, params, ray_origins, ray_directions,
                  init_trust_region_size=0.5, active_mask=None) -> RayMarchOutput:
        if ray_origins.ndim != 2 or ray_directions.shape != ray_origins.shape:
            raise ValueError('ray origins/directions must have matching (N, dim) shapes')
        if not math.isfinite(init_trust_region_size) or init_trust_region_size <= 0:
            raise ValueError('initial trust region must be finite and positive')
        if self.scene_bounds is not None and len(self.scene_bounds[0]) != ray_origins.shape[1]:
            raise ValueError('scene bounds must match ray dimension')
        dtype = jnp.result_type(ray_origins, ray_directions, jnp.float32)
        ray_origins = jnp.asarray(ray_origins, dtype=dtype)
        ray_directions = jnp.asarray(ray_directions, dtype=dtype)
        n = ray_origins.shape[0]
        requested = jnp.ones(n, dtype=bool) if active_mask is None else jnp.asarray(active_mask, dtype=bool)
        if requested.shape != (n,):
            raise ValueError('active_mask must have shape (N,)')
        output = RayMarchOutput(jnp.zeros(n, jnp.int32), jnp.zeros_like(ray_origins),
                                jnp.zeros(n, bool), jnp.zeros(n, jnp.int32))
        _, output, _ = run_bounded_batch(
            params, (ray_origins, ray_directions, requested, init_trust_region_size), output,
            initialize=self.initialize, step_batch=self.step_batch, finished_mask=finished_rays,
            write_finished=write_ray_results, mask_padding=mask_finished_padding,
            batch_size=n, max_active_states=self.max_active_states, n_compactions=self.n_compactions)
        return output

    def _initialize(self, inputs, indices, valid_slots):
        all_origins, all_directions, requested, init_trust_region_size = inputs
        ray_origins, ray_directions = all_origins[indices], all_directions[indices]
        requested = requested[indices] & valid_slots
        n = indices.size
        lengths = jnp.linalg.norm(ray_directions, axis=-1)
        valid = (jnp.isfinite(lengths) & (lengths > 0)
                 & jnp.all(jnp.isfinite(ray_origins), axis=-1))
        directions = jnp.where(valid[:, None], ray_directions / jnp.where(valid, lengths, 1)[:, None], 0.)
        origins = jnp.where(jnp.isfinite(ray_origins), ray_origins, 0.)
        entry, exit, intersects = jax.vmap(lambda o, d: ray_scene_interval(o, d, self.scene_bounds))(origins, directions)
        end = jnp.minimum(exit, self.max_distance)
        intersects = intersects & (entry <= end)
        status = jnp.where(~requested, Termination.INACTIVE,
                 jnp.where(~valid, Termination.INVALID,
                 jnp.where(~intersects, Termination.EXIT, Termination.ACTIVE)))
        active = status == Termination.ACTIVE
        state = RayMarchState(
            jnp.zeros(n, jnp.int32), jnp.where(active, entry, 0.).astype(origins.dtype),
            origins, directions, jnp.full(n, init_trust_region_size, dtype=origins.dtype),
            ~active, jnp.zeros(n, bool), jnp.where(active, end, 0.).astype(origins.dtype), status,
        )
        return state

    trace_rays = first_hit
