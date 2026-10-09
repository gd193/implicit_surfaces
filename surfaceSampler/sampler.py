import jax
import jax.numpy as jnp
import jax.random as jr

from typing import Callable, Tuple, Any
import math

from utils.sampling_primitives import sample_uniform_ray_bbox_2d, sample_uniform_ray_bbox_3d
from utils.types import RayBuffer, MultiHitRayState, MultiHitRayOutput, SurfaceSamples, Termination
from surfaces.queries import RayQuery
from utils.batch_driver import (run_bounded_batch, finished_rays, mask_finished_padding,
                                DEFAULT_MAX_ACTIVE_STATES, validate_max_active_states)



samping_ray_bbox_selector = {
    2: sample_uniform_ray_bbox_2d,
    3: sample_uniform_ray_bbox_3d
}


def write_multi_hit_done_to_output(state, output, indices):
    return MultiHitRayOutput(*(
        getattr(output, name).at[indices].set(getattr(state, name), mode='drop')
        for name in MultiHitRayOutput._fields
    ))


class LipschitzContinuation:
    """Minimum hit-band advance epsilon/L; only used after accepting a hit.

    L must apply to the field in the sampling region. This is a sampling
    continuation, not a certified empty-space step or exact root enumeration.
    """
    def __init__(self, bound: float):
        self.bound = float(bound)
        if not math.isfinite(self.bound) or self.bound <= 0:
            raise ValueError('lipschitz_bound must be finite and positive')

    def __call__(self, params, position, direction, query, epsilon):
        return jnp.asarray(epsilon / self.bound, dtype=position.dtype)


class SurfaceSampler:
    """
    samples uniformly distributed points on the level set of an implicit surface via random ray casting.
    https://arxiv.org/pdf/2506.05268

    epsilon is a function-value tolerance. After a hit, the minimum world-space
    advance is epsilon / lipschitz_bound. Supply a surface carrying that bound,
    an explicit bound, or a continuation callback. No unit bound is assumed.
    Incomplete sampling raises unless return_details or allow_incomplete is set.
    Prefer from_surface() for the surface-semantic interface. The old level_set
    constructor argument is ignored; callbacks already return shifted residuals.
    """
    def __init__(self, 
                 # ray_march_fn: (params, positions, directions, trust_region_size) -> (fn_val, t_step, inside_tr)
                 ray_march_fn: Callable[[Any, jnp.ndarray, jnp.ndarray, jnp.ndarray], tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]],
                 level_set: float, dim: int, max_step: int, epsilon: float, max_hits: int, init_trust_region_size: float, 
                 n_compactions: int = 10,
                 ray_sample_batch_size: int=1000000, ray_sample_max_iter: int=1000, 
                 print_progress: bool = False, debug: bool = False,
                 *, max_active_states: int = DEFAULT_MAX_ACTIVE_STATES,
                 lipschitz_bound: float | None = None,
                 continuation: Callable | None = None,
                 query: RayQuery | None = None,
                 update_tr_fn: Callable | None = None):
        if dim not in (2, 3) or max_step < 1 or max_hits < 1:
            raise ValueError('dim must be 2 or 3; max_step and max_hits must be positive')
        if not math.isfinite(epsilon) or epsilon <= 0:
            raise ValueError('epsilon must be finite and positive')
        if not math.isfinite(init_trust_region_size) or init_trust_region_size <= 0:
            raise ValueError('initial trust region must be finite and positive')
        self.max_active_states = validate_max_active_states(max_active_states)
        self.initialize = jax.jit(self._initialize)
        self.query = query if query is not None else RayQuery(ray_march_fn, update_tr=update_tr_fn)
        self.dim = dim
        self.max_step = max_step
        self.max_hits = max_hits
        self.epsilon = epsilon
        if lipschitz_bound is None and continuation is None:
            surface = getattr(ray_march_fn, '__self__', None)
            lipschitz_bound = getattr(surface, 'lipschitz_bound', None)
        if continuation is not None and lipschitz_bound is not None:
            raise ValueError('choose continuation or lipschitz_bound, not both')
        if continuation is None:
            if lipschitz_bound is None:
                raise ValueError('sampling needs lipschitz_bound or an explicit continuation policy')
            continuation = LipschitzContinuation(lipschitz_bound)
        self.lipschitz_bound = lipschitz_bound
        self.continuation = continuation
        self.init_trust_region_size = init_trust_region_size
        self.ray_sample_batch_size = ray_sample_batch_size
        self.ray_sample_max_iter = ray_sample_max_iter
        self.print_progress = print_progress
        self.debug = debug
        self.sampling_ray_bbox_fn = samping_ray_bbox_selector[dim]
        self.step_states_fn = jax.jit(jax.vmap(self.step_fn, in_axes=(None, 0)))
        self.n_compactions = n_compactions


    @classmethod
    def from_surface(cls, surface, *, max_step=256, epsilon=1e-4,
                     max_hits=16, init_trust_region_size=0.5,
                     update_tr_fn=None, **kwargs):
        return cls(surface.ray_step, level_set=0., dim=surface.dim,
                   max_step=max_step, epsilon=epsilon, max_hits=max_hits,
                   init_trust_region_size=init_trust_region_size,
                   query=RayQuery.from_surface(surface, update_tr=update_tr_fn), **kwargs)

    def sample_rays_batched(self, key: jax.Array, 
                            bbox_mins: jnp.ndarray, 
                            bbox_maxs: jnp.ndarray, 
                            num_rays: int, 
                            batch_size: int,
                            max_iters: int,
                            dtype: jnp.dtype) -> Tuple[RayBuffer, jnp.ndarray]:
        """
        Batched rejection sampling for uniform ray generation.
        """
        if batch_size < 1 or max_iters < 1:
            raise ValueError("ray sampling batch size and iteration budget must be positive")
        batch_size = min(batch_size, self.max_active_states)
        dim = bbox_mins.shape[0]
        generate_batch_fn = jax.vmap(self.sampling_ray_bbox_fn, in_axes=(0, None, None, None))

        # 2. Initialize  buffer
        init_dirs = jnp.zeros((num_rays, dim), dtype=dtype)
        init_pts = jnp.zeros((num_rays, dim), dtype=dtype)
        init_ts = jnp.zeros((num_rays,), dtype=dtype)
        init_buffer = RayBuffer(ray_origin=init_pts, ray_direction=init_dirs, max_ts=init_ts)

        # State: (key, buffer, valid_count, iteration)
        init_state = (key, init_buffer, jnp.array(0, dtype=jnp.int32), jnp.array(0, dtype=jnp.int32))

        def cond_fn(state):
            _, _, count, i = state
            return jnp.logical_and(count < num_rays, i < max_iters)

        def body_fn(state):
            curr_key, buffer, count, i = state
            key_batch, key_next = jr.split(curr_key)
            
            # Split key into a batch of keys for the vmap
            keys = jr.split(key_batch, batch_size)

            # Generate a batch of candidate rays
            is_hit, ray_direction, ray_origins, max_t_in_box = generate_batch_fn(keys, bbox_mins, bbox_maxs, dtype)

            # Compact the valid points
            sort_indices = jnp.argsort(~is_hit)
            sorted_dirs = ray_direction[sort_indices]
            sorted_pts = ray_origins[sort_indices]
            sorted_max_ts = max_t_in_box[sort_indices]

            # Arithmetic for bounds checking
            num_valid = jnp.sum(is_hit, dtype=count.dtype)
            space_left = num_rays - count
            n_to_write = jnp.minimum(num_valid, space_left)
            target_indices = count + jnp.arange(0, batch_size, dtype=count.dtype)
            write_mask = target_indices < (count + n_to_write)
            to_write_indices = jnp.where(write_mask, target_indices, num_rays) # use num_rays as an invalid index for out-of-bounds writes

            # Write valid rays into our global buffers
            new_dirs = buffer.ray_direction.at[to_write_indices].set(sorted_dirs, mode='drop')
            new_pts = buffer.ray_origin.at[to_write_indices].set(sorted_pts, mode='drop')
            new_ts = buffer.max_ts.at[to_write_indices].set(sorted_max_ts, mode='drop')

            new_buffer = RayBuffer(ray_direction=new_dirs, ray_origin=new_pts, max_ts=new_ts)

            return (key_next, new_buffer, count + n_to_write, i + 1)

        # 3. Execute Loop
        _, final_buffer, final_count, _ = jax.lax.while_loop(cond_fn, body_fn, init_state)
        
        is_finished = final_count >= num_rays
        
        return final_buffer, is_finished
    

    def step_fn(self, params: Any, state: MultiHitRayState) -> MultiHitRayState:
        position = state.ray_origin + state.ray_direction * state.t
        query = self.query(params, position, state.ray_direction, state.trust_region_size)
        in_band = jnp.abs(query.residual) < self.epsilon
        valid = jnp.isfinite(query.residual) & jnp.isfinite(query.distance) & (query.distance >= 0)
        record = in_band & ~state.in_hit_band & ~state.done_flag & valid
        room = state.hit_counts < self.max_hits
        record = record & room
        slot = jnp.minimum(state.hit_counts, self.max_hits - 1)
        hits = jnp.where(record, state.t_hits.at[slot].set(state.t), state.t_hits)
        count = state.hit_counts + record.astype(state.hit_counts.dtype)
        # The continuation is algorithm-owned and never replaces a zero safe
        # step outside the hit band. Crossing either sign is intentional.
        escape = self.continuation(params, position, state.ray_direction, query, self.epsilon)
        advance = jnp.where(in_band, jnp.maximum(query.distance, escape), query.distance)
        niter = state.niter + 1
        at_exit = state.t >= state.t_max
        overflow = in_band & ~state.in_hit_band & ~room
        capacity = count >= self.max_hits
        continuing = ~at_exit & ~capacity & (niter < self.max_step)
        valid = valid & (~continuing | (jnp.isfinite(advance) & (advance >= 0)
                        & jnp.isfinite(query.next_trust_region) & (query.next_trust_region > 0)
                        & (~in_band | (jnp.isfinite(escape) & (escape > 0)))))
        status = jnp.where(~valid, Termination.INVALID,
                 jnp.where(at_exit & ~overflow, Termination.EXIT,
                 jnp.where(capacity, Termination.HIT_CAPACITY,
                 jnp.where(niter >= self.max_step, Termination.MAX_STEPS, Termination.ACTIVE))))
        done = status != Termination.ACTIVE
        updated = state._replace(
            niter=niter, t=jnp.where(done, state.t, jnp.minimum(state.t + advance, state.t_max)),
            t_hits=hits, hit_counts=count, in_hit_band=in_band,
            trust_region_size=jnp.where(done, state.trust_region_size, query.next_trust_region),
            done_flag=done, status=status,
        )
        return jax.tree.map(lambda old, new: jnp.where(state.done_flag, old, new), state, updated)

    def trace_rays(self, params: Any, rays: RayBuffer, *, return_buffers=False):
        """Record multiple hits along supplied unit rays; always report status.

        max_ts is measured from each origin, in world units. A HIT_CAPACITY or
        MAX_STEPS result contains a prefix of the hits, not a complete traversal.
        """
        origins, directions, ends = rays
        if (origins.ndim != 2 or origins.shape[1] != self.dim
                or directions.shape != origins.shape or ends.shape != (origins.shape[0],)):
            raise ValueError('RayBuffer must contain (N, dim) origins/directions and (N,) max_ts')
        n = origins.shape[0]
        output = MultiHitRayOutput(jnp.zeros(n, jnp.int32), origins, directions,
                                   jnp.zeros((n, self.max_hits), origins.dtype),
                                   jnp.zeros(n, jnp.int32), jnp.zeros(n, jnp.int32))
        buffers = run_bounded_batch(
            params, rays, output, initialize=self.initialize, step_batch=self.step_states_fn,
            finished_mask=finished_rays, write_finished=write_multi_hit_done_to_output,
            mask_padding=mask_finished_padding, batch_size=n,
            max_active_states=self.max_active_states, n_compactions=self.n_compactions)
        return buffers if return_buffers else buffers[1]

    def _initialize(self, rays, indices, valid_slots):
        origins, directions, ends = jax.tree.map(lambda a: a[indices], rays)
        n = indices.size
        lengths = jnp.linalg.norm(directions, axis=-1)
        valid = (valid_slots & jnp.all(jnp.isfinite(origins), axis=-1)
                 & jnp.isfinite(lengths) & jnp.isclose(lengths, 1., rtol=1e-5, atol=1e-6)
                 & jnp.isfinite(ends) & (ends >= 0))
        status = jnp.where(valid, Termination.ACTIVE, Termination.INVALID)
        state = MultiHitRayState(
            niter=jnp.zeros(n, jnp.int32), t=jnp.zeros(n, origins.dtype),
            t_hits=jnp.zeros((n, self.max_hits), origins.dtype), t_max=ends,
            hit_counts=jnp.zeros(n, jnp.int32), in_hit_band=jnp.zeros(n, bool),
            ray_origin=origins, ray_direction=directions,
            trust_region_size=jnp.full(n, self.init_trust_region_size, origins.dtype),
            done_flag=~valid, status=status,
        )
        return state

    def sample_surface_points(self, params: Any, key: jax.Array, bbox_min: jax.Array,
                              bbox_max: jax.Array, num_rays: int, *,
                              return_details: bool = False, allow_incomplete: bool = False):
        """Generate uniform box rays and gather their hit-band entry points.

        return_details=True returns SurfaceSamples(points, rays), including
        per-ray completion reasons. By default incomplete sampling raises.
        allow_incomplete=True explicitly accepts a possibly biased partial set.
        """
        if num_rays < 0 or bbox_min.shape != (self.dim,) or bbox_max.shape != (self.dim,):
            raise ValueError('num_rays must be nonnegative and bounds must match dim')
        if not bool(jnp.all(jnp.isfinite(bbox_min) & jnp.isfinite(bbox_max) & (bbox_min < bbox_max))):
            raise ValueError('sampling bounds must be finite lower < upper')
        if num_rays == 0:
            rays = RayBuffer(jnp.zeros((0, self.dim), bbox_min.dtype),
                             jnp.zeros((0, self.dim), bbox_min.dtype), jnp.zeros(0, bbox_min.dtype))
        else:
            rays, generated = self.sample_rays_batched(
                key, bbox_min, bbox_max, num_rays, batch_size=min(self.ray_sample_batch_size, num_rays, self.max_active_states),
                max_iters=self.ray_sample_max_iter, dtype=bbox_min.dtype)
            if not bool(generated):
                raise RuntimeError('ray generation exhausted its rejection-sampling budget')
        buffers = self.trace_rays(params, rays, return_buffers=True)
        if self.debug:
            return buffers
        output = buffers[1]
        if not return_details and not allow_incomplete and not bool(jnp.all(output.complete)):
            raise RuntimeError('incomplete surface sampling; use return_details=True to inspect ray status')
        points = gather_hit_points(output)
        return SurfaceSamples(points, output) if return_details else points


def gather_hit_points(output_states: MultiHitRayOutput) -> jnp.ndarray:
    """
    Gathers hit points from the output state after ray marching is complete.
    """
    N_rays, max_hits = output_states.t_hits.shape
    dim = output_states.ray_origin.shape[1]

    hit_indices = jnp.arange(max_hits)
    valid_hits_mask = hit_indices < output_states.hit_counts[:, None]
    ray_idx, hit_idx = jnp.where(valid_hits_mask)

    valid_origins = output_states.ray_origin[ray_idx]       # (Valid_Hits, 3)
    valid_directions = output_states.ray_direction[ray_idx] # (Valid_Hits, 3)
    valid_t = output_states.t_hits[ray_idx, hit_idx]        # (Valid_Hits,)

    hit_coordinates = valid_origins + valid_directions * valid_t[:, None]
    return hit_coordinates
