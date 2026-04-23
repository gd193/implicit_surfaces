import jax
import jax.numpy as jnp
import jax.random as jr

from typing import Callable, Tuple, Any

from utils.sampling_primitives import sample_uniform_ray_bbox_2d, sample_uniform_ray_bbox_3d
from utils.types import RayBuffer, MultiHitRayState, MultiHitRayOutput
from utils.helpers import update_trust_region_size
from utils.jnp_array_compaction import perform_compaction, should_compact, uniform_size_generator


samping_ray_bbox_selector = {
    2: sample_uniform_ray_bbox_2d,
    3: sample_uniform_ray_bbox_3d
}


def write_multi_hit_done_to_output(current_state: MultiHitRayState, 
                                   output_state: MultiHitRayOutput, 
                                   write_indices: jnp.ndarray) -> MultiHitRayOutput:
    """
    Scatters completed rays from the current local batch into the global output buffers.
    """
    new_out_niter = output_state.niter.at[write_indices].set(
        current_state.niter, mode='drop'
    )
    new_out_origin = output_state.ray_origin.at[write_indices].set(
        current_state.ray_origin, mode='drop'
    )
    new_out_direction = output_state.ray_direction.at[write_indices].set(
        current_state.ray_direction, mode='drop'
    )
    new_out_t_hits = output_state.t_hits.at[write_indices].set(
        current_state.t_hits, mode='drop'
    )
    new_out_hit_counts = output_state.hit_counts.at[write_indices].set(
        current_state.hit_counts, mode='drop'
    )

    return output_state._replace(
        niter=new_out_niter,
        ray_origin=new_out_origin,
        ray_direction=new_out_direction,
        t_hits=new_out_t_hits,
        hit_counts=new_out_hit_counts
    )


def update_multi_hit_state(state: MultiHitRayState, valid_mask: jnp.ndarray) -> MultiHitRayState:
    forced_done_flags = jnp.logical_or(state.done_flag, ~valid_mask)
    return state._replace(done_flag=forced_done_flags)


class SurfaceSampler:
    """
    samples uniformly distributed points on the level set of an implicit surface via random ray casting.
    https://arxiv.org/pdf/2506.05268
    """
    def __init__(self, 
                 # ray_march_fn: (params, positions, directions, trust_region_size) -> (fn_val, t_step, inside_tr)
                 ray_march_fn: Callable[[Any, jnp.ndarray, jnp.ndarray, jnp.ndarray], tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]],
                 level_set: float, dim: int, max_step: int, epsilon: float, max_hits: int, init_trust_region_size: float, 
                 n_compactions: int = 10,
                 ray_sample_batch_size: int=1000000, ray_sample_max_iter: int=1000, 
                 print_progress: bool = False, debug: bool = False):
        self.ray_march_fn = ray_march_fn
        self.level_set = level_set
        self.dim = dim
        self.max_step = max_step
        self.max_hits = max_hits
        self.epsilon = epsilon
        self.init_trust_region_size = init_trust_region_size
        self.ray_sample_batch_size = ray_sample_batch_size
        self.ray_sample_max_iter = ray_sample_max_iter
        self.print_progress = print_progress
        self.debug = debug
        self.sampling_ray_bbox_fn = samping_ray_bbox_selector[dim]
        self.step_states_fn = jax.jit(jax.vmap(self.step_fn, in_axes=(None, 0)))
        self.n_compactions = n_compactions


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
            num_valid = jnp.sum(is_hit)
            space_left = num_rays - count
            n_to_write = jnp.minimum(num_valid, space_left)
            target_indices = count + jnp.arange(0, batch_size)
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
        """
        March ray to find hits with the surface, and update the state accordingly.
        Implementation of Algorithm 2 in https://arxiv.org/pdf/2506.05268
        """
        current_position = state.ray_origin + state.ray_direction * state.t
        fn_val, t_step, inside_tr = self.ray_march_fn(params, current_position, state.ray_direction, state.trust_region_size)
        fn_val, t_step, inside_tr = fn_val.squeeze(), t_step.squeeze(), inside_tr.squeeze() # ensure shapes are correct


        is_hitting_surface = jnp.abs(fn_val - self.level_set) < self.epsilon
        is_new_hit = jnp.logical_and(is_hitting_surface, state.inside_surface == False)
        can_record_hit = state.hit_counts < self.max_hits
        ray_still_valid = jnp.logical_not(state.done_flag)
        should_record_hit = jnp.logical_and(is_new_hit, jnp.logical_and(can_record_hit, ray_still_valid))
        step_size = jnp.maximum(t_step, jnp.array(self.epsilon, dtype=state.t.dtype)) # ensure we always make some progress

        # Update hit records
        safe_hit_idx = jnp.minimum(state.hit_counts, self.max_hits - 1) # ensure we don't write out of bounds
        new_t_hits = jnp.where(should_record_hit, state.t_hits.at[safe_hit_idx].set(state.t), state.t_hits)
        new_hit_counts = jnp.where(should_record_hit, state.hit_counts + 1, state.hit_counts)

        # Update ray marching state
        new_iter = jnp.where(ray_still_valid, state.niter + 1, state.niter)
        new_inside_surface = jnp.where(ray_still_valid, is_hitting_surface, state.inside_surface)
        new_t = jnp.where(ray_still_valid, state.t + step_size, state.t)
        new_trust_region_size = jnp.where(ray_still_valid, update_trust_region_size(state.trust_region_size, t_step, inside_tr), state.trust_region_size)
        new_done_flag = jnp.logical_or(state.done_flag, jnp.logical_or(new_t >= state.t_max, new_iter >= self.max_step))

        return MultiHitRayState(
            niter=new_iter,
            t=new_t,
            t_hits=new_t_hits,
            t_max=state.t_max,
            hit_counts=new_hit_counts,
            inside_surface=new_inside_surface,
            ray_origin=state.ray_origin,
            ray_direction=state.ray_direction,
            trust_region_size=new_trust_region_size,
            done_flag=new_done_flag,
        )


    def sample_surface_points(self, params: Any, key: jnp.ndarray, bbox_min: jnp.ndarray, bbox_max: jnp.ndarray, num_rays: int) -> jnp.ndarray:
        """
        Args:
            params: Parameters to pass to the implicit surface function.
            key: JAX random key for sampling.
            num_points: Number of rays used to sample on the surface.
        Returns:
            An array of shape (num_points, dim) containing the sampled points on the surface.
        """
        ### sample bbox rays
        raybuffer, is_finished = self.sample_rays_batched(key, bbox_min, bbox_max, num_rays, 
                                             batch_size=min(self.ray_sample_batch_size, num_rays), max_iters=self.ray_sample_max_iter, dtype=bbox_min.dtype)
        assert jnp.any(is_finished)
        active_states = MultiHitRayState(
            niter=jnp.zeros(num_rays, dtype=jnp.int32),
            t=jnp.zeros(num_rays, dtype=bbox_max.dtype),
            t_hits=jnp.zeros((num_rays, self.max_hits), dtype=bbox_max.dtype),
            t_max = raybuffer.max_ts,
            hit_counts=jnp.zeros(num_rays, dtype=jnp.int32),
            inside_surface=jnp.zeros(num_rays, dtype=jnp.bool_),
            ray_origin=raybuffer.ray_origin,
            ray_direction=raybuffer.ray_direction,
            trust_region_size=jnp.ones(num_rays, dtype=bbox_min.dtype) * self.init_trust_region_size,
            done_flag=jnp.zeros(num_rays, dtype=jnp.bool_),
        )
        active_indices = jnp.arange(num_rays)
        output_states = MultiHitRayOutput(
            niter=jnp.zeros(num_rays, dtype=jnp.int32),
            ray_origin=jnp.zeros_like(raybuffer.ray_origin),
            ray_direction=jnp.zeros_like(raybuffer.ray_direction),
            t_hits=jnp.zeros_like(active_states.t_hits),
            hit_counts=jnp.zeros_like(active_states.hit_counts)
        )

        ### Python control loop variables
        iter = 0
        current_compaction_index = 0
        compaction_sizes = uniform_size_generator(num_rays, n_compactions=self.n_compactions)
        INVALID_IND = num_rays + 1

        ### This cannot be jitted because of the dynamic compaction sizes (XLA needs to know static shapes and step numbers)
        while(True):
            active_states = self.step_states_fn(params, active_states)
            iter += 1
            N_valid = int(jnp.sum(~active_states.done_flag))
            is_done = N_valid == 0


            should_compact_flag, new_compaction_index, new_compaction_size = should_compact(N_valid, current_compaction_index, compaction_sizes)
            if should_compact_flag or is_done:

                current_compaction_index = new_compaction_index

                active_states, output_states, active_indices = perform_compaction(
                    active_states,
                    output_states,
                    active_indices,
                    active_states.done_flag,
                    fill_index_value=INVALID_IND,
                    new_compacted_size=new_compaction_size,
                    write_to_output_fn=write_multi_hit_done_to_output,
                    update_state_fn=update_multi_hit_state
                )
            if is_done:
                break

        if self.debug:
            return active_states, output_states, active_indices # type: ignore
        # Gather all hit points from output_states
        hit_points = gather_hit_points(output_states)
        return hit_points

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


