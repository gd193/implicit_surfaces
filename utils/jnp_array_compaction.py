import jax
import jax.numpy as jnp
from typing import Callable, List, Tuple
from math import floor
import functools
from utils.types import RayMarchState, RayMarchOutput

compaction_sizes = []


def init_compaction_sizes(generator_fn: Callable[[], List[int]]):
    global compaction_sizes
    compaction_sizes = generator_fn()


def uniform_size_generator(size: int, n_compactions: int) -> List[int]:
    def generator_fn():
        return [floor(size * i / n_compactions) for i in range(0, n_compactions + 1)][::-1]
    return generator_fn


def power_of_two_size_generator(start_base: int, n_compactions: int) -> List[int]:
    def generator_fn():
        return ([0] + [(2 ** i) for i in range(start_base, start_base+n_compactions)])[::-1]
    return generator_fn

@functools.lru_cache(maxsize=None)
def should_compact(N_valid: int, current_compaction_index: int) -> bool:
    global compaction_sizes
    for idx in range(len(compaction_sizes)-1, current_compaction_index, -1):
        if N_valid <= compaction_sizes[idx]:
            return True, idx, compaction_sizes[idx]
    return False, -1, -1


# def write_done_to_output_arrays(current_arrays: tuple, output_arrays: tuple, active_indices: jnp.ndarray) -> tuple:
#     curr_niters, curr_ray_origins, curr_hit_flags = current_arrays
#     out_niters, out_ray_origins, out_hit_flags = output_arrays

#     out_niters = out_niters.at[active_indices].set(curr_niters, mode='drop')
#     out_ray_origins = out_ray_origins.at[active_indices].set(curr_ray_origins, mode='drop')
#     out_hit_flags = out_hit_flags.at[active_indices].set(curr_hit_flags, mode='drop')

#     return out_niters, out_ray_origins, out_hit_flags


# def compact_arrays(fill_index_value: int, new_size: int, done_flags: jnp.ndarray, active_indices: jnp.ndarray, arrays: tuple[jnp.ndarray]) -> tuple[jnp.ndarray]:
#     N_in = jnp.sum(~done_flags)
#     out_mask = jnp.arange(0, new_size) < N_in
#     target_indices = jnp.nonzero(~done_flags, size=new_size, fill_value=fill_index_value)[0]
#     active_indices = active_indices.at[target_indices].get(mode='drop')
#     #done_flags = jnp.logical_or(done_flags, out_mask)
#     out_arrays = []
#     for array in arrays:
#         out_array = array.at[target_indices, ...].get(mode='drop')
#         out_arrays.append(out_array)
#     return tuple(out_arrays), jnp.where(out_mask, active_indices, fill_index_value)


# @functools.partial(jax.jit, static_argnames=('fill_index_value', 'new_compacted_size'))
# def perform_compaction(loop_state: tuple[jnp.ndarray, ...],
#                        output_state: tuple[jnp.ndarray, ...],
#                        active_indices: jnp.ndarray,
#                        fill_index_value: int,
#                        new_compacted_size: int) -> tuple[tuple[jnp.ndarray, ...], tuple[jnp.ndarray, ...], jnp.ndarray]:
#     niters, ray_positions, ray_directions, intervals, ray_origins, done_flags, hit_flags = loop_state

#     output_state = write_done_to_output_arrays(
#         (niters, ray_positions, hit_flags),
#         output_state,
#         jnp.where(done_flags, active_indices, fill_index_value)
#     )
#     new_loop_state, active_indices = compact_arrays(
#         fill_index_value,
#         new_compacted_size,
#         done_flags,
#         active_indices,
#         loop_state,
#     )
#     return new_loop_state, output_state, active_indices

def write_done_to_output_arrays(current_state: RayMarchState, output_state: RayMarchOutput, active_indices: jnp.ndarray) -> RayMarchOutput:
    """
    Scatters completed rays from the current local batch into the global output buffers.
    Only transfers fields present in RayMarchOutput.
    """
    # 1. Update niter
    new_out_niter = output_state.niter.at[active_indices].set(
        current_state.niter, mode='drop'
    )
    
    # 2. Update ray_position
    new_out_position = output_state.ray_position.at[active_indices].set(
        current_state.ray_position, mode='drop'
    )
    
    # 3. Update hit_flag
    new_out_hit = output_state.hit_flag.at[active_indices].set(
        current_state.hit_flag, mode='drop'
    )

    return output_state._replace(
        niter=new_out_niter,
        ray_position=new_out_position,
        hit_flag=new_out_hit
    )


def compact_state_tree(state: RayMarchState, 
                       new_size: int, 
                       fill_index_value: int) -> Tuple[RayMarchState, jnp.ndarray]:
    """
    Compacts the entire RayMarchState PyTree based on its own done_flags.
    Returns (new_compacted_state, valid_mask).
    """
    done_flags = state.done_flag
    
    is_active = ~done_flags
    N_in = jnp.sum(is_active)
    
    target_indices = jnp.nonzero(is_active, size=new_size, fill_value=fill_index_value)[0]
    valid_mask = jnp.arange(new_size) < N_in
    safe_gather_indices = jnp.where(valid_mask, target_indices, 0)
    
    def compact_leaf(arr):
        return arr.at[safe_gather_indices].get(mode='drop')

    new_state = jax.tree_util.tree_map(compact_leaf, state)
    
    return new_state, valid_mask, target_indices


@functools.partial(jax.jit, static_argnames=('fill_index_value', 'new_compacted_size'))
def perform_compaction(loop_state: RayMarchState,
                       output_state: RayMarchOutput,
                       active_indices: jnp.ndarray,
                       fill_index_value: int,
                       new_compacted_size: int) -> Tuple[RayMarchState, RayMarchOutput, jnp.ndarray]:
    
    write_indices = jnp.where(loop_state.done_flag, active_indices, fill_index_value)

    new_output_state = write_done_to_output_arrays(
        loop_state,
        output_state,
        write_indices
    )
    compacted_state, valid_mask, target_gather_indices = compact_state_tree(
        loop_state, 
        new_compacted_size, 
        fill_index_value
    )

    new_active_indices = active_indices.at[target_gather_indices].get(mode='drop')
    # Mask padding indices to fill_value so we don't accidentally write to index 0 later
    new_active_indices = jnp.where(valid_mask, new_active_indices, fill_index_value)
    forced_done_flags = jnp.logical_or(compacted_state.done_flag, ~valid_mask)
    # Update the state with the fixed flags
    compacted_state = compacted_state._replace(done_flag=forced_done_flags)

    return compacted_state, new_output_state, new_active_indices