import jax
import jax.numpy as jnp
from typing import Callable, List, Tuple, Any
from math import floor
import functools


def uniform_size_generator(size: int, n_compactions: int) -> List[int]:
    if n_compactions <= 0:
        return [size]
    return [floor(size * i / n_compactions) for i in range(0, n_compactions + 1)][::-1]


def power_of_two_size_generator(start_base: int, n_compactions: int) -> List[int]:
    return ([0] + [(2 ** i) for i in range(start_base, start_base+n_compactions)])[::-1]


def should_compact(N_valid: int, current_compaction_index: int, compaction_sizes: List[int]) -> Tuple[bool, int, int]:
    for idx in range(len(compaction_sizes)-1, current_compaction_index, -1):
        if N_valid <= compaction_sizes[idx]:
            return True, idx, compaction_sizes[idx]
    return False, -1, -1


# `fill_index_value` was retained for compatibility with `perform_compaction`
# (used downstream as the OOB sentinel when scattering finished rays into the
# output buffer). Inside `compact_tree` itself the fill is masked away by
# `safe_gather_indices`, so for the WoS hot path we use this internal-only
# variant whose static_args are just `new_size`. Single cache axis → far
# fewer JIT compiles when the bucket schedule transitions through many sizes.
@functools.partial(jax.jit, static_argnames=('new_size',))
def compact_tree(state: Any,
                 is_active: jnp.ndarray,
                 new_size: int) -> Tuple[Any, jnp.ndarray, jnp.ndarray]:
    """
    Compacts an arbitrary PyTree state based on a boolean is_active mask.
    Returns (new_compacted_state, valid_mask, target_indices).

    Precondition: `new_size >= jnp.sum(is_active)`. `jnp.nonzero(..., size=new_size)` has
    static output shape, so if more rays are active than `new_size`, the overflow is
    silently truncated (the scheduler in `should_compact` is responsible for keeping
    `new_size` large enough). Callers using a custom compaction schedule must preserve
    this invariant.

    JIT-cached on `new_size`. Pre-warming via
    `WoSIntegrator.pre_warm_compaction_buckets` populates this cache before
    timed runs (one compile per bucket size; 800 ms each on the 1 M-buffer
    hot path otherwise).
    """
    N_in = jnp.sum(is_active)

    # fill_value here is masked away by safe_gather_indices, so any constant works.
    target_indices = jnp.nonzero(is_active, size=new_size, fill_value=0)[0]
    valid_mask = jnp.arange(new_size) < N_in
    safe_gather_indices = jnp.where(valid_mask, target_indices, 0)

    def compact_leaf(arr):
        return arr.at[safe_gather_indices].get(mode='drop')

    new_state = jax.tree_util.tree_map(compact_leaf, state)

    return new_state, valid_mask, target_indices


def compact_tree_with_fill(state: Any,
                           is_active: jnp.ndarray,
                           new_size: int,
                           fill_index_value: int) -> Tuple[Any, jnp.ndarray, jnp.ndarray]:
    """Backward-compat wrapper: returns target_indices that are *also* set to
    `fill_index_value` for padding slots. Used by `perform_compaction` whose
    caller scatters finished rays via the same indices into an output buffer.
    """
    new_state, valid_mask, target_indices_compact = compact_tree(state, is_active, new_size)
    target_indices_full = jnp.where(valid_mask, target_indices_compact, fill_index_value)
    return new_state, valid_mask, target_indices_full


@functools.partial(jax.jit, static_argnames=('fill_index_value', 'new_compacted_size', 'write_to_output_fn', 'update_state_fn'))
def perform_compaction(loop_state: Any,
                       output_state: Any,
                       active_indices: jnp.ndarray,
                       is_done: jnp.ndarray,
                       fill_index_value: int,
                       new_compacted_size: int,
                       write_to_output_fn: Callable,
                       update_state_fn: Callable) -> Tuple[Any, Any, jnp.ndarray]:
    """
    Compact `loop_state` to `new_compacted_size` and scatter finished rays into `output_state`.

    Note on retracing: `write_to_output_fn` and `update_state_fn` are declared static, so JAX keys
    its compilation cache by their identity/hash. Define them once at module/class scope (or bind
    them once via `functools.partial` outside the loop) and reuse the same object across iterations
    -- otherwise every call constructs a fresh callable, misses the cache, and triggers a full
    retrace + recompile each step.
    """
    write_indices = jnp.where(is_done, active_indices, fill_index_value)

    new_output_state = write_to_output_fn(loop_state, output_state, write_indices)

    is_active = ~is_done
    compacted_state, valid_mask, target_gather_indices = compact_tree_with_fill(
        loop_state,
        is_active,
        new_compacted_size,
        fill_index_value,
    )

    new_active_indices = active_indices.at[target_gather_indices].get(mode='drop')
    # Mask padding indices to fill_value so we don't accidentally write to index 0 later
    new_active_indices = jnp.where(valid_mask, new_active_indices, fill_index_value)
    
    # Let the user update specific tracking fields like 'done_flag' or 'niter' given the valid mask
    compacted_state = update_state_fn(compacted_state, valid_mask)

    return compacted_state, new_output_state, new_active_indices