"""Bounded, refillable buffers of independent states; no geometry semantics."""
from functools import partial
from numbers import Integral
from typing import Any, Callable

import jax
import jax.numpy as jnp

from utils.jnp_array_compaction import perform_compaction, should_compact, uniform_size_generator

DEFAULT_MAX_ACTIVE_STATES = 4096


def validate_max_active_states(value):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
        raise ValueError('max_active_states must be a positive integer')
    return int(value)


@partial(jax.jit, static_argnames=('initialize', 'write_finished'))
def _refill(inputs, state, output, indices, done, next_index, total, *,
            initialize, write_finished):
    # Flush before replacing a lane, including terminal states at initialization.
    output = write_finished(state, output, jnp.where(done, indices, total))
    pending_indices = next_index + jnp.cumsum(done.astype(jnp.int32)) - 1
    take = done & (pending_indices < total)
    fresh = initialize(inputs, jnp.minimum(pending_indices, total - 1), take)
    state = jax.tree.map(
        lambda old, new: jnp.where(take.reshape(take.shape + (1,)*(old.ndim-1)), new, old),
        state, fresh)
    indices = jnp.where(take, pending_indices, jnp.where(done, total, indices))
    return state, output, indices, next_index + jnp.sum(take)


def run_bounded_batch(params: Any, inputs: Any, output: Any, *,
                      initialize: Callable, step_batch: Callable,
                      finished_mask: Callable, write_finished: Callable,
                      mask_padding: Callable, batch_size: int,
                      max_active_states: int, n_compactions: int):
    """Drain queued tasks through a bounded live-state buffer.

    initialize(inputs, indices, valid) creates only buffer-sized state. Results
    are written once in original task order. Refill precedes stepping, so queued
    work is not blocked by long-lived tasks. Shrink only after the queue empties.
    Input/output storage is caller-owned and may scale with total task count.
    Return (remaining buffer, output, original indices) for diagnostics.
    """
    capacity = min(batch_size, validate_max_active_states(max_active_states))
    indices = jnp.arange(capacity)
    state = initialize(inputs, indices, jnp.ones(capacity, bool))
    if batch_size == 0:
        return state, output, indices
    next_index = capacity
    sizes = uniform_size_generator(capacity, n_compactions)
    bucket = 0
    while True:
        done = finished_mask(state)
        active = int(jnp.sum(~done))
        if next_index < batch_size and active < indices.size:
            state, output, indices, next_index = _refill(
                inputs, state, output, indices, done, next_index, batch_size,
                initialize=initialize, write_finished=write_finished)
            next_index = int(next_index)
            # Fresh tasks may already be invalid/inactive. Drain them before
            # querying and refill again rather than letting them block the queue.
            continue
        if active == 0:
            break
        if next_index == batch_size:
            compact, next_bucket, size = should_compact(active, bucket, sizes)
            if compact:
                bucket = next_bucket
                state, output, indices = perform_compaction(
                    state, output, indices, done, batch_size, size,
                    write_to_output_fn=write_finished, update_state_fn=mask_padding)
        state = step_batch(params, state)
    output = write_finished(state, output, indices)
    return state, output, indices


def bounded_map(fn, *arrays, max_active_states):
    """Evaluate independent shading queries in bounded, padded chunks.

    This is for stateless work; iterative traversal uses run_bounded_batch.
    Outputs retain input order. Padding is discarded before assembly.
    """
    limit = validate_max_active_states(max_active_states)
    n = jax.tree.leaves(arrays)[0].shape[0]
    if n == 0:
        return fn(*arrays)
    size = min(n, limit)
    pieces = []
    for start in range(0, n, size):
        count = min(size, n-start)
        indices = jnp.minimum(start + jnp.arange(size), n-1)
        chunk = jax.tree.map(lambda a: a[indices], arrays)
        result = fn(*chunk)
        jax.block_until_ready(result)
        pieces.append(jax.tree.map(lambda a: a[:count], result))
    return jax.tree.map(lambda *parts: jnp.concatenate(parts), *pieces)


def finished_rays(state):
    return state.done_flag


def mask_finished_padding(state, valid):
    return state._replace(done_flag=state.done_flag | ~valid)
