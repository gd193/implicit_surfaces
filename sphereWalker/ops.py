"""Stateless JAX primitives shared by the sphereWalker integrators."""

from functools import partial
from numbers import Integral

import jax
import jax.numpy as jnp
import jax.random as jr

from sphereWalker.types import WalkerTermination as WT
from utils.jnp_array_compaction import (
    compact_tree,
    geometric_size_generator,
    uniform_size_generator,
)
from utils.sampling_primitives import (
    sample_uniform_1ball,
    sample_uniform_1sphere,
    sample_uniform_2ball,
    sample_uniform_2sphere,
    weighted_greens_fn_1ball,
    weighted_greens_fn_2ball,
)


sphere_sampling_fn_select = {
    2: sample_uniform_1sphere,
    3: sample_uniform_2sphere,
}

ball_sampling_fn_select = {
    2: sample_uniform_1ball,
    3: sample_uniform_2ball,
}

weight_greens_fn_select = {
    2: weighted_greens_fn_1ball,
    3: weighted_greens_fn_2ball,
}


def walker_status(state, residual, radius, next_tr, eps, max_steps):
    """A terminal condition is not automatically a boundary hit."""
    # TODO: Replace residual-based termination with a bounded closest-point
    # query: minimize ||y - x||^2 subject to F(y) = 0 using interval bounds
    # and branch-and-bound. See Huang, "Geometric Queries on Closed Implicit
    # Surfaces for Walk on Stars", Sec. 3.1, Eq. (1):
    # algo_paper/3757376.3771378.pdf (doi:10.1145/3757376.3771378).
    # Retain cheap empty-sphere queries for walking; return a boundary point,
    # outward normal, distance lower/upper bounds, and convergence status.
    # A lower bound supports safe steps; an upper bound <= eps proves proximity.
    # Use the located boundary point for g, D, V_n, and the differential offset,
    # checking that the offset stays inside the domain. Projection failure must
    # remain explicit. Only tighten the pruning upper bound with a feasible
    # point or certified root-containing box; an UNKNOWN box is insufficient.
    hit = jnp.abs(residual) <= eps
    exhausted = state.nSteps >= max_steps
    valid = (jnp.isfinite(residual) & jnp.isfinite(radius) & (radius >= 0)
             & jnp.all(jnp.isfinite(state.position)) & jnp.all(jnp.isfinite(state.w)))
    valid &= hit | exhausted | (jnp.isfinite(next_tr) & (next_tr > 0))
    return jnp.asarray(jnp.where(~valid, WT.INVALID,
                       jnp.where(hit, WT.BOUNDARY_HIT,
                       jnp.where(exhausted, WT.MAX_STEPS, WT.ACTIVE))), dtype=jnp.int32)


def finish_step(state, updated):
    """Preserve terminal states until they are reduced or change phase."""
    # Loop-carried state keeps the input geometry precision even under x64.
    updated = jax.tree.map(lambda old, new: jnp.asarray(new, dtype=old.dtype), state, updated)
    finite = jnp.all(jnp.isfinite(updated.position)) & jnp.all(jnp.isfinite(updated.w))
    updated = updated._replace(status=jnp.where(finite, updated.status, WT.INVALID),
                               done_flag=updated.done_flag | ~finite)
    return jax.tree.map(lambda old, new: jnp.where(state.done_flag, old, new), state, updated)


def _require_positive_int(name, value):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
        raise ValueError(f'{name} must be a positive integer')
    return int(value)


def average_draws(key, sample_fn, *, count: int, chunk_size: int = 32):
    """Mean of `count` draws of `sample_fn(key)`, vmapped in fixed-width chunks.

    Storage is independent of the budget; one copy of the sampler is traced.
    Draw n uses fold_in(key, n), so results do not depend on chunk_size.
    """
    count = _require_positive_int('count', count)
    chunk_size = _require_positive_int('chunk_size', chunk_size)
    width = min(count, chunk_size)
    offsets0 = jnp.arange(width)
    zeros = jax.tree.map(lambda s: jnp.zeros(s.shape, jnp.result_type(s.dtype, 0.)),
                         jax.eval_shape(sample_fn, jr.fold_in(key, 0)))

    def add_block(i, total):
        offsets = i * width + offsets0
        values = jax.vmap(sample_fn)(jax.vmap(lambda n: jr.fold_in(key, n))(offsets))

        def add(a, b):
            mask = (offsets < count).reshape((width,) + (1,) * (b.ndim - 1))
            return a + jnp.sum(jnp.where(mask, jnp.asarray(b, a.dtype), jnp.asarray(0, a.dtype)), axis=0)
        return jax.tree.map(add, total, values)

    total = jax.lax.fori_loop(0, (count + width - 1) // width, add_block, zeros)
    return jax.tree.map(lambda x: x / count, total)


def run_chunked(mask, bucket_size, body_fn, init_carry):
    """Run `body_fn` over the set lanes of `mask` in chunks of `bucket_size`.

    body_fn(idx, safe_idx, active, carry) -> carry. `idx` holds buffer indices
    (n for padding, so `.at[idx]...(mode='drop')` scatters skip it); `safe_idx`
    replaces padding with the first selected lane, so gathers only ever see
    real selected states (never index 0 or zero-filled data). The body runs
    ceil(count / bucket) times, i.e. never when no lane is selected.
    """
    bucket_size = _require_positive_int('bucket_size', bucket_size)
    n = mask.shape[0]
    if n == 0:
        return init_carry
    bucket = min(bucket_size, n)
    count = jnp.sum(mask, dtype=jnp.int32)
    indices = jnp.nonzero(mask, size=n, fill_value=n)[0].astype(jnp.int32)
    # Pad so every chunk slice stays in bounds.
    indices = jnp.concatenate([indices, jnp.full((bucket,), n, jnp.int32)])

    def cond(carry):
        return carry[0] * bucket < count

    def body(carry):
        chunk, inner = carry
        idx = jax.lax.dynamic_slice(indices, (chunk * bucket,), (bucket,))
        active = idx < n
        safe_idx = jnp.where(active, idx, indices[0])
        return chunk + jnp.int32(1), body_fn(idx, safe_idx, active, inner)

    return jax.lax.while_loop(cond, body, (jnp.int32(0), init_carry))[1]


def first_ball_source_weight(dim, radius, unit_radius, ball_direction, source):
    """First-ball source gradient sample, Sawhney & Crane (2020), Eq. 14.

    The source radius is uniform in the first ball (t = r/R, PDF included),
    which keeps the weight R (1 - t^dim) bounded. The product is formed in
    the promoted precision of `source` and the geometry; callers cast it.
    """
    return (jnp.asarray(source)[..., None]
            * (radius * (1.0 - unit_radius ** dim)) * ball_direction)


def get_scatter_fn(free_indices: jnp.ndarray, is_valid_mask: jnp.ndarray):
    """Reset freed slots, then write new tasks into the valid ones.

    Invalid tasks target the buffer length, which the scatter drops; no
    problem size enters the kernel, so it is reused across sample counts.
    """
    def scatter(arr, val, reset_value):
        arr = arr.at[free_indices].set(reset_value, mode='drop')
        target_indices = jnp.where(is_valid_mask, free_indices, arr.shape[0])
        return arr.at[target_indices].set(val, mode='drop')
    return scatter


def create_zero_buffer(leaf_array: jnp.ndarray, buffer_size: int) -> jnp.ndarray:
    """leaf_array.shape (1, *D) -> zeros of shape (buffer_size, *D)."""
    return jnp.zeros((buffer_size,) + leaf_array.shape[1:], dtype=leaf_array.dtype)


def mask_leaf(leaf_array: jnp.ndarray, mask: jnp.ndarray) -> jnp.ndarray:
    """Zero the rows of a (N, *D) leaf where the (N,) mask is False."""
    mask = jnp.reshape(mask, (leaf_array.shape[0],) + (1,) * (leaf_array.ndim - 1))
    return jnp.where(mask, leaf_array, jnp.array(0, dtype=leaf_array.dtype))


def pad_to_bucket(array: jnp.ndarray, bucket_size: int = 1024):
    """Pad to the next multiple of bucket_size, repeating the first row.

    Repeating a real row (not zeros) avoids evaluating callbacks at invented
    points. Returns (padded_array, original_length).
    """
    N_original = array.shape[0]
    N_padded = ((N_original - 1) // bucket_size + 1) * bucket_size
    pad_len = N_padded - N_original
    if pad_len > 0:
        padding = jnp.tile(array[0:1], (pad_len,) + (1,) * (array.ndim - 1))
        return jnp.concatenate([array, padding], axis=0), N_original
    return array, N_original


def compaction_schedule(size, count, mode):
    match mode:
        case 'uniform':
            return uniform_size_generator(size, count)
        case 'geometric':
            return geometric_size_generator(size, count)
        case _:
            raise ValueError('compaction_mode must be uniform or geometric')


@partial(jax.jit, static_argnames=('new_size',))
def _compact_step(states, parents, is_active, new_size):
    """One full compaction transition jit'd as a single kernel.

    Bundles `compact_tree(...)`, the parent gather and the invalid-parent fill,
    which re-traced per (input_size, new_size) pair when run eagerly. Module
    level so the JIT cache survives across integrator instances.
    """
    new_states, valid_mask, target_indices = compact_tree(states, is_active, new_size)
    new_parents = parents.at[target_indices].get(mode='drop')
    new_parents = jnp.where(valid_mask, new_parents,
                            jnp.array(-1, dtype=new_parents.dtype))
    return new_states, new_parents
