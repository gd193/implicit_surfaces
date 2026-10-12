import jax
import jax.numpy as jnp
import jax.random as jr

from typing import Callable, Tuple, Any


def sample_points_batched(fn: Callable[[Any, jnp.ndarray], jnp.ndarray],
                          params: Any,
                          accept_sample_fn: Callable[[jnp.ndarray], jnp.ndarray],
                          init_buffer: Any,
                          key: jnp.ndarray,
                          aa: jnp.ndarray,
                          bb: jnp.ndarray,
                          num_points: int,
                          batch_size: int,
                          max_iters: int = 1000) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Sample points uniformly in the box defined by aa and bb, and classify them using the provided function.
    Args:
        fn: A function that takes (params, x) and returns a scalar value for each point x.
        params: Parameters to pass to the function fn.
        accept_sample_fn: A function that takes the output of fn and returns a boolean array indicating which points are accepted (0 no or 1 yes).
        init_buffer: Unused; kept for call compatibility.
        key: JAX random key for sampling.
        aa: Lower corner of the bounding box (shape (dim,)).
        bb: Upper corner of the bounding box (shape (dim,)).
        num_points: Total number of points to sample.
        batch_size: Number of points to sample in each batch for classification (any size).
    Returns:
        points: An array of shape (num_points, dim) containing the sampled points,
            in float_dtype(aa, bb). Rows past the accepted count are zero.
        is_finished: True iff num_points points were accepted within max_iters batches.
    """
    dtype = jnp.result_type(aa, bb, jnp.float32)
    aa, bb = jnp.asarray(aa, dtype), jnp.asarray(bb, dtype)
    dim = aa.shape[0]

    init_buffer = jnp.zeros((num_points, dim), dtype)
    init_state = (key, init_buffer, jnp.array(0, dtype=jnp.int32), jnp.array(0, dtype=jnp.int32))  # (key, buffer, count, iteration)

    def cond_fn(state):
        _, _, count, i = state
        # Stop when buffer is full OR max_iters reached
        return jnp.logical_and(count < num_points, i < max_iters)

    def body_fn(state):
        key, buffer, count, i = state
        key_batch, key_next = jr.split(key)

        # Sample a batch of points uniformly in the box
        batch_points = jr.uniform(key_batch, shape=(batch_size, dim), dtype=dtype, minval=aa, maxval=bb)
        fn_values = jax.vmap(fn, in_axes=(None, 0))(params, batch_points)
        valid_mask = jax.vmap(accept_sample_fn)(fn_values).astype(bool)

        # Each accepted point goes to count + its rank among accepted points;
        # rejected points and overflow past num_points are dropped, so earlier
        # rows are never overwritten.
        rank = jnp.cumsum(valid_mask, dtype=jnp.int32) - 1
        target = jnp.where(valid_mask, count + rank, jnp.int32(num_points))
        buffer_updated = buffer.at[target].set(batch_points, mode='drop')
        n_written = jnp.minimum(jnp.sum(valid_mask, dtype=jnp.int32), num_points - count)
        return (key_next, buffer_updated, count + n_written, i + 1)

    _, final_buffer, final_count, _ = jax.lax.while_loop(cond_fn, body_fn, init_state)
    is_finished = final_count >= num_points
    return final_buffer, is_finished


def sample_domain_points(surface, params: Any, key: jnp.ndarray, aa: jnp.ndarray, bb: jnp.ndarray,
                         num_points: int, batch_size: int = 65536,
                         max_iters: int = 1000) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Uniform points in the box [aa, bb) that lie inside the surface's domain,
    i.e. signed_value(params, x) < 0 (level-set aware)."""
    return sample_points_batched(surface.signed_value, params, lambda v: v < 0, None, key,
                                 aa, bb, num_points, batch_size, max_iters)
