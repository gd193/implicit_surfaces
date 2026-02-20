import jax
import jax.numpy as jnp
import jax.random as jr

from typing import Callable, Tuple, Any, cast


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
        key: JAX random key for sampling.
        aa: Lower corner of the bounding box (shape (dim,)).
        bb: Upper corner of the bounding box (shape (dim,)).
        num_points: Total number of points to sample.
        batch_size: Number of points to sample in each batch for classification.
    Returns:
        points: An array of shape (num_points, dim) containing the sampled points.
    """
    dim = aa.shape[0]
    num_batches = (num_points + batch_size - 1) // batch_size  # ceil division

    init_buffer = jnp.zeros((num_points, dim))
    init_state = (key, init_buffer, jnp.array(0, dtype=jnp.int32), jnp.array(0, dtype=jnp.int32))  # (key, buffer, count, iteration)

    def cond_fn(state):
        _, _, count, i = state
        # Stop when buffer is full OR max_iters reached
        return jnp.logical_and(count < num_points, i < max_iters)
    
    def body_fn(state):
        key, buffer, count, i = state
        key_batch, key_next = jr.split(key)

        # Sample a batch of points uniformly in the box
        batch_points = jr.uniform(key_batch, shape=(batch_size, dim), minval=aa, maxval=bb)
        fn_values = jax.vmap(fn, in_axes=(None, 0))(params, batch_points)
        valid_mask = jax.vmap(accept_sample_fn)(fn_values)

        # compact valid points and update buffer
        sort_indices = jnp.argsort(~valid_mask)  # valid points (True) will come first
        sorted_points = batch_points[sort_indices]

        # write size arithmetic
        num_valid = jnp.sum(valid_mask)
        space_left = num_points - count
        n_to_write = jnp.minimum(num_valid, space_left)

        # write valid points to buffer and return
        buffer_updated = jax.lax.dynamic_update_slice(buffer, sorted_points, (count, 0))
        return (key_next, buffer_updated, count + n_to_write, i + 1)
    
    _, final_buffer, final_count, _ = jax.lax.while_loop(cond_fn, body_fn, init_state)
    is_finished = final_count >= num_points
    return final_buffer, is_finished