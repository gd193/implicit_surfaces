import jax
import jax.numpy as jnp
import jax.random as jr
from typing import Tuple


def sample_uniform_1sphere(key: jr.PRNGKey, dtype: jnp.dtype) -> jnp.array:
    """
    Sample a point uniformly on the surface of a unit 1-sphere (circle in 2D).
    Args:
        key: jax.random.PRNGKey
        
    Returns:
        jnp.array of shape (2,) sampled point on the unit circle
    """
    theta = jr.uniform(key, minval=0.0, maxval=2 * jnp.pi, dtype=dtype)
    return jnp.array([jnp.cos(theta), jnp.sin(theta)])


def sample_uniform_2sphere(key: jr.PRNGKey, dtype: jnp.dtype) -> jnp.array:
    """
    Sample a point uniformly on the surface of a unit 2-sphere (sphere in 3D).
    Taken from Mitsuba https://github.com/mitsuba-renderer/mitsuba3/blob/master/include/mitsuba/core/warp.h square_to_uniform_sphere
    Args:
        key: jax.random.PRNGKey
    Returns:
        jnp.array of shape (3,) sampled point on the unit sphere
    """
    key_z, key_phi = jr.split(key)
    z = jr.uniform(key_z, minval=-1.0, maxval=1.0, dtype=dtype)
    r = jnp.sqrt(jnp.maximum(0.0, 1.0 - z * z))
    phi = jr.uniform(key_phi, minval=0.0, maxval=2 * jnp.pi, dtype=dtype)
    return jnp.array([r * jnp.cos(phi), r * jnp.sin(phi), z])


def sample_uniform_1ball(key: jr.PRNGKey, dtype: jnp.dtype) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Sample a point uniformly inside a unit 1-ball (disk in 2D).
    Args:
        key: jax.random.PRNGKey
    Returns:
        jnp.array of shape (1,) sampled radius
        jnp.array of shape (2,) sampled direction in the unit disk
    """
    key_r, key_theta = jr.split(key)
    direction = sample_uniform_1sphere(key_theta, dtype=dtype)
    radius = jnp.sqrt(jr.uniform(key_r, minval=jnp.finfo(dtype).tiny, maxval=1.0, dtype=dtype))
    return radius, direction


def sample_uniform_2ball(key: jr.PRNGKey, dtype: jnp.dtype) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Sample a point uniformly inside a unit 2-ball (ball in 3D).
    Args:
        key: jax.random.PRNGKey
        dtype: jnp.dtype
    Returns:
        jnp.array of shape (1,) sampled radius
        jnp.array of shape (3,) sampled direction in the unit ball
    """
    key_r, key_direction = jr.split(key)
    direction = sample_uniform_2sphere(key_direction, dtype=dtype)
    radius = jnp.cbrt(jr.uniform(key_r, minval=jnp.finfo(dtype).tiny, maxval=1.0, dtype=dtype))
    return radius, direction


def weighted_greens_fn_1ball(r: jnp.ndarray, R: jnp.ndarray) -> jnp.ndarray:
    """
    Weighted Green's function for 1 ball (ball in 2D) with outer radius R.
    Args:
        r: jnp.ndarray of shape (...,) radius
        R: jnp.ndarray of shape (...,) outer radius
    Returns:
        jnp.ndarray of shape (...,) weighted Green's function value
    """
    ball_volume = jnp.pi * R * R
    green_fn_val = 1 / (2.0 * jnp.pi) * (jnp.log(R / r))
    return green_fn_val * ball_volume


def weighted_greens_fn_2ball(r: jnp.ndarray, R: jnp.ndarray) -> jnp.ndarray:
    """
    Weighted Green's function for 2 ball (ball in 3D) with outer radius R.
    Args:
        r: jnp.ndarray of shape (...,) radius
        R: jnp.ndarray of shape (...,) outer radius
    Returns:
        jnp.ndarray of shape (...,) weighted Green's function value
    """
    ball_volume = (4.0 / 3.0) * jnp.pi * R * R * R
    green_fn_val = 1 / (4.0 * jnp.pi) * (R - r) / (r * R)
    return green_fn_val * ball_volume
