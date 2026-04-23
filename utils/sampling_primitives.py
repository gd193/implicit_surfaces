import jax
import jax.numpy as jnp
import jax.random as jr
from typing import Tuple
import math


def sample_uniform_ray_bbox_2d(key: jax.Array, bbox_mins: jnp.ndarray, bbox_maxs: jnp.ndarray, dtype: jnp.dtype) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Sample a ray origin uniformly within a bounding box defined by bbox_mins and bbox_maxs.
    Algorithm 1 from https://arxiv.org/pdf/2506.05268
    Args:
        key: jax.random.PRNGKey
        bbox_mins: jax.ndarray of shape (2,) representing the minimum corner of the bounding box
        bbox_maxs: jax.ndarray of shape (2,) representing the maximum corner of the bounding box
        dtype: jnp.dtype for the output array
    Returns:
        intersects_flag: boolean indicating whether the sampled ray intersects the bounding box
        direction: jnp.ndarray of shape (2,) representing the sampled ray direction
        intersection_point: jnp.ndarray of shape (2,) representing the intersection point of the ray with the bounding box
        max_t_in_box: jnp.ndarray representing the maximum t value for which the ray is inside the bounding box
    """
    # box Properties
    box_center = (bbox_mins + bbox_maxs) / 2.0
    box_size = bbox_maxs - bbox_mins
    box_diagonal_length = jnp.linalg.norm(box_size)
    box_bounding_radius = box_diagonal_length / 2.0

    # random unit direction
    key_direction, key_offset = jr.split(key)
    direction = sample_uniform_1sphere(key_direction, dtype=dtype)
    
    # get normal vector by [x, y] -> [-y, x]
    n_vec = jnp.array([-direction[1], direction[0]])

    # random offset value sampled uniformly in range(-radius, radius)
    u = jr.uniform(key_offset, minval=-box_bounding_radius, maxval=box_bounding_radius, dtype=dtype)
    
    plane_origin = box_center + u * n_vec
    ray_origin = plane_origin - direction * box_bounding_radius

    # determine if the ray o+d*t intersects the bbox using the slab method
    t_min = (bbox_mins - ray_origin) / (direction + jnp.sign(direction) * 1e-12)
    t_max = (bbox_maxs - ray_origin) / (direction + jnp.sign(direction) * 1e-12)
    t_close = jnp.minimum(t_min, t_max)
    t_close = jnp.max(t_close)
    t_far = jnp.maximum(t_min, t_max)
    t_far = jnp.min(t_far)

    # if t_close <= t_far, the ray intersects the bbox, otherwise it does not
    intersects_flag = jnp.logical_and(t_close <= t_far, t_far > 0.0)
    intersection_point = ray_origin + direction * t_close
    max_t_in_box = t_far - t_close

    return intersects_flag, direction, intersection_point, max_t_in_box


def sample_uniform_ray_bbox_3d(key: jax.Array, bbox_mins: jnp.ndarray, bbox_maxs: jnp.ndarray, dtype: jnp.dtype) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Sample a ray origin uniformly within a bounding box defined by bbox_mins and bbox_maxs.
    Algorithm 1 from https://arxiv.org/pdf/2506.05268, also 
    https://github.com/iszihan/implicit-uniform-sampler/blob/main/ImplicitUniformSampler/sampler.py
    Args:
        key: jax.random.PRNGKey
        bbox_mins: jax.ndarray of shape (3,) representing the minimum corner of the bounding box
        bbox_maxs: jax.ndarray of shape (3,) representing the maximum corner of the bounding box
        dtype: jnp.dtype for the output array
    Returns:
        intersects_flag: boolean indicating whether the sampled ray intersects the bounding box
        direction: jnp.ndarray of shape (3,) representing the sampled ray direction
        intersection_point: jnp.ndarray of shape (3,) representing the intersection point of the ray with the bounding box
        max_t_in_box: jnp.ndarray representing the maximum t value for which the ray is inside the bounding box
    """
    # box properties
    box_center = (bbox_mins + bbox_maxs) / 2.0
    box_size = bbox_maxs - bbox_mins
    box_diagonal_length = jnp.sqrt(jnp.sum(box_size ** 2))
    box_bounding_radius = box_diagonal_length / 2.0

    # random unit direction
    key_direction, key_offset = jr.split(key)
    direction = sample_uniform_2sphere(key_direction, dtype=dtype) 

    #get normal and binormal vectors to d using SVD, full_matrices=True to get the full set of singular vectors
    _, __, vh = jnp.linalg.svd(jnp.expand_dims(direction, axis=0), full_matrices=True)
    n_vec = vh[1] # normal vector to d
    b_vec = vh[2] # binormal vector to d

    #random offset values uniformly sampled in range(0, bbox diagonal length)
    Us = jr.uniform(key_offset, shape=(2,), minval=0.0, maxval=1.0, dtype=dtype) * 2.0 - 1.0
    Us *= box_bounding_radius
    u1, u2 = Us
    
    ray_origin = u1 * n_vec + u2 * b_vec + box_center
    ray_origin = ray_origin - direction * box_bounding_radius

    #determine if the ray o+d*t intersects the bbox using the slab method
    t_min = (bbox_mins - ray_origin) / (direction + jnp.sign(direction) * 1e-12)
    t_max = (bbox_maxs - ray_origin) / (direction + jnp.sign(direction) * 1e-12)
    t_close = jnp.minimum(t_min, t_max)
    t_close = jnp.max(t_close)
    t_far = jnp.maximum(t_min, t_max)
    t_far = jnp.min(t_far)

    # if t_close <= t_far, the ray intersects the bbox, otherwise it does not
    intersects_flag = jnp.logical_and(t_close <= t_far, t_far > 0.0)
    intersection_point = ray_origin + direction * t_close
    max_t_in_box = t_far - t_close

    return intersects_flag, direction, intersection_point, max_t_in_box


def sample_uniform_1sphere(key: jax.Array, dtype: jnp.dtype) -> jnp.ndarray:
    """
    Sample a point uniformly on the surface of a unit 1-sphere (circle in 2D).
    Args:
        key: jax.random.PRNGKey
        
    Returns:
        jnp.ndarray of shape (2,) sampled point on the unit circle
    """
    theta = jr.uniform(key, minval=0.0, maxval=2 * jnp.pi, dtype=dtype)
    return jnp.array([jnp.cos(theta), jnp.sin(theta)])


def sample_uniform_2sphere(key: jax.Array, dtype: jnp.dtype) -> jnp.ndarray:
    """
    Sample a point uniformly on the surface of a unit 2-sphere (sphere in 3D).
    Taken from Mitsuba https://github.com/mitsuba-renderer/mitsuba3/blob/master/include/mitsuba/core/warp.h square_to_uniform_sphere
    Args:
        key: jax.random.PRNGKey
    Returns:
        jnp.ndarray of shape (3,) sampled point on the unit sphere
    """
    uv = jr.uniform(key, shape=(2,), minval=0.0, maxval=1.0, dtype=dtype)
    z = 2.0 * uv[0] - 1.0
    r = jnp.sqrt(jnp.maximum(0.0, 1.0 - z * z))
    phi = 2.0 * jnp.pi * uv[1]
    return jnp.array([r * jnp.cos(phi), r * jnp.sin(phi), z])


def sample_uniform_1ball(key: jax.Array, dtype: jnp.dtype) -> Tuple[jnp.ndarray, jnp.ndarray]:
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


def sample_uniform_2ball(key: jax.Array, dtype: jnp.dtype) -> Tuple[jnp.ndarray, jnp.ndarray]:
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
