import jax
import jax.numpy as jnp
import jax.random as jr
from typing import Tuple
from utils.types import BoxBoundarySample


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


def sample_uniform_2sphere(key: jax.Array, dtype: jnp.dtype, shape=()) -> jnp.ndarray:
    """
    Sample a point uniformly on the surface of a unit 2-sphere (sphere in 3D).
    Taken from Mitsuba https://github.com/mitsuba-renderer/mitsuba3/blob/master/include/mitsuba/core/warp.h square_to_uniform_sphere
    Args:
        key: jax.random.PRNGKey
        dtype: floating-point dtype
        shape: leading batch dimensions
    Returns:
        jnp.ndarray of shape (*shape, 3) sampled points on the unit sphere
    """
    uv = jr.uniform(key, shape=(*shape, 2), minval=0.0, maxval=1.0, dtype=dtype)
    z = 2.0 * uv[..., 0] - 1.0
    r = jnp.sqrt(jnp.maximum(0.0, 1.0 - z * z))
    phi = 2.0 * jnp.pi * uv[..., 1]
    return jnp.stack([r * jnp.cos(phi), r * jnp.sin(phi), z], axis=-1)


def unit_interval(key: jax.Array, dtype: jnp.dtype) -> jnp.ndarray:
    """
    Uniform sample in (0, 1]: 1 - U[0, 1). Its smallest value is the grid step
    (2^-24 in float32), not 0. uniform(minval=tiny) instead maps the 0 draw
    (probability 2^-23 in float32) to tiny = 1e-38, a ball radius of ~1e-13 R
    and a Green's weight ~R^2 / r near 1e12 in 3D: a few such outliers per
    50M source samples.
    """
    return 1.0 - jr.uniform(key, dtype=dtype)


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
    radius = jnp.sqrt(unit_interval(key_r, dtype))
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
    radius = jnp.cbrt(unit_interval(key_r, dtype))
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
    # Guard against r=0 / R=0 (sample_uniform_*ball can produce r=0; an empty-
    # sphere classifier failure can produce R=0). The Greens function is a
    # density and contributes zero to a volume integral on a measure-zero set,
    # so the limit value is 0. Without this we'd produce ±inf · 0 → NaN that
    # poisons accumulator buffers.
    safe = jnp.logical_and(r > 0, R > 0)
    safe_r = jnp.where(safe, r, 1.0)
    safe_R = jnp.where(safe, R, 1.0)
    green_fn_val = 1 / (2.0 * jnp.pi) * (jnp.log(safe_R / safe_r))
    green_fn_val = jnp.where(safe, green_fn_val, 0.0)
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
    # Guard against r=0 / R=0 (see 1ball above for why).
    safe = jnp.logical_and(r > 0, R > 0)
    safe_denom = jnp.where(safe, r * R, 1.0)
    green_fn_val = 1 / (4.0 * jnp.pi) * (R - r) / safe_denom
    green_fn_val = jnp.where(safe, green_fn_val, 0.0)
    return green_fn_val * ball_volume


def sample_green_radius(key, dim, dtype):
    """Normalized radial density proportional to the ball Green function.

    2D: 4t log(1/t), sampled as sqrt(U V).
    3D: 6t(1-t), sampled as the median of three uniforms (Beta(2,2)).
    Its integral over the ball is R^2/(2*dim).
    """
    keys = jr.split(key, 3)
    u = jax.vmap(lambda k: unit_interval(k, dtype))(keys)
    if dim == 2:
        return jnp.sqrt(u[0] * u[1])
    if dim == 3:
        a, b, c = u
        return jnp.maximum(jnp.minimum(a, b), jnp.minimum(jnp.maximum(a, b), c))
    raise ValueError("Green sampling supports dimensions 2 and 3")


def sample_green_1ball(key, dtype):
    """Green-weighted sample in the unit disk."""
    kr, kd = jr.split(key)
    return sample_green_radius(kr, 2, dtype), sample_uniform_1sphere(kd, dtype)


def sample_green_2ball(key, dtype):
    """Green-weighted sample in the unit ball."""
    kr, kd = jr.split(key)
    return sample_green_radius(kr, 3, dtype), sample_uniform_2sphere(kd, dtype)


def sample_box_boundary_in_ball(key, position, radius, lower, upper):
    """Sample a box face in a ball, in 2D or 3D, without Green weighting.

    In 3D a uniform radial distance gives conditional area PDF 1/(2*pi*a*rho).
    In 2D t=U*V with a random sign gives conditional length PDF -log(t)/(2*a).
    Each of the 2*dim face planes is chosen with equal probability.
    """
    dim = position.shape[0]
    if dim not in (2, 3):
        raise ValueError('box boundary sampling supports dimensions 2 and 3')
    lo, hi = jnp.asarray(lower, position.dtype), jnp.asarray(upper, position.dtype)
    if lo.shape != position.shape or hi.shape != position.shape:
        raise ValueError('box corners must match the position shape')
    face_key, radial_key, angle_key, second_key = jr.split(key, 4)
    face = jr.randint(face_key, (), 0, 2*dim)
    axis = face % dim
    coordinate = jnp.where(face < dim, lo[axis], hi[axis])
    normal = jax.nn.one_hot(axis, dim, dtype=position.dtype)*jnp.where(face < dim, -1, 1)
    center = position.at[axis].set(coordinate)
    distance_to_plane = jnp.abs(coordinate-position[axis])
    safe_radius = jnp.where(radius > 0, radius, 1.)
    a = jnp.sqrt(jnp.maximum((safe_radius-distance_to_plane)*(safe_radius+distance_to_plane), 0.))
    t = unit_interval(radial_key, position.dtype)
    if dim == 3:
        rho = a*t
        angle = 2*jnp.pi*unit_interval(angle_key, position.dtype)
        sample = center.at[(axis+1) % dim].add(rho*jnp.cos(angle))
        sample = sample.at[(axis+2) % dim].add(rho*jnp.sin(angle))
    else:
        t = t*unit_interval(second_key, position.dtype)
        rho = a*t
        sign = jnp.where(jr.bernoulli(angle_key), 1., -1.)
        sample = center.at[(axis+1) % dim].add(sign*rho)
    r = jnp.hypot(distance_to_plane, rho)
    valid = ((radius > distance_to_plane) & (r < radius)
             & jnp.all(sample >= lo) & jnp.all(sample <= hi))
    sample = jnp.where(valid, sample, jnp.clip(center, lo, hi))
    return BoxBoundarySample(sample, normal, r, a, t, valid)


def weighted_greens_fn_box_boundary(sample: BoxBoundarySample, radius):
    """Positive ball Laplace Green function divided by the face-sampling PDF.

    Algebraically cancel the singular Green factor against the sampler density
    before evaluation, as for the volume-weighted helpers. No boundary flux or
    contact/solid-angle correction is included; those belong to the estimator.
    """
    dim = sample.position.shape[0]
    r, a, t = sample.distance, sample.face_radius, sample.radial_fraction
    safe_radius = jnp.where(radius > 0, radius, 1.)
    safe_r = jnp.where(r > 0, r, 1.)
    if dim == 3:
        weight = dim*a*((a*t)/safe_r)*jnp.maximum(1-r/safe_radius, 0.)
    else:
        log_ratio = jnp.log1p(jnp.maximum(safe_radius-r, 0.)/safe_r)
        log_density = jnp.where(t < 1, -jnp.log(t), 1.)
        weight = (2*dim*a/jnp.pi)*(log_ratio/log_density)
    return jnp.where(sample.valid, weight, 0.)
