"""Primitives for any dimension (sphere_sdf is a circle in 2D). The *_sdf
suffix means exact signed distance, negative inside.
Source: Inigo Quilez (https://iquilezles.org/articles/distfunctions2d/) and (https://iquilezles.org/articles/distfunctions/)
"""
import jax
import jax.numpy as jnp


def safe_norm(x, ord=2, axis=None, keepdims=False, eps=1e-12):
    """
    Compute the norm of x while ensuring numerical stability by adding a small epsilon to prevent division by zero.
    """
    return jnp.sqrt(jnp.sum(jax.lax.integer_pow(jnp.abs(x), ord), axis=axis, keepdims=keepdims) + eps)


def sphere_sdf(params, x):
    """
    A simple signed distance function for a sphere defined by its center and radius.
    params: a dictionary containing:
        - 'center' (jnp.ndarray)
        - 'radius' (float)
    x: the point at which to evaluate the SDF
    """
    center = params['center']
    radius = params['radius']
    if center.shape != x.shape:
        raise ValueError(f"Center and point must have the same dimensions (got {center.shape} and {x.shape})")
    if isinstance(radius, jnp.ndarray) and radius.shape != ():
        raise ValueError(f"Radius must be a scalar (got shape {radius.shape})")
    return safe_norm(x - center, 2) - radius


def bbox_sdf(params, x):
    """
    A simple signed distance function for an axis-aligned bounding box defined by its min and max corners.
    params: a dictionary containing :
        - 'min_corner' (jnp.ndarray)
        - 'max_corner' (jnp.ndarray)
    x: the point at which to evaluate the SDF
    """
    min_corner = params['min_corner']
    max_corner = params['max_corner']
    if min_corner.shape != x.shape:
        raise ValueError(f"Min corner and point must have the same dimensions (got {min_corner.shape} and {x.shape})")
    if max_corner.shape != x.shape:
        raise ValueError(f"Max corner and point must have the same dimensions (got {max_corner.shape} and {x.shape})")
    
    center = (min_corner + max_corner) * 0.5
    half_extents = (max_corner - min_corner) * 0.5
    q = jnp.abs(x - center) - half_extents
    outside_dist = safe_norm(jnp.maximum(q, 0), 2)
    inside_dist = jnp.minimum(jnp.max(q), 0)
    return outside_dist + inside_dist


def rounded_box_sdf(params, x):
    """
    Signed distance to an axis-aligned box with rounded edges, any dimension.
    params: a dictionary containing:
        - 'center' (jnp.ndarray)
        - 'half_extents' (jnp.ndarray): half size including the rounding
        - 'radius' (float): edge rounding radius, at most min(half_extents)
    x: the point at which to evaluate the SDF
    """
    q = jnp.abs(x - params['center']) - params['half_extents'] + params['radius']
    return safe_norm(jnp.maximum(q, 0)) + jnp.minimum(jnp.max(q), 0) - params['radius']


def capsule_sdf(params, x):
    """
    Signed distance to a capsule (segment a-b swept by a ball), any dimension.
    params: a dictionary containing:
        - 'a', 'b' (jnp.ndarray): segment end points
        - 'radius' (float)
    x: the point at which to evaluate the SDF
    """
    pa, ba = x - params['a'], params['b'] - params['a']
    h = jnp.clip(jnp.dot(pa, ba) / jnp.dot(ba, ba), 0., 1.)
    return safe_norm(pa - ba * h) - params['radius']
