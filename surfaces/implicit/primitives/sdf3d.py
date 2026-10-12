"""3D primitives: points x of shape (3,). Exact signed distances, negative inside.
Source: Inigo Quilez (https://iquilezles.org/articles/distfunctions/)
"""
import jax.numpy as jnp

from .sdfnd import safe_norm


def torus_sdf(params, x):
    """
    Signed distance to a torus around the z axis.
    params: a dictionary containing:
        - 'center' (jnp.ndarray) of shape (3,)
        - 'major' (float): distance from the center to the tube's center line
        - 'minor' (float): tube radius
    x: (3,) point
    """
    p = x - params['center']
    q = jnp.stack([safe_norm(p[:2]) - params['major'], p[2]])
    return safe_norm(q) - params['minor']
