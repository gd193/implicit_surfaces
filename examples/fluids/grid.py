"""Clamped trilinear interpolation for grid-backed fields."""
from itertools import product

import jax.numpy as jnp


def sample_vector_grid(field, x, lower, spacing):
    """Sample all channels with the same cell-centered, clamped interpolant."""
    index = (x-jnp.asarray(lower, x.dtype))/jnp.asarray(spacing, x.dtype)
    base = jnp.floor(index).astype(jnp.int32)
    fraction = index-base.astype(index.dtype)
    flat = field.reshape((-1, field.shape[-1]))
    total = jnp.zeros(field.shape[-1], field.dtype)
    # Scalar flat indices fuse into the gathers. Building separate N-by-3
    # corner index arrays through map_coordinates adds kernels to every walk.
    for a, b, c in product((0, 1), repeat=3):
        i = jnp.clip(base[0]+a, 0, field.shape[0]-1)
        j = jnp.clip(base[1]+b, 0, field.shape[1]-1)
        k = jnp.clip(base[2]+c, 0, field.shape[2]-1)
        weight = ((fraction[0] if a else 1-fraction[0])
                  *(fraction[1] if b else 1-fraction[1])
                  *(fraction[2] if c else 1-fraction[2]))
        total = total+weight*flat[(i*field.shape[1]+j)*field.shape[2]+k]
    return total


def sample_vector_grid_nested(field, x, lower, spacing):
    """Clamped trilinear sampling with seven nested vector interpolations."""
    index = (x-jnp.asarray(lower, x.dtype))/jnp.asarray(spacing, x.dtype)
    base = jnp.floor(index).astype(jnp.int32)
    fraction = index-base.astype(index.dtype)
    low = [jnp.clip(base[a], 0, field.shape[a]-1) for a in range(3)]
    high = [jnp.clip(base[a]+1, 0, field.shape[a]-1) for a in range(3)]
    flat = field.reshape((-1, field.shape[-1]))

    def at(i, j, k):
        return flat.at[(i*field.shape[1]+j)*field.shape[2]+k].get(
            mode='promise_in_bounds', wrap_negative_indices=False)

    def mix(a, b, t):
        return (1-t)*a+t*b

    def along_z(i, j):
        return mix(at(i, j, low[2]), at(i, j, high[2]), fraction[2])

    def along_y(i):
        return mix(along_z(i, low[1]), along_z(i, high[1]), fraction[1])

    return mix(along_y(low[0]), along_y(high[0]), fraction[0])
