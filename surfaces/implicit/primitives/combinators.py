"""Combinators. They act on SDF values and return 1-Lipschitz fields, not
exact SDFs: |f(x)| <= distance to {f = 0} still holds, so empty-sphere
radii stay safe (use lipschitz_bound=1), but steps can be shorter.
"""
import jax.numpy as jnp


def union(a, b):
    """min(a, b). Exact outside, underestimates distance inside."""
    return jnp.minimum(a, b)


def intersection(a, b):
    """max(a, b). Exact inside, underestimates distance outside."""
    return jnp.maximum(a, b)


def subtraction(a, b):
    """a with b removed: max(a, -b). Underestimates distance."""
    return jnp.maximum(a, -b)


def smooth_union(a, b, k):
    """
    Polynomial smooth minimum with blend radius k > 0 (adds material in the
    blend). Its gradient is a convex combination of the inputs' gradients,
    so it stays 1-Lipschitz.
    source: https://iquilezles.org/articles/smin/
    """
    h = jnp.maximum(k - jnp.abs(a - b), 0.) / k
    return jnp.minimum(a, b) - h * h * k * 0.25
