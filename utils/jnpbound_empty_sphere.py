import jax.numpy as jnp
from jnp_bounds.types import MixedIntervalPolynomial
from jnp_bounds.mixed_polynomials_arithmetic import coefficient
from jnp_bounds import interval_arithmetic

from typing import Union


def eigenvalues_of_hermitian_2x2(a, b, c):
    """
    Computes eigenvalues using the deviatoric (geometric) decomposition.
    More stable and fewer FLOPs.
    source: https://hal.science/hal-01501221/document
    """
    mu = (a + b) * 0.5
    delta = (a - b) * 0.5
    radius = jnp.hypot(delta, c)
    
    lambda1 = mu - radius
    lambda2 = mu + radius
    
    return lambda1, lambda2


def eigenvalues_of_hermitian_3x3(a, b, c, d, e, f):
    """
    Computes the three eigenvalues of a real symmetric matrix:
    | a d f |
    | d b e |
    | f e c |
    using the closed-form solution for eigenvalues of a cubic characteristic polynomial.
    source: https://hal.science/hal-01501221/document
    """
    ab, bc, ca = a - b, b - c, c - a
    # Scale the traceless part before forming the cubic discriminant.
    scale = jnp.maximum(jnp.maximum(jnp.abs(ab), jnp.abs(bc)), jnp.abs(ca))
    scale = jnp.maximum(scale, jnp.maximum(jnp.maximum(jnp.abs(d), jnp.abs(e)), jnp.abs(f)))
    divisor = jnp.where(scale == 0, 1., scale)
    ab, bc, ca = ab / divisor, bc / divisor, ca / divisor
    d, e, f = d / divisor, e / divisor, f / divisor
    da, db, dc = ab - ca, bc - ab, ca - bc
    x1 = 0.5 * (ab**2 + bc**2 + ca**2) + 3 * (d**2 + e**2 + f**2)
    x2 = -da * db * dc + 9 * (dc * d**2 + db * f**2 + da * e**2) - 54 * d * e * f
    inner_sqrt = jnp.maximum(0.0, 4 * x1**3 - x2**2)

    phi = jnp.arctan2(jnp.sqrt(inner_sqrt), x2)

    spread = 2 * (scale * jnp.sqrt(x1))
    lambda1 = (a + b + c - spread * jnp.cos(phi / 3)) / 3
    lambda2 = (a + b + c + spread * jnp.cos((phi - jnp.pi) / 3)) / 3
    lambda3 = (a + b + c + spread * jnp.cos((phi + jnp.pi) / 3)) / 3

    return lambda1, lambda2, lambda3


def max_abs(enc):
    return jnp.maximum(jnp.abs(interval_arithmetic.lower(enc)), jnp.abs(interval_arithmetic.upper(enc)))
    

def _quadratic_radius(const_term, linear_bound, curvature_bound):
    gap = jnp.abs(const_term)
    flat = (linear_bound == 0) & (curvature_bound == 0)
    half_linear = 0.5 * linear_bound
    denominator = half_linear + jnp.hypot(half_linear, jnp.sqrt(gap) * jnp.sqrt(curvature_bound))
    radius = gap / jnp.where(denominator == 0, 1., denominator)
    # Reserve +inf for a finite, off-surface constant enclosure.
    usable = (denominator > 0) & jnp.isfinite(radius)
    radius = jnp.where(usable, radius, 0.)
    return jnp.where(flat & jnp.isfinite(gap) & (gap > 0), jnp.inf, radius)


def bounded_dist_isosurf_2nd_order_2d(taylor_enclosure: MixedIntervalPolynomial, 
                                      upper_or_lower: Union[float, jnp.ndarray], 
                                      isolevel: float=0.0) -> jnp.ndarray:

    a00 = coefficient(taylor_enclosure, (0, 0))
    a10 = coefficient(taylor_enclosure, (1, 0))
    a01 = coefficient(taylor_enclosure, (0, 1))
    enc_a20 = coefficient(taylor_enclosure, (2, 0))
    enc_a02 = coefficient(taylor_enclosure, (0, 2))
    enc_a11 = coefficient(taylor_enclosure, (1, 1))

    # Normalize flag and select bounds branchlessly
    is_upper = (upper_or_lower == 1)
    a20_val = jnp.where(is_upper, interval_arithmetic.upper(enc_a20), interval_arithmetic.lower(enc_a20))
    a02_val = jnp.where(is_upper, interval_arithmetic.upper(enc_a02), interval_arithmetic.lower(enc_a02))
    a11_val = jnp.maximum(jnp.abs(interval_arithmetic.lower(enc_a11)), jnp.abs(interval_arithmetic.upper(enc_a11)))

    lambda1, lambda2 = eigenvalues_of_hermitian_2x2(a20_val, a02_val, 0.5 * a11_val)
    to_bound_eigenvalue = jnp.maximum(jnp.abs(lambda1), jnp.abs(lambda2))

    const_term = a00 - isolevel
    linear_bound = jnp.hypot(a10, a01)
    return _quadratic_radius(const_term, linear_bound, to_bound_eigenvalue)


def bounded_dist_isosurf_2nd_order_1d(taylor_enclosure: MixedIntervalPolynomial, 
                                      upper_or_lower: Union[float, jnp.ndarray], 
                                      isolevel: float=0.0) -> jnp.ndarray:
    """
    Computes a lower bound on the distance to the isosurface defined by the 1D second-order polynomial.
    """

    a0 = coefficient(taylor_enclosure, (0,))
    a1 = coefficient(taylor_enclosure, (1,))
    enc_a2 = coefficient(taylor_enclosure, (2,))

    # Normalize flag and select bounds branchlessly
    is_upper = (upper_or_lower == 1)
    a2_val = jnp.where(is_upper, interval_arithmetic.upper(enc_a2), interval_arithmetic.lower(enc_a2))
    
    const_term = a0 - isolevel
    linear_bound = jnp.abs(a1)
    to_bound_eigenvalue = jnp.abs(a2_val)  # In 1D, the absolute coefficient IS the eigenvalue

    return _quadratic_radius(const_term, linear_bound, to_bound_eigenvalue)


def bounded_dist_isosurf_2nd_order_3d(taylor_enclosure: MixedIntervalPolynomial, 
                                      upper_or_lower: Union[float, jnp.ndarray], 
                                      isolevel: float=0.0) -> jnp.ndarray:
    """
    Computes a lower bound on the distance to the isosurface defined by the 3D second-order polynomial.
    Uses the method from "Efficient and Robust Isosurface Extraction on Unstructured Grids Using Taylor Models"
    """

    a000 = coefficient(taylor_enclosure, (0, 0, 0))
    a100 = coefficient(taylor_enclosure, (1, 0, 0))
    a010 = coefficient(taylor_enclosure, (0, 1, 0))
    a001 = coefficient(taylor_enclosure, (0, 0, 1))

    enc_a200 = coefficient(taylor_enclosure, (2, 0, 0))
    enc_a020 = coefficient(taylor_enclosure, (0, 2, 0))
    enc_a002 = coefficient(taylor_enclosure, (0, 0, 2))
    enc_a110 = coefficient(taylor_enclosure, (1, 1, 0))
    enc_a101 = coefficient(taylor_enclosure, (1, 0, 1))
    enc_a011 = coefficient(taylor_enclosure, (0, 1, 1))

    # Normalize flag and select bounds branchlessly
    is_upper = (upper_or_lower == 1)
    a200_val = jnp.where(is_upper, interval_arithmetic.upper(enc_a200), interval_arithmetic.lower(enc_a200))
    a020_val = jnp.where(is_upper, interval_arithmetic.upper(enc_a020), interval_arithmetic.lower(enc_a020))
    a002_val = jnp.where(is_upper, interval_arithmetic.upper(enc_a002), interval_arithmetic.lower(enc_a002))

    # Mixed terms rely on worst-case for global bounds
    a110_val = max_abs(enc_a110)
    a101_val = max_abs(enc_a101)
    a011_val = max_abs(enc_a011)

    const_term = a000 - isolevel
    linear_bound = jnp.hypot(jnp.hypot(a100, a010), a001)
    lambda_1_1, lambda_2_1, lambda_3_1 = eigenvalues_of_hermitian_3x3(a200_val, a020_val, a002_val, 
                                                       0.5 * a110_val, 0.5 * a011_val, 0.5 * a101_val)
    to_bound_eigenvalue_1 = jnp.maximum(jnp.abs(lambda_1_1), jnp.maximum(jnp.abs(lambda_2_1), jnp.abs(lambda_3_1)))
    lambda_1_2, lambda_2_2, lambda_3_2 = eigenvalues_of_hermitian_3x3(a200_val, a020_val, a002_val, 
                                                       0.5 * a110_val, 0.5 * a011_val,  -0.5 * a101_val)
    to_bound_eigenvalue_2 = jnp.maximum(jnp.abs(lambda_1_2), jnp.maximum(jnp.abs(lambda_2_2), jnp.abs(lambda_3_2)))
    to_bound_eigenvalue = jnp.maximum(to_bound_eigenvalue_1, to_bound_eigenvalue_2)

    return _quadratic_radius(const_term, linear_bound, to_bound_eigenvalue)


def empty_sphere_2nd_order_tensor(taylor_enclosure: MixedIntervalPolynomial, 
                                  upper_or_lower: Union[float, jnp.ndarray], 
                                  dim: int, 
                                  isolevel: float=0.0) -> jnp.ndarray:
    if dim == 3:
        return bounded_dist_isosurf_2nd_order_3d(taylor_enclosure, upper_or_lower, isolevel)
    elif dim == 2:
        return bounded_dist_isosurf_2nd_order_2d(taylor_enclosure, upper_or_lower, isolevel)
    elif dim == 1:
        return bounded_dist_isosurf_2nd_order_1d(taylor_enclosure, upper_or_lower, isolevel)
    else:
        raise ValueError(f"Dimension {dim} not supported for empty sphere computation.")
