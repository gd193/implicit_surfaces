import jax.numpy as jnp
from jnp_bounds.types import MixedIntervalPolynomial
from jnp_bounds.mixed_polynomials_arithmetic import coefficient
from jnp_bounds import interval_arithmetic

from typing import Union
from jnp_bounds.polynomial_roots import solve_3rd_order_bound






def eigenvalues_of_hermitian_2x2(a, b, c):
    """
    Computes eigenvalues using the deviatoric (geometric) decomposition.
    More stable and fewer FLOPs.
    source: https://hal.science/hal-01501221/document
    """
    mu = (a + b) * 0.5
    delta = (a - b) * 0.5
    radius = jnp.sqrt(delta**2 + c**2)
    
    lambda1 = mu - radius
    lambda2 = mu + radius
    
    return lambda1, lambda2


def eigenvalues_of_hermitian_3x3(a, b, c, d, e, f):
    """
    Computes the maximum absolute eigenvalue of a 3x3 Hermitian matrix:
    | a d f |
    | d b e |
    | f e c |
    using the closed-form solution for eigenvalues of a cubic characteristic polynomial.
    source: https://hal.science/hal-01501221/document
    """
    x1 = a**2 + b**2 + c**2 - a*b - a*c - b*c + 3*(d**2 + e**2 + f**2)
    x2 = -(2*a - b - c)*(2*b - a - c)*(2*c - a - b) + \
            9 * ((2*c - a - b)*d**2 + (2*b - a - c)*f**2 + (2*a - b - c)*e**2) - 54 * d*e*f
    
    # JAX Safety: prevent negative values caused by floating-point drift
    safe_x1 = jnp.maximum(0.0, x1)
    inner_sqrt = jnp.maximum(0.0, 4 * safe_x1**3 - x2**2)

    phi = jnp.arctan2(jnp.sqrt(inner_sqrt), x2)
    phi = jnp.where(x2 <= 0, phi + jnp.pi, phi)

    lambda1 = (a + b + c - 2 * jnp.sqrt(safe_x1) * jnp.cos(phi / 3)) / 3
    lambda2 = (a + b + c + 2 * jnp.sqrt(safe_x1) * jnp.cos((phi - jnp.pi) / 3)) / 3
    lambda3 = (a + b + c + 2 * jnp.sqrt(safe_x1) * jnp.cos((phi + jnp.pi) / 3)) / 3

    return lambda1, lambda2, lambda3


def max_abs(enc):
    return jnp.maximum(jnp.abs(interval_arithmetic.lower(enc)), jnp.abs(interval_arithmetic.upper(enc)))
    

def bounded_dist_isosurf_1st_order_1d(taylor_enclosure, upper_or_lower: Union[float, jnp.ndarray], isolevel: float=0.0) -> jnp.ndarray:

    a0 = coefficient(taylor_enclosure, (0,))
    enc_a1 = coefficient(taylor_enclosure, (1,))

    K0 = jnp.abs(a0 - isolevel)
    K1 = max_abs(enc_a1)
    
    return jnp.where(K1 > 1e-12, K0 / K1, jnp.inf)


def bounded_dist_isosurf_1st_order_2d(taylor_enclosure, upper_or_lower: Union[float, jnp.ndarray], isolevel: float=0.0) -> jnp.ndarray:

    a00 = coefficient(taylor_enclosure, (0, 0))
    enc_a10 = coefficient(taylor_enclosure, (1, 0))
    enc_a01 = coefficient(taylor_enclosure, (0, 1))

    K0 = jnp.abs(a00 - isolevel)
    K1 = jnp.sqrt(max_abs(enc_a10)**2 + max_abs(enc_a01)**2)
    
    return jnp.where(K1 > 1e-12, K0 / K1, jnp.inf)


def bounded_dist_isosurf_1st_order_3d(taylor_enclosure, upper_or_lower: Union[float, jnp.ndarray], isolevel: float=0.0) -> jnp.ndarray:

    a000 = coefficient(taylor_enclosure, (0, 0, 0))
    enc_a100 = coefficient(taylor_enclosure, (1, 0, 0))
    enc_a010 = coefficient(taylor_enclosure, (0, 1, 0))
    enc_a001 = coefficient(taylor_enclosure, (0, 0, 1))

    K0 = jnp.abs(a000 - isolevel)
    K1 = jnp.sqrt(max_abs(enc_a100)**2 + max_abs(enc_a010)**2 + max_abs(enc_a001)**2)
    
    return jnp.where(K1 > 1e-12, K0 / K1, jnp.inf)    


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
    linear_bound = jnp.sqrt(a10**2 + a01**2)
    bound = 2 * jnp.abs(const_term) / (linear_bound + jnp.sqrt(linear_bound ** 2 + 4 * jnp.abs(const_term) * to_bound_eigenvalue))
    return bound


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
    
    const_term = jnp.abs(a0 - isolevel)
    linear_bound = jnp.abs(a1)
    to_bound_eigenvalue = jnp.abs(a2_val)  # In 1D, the absolute coefficient IS the eigenvalue

    # 2. Branchless single-line Citardauq bound!
    bound = 2 * const_term / (linear_bound + jnp.sqrt(linear_bound ** 2 + 4 * const_term * to_bound_eigenvalue))

    return bound

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
    linear_bound = jnp.sqrt(a100**2 + a010**2 + a001**2)
    lambda_1_1, lambda_2_1, lambda_3_1 = eigenvalues_of_hermitian_3x3(a200_val, a020_val, a002_val, 
                                                       0.5 * a110_val, 0.5 * a011_val, 0.5 * a101_val)
    to_bound_eigenvalue_1 = jnp.maximum(jnp.abs(lambda_1_1), jnp.maximum(jnp.abs(lambda_2_1), jnp.abs(lambda_3_1)))
    lambda_1_2, lambda_2_2, lambda_3_2 = eigenvalues_of_hermitian_3x3(a200_val, a020_val, a002_val, 
                                                       0.5 * a110_val, 0.5 * a011_val,  -0.5 * a101_val)
    to_bound_eigenvalue_2 = jnp.maximum(jnp.abs(lambda_1_2), jnp.maximum(jnp.abs(lambda_2_2), jnp.abs(lambda_3_2)))
    to_bound_eigenvalue = jnp.maximum(to_bound_eigenvalue_1, to_bound_eigenvalue_2)

    bound = 2 * jnp.abs(const_term) / (linear_bound + jnp.sqrt(linear_bound ** 2 + 4 * jnp.abs(const_term) * to_bound_eigenvalue))
    return bound


def bounded_dist_isosurf_3rd_order_1d(taylor_enclosure, upper_or_lower: Union[float, jnp.ndarray], isolevel: float=0.0) -> jnp.ndarray:

    a0 = coefficient(taylor_enclosure, (0,))
    a1 = coefficient(taylor_enclosure, (1,))
    a2 = coefficient(taylor_enclosure, (2,))
    enc_a3 = coefficient(taylor_enclosure, (3,))
    
    K0 = jnp.abs(a0 - isolevel)
    K1 = jnp.abs(a1)
    K2 = jnp.abs(a2)
    K3 = max_abs(enc_a3)

    # Solve K3*d^3 + K2*d^2 + K1*d - K0 = 0
    bound = solve_3rd_order_bound(K0, K1, K2, K3, steps=4)
    return bound


def bounded_dist_isosurf_3rd_order_2d(taylor_enclosure, upper_or_lower: Union[float, jnp.ndarray], isolevel: float=0.0) -> jnp.ndarray:

    # Extract 0th, 1st and 2nd order coefficients
    a00 = coefficient(taylor_enclosure, (0, 0))
    a10 = coefficient(taylor_enclosure, (1, 0))
    a01 = coefficient(taylor_enclosure, (0, 1))
    a20 = coefficient(taylor_enclosure, (2, 0))
    a02 = coefficient(taylor_enclosure, (0, 2))
    a11 = coefficient(taylor_enclosure, (1, 1))

    # Extract 3rd order
    enc_a30 = coefficient(taylor_enclosure, (3, 0))
    enc_a03 = coefficient(taylor_enclosure, (0, 3))
    enc_a21 = coefficient(taylor_enclosure, (2, 1))
    enc_a12 = coefficient(taylor_enclosure, (1, 2))

    # K0 & K1 bounds
    K0 = jnp.abs(a00 - isolevel)
    K1 = jnp.sqrt(a10**2 + a01**2)

    # K2 bound (max absolute eigenvalue of symmetric Hessian)
    lambda_1, lambda_2 = eigenvalues_of_hermitian_2x2(a20, a02, 0.5 * a11)
    K2 = jnp.maximum(jnp.abs(lambda_1), jnp.abs(lambda_2))

    # K3 bound (Frobenius norm of symmetric cubic tensor)
    K3 = jnp.sqrt(max_abs(enc_a30)**2 + 
                  max_abs(enc_a03)**2 + 
                  (1.0/3.0) * max_abs(enc_a21)**2 + 
                  (1.0/3.0) * max_abs(enc_a12)**2)

    # Solve bounding polynomial
    bound = solve_3rd_order_bound(K0, K1, K2, K3, steps=4)
    return bound


def bounded_dist_isosurf_3rd_order_3d(taylor_enclosure, upper_or_lower: Union[float, jnp.ndarray], isolevel: float=0.0) -> jnp.ndarray:

    # Extract 0th, 1st and 2nd order coefficients
    a000 = coefficient(taylor_enclosure, (0, 0, 0))
    a100 = coefficient(taylor_enclosure, (1, 0, 0))
    a010 = coefficient(taylor_enclosure, (0, 1, 0))
    a001 = coefficient(taylor_enclosure, (0, 0, 1))
    a200 = coefficient(taylor_enclosure, (2, 0, 0))
    a020 = coefficient(taylor_enclosure, (0, 2, 0))
    a002 = coefficient(taylor_enclosure, (0, 0, 2))
    a110 = coefficient(taylor_enclosure, (1, 1, 0))
    a101 = coefficient(taylor_enclosure, (1, 0, 1))
    a011 = coefficient(taylor_enclosure, (0, 1, 1))

    # Extract 3rd order
    enc_a300 = coefficient(taylor_enclosure, (3, 0, 0))
    enc_a030 = coefficient(taylor_enclosure, (0, 3, 0))
    enc_a003 = coefficient(taylor_enclosure, (0, 0, 3))
    enc_a210 = coefficient(taylor_enclosure, (2, 1, 0))
    enc_a201 = coefficient(taylor_enclosure, (2, 0, 1))
    enc_a120 = coefficient(taylor_enclosure, (1, 2, 0))
    enc_a021 = coefficient(taylor_enclosure, (0, 2, 1))
    enc_a102 = coefficient(taylor_enclosure, (1, 0, 2))
    enc_a012 = coefficient(taylor_enclosure, (0, 1, 2))
    enc_a111 = coefficient(taylor_enclosure, (1, 1, 1))

    # K0 & K1 bounds
    K0 = jnp.abs(a000 - isolevel)
    K1 = jnp.sqrt(a100**2 + a010**2 + a001**2)

    # K2 bound (max absolute eigenvalue of symmetric Hessian)
    lambda_1, lambda_2, lambda_3 = eigenvalues_of_hermitian_3x3(
        a200, a020, a002, 
        0.5 * a110, 0.5 * a011, 0.5 * a101
    )
    
    # K2 is the spectral radius (largest absolute eigenvalue)
    K2 = jnp.maximum(jnp.abs(lambda_1), jnp.maximum(jnp.abs(lambda_2), jnp.abs(lambda_3)))

    # K3 bound (Frobenius norm of symmetric cubic tensor)
    K3 = jnp.sqrt(max_abs(enc_a300)**2 + max_abs(enc_a030)**2 + max_abs(enc_a003)**2 + 
                  (1.0/3.0) * (max_abs(enc_a210)**2 + max_abs(enc_a201)**2 + 
                               max_abs(enc_a120)**2 + max_abs(enc_a021)**2 + 
                               max_abs(enc_a102)**2 + max_abs(enc_a012)**2) + 
                  (1.0/6.0) * max_abs(enc_a111)**2)

    # Solve bounding polynomial
    bound = solve_3rd_order_bound(K0, K1, K2, K3, steps=4)
    return bound


def empty_sphere_1st_order_tensor(taylor_enclosure: MixedIntervalPolynomial, 
                                  upper_or_lower: Union[float, jnp.ndarray], 
                                  dim: int, 
                                  isolevel: float=0.0) -> jnp.ndarray:
    if dim == 3:
        return bounded_dist_isosurf_1st_order_3d(taylor_enclosure, upper_or_lower, isolevel)
    elif dim == 2:
        return bounded_dist_isosurf_1st_order_2d(taylor_enclosure, upper_or_lower, isolevel)
    elif dim == 1:
        return bounded_dist_isosurf_1st_order_1d(taylor_enclosure, upper_or_lower, isolevel)
    else:
        raise ValueError(f"Dimension {dim} not supported for empty sphere computation.")

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
    

def empty_sphere_3rd_order_tensor(taylor_enclosure: MixedIntervalPolynomial, 
                                  upper_or_lower: Union[float, jnp.ndarray],
                                  dim: int,
                                  isolevel: float=0.0) -> jnp.ndarray:
    if dim == 3:
        return bounded_dist_isosurf_3rd_order_3d(taylor_enclosure, upper_or_lower, isolevel)
    elif dim == 2:
        return bounded_dist_isosurf_3rd_order_2d(taylor_enclosure, upper_or_lower, isolevel)
    elif dim == 1:
        return bounded_dist_isosurf_3rd_order_1d(taylor_enclosure, upper_or_lower, isolevel)
    else:
        raise ValueError(f"Dimension {dim} not supported for empty sphere computation.")
