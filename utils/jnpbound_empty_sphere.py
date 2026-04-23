import jax
import jax.numpy as jnp
import jnp_bounds.autobounds as jb
from jnp_bounds.types import MixedIntervalPolynomial
from jnp_bounds.utils.dim_degree_coefficients import dim_degree_to_exponents_cache

from typing import Union

multi_index_1d_1st_order = dim_degree_to_exponents_cache.static[(1, 1)][0]
multi_index_1d_2nd_order = dim_degree_to_exponents_cache.static[(1, 2)][0]
multi_index_1d_3rd_order = dim_degree_to_exponents_cache.static[(1, 3)][0]

multi_index_2d_1st_order = dim_degree_to_exponents_cache.static[(2, 1)][0]
multi_index_2d_2nd_order = dim_degree_to_exponents_cache.static[(2, 2)][0]
multi_index_2d_3rd_order = dim_degree_to_exponents_cache.static[(2, 3)][0]

multi_index_3d_1st_order = dim_degree_to_exponents_cache.static[(3, 1)][0]
multi_index_3d_2nd_order = dim_degree_to_exponents_cache.static[(3, 2)][0]
multi_index_3d_3rd_order = dim_degree_to_exponents_cache.static[(3, 3)][0]



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
    return jnp.maximum(jnp.abs(enc.lower), jnp.abs(enc.upper))
    

def bounded_dist_isosurf_1st_order_1d(taylor_enclosure, upper_or_lower: Union[float, jnp.ndarray], isolevel: float=0.0) -> jnp.ndarray:
    index_map = lambda idxs: multi_index_1d_1st_order.index(idxs)

    a0 = taylor_enclosure[index_map([0])]
    enc_a1 = taylor_enclosure[index_map([1])]

    K0 = jnp.abs(a0 - isolevel)
    K1 = max_abs(enc_a1)
    
    return jnp.where(K1 > 1e-12, K0 / K1, jnp.inf)


def bounded_dist_isosurf_1st_order_2d(taylor_enclosure, upper_or_lower: Union[float, jnp.ndarray], isolevel: float=0.0) -> jnp.ndarray:
    index_map = lambda idxs: multi_index_2d_1st_order.index(idxs)

    a00 = taylor_enclosure[index_map([0, 0])]
    enc_a10 = taylor_enclosure[index_map([1, 0])]
    enc_a01 = taylor_enclosure[index_map([0, 1])]

    K0 = jnp.abs(a00 - isolevel)
    K1 = jnp.sqrt(max_abs(enc_a10)**2 + max_abs(enc_a01)**2)
    
    return jnp.where(K1 > 1e-12, K0 / K1, jnp.inf)


def bounded_dist_isosurf_1st_order_3d(taylor_enclosure, upper_or_lower: Union[float, jnp.ndarray], isolevel: float=0.0) -> jnp.ndarray:
    index_map = lambda idxs: multi_index_3d_1st_order.index(idxs)

    a000 = taylor_enclosure[index_map([0, 0, 0])]
    enc_a100 = taylor_enclosure[index_map([1, 0, 0])]
    enc_a010 = taylor_enclosure[index_map([0, 1, 0])]
    enc_a001 = taylor_enclosure[index_map([0, 0, 1])]

    K0 = jnp.abs(a000 - isolevel)
    K1 = jnp.sqrt(max_abs(enc_a100)**2 + max_abs(enc_a010)**2 + max_abs(enc_a001)**2)
    
    return jnp.where(K1 > 1e-12, K0 / K1, jnp.inf)    


def bounded_dist_isosurf_2nd_order_2d(taylor_enclosure: MixedIntervalPolynomial, 
                                      upper_or_lower: Union[float, jnp.ndarray], 
                                      isolevel: float=0.0) -> jnp.ndarray:
    index_map = lambda idxs: multi_index_2d_2nd_order.index(idxs)

    a00 = taylor_enclosure[index_map([0, 0])]
    a10 = taylor_enclosure[index_map([1, 0])]
    a01 = taylor_enclosure[index_map([0, 1])]
    enc_a20 = taylor_enclosure[index_map([2, 0])]
    enc_a02 = taylor_enclosure[index_map([0, 2])]
    enc_a11 = taylor_enclosure[index_map([1, 1])]

    # Normalize flag and select bounds branchlessly
    is_upper = (upper_or_lower == 1)
    a20_val = jnp.where(is_upper, enc_a20.upper, enc_a20.lower)
    a02_val = jnp.where(is_upper, enc_a02.upper, enc_a02.lower)
    a11_val = jnp.maximum(jnp.abs(enc_a11.lower), jnp.abs(enc_a11.upper))

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
    index_map = lambda idxs: multi_index_1d_2nd_order.index(idxs)

    a0 = taylor_enclosure[index_map([0])]
    a1 = taylor_enclosure[index_map([1])]
    enc_a2 = taylor_enclosure[index_map([2])]

    # Normalize flag and select bounds branchlessly
    is_upper = (upper_or_lower == 1)
    a2_val = jnp.where(is_upper, enc_a2.upper, enc_a2.lower)
    
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
    index_map = lambda idxs: multi_index_3d_2nd_order.index(idxs)

    a000 = taylor_enclosure[index_map([0, 0, 0])]
    a100 = taylor_enclosure[index_map([1, 0, 0])]
    a010 = taylor_enclosure[index_map([0, 1, 0])]
    a001 = taylor_enclosure[index_map([0, 0, 1])]

    enc_a200 = taylor_enclosure[index_map([2, 0, 0])]
    enc_a020 = taylor_enclosure[index_map([0, 2, 0])]
    enc_a002 = taylor_enclosure[index_map([0, 0, 2])]
    enc_a110 = taylor_enclosure[index_map([1, 1, 0])]
    enc_a101 = taylor_enclosure[index_map([1, 0, 1])]
    enc_a011 = taylor_enclosure[index_map([0, 1, 1])]

    # Normalize flag and select bounds branchlessly
    is_upper = (upper_or_lower == 1)
    a200_val = jnp.where(is_upper, enc_a200.upper, enc_a200.lower)
    a020_val = jnp.where(is_upper, enc_a020.upper, enc_a020.lower)
    a002_val = jnp.where(is_upper, enc_a002.upper, enc_a002.lower)

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
    index_map = lambda idxs: multi_index_1d_3rd_order.index(idxs)

    a0 = taylor_enclosure[index_map([0])]
    a1 = taylor_enclosure[index_map([1])]
    a2 = taylor_enclosure[index_map([2])]
    enc_a3 = taylor_enclosure[index_map([3])]
    
    K0 = jnp.abs(a0 - isolevel)
    K1 = jnp.abs(a1)
    K2 = jnp.abs(a2)
    K3 = max_abs(enc_a3)

    # Solve K3*d^3 + K2*d^2 + K1*d - K0 = 0
    bound = solve_3rd_order_bound(K0, K1, K2, K3, steps=4)
    return bound


def bounded_dist_isosurf_3rd_order_2d(taylor_enclosure, upper_or_lower: Union[float, jnp.ndarray], isolevel: float=0.0) -> jnp.ndarray:
    index_map = lambda idxs: multi_index_2d_3rd_order.index(idxs)

    # Extract 0th, 1st and 2nd order coefficients
    a00 = taylor_enclosure[index_map([0, 0])]
    a10 = taylor_enclosure[index_map([1, 0])]
    a01 = taylor_enclosure[index_map([0, 1])]
    a20 = taylor_enclosure[index_map([2, 0])]
    a02 = taylor_enclosure[index_map([0, 2])]
    a11 = taylor_enclosure[index_map([1, 1])]

    # Extract 3rd order
    enc_a30 = taylor_enclosure[index_map([3, 0])]
    enc_a03 = taylor_enclosure[index_map([0, 3])]
    enc_a21 = taylor_enclosure[index_map([2, 1])]
    enc_a12 = taylor_enclosure[index_map([1, 2])]

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
    index_map = lambda idxs: multi_index_3d_3rd_order.index(idxs)

    # Extract 0th, 1st and 2nd order coefficients
    a000 = taylor_enclosure[index_map([0, 0, 0])]
    a100 = taylor_enclosure[index_map([1, 0, 0])]
    a010 = taylor_enclosure[index_map([0, 1, 0])]
    a001 = taylor_enclosure[index_map([0, 0, 1])]
    a200 = taylor_enclosure[index_map([2, 0, 0])]
    a020 = taylor_enclosure[index_map([0, 2, 0])]
    a002 = taylor_enclosure[index_map([0, 0, 2])]
    a110 = taylor_enclosure[index_map([1, 1, 0])]
    a101 = taylor_enclosure[index_map([1, 0, 1])]
    a011 = taylor_enclosure[index_map([0, 1, 1])]

    # Extract 3rd order
    enc_a300 = taylor_enclosure[index_map([3, 0, 0])]
    enc_a030 = taylor_enclosure[index_map([0, 3, 0])]
    enc_a003 = taylor_enclosure[index_map([0, 0, 3])]
    enc_a210 = taylor_enclosure[index_map([2, 1, 0])]
    enc_a201 = taylor_enclosure[index_map([2, 0, 1])]
    enc_a120 = taylor_enclosure[index_map([1, 2, 0])]
    enc_a021 = taylor_enclosure[index_map([0, 2, 1])]
    enc_a102 = taylor_enclosure[index_map([1, 0, 2])]
    enc_a012 = taylor_enclosure[index_map([0, 1, 2])]
    enc_a111 = taylor_enclosure[index_map([1, 1, 1])]

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


def solve_3rd_order_bound(K0: jnp.ndarray, K1: jnp.ndarray, K2: jnp.ndarray, K3: jnp.ndarray, steps=4):
    """
    Highly optimized branchless solver for K3*d^3 + K2*d^2 + K1*d - K0 = 0
    Assumes K0, K1, K2, K3 are all >= 0.
    """
    # Initial guess: min over single-term upper bounds on the root d*.
    # Since K0, K1, K2, K3 >= 0, each term in K1*d + K2*d^2 + K3*d^3 = K0 is non-negative,
    # so individually K_i * d^i <= K0, giving d <= (K0/K_i)^(1/i) for each i. The minimum
    # across i is the tightest such upper bound, and Newton on a monotonically increasing
    # convex function converges quadratically from above.
    # Adding 1e-12 prevents division by zero if a Taylor order is 0.
    b1 = K0 / (K1 + 1e-12)
    b2 = jnp.sqrt(K0 / (K2 + 1e-12))
    b3 = jnp.cbrt(K0 / (K3 + 1e-12))

    d = jnp.minimum(jnp.minimum(b1, b2), b3)

    # Unrolled Newton-Raphson
    for _ in range(steps):
        # Horner's method for fast polynomial evaluation: d*(K1 + d*(K2 + d*K3))
        val = d * (K1 + d * (K2 + d * K3)) - K0
        # Derivative: 3*K3*d^2 + 2*K2*d + K1
        deriv = K1 + d * (2.0 * K2 + d * 3.0 * K3)
        # Newton step
        d = d - val / (deriv + 1e-12)

    # Fallback check: if the surface is impossible to hit (K1, K2, K3 all 0 but K0 > 0)
    is_valid = (K1 > 1e-12) | (K2 > 1e-12) | (K3 > 1e-12) | (K0 < 1e-12)
    
    return jnp.where(is_valid, d, jnp.inf)