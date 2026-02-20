import jax
import jax.numpy as jnp
import jnp_bounds.autobounds as jb
from jnp_bounds.types import MixedIntervalPolynomial
from jnp_bounds.utils.dim_degree_coefficients import dim_degree_to_exponents_cache
from typing import Union

multi_indeces_1d_2nd_order = dim_degree_to_exponents_cache.static[(1, 2)][0]
multi_idx_2d_2nd_order = dim_degree_to_exponents_cache.static[(2, 2)][0]
multi_idx_3d_2nd_order = dim_degree_to_exponents_cache.static[(3, 2)][0]


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
    phi = jnp.arctan2(jnp.sqrt(4 * x1**3 - x2**2), x2)
    phi = jnp.where(x2 <= 0, phi + jnp.pi, phi)
    phi = jnp.where(x2 == 0, jnp.pi / 2, phi)

    lambda1 = (a + b + c - 2 * jnp.sqrt(x1) * jnp.cos(phi / 3)) / 3
    lambda2 = (a + b + c + 2 * jnp.sqrt(x1) * jnp.cos((phi - jnp.pi) / 3)) / 3
    lambda3 = (a + b + c + 2 * jnp.sqrt(x1) * jnp.cos((phi + jnp.pi) / 3)) / 3

    return lambda1, lambda2, lambda3
    

def empty_sphere_2nd_order_tensor(taylor_enclosure: MixedIntervalPolynomial, 
                                  upper_or_lower: Union[str, jnp.ndarray], 
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
    

def bounded_dist_isosurf_2nd_order_2d(taylor_enclosure: MixedIntervalPolynomial, 
                                      upper_or_lower: Union[str, jnp.ndarray], 
                                      isolevel: float=0.0) -> jnp.ndarray:
    index_map = lambda idxs: multi_idx_2d_2nd_order.index(idxs)

    a00 = taylor_enclosure[index_map([0, 0])]
    a10 = taylor_enclosure[index_map([1, 0])]
    a01 = taylor_enclosure[index_map([0, 1])]
    a20 = taylor_enclosure[index_map([2, 0])]
    a02 = taylor_enclosure[index_map([0, 2])]
    a11 = taylor_enclosure[index_map([1, 1])]

    if upper_or_lower == "lower":
        a20_val = a20.lower
        a02_val = a02.lower
    elif upper_or_lower == "upper":
        a20_val = a20.upper
        a02_val = a02.upper
    else:
        a20_val = jnp.where(upper_or_lower==1, a20.upper, a20.lower)
        a02_val = jnp.where(upper_or_lower==1, a02.upper, a02.lower)

    a11_val = jnp.maximum(jnp.abs(a11.lower), jnp.abs(a11.upper))
    lambda1, lambda2 = eigenvalues_of_hermitian_2x2(a20_val, a02_val, 0.5 * a11_val)
    to_bound_eigenvalue = jnp.maximum(jnp.abs(lambda1), jnp.abs(lambda2))

    const_term = a00 - isolevel
    linear_bound = jnp.sqrt(a10**2 + a01**2)
    bound = 2 * jnp.abs(const_term) / (linear_bound + jnp.sqrt(linear_bound ** 2 + 4 * jnp.abs(const_term) * to_bound_eigenvalue))
    return bound


def bounded_dist_isosurf_2nd_order_1d(taylor_enclosure: MixedIntervalPolynomial, 
                                      upper_or_lower: Union[str, jnp.ndarray], 
                                      isolevel: float=0.0) -> jnp.ndarray:
    """
    Computes the bounded distance to intersection of the 1D second-order polynomial with the isolevel value 
    """
    index_map = lambda idxs: multi_indeces_1d_2nd_order.index(idxs)

    a0 = taylor_enclosure[index_map([0])]
    a1 = taylor_enclosure[index_map([1])]
    a2 = taylor_enclosure[index_map([2])]

    const_term = a0 - isolevel
    linear_term = a1
    if upper_or_lower == "lower":
        a2_val = a2.lower
    elif upper_or_lower == "upper":
        a2_val = a2.upper
    else:
        a2_val = jnp.where(upper_or_lower==1, a2.upper, a2.lower)
    
    discriminant = a1**2 - 4 * a2_val * const_term
    no_solution = discriminant < 0
    sqrt_disc = jnp.sqrt(jnp.maximum(0., discriminant))
    bound = jnp.minimum(jnp.abs(root1), jnp.abs(root2))

    # numberical stability trick for quadratic formula
    q = -0.5 * (a1 + jnp.sign(a1) * sqrt_disc)
    root1 = q / a2_val
    root2 = const_term / q

    dist = jnp.minimum(jnp.abs(root1), jnp.abs(root2))
    return jnp.where(no_solution, jnp.inf, dist)


def bounded_dist_isosurf_2nd_order_3d(taylor_enclosure: MixedIntervalPolynomial, 
                                      upper_or_lower: Union[str, jnp.ndarray], 
                                      isolevel: float=0.0) -> jnp.ndarray:
    """
    Computes a lower bound on the distance to the isosurface defined by the 3D second-order polynomial.
    Uses the method from "Efficient and Robust Isosurface Extraction on Unstructured Grids Using Taylor Models"
    """
    index_map = lambda idxs: multi_idx_3d_2nd_order.index(idxs)

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

    if upper_or_lower == "lower":
        a200_val = a200.lower
        a020_val = a020.lower
        a002_val = a002.lower
    elif upper_or_lower == "upper":
        a200_val = a200.upper
        a020_val = a020.upper
        a002_val = a002.upper
    else:
        a200_val = jnp.where(upper_or_lower==1, a200.upper, a200.lower)
        a020_val = jnp.where(upper_or_lower==1, a020.upper, a020.lower)
        a002_val = jnp.where(upper_or_lower==1, a002.upper, a002.lower)

    a110_val = jnp.maximum(jnp.abs(a110.lower), jnp.abs(a110.upper))
    a101_val = jnp.maximum(jnp.abs(a101.lower), jnp.abs(a101.upper))
    a011_val = jnp.maximum(jnp.abs(a011.lower), jnp.abs(a011.upper))

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
