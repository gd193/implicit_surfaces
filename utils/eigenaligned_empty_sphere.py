import jax.numpy as jnp
from jnp_bounds.types import MixedIntervalPolynomial
from jnp_bounds.mixed_polynomials_arithmetic import coefficient
from jnp_bounds import interval_arithmetic
from jnp_bounds.polynomial_roots import quadratic_distance_root
from typing import Union

# Import helpers from the original module
from .jnpbound_empty_sphere import (
    eigenvalues_of_hermitian_3x3
)


def bounded_dist_isosurf_2nd_order_3d_eigenaligned(taylor_enclosure: MixedIntervalPolynomial, 
                                                   upper_or_lower: Union[float, jnp.ndarray], 
                                                   isolevel: float=0.0) -> jnp.ndarray:
    """
    Computes a lower bound on the distance to the isosurface defined by the 3D second-order polynomial.
    Uses the exact 3D quadratic distance (secular equation) for the nominal Hessian,
    coupled with an isotropic spectral error bound. This avoids pessimistic alignment assumptions
    and yields much larger valid sphere radii for Walk on Spheres.
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

    c = jnp.abs(a000 - isolevel)
    g = jnp.array([a100, a010, a001])

    # 1. Decompose into H_mid and H_rad
    # Midpoint of the interval Hessian reconstructs the precise curvature
    H_mid_xx = 0.5 * (interval_arithmetic.upper(enc_a200) + interval_arithmetic.lower(enc_a200))
    H_mid_yy = 0.5 * (interval_arithmetic.upper(enc_a020) + interval_arithmetic.lower(enc_a020))
    H_mid_zz = 0.5 * (interval_arithmetic.upper(enc_a002) + interval_arithmetic.lower(enc_a002))
    H_mid_xy = 0.25 * (interval_arithmetic.upper(enc_a110) + interval_arithmetic.lower(enc_a110))
    H_mid_xz = 0.25 * (interval_arithmetic.upper(enc_a101) + interval_arithmetic.lower(enc_a101))
    H_mid_yz = 0.25 * (interval_arithmetic.upper(enc_a011) + interval_arithmetic.lower(enc_a011))

    # Radius of the interval Hessian captures the local variation/error
    H_rad_xx = 0.5 * (interval_arithmetic.upper(enc_a200) - interval_arithmetic.lower(enc_a200))
    H_rad_yy = 0.5 * (interval_arithmetic.upper(enc_a020) - interval_arithmetic.lower(enc_a020))
    H_rad_zz = 0.5 * (interval_arithmetic.upper(enc_a002) - interval_arithmetic.lower(enc_a002))
    H_rad_xy = 0.25 * (interval_arithmetic.upper(enc_a110) - interval_arithmetic.lower(enc_a110))
    H_rad_xz = 0.25 * (interval_arithmetic.upper(enc_a101) - interval_arithmetic.lower(enc_a101))
    H_rad_yz = 0.25 * (interval_arithmetic.upper(enc_a011) - interval_arithmetic.lower(enc_a011))

    # 2. Spectral radius of error
    lrad1, lrad2, lrad3 = eigenvalues_of_hermitian_3x3(
        H_rad_xx, H_rad_yy, H_rad_zz, H_rad_xy, H_rad_yz, H_rad_xz
    )
    K_E = jnp.maximum(jnp.abs(lrad1), jnp.maximum(jnp.abs(lrad2), jnp.abs(lrad3)))

    # 3. Analyze exact nominal Hessian
    H_mid_mat = jnp.array([
        [H_mid_xx, H_mid_xy, H_mid_xz],
        [H_mid_xy, H_mid_yy, H_mid_yz],
        [H_mid_xz, H_mid_yz, H_mid_zz]
    ])
    
    # lam contains eigenvalues, Q contains normalized eigenvectors in columns
    lam, Q = jnp.linalg.eigh(H_mid_mat)
    
    # Project gradient into eigen basis
    v = Q.T @ g  

    # Numerical equation solving belongs to the enclosure library.
    return quadratic_distance_root(c, v, lam, K_E)
