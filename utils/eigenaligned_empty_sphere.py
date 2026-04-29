import jax
import jax.numpy as jnp
from jnp_bounds.types import MixedIntervalPolynomial
from jnp_bounds.utils.dim_degree_coefficients import dim_degree_to_exponents_cache
from typing import Union

# Import helpers from the original module
from .jnpbound_empty_sphere import (
    eigenvalues_of_hermitian_3x3,
    max_abs
)

multi_index_3d_2nd_order = dim_degree_to_exponents_cache.static[(3, 2)][0]

def bounded_dist_isosurf_2nd_order_3d_eigenaligned(taylor_enclosure: MixedIntervalPolynomial, 
                                                   upper_or_lower: Union[float, jnp.ndarray], 
                                                   isolevel: float=0.0) -> jnp.ndarray:
    """
    Computes a lower bound on the distance to the isosurface defined by the 3D second-order polynomial.
    Uses the exact 3D quadratic distance (secular equation) for the nominal Hessian,
    coupled with an isotropic spectral error bound. This avoids pessimistic alignment assumptions
    and yields much larger valid sphere radii for Walk on Spheres.
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

    c = jnp.abs(a000 - isolevel)
    g = jnp.array([a100, a010, a001])

    # 1. Decompose into H_mid and H_rad
    # Midpoint of the interval Hessian reconstructs the precise curvature
    H_mid_xx = 0.5 * (enc_a200.upper + enc_a200.lower)
    H_mid_yy = 0.5 * (enc_a020.upper + enc_a020.lower)
    H_mid_zz = 0.5 * (enc_a002.upper + enc_a002.lower)
    H_mid_xy = 0.25 * (enc_a110.upper + enc_a110.lower)
    H_mid_xz = 0.25 * (enc_a101.upper + enc_a101.lower)
    H_mid_yz = 0.25 * (enc_a011.upper + enc_a011.lower)

    # Radius of the interval Hessian captures the local variation/error
    H_rad_xx = 0.5 * (enc_a200.upper - enc_a200.lower)
    H_rad_yy = 0.5 * (enc_a020.upper - enc_a020.lower)
    H_rad_zz = 0.5 * (enc_a002.upper - enc_a002.lower)
    H_rad_xy = 0.25 * (enc_a110.upper - enc_a110.lower)
    H_rad_xz = 0.25 * (enc_a101.upper - enc_a101.lower)
    H_rad_yz = 0.25 * (enc_a011.upper - enc_a011.lower)

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

    # 4. Fast JAX 1D Newton solver for Lagrange multiplier mu
    # We want a root strictly larger than -lam.min()
    # Initialize from the safe isotropic guess
    mu = jnp.maximum(-jnp.min(lam) + 1e-6, jnp.linalg.norm(v) / (1e-6 + c))

    def newton_step(mu_curr):
        denom = lam + mu_curr
        denom_sq = denom ** 2
        denom_cb = denom ** 3
        
        # Value of secular equation root
        num = lam + K_E + 2.0 * mu_curr
        value = c - jnp.sum((num * v**2) / (4.0 * denom_sq))
        
        # Derivative 
        deriv = jnp.sum(((mu_curr + K_E) * v**2) / (2.0 * denom_cb))
        
        return mu_curr - value / (deriv + 1e-12)

    # Branchless loop for Newton-Raphson
    for _ in range(6):
        mu = newton_step(mu)

    # Convert the converged mu into precise bounding radius
    R_sq = jnp.sum((v**2) / (4.0 * (lam + mu)**2))
    bound = jnp.sqrt(R_sq)
    
    # Fallback to infinity if gradient is exactly zero to avoid division issues, 
    # standard isotropic method is needed if g=0, but g=0 is extremely rare in NN distance fields.
    return jnp.where(jnp.linalg.norm(g) > 1e-12, bound, jnp.inf)
