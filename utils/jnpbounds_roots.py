"""
Polynomial root-finders along a ray for Taylor enclosures of orders 1, 2, and 3.

Sentinel convention for `roots_*_order_polynomial`:
    Public return:  `t == -jnp.inf`  iff no positive real root exists.
                    Always branch on the returned `hit` flag, never on the sign of `t`.
    Cubic-internal: `+jnp.inf` is used as the neutral element for `jnp.minimum`,
                    marking "this candidate branch produced no usable root."
                    It is converted to the public `-jnp.inf` sentinel before return.
"""
import jax
import jax.numpy as jnp
import jnp_bounds.autobounds as jb
from jnp_bounds.utils.dim_degree_coefficients import dim_degree_to_exponents_cache
from functools import partial

multi_index_1d_1st_order = dim_degree_to_exponents_cache.static[(1, 1)][0]
multi_index_1d_2nd_order = dim_degree_to_exponents_cache.static[(1, 2)][0]
multi_index_1d_3rd_order = dim_degree_to_exponents_cache.static[(1, 3)][0]

multi_index_2d_1st_order = dim_degree_to_exponents_cache.static[(2, 1)][0]
multi_index_2d_2nd_order = dim_degree_to_exponents_cache.static[(2, 2)][0]
multi_index_2d_3rd_order = dim_degree_to_exponents_cache.static[(2, 3)][0]

multi_index_3d_1st_order = dim_degree_to_exponents_cache.static[(3, 1)][0]
multi_index_3d_2nd_order = dim_degree_to_exponents_cache.static[(3, 2)][0]
multi_index_3d_3rd_order = dim_degree_to_exponents_cache.static[(3, 3)][0]


def transform_1st_order_1d(taylor_enclosure, ray_direction, upper_or_lower):
    """
    Transform 1st order 1D polynomial coefficients into a 1st order polynomial along a ray.
    """
    index_map = lambda idxs: multi_index_1d_1st_order.index(idxs)

    a0 = taylor_enclosure[index_map([0])]
    enc_a1 = taylor_enclosure[index_map([1])]
    is_upper = (upper_or_lower == 1)

    # If is_upper is True:  choose a1.upper when ray >= 0
    # If is_upper is False: choose a1.upper when ray < 0
    choose_upper = (ray_direction[0] >= 0.0) == is_upper
    
    a1 = jnp.where(choose_upper, enc_a1.upper, enc_a1.lower)

    const_term = a0
    linear_term = a1 * ray_direction[0]

    return jnp.array([const_term, linear_term])
                                                               

def transform_2nd_order_1d(taylor_enclosure, ray_direction, upper_or_lower):
    index_map = lambda idxs: multi_index_1d_2nd_order.index(idxs)

    a0 = taylor_enclosure[index_map([0])]
    a1 = taylor_enclosure[index_map([1])]
    enc_a2 = taylor_enclosure[index_map([2])]

    is_upper = (upper_or_lower == 1)
    a2 = jnp.where(is_upper, enc_a2.upper, enc_a2.lower)

    const_term = a0
    linear_term = a1 * ray_direction[0]
    quadratic_term = a2 * ray_direction[0]**2

    return jnp.array([const_term, linear_term, quadratic_term])


def transform_3rd_order_1d(taylor_enclosure, ray_direction, upper_or_lower):
    index_map = lambda idxs: multi_index_1d_3rd_order.index(idxs)

    a0 = taylor_enclosure[index_map([0])]
    a1 = taylor_enclosure[index_map([1])]
    a2 = taylor_enclosure[index_map([2])]
    enc_a3 = taylor_enclosure[index_map([3])]

    is_upper = (upper_or_lower == 1)
    choose_upper = (ray_direction[0] >= 0.0) == is_upper
    a3 = jnp.where(choose_upper, enc_a3.upper, enc_a3.lower)

    const_term = a0
    linear_term = a1 * ray_direction[0]
    quadratic_term = a2 * ray_direction[0]**2
    cubic_term = a3 * ray_direction[0]**3

    return jnp.array([const_term, linear_term, quadratic_term, cubic_term])


def transform_1st_order_2d(taylor_enclosure, ray_direction, upper_or_lower):
    index_map = lambda idxs: multi_index_2d_1st_order.index(idxs)

    a00 = taylor_enclosure[index_map([0, 0])]

    enc_a10 = taylor_enclosure[index_map([1, 0])]
    enc_a01 = taylor_enclosure[index_map([0, 1])]

    is_upper = (upper_or_lower == 1)

    choose_upper_x = (ray_direction[0] >= 0.0) == is_upper
    choose_upper_y = (ray_direction[1] >= 0.0) == is_upper

    a10 = jnp.where(choose_upper_x, enc_a10.upper, enc_a10.lower)
    a01 = jnp.where(choose_upper_y, enc_a01.upper, enc_a01.lower)

    # compute polynomial terms along the ray
    const_term = a00
    linear_term = a10 * ray_direction[0] + a01 * ray_direction[1]

    return jnp.array([const_term, linear_term])


def transform_2nd_order_2d(taylor_enclosure, ray_direction, upper_or_lower):
    index_map = lambda idxs: multi_index_2d_2nd_order.index(idxs)

    a00 = taylor_enclosure[index_map([0, 0])]
    a10 = taylor_enclosure[index_map([1, 0])]
    a01 = taylor_enclosure[index_map([0, 1])]
    
    # intervals
    enc_a20 = taylor_enclosure[index_map([2, 0])]
    enc_a02 = taylor_enclosure[index_map([0, 2])]
    enc_a11 = taylor_enclosure[index_map([1, 1])]

    is_upper = (upper_or_lower == 1)

    # Select a20 and a02 
    a20 = jnp.where(is_upper, enc_a20.upper, enc_a20.lower)
    a02 = jnp.where(is_upper, enc_a02.upper, enc_a02.lower)

    # Select a11
    signs_match = (ray_direction[0] * ray_direction[1]) >= 0.0
    choose_upper_a11 = (signs_match == is_upper)
    a11 = jnp.where(choose_upper_a11, enc_a11.upper, enc_a11.lower)

    # Compute the polynomial terms
    const_term = a00 
    linear_term = a01 * ray_direction[1] + a10 * ray_direction[0]
    quadratic_term = a02 * ray_direction[1]**2 + a20 * ray_direction[0]**2 + a11 * ray_direction[0] * ray_direction[1]

    return jnp.array([const_term, linear_term, quadratic_term])


def transform_3rd_order_2d(taylor_enclosure, ray_direction, upper_or_lower):
    index_map = lambda idxs: multi_index_2d_3rd_order.index(idxs)

    a00 = taylor_enclosure[index_map([0, 0])]
    a10 = taylor_enclosure[index_map([1, 0])]
    a01 = taylor_enclosure[index_map([0, 1])]
    
    a20 = taylor_enclosure[index_map([2, 0])]
    a02 = taylor_enclosure[index_map([0, 2])]
    a11 = taylor_enclosure[index_map([1, 1])]

    enc_a30 = taylor_enclosure[index_map([3, 0])]
    enc_a03 = taylor_enclosure[index_map([0, 3])]
    enc_a21 = taylor_enclosure[index_map([2, 1])]
    enc_a12 = taylor_enclosure[index_map([1, 2])]

    is_upper = (upper_or_lower == 1)

    # determine signs for 3rd order terms
    choose_upper_x = (ray_direction[0] >= 0.0) == is_upper
    choose_upper_y = (ray_direction[1] >= 0.0) == is_upper

    # select 3rd order terms
    a30 = jnp.where(choose_upper_x, enc_a30.upper, enc_a30.lower)
    a12 = jnp.where(choose_upper_x, enc_a12.upper, enc_a12.lower)
    a03 = jnp.where(choose_upper_y, enc_a03.upper, enc_a03.lower)
    a21 = jnp.where(choose_upper_y, enc_a21.upper, enc_a21.lower)

    # compute polynomial terms
    const_term = a00
    linear_term = a10 * ray_direction[0] + a01 * ray_direction[1]
    
    quadratic_term = (a20 * ray_direction[0]**2 + 
                      a02 * ray_direction[1]**2 + 
                      a11 * ray_direction[0] * ray_direction[1])
                      
    cubic_term = (a30 * ray_direction[0]**3 + 
                  a03 * ray_direction[1]**3 + 
                  a21 * ray_direction[0]**2 * ray_direction[1] + 
                  a12 * ray_direction[0] * ray_direction[1]**2)

    return jnp.array([const_term, linear_term, quadratic_term, cubic_term])


def transform_1st_order_3d(taylor_enclosure, ray_direction, upper_or_lower):
    index_map = lambda idxs: multi_index_3d_1st_order.index(idxs)

    a000 = taylor_enclosure[index_map([0, 0, 0])]

    enc_a100 = taylor_enclosure[index_map([1, 0, 0])]
    enc_a010 = taylor_enclosure[index_map([0, 1, 0])]
    enc_a001 = taylor_enclosure[index_map([0, 0, 1])]

    is_upper = (upper_or_lower == 1)

    choose_upper_x = (ray_direction[0] >= 0.0) == is_upper
    choose_upper_y = (ray_direction[1] >= 0.0) == is_upper
    choose_upper_z = (ray_direction[2] >= 0.0) == is_upper

    a100 = jnp.where(choose_upper_x, enc_a100.upper, enc_a100.lower)
    a010 = jnp.where(choose_upper_y, enc_a010.upper, enc_a010.lower)
    a001 = jnp.where(choose_upper_z, enc_a001.upper, enc_a001.lower)

    const_term = a000
    linear_term = a100 * ray_direction[0] + a010 * ray_direction[1] + a001 * ray_direction[2]

    return jnp.array([const_term, linear_term])


def transform_2nd_order_3d(taylor_enclosure, ray_direction, upper_or_lower):
    """
    Transform 2nd order 3D polynomial coefficients into a 2nd order polynomial along a ray.
    Assuming that ray_origin is at [0, 0, 0].
    Reason: autobounds are defined relative to a local origin.
    Quadric intersection math: http://www.bmsc.washington.edu/people/merritt/graphics/quadrics.html
    """
    index_map = lambda idxs: multi_index_3d_2nd_order.index(idxs)

    a000 = taylor_enclosure[index_map([0, 0, 0])]
    a100 = taylor_enclosure[index_map([1, 0, 0])]
    a010 = taylor_enclosure[index_map([0, 1, 0])]
    a001 = taylor_enclosure[index_map([0, 0, 1])]

    #intervals
    enc_a200 = taylor_enclosure[index_map([2, 0, 0])]
    enc_a020 = taylor_enclosure[index_map([0, 2, 0])]
    enc_a002 = taylor_enclosure[index_map([0, 0, 2])]
    
    enc_a110 = taylor_enclosure[index_map([1, 1, 0])]
    enc_a101 = taylor_enclosure[index_map([1, 0, 1])]
    enc_a011 = taylor_enclosure[index_map([0, 1, 1])]

    is_upper = (upper_or_lower == 1)

    # Pure quadratic terms
    # Because they are multiplied by non-negative squares (e.g., d_x^2), 
    # they strictly follow the `is_upper` flag.
    a200 = jnp.where(is_upper, enc_a200.upper, enc_a200.lower)
    a020 = jnp.where(is_upper, enc_a020.upper, enc_a020.lower)
    a002 = jnp.where(is_upper, enc_a002.upper, enc_a002.lower)

    # Mixed quadratic terms
    # Because they are multiplied by d_i * d_j, they depend on the XNOR of the products and is_upper.
    choose_upper_a110 = ((ray_direction[0] * ray_direction[1]) >= 0.0) == is_upper
    choose_upper_a101 = ((ray_direction[0] * ray_direction[2]) >= 0.0) == is_upper
    choose_upper_a011 = ((ray_direction[1] * ray_direction[2]) >= 0.0) == is_upper

    a110 = jnp.where(choose_upper_a110, enc_a110.upper, enc_a110.lower)
    a101 = jnp.where(choose_upper_a101, enc_a101.upper, enc_a101.lower)
    a011 = jnp.where(choose_upper_a011, enc_a011.upper, enc_a011.lower)

    #final polynomial terms along the ray
    const_term = a000
    linear_term = a001 * ray_direction[2] + a010 * ray_direction[1] + a100 * ray_direction[0]
    
    quadratic_term = (a002 * ray_direction[2]**2 + 
                      a020 * ray_direction[1]**2 + 
                      a200 * ray_direction[0]**2 + 
                      a011 * ray_direction[1] * ray_direction[2] + 
                      a101 * ray_direction[0] * ray_direction[2] + 
                      a110 * ray_direction[0] * ray_direction[1])
    
    return jnp.array([const_term, linear_term, quadratic_term])


def transform_3rd_order_3d(taylor_enclosure, ray_direction, upper_or_lower):
    index_map = lambda idxs: multi_index_3d_3rd_order.index(idxs)

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

    # Normalize flag
    is_upper = (upper_or_lower == 1)

    # determine signs for 3rd order terms
    choose_upper_x = (ray_direction[0] >= 0.0) == is_upper
    choose_upper_y = (ray_direction[1] >= 0.0) == is_upper
    choose_upper_z = (ray_direction[2] >= 0.0) == is_upper
    choose_upper_xyz = ((ray_direction[0] * ray_direction[1] * ray_direction[2]) >= 0.0) == is_upper

    a300 = jnp.where(choose_upper_x, enc_a300.upper, enc_a300.lower)
    a120 = jnp.where(choose_upper_x, enc_a120.upper, enc_a120.lower)
    a102 = jnp.where(choose_upper_x, enc_a102.upper, enc_a102.lower)
    a030 = jnp.where(choose_upper_y, enc_a030.upper, enc_a030.lower)
    a210 = jnp.where(choose_upper_y, enc_a210.upper, enc_a210.lower)
    a012 = jnp.where(choose_upper_y, enc_a012.upper, enc_a012.lower)
    a003 = jnp.where(choose_upper_z, enc_a003.upper, enc_a003.lower)
    a201 = jnp.where(choose_upper_z, enc_a201.upper, enc_a201.lower)
    a021 = jnp.where(choose_upper_z, enc_a021.upper, enc_a021.lower)
    a111 = jnp.where(choose_upper_xyz, enc_a111.upper, enc_a111.lower)

    # compute polynomial terms
    const_term = a000
    
    linear_term = (a100 * ray_direction[0] + 
                   a010 * ray_direction[1] + 
                   a001 * ray_direction[2])
    
    quadratic_term = (a200 * ray_direction[0]**2 + 
                      a020 * ray_direction[1]**2 + 
                      a002 * ray_direction[2]**2 + 
                      a110 * ray_direction[0] * ray_direction[1] + 
                      a101 * ray_direction[0] * ray_direction[2] + 
                      a011 * ray_direction[1] * ray_direction[2])
                      
    cubic_term = (a300 * ray_direction[0]**3 + 
                  a030 * ray_direction[1]**3 + 
                  a003 * ray_direction[2]**3 + 
                  a210 * ray_direction[0]**2 * ray_direction[1] + 
                  a201 * ray_direction[0]**2 * ray_direction[2] + 
                  a120 * ray_direction[0] * ray_direction[1]**2 + 
                  a021 * ray_direction[1]**2 * ray_direction[2] + 
                  a102 * ray_direction[0] * ray_direction[2]**2 + 
                  a012 * ray_direction[1] * ray_direction[2]**2 + 
                  a111 * ray_direction[0] * ray_direction[1] * ray_direction[2])

    return jnp.array([const_term, linear_term, quadratic_term, cubic_term])


@partial(jax.jit, static_argnames=('dim'))
def roots_1st_order_tensor(taylor_enclosure, ray_direction, upper_or_lower, dim):
    if dim == 3:
        poly_coeffs = transform_1st_order_3d(taylor_enclosure, ray_direction, upper_or_lower)
    elif dim == 2:
        poly_coeffs = transform_1st_order_2d(taylor_enclosure, ray_direction, upper_or_lower)
    elif dim == 1:
        poly_coeffs = transform_1st_order_1d(taylor_enclosure, ray_direction, upper_or_lower)
    else:
        raise ValueError(f"Unsupported dimension: {dim}. Only 1D, 2D, and 3D are supported.")
    return roots_1st_order_polynomial(poly_coeffs)


@partial(jax.jit, static_argnames=('dim'))
def roots_2nd_order_tensor(taylor_enclosure, ray_direction, upper_or_lower, dim):
    if dim == 3:
        poly_coeffs = transform_2nd_order_3d(taylor_enclosure, ray_direction, upper_or_lower)
    elif dim == 2:
        poly_coeffs = transform_2nd_order_2d(taylor_enclosure, ray_direction, upper_or_lower)
    elif dim == 1:
        poly_coeffs = transform_2nd_order_1d(taylor_enclosure, ray_direction, upper_or_lower)
    else:
        raise ValueError(f"Unsupported dimension: {dim}. Only 1D, 2D, and 3D are supported.")
    return roots_2nd_order_polynomial(poly_coeffs)


@partial(jax.jit, static_argnames=('dim'))
def roots_3rd_order_tensor(taylor_enclosure, ray_direction, upper_or_lower, dim):
    if dim == 3:
        poly_coeffs = transform_3rd_order_3d(taylor_enclosure, ray_direction, upper_or_lower)
    elif dim == 2:
        poly_coeffs = transform_3rd_order_2d(taylor_enclosure, ray_direction, upper_or_lower)
    elif dim == 1:
        poly_coeffs = transform_3rd_order_1d(taylor_enclosure, ray_direction, upper_or_lower)
    else:
        raise ValueError(f"Unsupported dimension: {dim}. Only 1D, 2D, and 3D are supported.")
    return roots_3rd_order_polynomial(poly_coeffs)


def roots_1st_order_polynomial(poly_coeffs):
    a = poly_coeffs[0]
    b = poly_coeffs[1]

    does_intersec = jnp.abs(b) > 1e-12
    safe_b = jnp.where(does_intersec, b, 1.0)
    t_raw = -a / safe_b
    hit = jnp.logical_and(does_intersec, t_raw > 0)
    t = jnp.where(hit, t_raw, -jnp.inf)
    return hit, t



def roots_2nd_order_polynomial(poly_coeffs):
    c = poly_coeffs[0] # constant term
    b = poly_coeffs[1] # linear term
    a = poly_coeffs[2] # quadratic term

    # Safe sqrt to avoid NaNs
    discriminant = b**2 - 4*a*c
    sqrt_discriminant = jnp.sqrt(jnp.maximum(0., discriminant))

    # Citardauq Formula for numerical stability
    # This avoids subtractive cancellation when b^2 >> 4ac
    q = -0.5 * (b + jnp.sign(b) * sqrt_discriminant)
    
    # Handle linear case (a -> 0) safely to avoid division by zero
    # If a is tiny, fallback to linear solution -c/b, else use quadratic
    root_0 = jnp.where(jnp.abs(a) > 1e-12, q / a, -c / (b + 1e-12))
    root_1 = jnp.where(jnp.abs(a) > 1e-12, c / q, root_0)
    
    root_near = jnp.minimum(root_0, root_1)
    root_far = jnp.maximum(root_0, root_1)

    t = jnp.where(root_near > 0, root_near, root_far)
    t = jnp.where(discriminant >= 0, t, -jnp.inf)
    hit = t > 0
    
    return hit, t


def roots_3rd_order_polynomial(poly_coeffs):
    d = poly_coeffs[0] # constant term
    c = poly_coeffs[1] # linear term
    b = poly_coeffs[2] # quadratic term
    a = poly_coeffs[3] # cubic term

    # 1. FALLBACK: Quadratic Case (a -> 0)
    discriminant_q = c**2 - 4 * b * d
    sqrt_disc_q = jnp.sqrt(jnp.maximum(0.0, discriminant_q))
    q_q = -0.5 * (c + jnp.sign(c) * sqrt_disc_q)
    
    safe_b = jnp.where(jnp.abs(b) > 1e-12, b, 1.0)
    safe_q_q = jnp.where(jnp.abs(q_q) > 1e-12, q_q, 1.0)
    
    root0_q = jnp.where(jnp.abs(b) > 1e-12, q_q / safe_b, -d / (c + 1e-12))
    root1_q = jnp.where(jnp.abs(b) > 1e-12, d / safe_q_q, root0_q)
    
    r1_quad = jnp.where(discriminant_q >= 0, root0_q, jnp.inf)
    r2_quad = jnp.where(discriminant_q >= 0, root1_q, jnp.inf)
    r3_quad = jnp.inf

    # 2. Standard Cubic Case
    is_cubic = jnp.abs(a) > 1e-12
    safe_a = jnp.where(is_cubic, a, 1.0)
    B = b / safe_a
    C = c / safe_a
    D = d / safe_a

    Q = (B**2 - 3.0*C) / 9.0
    R = (2.0 * B**3 - 9.0 * B * C + 27.0 * D) / 54.0

    safe_Q = jnp.maximum(Q, 0.0)
    is_multi = (R**2 < Q**3)

    # Branch A: three real roots
    arccos_input = jnp.clip(R / jnp.sqrt(safe_Q**3 + 1e-12), -1.0, 1.0)
    theta = jnp.arccos(arccos_input)
    
    sqrt_Q = jnp.sqrt(safe_Q)
    t1_multi = -2.0 * sqrt_Q * jnp.cos(theta / 3.0) - B / 3.0
    t2_multi = -2.0 * sqrt_Q * jnp.cos((theta + 2.0 * jnp.pi) / 3.0) - B / 3.0
    t3_multi = -2.0 * sqrt_Q * jnp.cos((theta - 2.0 * jnp.pi) / 3.0) - B / 3.0
    
    # Branch B: one real root
    diff = jnp.maximum(R**2 - Q**3, 0.0)
    sqrt_diff = jnp.sqrt(diff)

    abs_R_plus_sqrt = jnp.abs(R) + sqrt_diff
    E = -jnp.sign(R) * jnp.cbrt(abs_R_plus_sqrt)

    E_is_valid = jnp.abs(E) > 1e-12
    safe_E = jnp.asarray(jnp.where(E_is_valid, E, 1.0))
    F = (Q / safe_E) * E_is_valid

    t1_single = E + F - B / 3.0
    t2_single = jnp.inf
    t3_single = jnp.inf

    # Merge Cubic Branches
    r1_cubic = jnp.asarray(jnp.where(is_multi, t1_multi, t1_single))
    r2_cubic = jnp.asarray(jnp.where(is_multi, t2_multi, t2_single))
    r3_cubic = jnp.asarray(jnp.where(is_multi, t3_multi, t3_single))
    
    # 3. Final Merge (Cubic vs Quadratic Fallback)
    r1_merged = jnp.where(is_cubic, r1_cubic, r1_quad)
    r2_merged = jnp.where(is_cubic, r2_cubic, r2_quad)
    r3_merged = jnp.where(is_cubic, r3_cubic, r3_quad)

    r1_pos = jnp.asarray(jnp.where(r1_merged > 0, r1_merged, jnp.inf))
    r2_pos = jnp.asarray(jnp.where(r2_merged > 0, r2_merged, jnp.inf))
    r3_pos = jnp.asarray(jnp.where(r3_merged > 0, r3_merged, jnp.inf))

    # Find the smallest positive root
    t_min = jnp.minimum(jnp.minimum(r1_pos, r2_pos), r3_pos)
    hit = t_min < jnp.inf
    t = jnp.asarray(jnp.where(hit, t_min, -jnp.inf))

    return hit, t    