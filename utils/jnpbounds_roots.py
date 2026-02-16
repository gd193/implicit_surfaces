import jax
import jax.numpy as jnp
import jnp_bounds.autobounds as jb
from jnp_bounds.utils.dim_degree_coefficients import dim_degree_to_exponents_cache
from functools import partial

multi_indeces_1d_2nd_order = dim_degree_to_exponents_cache.static[(1, 2)][0]
multi_idx_2d_2nd_order = dim_degree_to_exponents_cache.static[(2, 2)][0]
multi_idx_3d_2nd_order = dim_degree_to_exponents_cache.static[(3, 2)][0]
                                                               

def transform_2nd_order_1d(taylor_enclosure, ray_direction, upper_or_lower):
    index_map = lambda idxs: multi_indeces_1d_2nd_order.index(idxs)

    a0 = taylor_enclosure[index_map([0])]
    a1 = taylor_enclosure[index_map([1])]
    a2 = taylor_enclosure[index_map([2])]

    def lower_transform(ray_direction):
        return jnp.where(ray_direction[0] >= 0., a2.lower, a2.upper)
    def upper_transform(ray_direction):
        return jnp.where(ray_direction[0] >= 0., a2.upper, a2.lower)
    
    if isinstance(upper_or_lower, jnp.ndarray):
        a2 = jax.lax.cond(upper_or_lower==jnp.array(1), upper_transform, lower_transform, ray_direction)
    elif isinstance(upper_or_lower, str):
        if upper_or_lower == "upper":
            a2 = upper_transform(ray_direction)
        elif upper_or_lower == "lower":
            a2 = lower_transform(ray_direction)
        else:
            raise ValueError("upper_or_lower string must be 'upper' or 'lower'")
    else:
        raise ValueError("upper_or_lower must be a string or jnp.array")
    

    const_term = a0 # + a1 * ray_origin[0] + a2 * ray_origin[0]**2
    linear_term = a1 * ray_direction[0] # + 2 * a2 * ray_origin[0] * ray_direction[0]
    quadratic_term = a2 * ray_direction[0]**2

    return jnp.array([const_term, linear_term, quadratic_term])

def transform_2nd_order_2d(taylor_enclosure, ray_direction, upper_or_lower):
    index_map = lambda idxs: multi_idx_2d_2nd_order.index(idxs)

    a00 = taylor_enclosure[index_map([0, 0])]
    a10 = taylor_enclosure[index_map([1, 0])]
    a01 = taylor_enclosure[index_map([0, 1])]
    a20_lower = taylor_enclosure[index_map([2, 0])].lower
    a02_lower = taylor_enclosure[index_map([0, 2])].lower
    a11_lower = taylor_enclosure[index_map([1, 1])].lower
    a20_upper = taylor_enclosure[index_map([2, 0])].upper
    a02_upper = taylor_enclosure[index_map([0, 2])].upper
    a11_upper = taylor_enclosure[index_map([1, 1])].upper

    def lower_transform(ray_direction):
        a20 = a20_lower
        a02 = a02_lower
        a11 = jnp.where(jnp.sign(ray_direction[0])*jnp.sign(ray_direction[1]) >= 0., a11_lower, a11_upper)
        return a20, a02, a11
    
    def upper_transform(ray_direction):
        a20 = a20_upper
        a02 = a02_upper
        a11 = jnp.where(jnp.sign(ray_direction[0])*jnp.sign(ray_direction[1]) >= 0., a11_upper, a11_lower)
        return a20, a02, a11
    
    if isinstance(upper_or_lower, jnp.ndarray):
        a20, a02, a11 = jax.lax.cond(upper_or_lower==jnp.array(1), upper_transform, lower_transform, ray_direction)
    elif isinstance(upper_or_lower, str):
        if upper_or_lower == "upper":
            a20, a02, a11 = upper_transform(ray_direction)
        elif upper_or_lower == "lower":
            a20, a02, a11 = lower_transform(ray_direction)
        else:
            raise ValueError("upper_or_lower string must be 'upper' or 'lower'")
    else:
        raise ValueError("upper_or_lower must be a string or jnp.array")
    

    const_term = a00 \
        #+ a01 * ray_origin[1] + a10 * ray_origin[0] \
        #+ a02 * ray_origin[1]**2 + a20 * ray_origin[0]**2 + a11 * ray_origin[0] * ray_origin[1]
    
    linear_term = a01 * ray_direction[1] + a10 * ray_direction[0] \
        #+ 2 * a02 * ray_origin[1] * ray_direction[1] + 2 * a20 * ray_origin[0] * ray_direction[0] \
        #+ a11 * (ray_origin[0] * ray_direction[1] + ray_origin[1] * ray_direction[0])
    
    quadratic_term = a02 * ray_direction[1]**2 + a20 * ray_direction[0]**2 + a11 * ray_direction[0] * ray_direction[1]

    return jnp.array([const_term, linear_term, quadratic_term])


def transform_2nd_order_3d(taylor_enclosure, ray_direction, upper_or_lower):
    """
    Transform 2nd order 3D polynomial coefficients into a 2nd order polynomial along a ray.
    Assuming that ray_origin is at [0, 0, 0].
    Reason: autobounds are defined relative to a local origin.
    Quadric intersection math: http://www.bmsc.washington.edu/people/merritt/graphics/quadrics.html
    """
    index_map = lambda idxs: multi_idx_3d_2nd_order.index(idxs)

    a000 = taylor_enclosure[index_map([0, 0, 0])]
    a100 = taylor_enclosure[index_map([1, 0, 0])]
    a010 = taylor_enclosure[index_map([0, 1, 0])]
    a001 = taylor_enclosure[index_map([0, 0, 1])]
    a110_lower = taylor_enclosure[index_map([1, 1, 0])].lower
    a101_lower = taylor_enclosure[index_map([1, 0, 1])].lower
    a011_lower = taylor_enclosure[index_map([0, 1, 1])].lower
    a200_lower = taylor_enclosure[index_map([2, 0, 0])].lower
    a020_lower = taylor_enclosure[index_map([0, 2, 0])].lower
    a002_lower = taylor_enclosure[index_map([0, 0, 2])].lower
    a110_upper = taylor_enclosure[index_map([1, 1, 0])].upper
    a101_upper = taylor_enclosure[index_map([1, 0, 1])].upper
    a011_upper = taylor_enclosure[index_map([0, 1, 1])].upper
    a200_upper = taylor_enclosure[index_map([2, 0, 0])].upper
    a020_upper = taylor_enclosure[index_map([0, 2, 0])].upper
    a002_upper = taylor_enclosure[index_map([0, 0, 2])].upper

    def lower_transform(ray_direction):
        a110 = jnp.where(jnp.sign(ray_direction[0])*jnp.sign(ray_direction[1]) >= 0., a110_lower, a110_upper)
        a101 = jnp.where(jnp.sign(ray_direction[0])*jnp.sign(ray_direction[2]) >= 0., a101_lower, a101_upper)
        a011 = jnp.where(jnp.sign(ray_direction[1])*jnp.sign(ray_direction[2]) >= 0., a011_lower, a011_upper)
        a200 = a200_lower
        a020 = a020_lower
        a002 = a002_lower
        return a110, a101, a011, a200, a020, a002
    
    def upper_transform(ray_direction):
        a110 = jnp.where(jnp.sign(ray_direction[0])*jnp.sign(ray_direction[1]) >= 0., a110_upper, a110_lower)
        a101 = jnp.where(jnp.sign(ray_direction[0])*jnp.sign(ray_direction[2]) >= 0., a101_upper, a101_lower)
        a011 = jnp.where(jnp.sign(ray_direction[1])*jnp.sign(ray_direction[2]) >= 0., a011_upper, a011_lower)
        a200 = a200_upper
        a020 = a020_upper
        a002 = a002_upper
        return a110, a101, a011, a200, a020, a002

    if isinstance(upper_or_lower, jnp.ndarray):
        a110, a101, a011, a200, a020, a002 = jax.lax.cond(upper_or_lower==jnp.array(1), upper_transform, lower_transform, ray_direction)
    elif isinstance(upper_or_lower, str):
        if upper_or_lower == "upper":
            a110, a101, a011, a200, a020, a002 = upper_transform(ray_direction)
        elif upper_or_lower == "lower":
            a110, a101, a011, a200, a020, a002 = lower_transform(ray_direction)
        else:
            raise ValueError("upper_or_lower string must be 'upper' or 'lower'")
    else:
        raise ValueError("upper_or_lower must be a string or jnp.array")

    const_term = a000 \
        #+ a001 * ray_origin[2] + a010 * ray_origin[1] + a100 * ray_origin[0] + \
        #a002 * ray_origin[2]**2 + a020 * ray_origin[1]**2 + a200 * ray_origin[0]**2 + \
        #a011 * ray_origin[1] * ray_origin[2] + a101 * ray_origin[0] * ray_origin[2] + a110 * ray_origin[0] * ray_origin[1]
    
    linear_term = a001 * ray_direction[2] + a010 * ray_direction[1] + a100 * ray_direction[0] \
        #+ 2 * a002 * ray_origin[2] * ray_direction[2] + 2 * a020 * ray_origin[1] * ray_direction[1] + 2 * a200 * ray_origin[0] * ray_direction[0] + \
        #a011 * (ray_origin[1] * ray_direction[2] + ray_origin[2] * ray_direction[1]) + \
        #a101 * (ray_origin[0] * ray_direction[2] + ray_origin[2] * ray_direction[0]) + \
        #a110 * (ray_origin[0] * ray_direction[1] + ray_origin[1] * ray_direction[0])
    
    quadratic_term = a002 * ray_direction[2]**2 + a020 * ray_direction[1]**2 + a200 * ray_direction[0]**2 + \
        a011 * ray_direction[1] * ray_direction[2] + a101 * ray_direction[0] * ray_direction[2] + a110 * ray_direction[0] * ray_direction[1]
    
    return jnp.array([const_term, linear_term, quadratic_term])


def roots_2nd_order_tensor(taylor_enclosure, ray_direction, upper_or_lower, dim):
    if dim == 3:
        poly_coeffs = transform_2nd_order_3d(taylor_enclosure, ray_direction, upper_or_lower)
    elif dim == 2:
        poly_coeffs = transform_2nd_order_2d(taylor_enclosure, ray_direction, upper_or_lower)
    elif dim == 1:
        poly_coeffs = transform_2nd_order_1d(taylor_enclosure, ray_direction, upper_or_lower)
    return roots_2nd_order_polynomial(poly_coeffs, ray_direction)


def roots_2nd_order_polynomial(poly_coeffs, ray_direction):
    c = poly_coeffs[0] # constant term
    b = poly_coeffs[1] # linear term
    a = poly_coeffs[2] # quadratic term

    # Safe sqrt to avoid NaNs in gradients if discriminant < 0
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