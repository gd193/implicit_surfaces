import jax
import jax.numpy as jnp


multi_indeces_1d_2nd_order = jnp.array([
    [0],  # a0
    [1],  # a1
    [2],  # a2
])

multi_indeces_2nd_order = jnp.array([
    [0, 0], # a00
    [0, 1], # a01
    [0, 2], # a02
    [1, 0], # a10
    [1, 1], # a11
    [2, 0], # a20
])

multi_idx_3d_2nd_order = (
    (0, 0, 0),  # a000
    (1, 0, 0),  # a100
    (0, 1, 0),  # a010
    (0, 0, 1),  # a001
    (1, 1, 0),  # a110
    (1, 0, 1),  # a101
    (0, 1, 1),  # a011
    (2, 0, 0),  # a200
    (0, 2, 0),  # a020
    (0, 0, 2),  # a002  
)

def transform_2nd_order_tensor(tensor):

    if tensor.ndim == 4:
        coeffs = jnp.zeros((16,))
        coeffs = coeffs.at[0].set(tensor[0, *multi_idx_3d_2nd_order[0]])
        coeffs = coeffs.at[1].set(tensor[0, *multi_idx_3d_2nd_order[1]])
        coeffs = coeffs.at[2].set(tensor[0, *multi_idx_3d_2nd_order[2]])
        coeffs = coeffs.at[3].set(tensor[0, *multi_idx_3d_2nd_order[3]])
        coeffs = coeffs.at[4].set(tensor[0, *multi_idx_3d_2nd_order[4]])
        coeffs = coeffs.at[5].set(tensor[0, *multi_idx_3d_2nd_order[5]])
        coeffs = coeffs.at[6].set(tensor[0, *multi_idx_3d_2nd_order[6]])
        coeffs = coeffs.at[7].set(tensor[0, *multi_idx_3d_2nd_order[7]])
        coeffs = coeffs.at[8].set(tensor[0, *multi_idx_3d_2nd_order[8]])
        coeffs = coeffs.at[9].set(tensor[0, *multi_idx_3d_2nd_order[9]])
        coeffs = coeffs.at[10].set(tensor[1, *multi_idx_3d_2nd_order[4]])
        coeffs = coeffs.at[11].set(tensor[1, *multi_idx_3d_2nd_order[5]])
        coeffs = coeffs.at[12].set(tensor[1, *multi_idx_3d_2nd_order[6]])
        coeffs = coeffs.at[13].set(tensor[1, *multi_idx_3d_2nd_order[7]])
        coeffs = coeffs.at[14].set(tensor[1, *multi_idx_3d_2nd_order[8]])
        coeffs = coeffs.at[15].set(tensor[1, *multi_idx_3d_2nd_order[9]])

        return coeffs
    
    elif tensor.ndim == 3:
        coeffs = jnp.zeros((10,))
        coeffs = coeffs.at[0].set(tensor[0, 0])  # a00
        coeffs = coeffs.at[1].set(tensor[0, 1])  # a01
        coeffs = coeffs.at[2].set(tensor[0, 2])  # a02
        coeffs = coeffs.at[3].set(tensor[1, 0])  # a10
        coeffs = coeffs.at[4].set(tensor[1, 1])  # a11
        coeffs = coeffs.at[5].set(tensor[2, 0])  # a20
        return coeffs
    
    elif tensor.ndim == 2:
        coeffs = jnp.zeros((4,))
        return coeffs
    
    else:
        raise ValueError("Tensor must be 1D, 2D, or 3D. with an extra upper/lower dimension.")

def transform_2nd_order_1d(coeffs, ray_direction, upper_or_lower):

    a0 = coeffs[0]
    a1 = coeffs[1]
    a2 = coeffs[2]

    const_term = a0 + a1 * ray_origin[0] + a2 * ray_origin[0]**2
    linear_term = a1 * ray_direction[0] + 2 * a2 * ray_origin[0] * ray_direction[0]
    quadratic_term = a2 * ray_direction[0]**2

    return jnp.array([const_term, linear_term, quadratic_term])

def transform_2nd_order_2d(coeffs, ray_direction, upper_or_lower):

    a00 = coeffs[0]
    a01 = coeffs[1]
    a02 = coeffs[2]
    a10 = coeffs[3]
    a11 = coeffs[4]
    a20 = coeffs[5]

    const_term = a00 + \
        a01 * ray_origin[1] + a10 * ray_origin[0] + \
        a02 * ray_origin[1]**2 + a20 * ray_origin[0]**2 + a11 * ray_origin[0] * ray_origin[1]
    
    linear_term = a01 * ray_direction[1] + a10 * ray_direction[0] + \
        2 * a02 * ray_origin[1] * ray_direction[1] + 2 * a20 * ray_origin[0] * ray_direction[0] + \
        a11 * (ray_origin[0] * ray_direction[1] + ray_origin[1] * ray_direction[0])
    
    quadratic_term = a02 * ray_direction[1]**2 + a20 * ray_direction[0]**2 + a11 * ray_direction[0] * ray_direction[1]

    return jnp.array([const_term, linear_term, quadratic_term])


def transform_2nd_order_3d(coeffs, ray_direction, upper_or_lower):
    """
    Transform 2nd order 3D polynomial coefficients into a 2nd order polynomial along a ray.
    Assuming that ray_origin is at [0, 0, 0].
    Reason: autobounds are defined relative to a local origin.
    Quadric intersection math: http://www.bmsc.washington.edu/people/merritt/graphics/quadrics.html
    """
    a000 = coeffs[0]
    a100 = coeffs[1]
    a010 = coeffs[2]
    a001 = coeffs[3]
    a110_lower = coeffs[4]
    a101_lower = coeffs[5]
    a011_lower = coeffs[6]
    a200_lower = coeffs[7]
    a020_lower = coeffs[8]
    a002_lower = coeffs[9]
    a110_upper = coeffs[10]
    a101_upper = coeffs[11]
    a011_upper = coeffs[12]
    a200_upper = coeffs[13]
    a020_upper = coeffs[14]
    a002_upper = coeffs[15]

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


def roots_2nd_order_tensor(tensor, ray_direction, upper_or_lower):
    flattened_coeffs = transform_2nd_order_tensor(tensor)
    if tensor.ndim == 4:
        poly_coeffs = transform_2nd_order_3d(flattened_coeffs, ray_direction, upper_or_lower)
    elif tensor.ndim == 3:
        poly_coeffs = transform_2nd_order_2d(flattened_coeffs, ray_direction, upper_or_lower)
    elif tensor.ndim == 2:
        poly_coeffs = transform_2nd_order_1d(flattened_coeffs, ray_direction, upper_or_lower)
    return roots_2nd_order_polynomial(poly_coeffs, ray_direction)


def roots_2nd_order_polynomial(poly_coeffs, ray_direction):
    c = poly_coeffs[0] # constant term
    b = poly_coeffs[1] # linear term
    a = poly_coeffs[2] # quadratic term

    discriminant = b**2 - 4*a*c
    sqrt_discriminant = jnp.sqrt(discriminant)

    root_0 = (-b - jnp.sign(b)*sqrt_discriminant) / (2*a)
    root_1 = c / (a*root_0)
    root_near = jnp.minimum(root_0, root_1)
    root_far = jnp.maximum(root_0, root_1)

    t = jnp.where(root_near > 0, root_near, root_far)
    t = jnp.where(discriminant >= 0, t, -jnp.inf)
    hit = t > 0
    
    return hit, t