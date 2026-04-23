import jax 
import jax.numpy as jnp
from functools import partial


def power_series_polynomial(power_series_coefficients: jnp.array, x: jnp.array) -> jnp.array:
    """
    Evaluate a polynomial given by its power series coefficients at a point x.
    The indices of the coefficients correspond to the powers of x.
    Args:
        power_series_coefficients: coefficient jnp.array (l1, ..., ln), polynomial is of degree li in x[i]
        x: jnp.array (n,), point at which to evaluate the polynomial
    Returns:
        scalar jnp.array (,) evaluated polynomial at point x
    """
    dim = power_series_coefficients.ndim
    idxs = jnp.indices(power_series_coefficients.shape).reshape(dim, -1).T
    monomials = jnp.prod(jnp.power(x, idxs), axis=-1)
    return jnp.sum(power_series_coefficients.ravel() * monomials)

"""
Convert autobounds coefficients to power series coefficient tensor.
This implementation is kinda hot garbage, but it works for now.
I don't know how to elegantly do the conversion without hardcoding the tensor indices,
especially since ab_bounds are tuples of jnp.arrays, so it's kinda weird.
"""
################################
# 1d conversion functions
################################
def add_0th_order_1d(ps_vector, coeff):
    ps_vector = ps_vector.at[0].set(coeff[0][0])
    return ps_vector

def add_1st_order_1d(ps_vector, coeff):
    ps_vector = ps_vector.at[1].set(coeff[1][0, 0])
    return ps_vector

def add_2nd_order_1d(ps_vector, coeff):
    ps_vector = ps_vector.at[2].set(coeff[2][0, 0, 0])
    return ps_vector

def add_3rd_order_1d(ps_vector, coeff):
    ps_vector = ps_vector.at[3].set(coeff[3][0, 0, 0, 0])
    return ps_vector

@partial(jax.jit, static_argnames=('degree'))
def ab_to_ps_1d(ab_coefficients: tuple, degree: int) -> jnp.array:
    ps_vector = jnp.zeros((degree + 1))
    ps_vector = add_0th_order_1d(ps_vector, ab_coefficients)
    if degree >= 1:
        ps_vector = add_1st_order_1d(ps_vector, ab_coefficients)
    if degree >= 2:
        ps_vector = add_2nd_order_1d(ps_vector, ab_coefficients)
    if degree >= 3:
        ps_vector = add_3rd_order_1d(ps_vector, ab_coefficients)
    return ps_vector


################################
# 2d conversion functions
################################
def add_0th_order2d(ps_matrix, coeff):
    ps_matrix = ps_matrix.at[0, 0].set(coeff[0][0])
    return ps_matrix

def add_1st_order2d(ps_matrix, coeff):
    ps_matrix = ps_matrix.at[0, 1].set(coeff[1][0][0])
    ps_matrix = ps_matrix.at[1, 0].set(coeff[1][0][1])
    return ps_matrix

def add_2nd_order2d(ps_matrix, coeff):
    ps_matrix = ps_matrix.at[2, 0].set(coeff[2][0][0, 0])
    ps_matrix = ps_matrix.at[0, 2].set(coeff[2][0][1, 1])
    ps_matrix = ps_matrix.at[1, 1].set(coeff[2][0][0, 1] + coeff[2][0][1, 0])
    return ps_matrix

def add_3rd_order2d(ps_matrix, coeff):
    ps_matrix = ps_matrix.at[3, 0].set(coeff[3][0][0, 0, 0])
    ps_matrix = ps_matrix.at[0, 3].set(coeff[3][0][1, 1, 1])
    ps_matrix = ps_matrix.at[1, 2].set(coeff[3][0][0, 1, 1] + coeff[3][0][1, 0, 1] + coeff[3][0][1, 1, 0])
    ps_matrix = ps_matrix.at[2, 1].set(coeff[3][0][1, 0, 0] + coeff[3][0][0, 1, 0] + coeff[3][0][0, 0, 1])
    return ps_matrix

@partial(jax.jit, static_argnames=('degree'))
def ab_to_ps_2d(ab_coefficients: tuple, degree: int) -> jnp.array:
    ps_matrix = jnp.zeros((degree + 1, degree + 1))
    if degree == 0:
        ps_matrix = add_0th_order2d(ps_matrix, ab_coefficients)
    elif degree == 1:
        ps_matrix = add_1st_order2d(add_0th_order2d(ps_matrix, ab_coefficients), ab_coefficients)
    elif degree == 2:
        ps_matrix = add_2nd_order2d(add_1st_order2d(add_0th_order2d(ps_matrix, ab_coefficients), ab_coefficients), ab_coefficients)
    elif degree == 3:
        ps_matrix = add_3rd_order2d(add_2nd_order2d(add_1st_order2d(add_0th_order2d(ps_matrix, ab_coefficients), ab_coefficients), ab_coefficients), ab_coefficients)
    else:
        raise ValueError("Unsupported degree")
                           
    return ps_matrix

################################
# 3d conversion functions
################################
def add_0th_order_3d(ps_tensor, coeff):
    coeff0 = coeff[0][0]
    
    ps_tensor = ps_tensor.at[0, 0, 0].set(coeff0) # constant term
    return ps_tensor

def add_1st_order_3d(ps_tensor, coeff):
    coeff1 = coeff[1][0]

    ps_tensor = ps_tensor.at[1, 0, 0].set(coeff1[0]) # x0
    ps_tensor = ps_tensor.at[0, 1, 0].set(coeff1[1]) # x1
    ps_tensor = ps_tensor.at[0, 0, 1].set(coeff1[2]) # x2

    return ps_tensor


def add_2nd_order_3d(ps_tensor, coeff):
    coeff2 = coeff[2][0]

    ps_tensor = ps_tensor.at[2, 0, 0].set(coeff2[0, 0]) # x0²
    ps_tensor = ps_tensor.at[0, 2, 0].set(coeff2[1, 1]) # x1²
    ps_tensor = ps_tensor.at[0, 0, 2].set(coeff2[2, 2]) # x2²

    ps_tensor = ps_tensor.at[1, 1, 0].set(coeff2[0, 1] + coeff2[1, 0]) # x0 x1
    ps_tensor = ps_tensor.at[1, 0, 1].set(coeff2[0, 2] + coeff2[2, 0]) # x0 x2
    ps_tensor = ps_tensor.at[0, 1, 1].set(coeff2[1, 2] + coeff2[2, 1]) # x1 x2

    return ps_tensor

def add_3rd_order_3d(ps_tensor, coeff):
    coeff3 = coeff[3][0]
    ps_tensor = ps_tensor.at[3, 0, 0].set(coeff3[0, 0, 0])
    ps_tensor = ps_tensor.at[0, 3, 0].set(coeff3[1, 1, 1])
    ps_tensor = ps_tensor.at[0, 0, 3].set(coeff3[2, 2, 2])

    ps_tensor = ps_tensor.at[2, 1, 0].set(coeff3[0, 0, 1] + coeff3[0, 1, 0] + coeff3[1, 0, 0]) #x0² x1
    ps_tensor = ps_tensor.at[2, 0, 1].set(coeff3[0, 0, 2] + coeff3[0, 2, 0] + coeff3[2, 0, 0]) #x0² x2

    ps_tensor = ps_tensor.at[1, 2, 0].set(coeff3[0, 1, 1] + coeff3[1, 0, 1] + coeff3[1, 1, 0]) #x0 x1²
    ps_tensor = ps_tensor.at[1, 0, 2].set(coeff3[0, 2, 2] + coeff3[2, 0, 2] + coeff3[2, 2, 0]) #x0 x2²

    ps_tensor = ps_tensor.at[0, 2, 1].set(coeff3[1, 1, 2] + coeff3[1, 2, 1] + coeff3[2, 1, 1]) #x1² x2
    ps_tensor = ps_tensor.at[0, 1, 2].set(coeff3[2, 2, 1] + coeff3[2, 1, 2] + coeff3[1, 2, 2]) #x1 x2²

    ps_tensor = ps_tensor.at[1, 1, 1].set(coeff3[0, 1, 2] + coeff3[2, 0, 1] + coeff3[2, 1, 0] + #x0 x1 x2
                                          coeff3[1, 0, 2] + coeff3[1, 2, 0] + coeff3[0, 2, 1])
    
    return ps_tensor

@partial(jax.jit, static_argnames=('degree'))
def ab_to_ps_3d(ab_coefficients: tuple, degree: int) -> jnp.array:
    ps_tensor = jnp.zeros((degree + 1, degree + 1, degree + 1))

    if degree == 0:
        ps_tensor = add_0th_order_3d(ps_tensor, ab_coefficients)
    elif degree == 1:
        ps_tensor = add_1st_order_3d(add_0th_order_3d(ps_tensor, ab_coefficients), ab_coefficients)
    elif degree == 2:
        ps_tensor = add_2nd_order_3d(add_1st_order_3d(add_0th_order_3d(ps_tensor, ab_coefficients), ab_coefficients), ab_coefficients)
    elif degree == 3:
        ps_tensor = add_3rd_order_3d(add_2nd_order_3d(add_1st_order_3d(add_0th_order_3d(ps_tensor, ab_coefficients), ab_coefficients), ab_coefficients), ab_coefficients)
    else:
        raise ValueError("Unsupported degree")                          
    return ps_tensor




@partial(jax.jit, static_argnames=('dimension', 'degree'))
def autobounds_coefficients_to_power_series(ab_coefficients: tuple, dimension: int, degree: int) -> jnp.array:
    """
    Convert autobounds coefficient tuple to power series coefficient tensor.
    Args:
        ab_coefficients: autobounds.coefficients tuple [jnp.array,...] if degree l
        dimension: int, dimension of multivariate polynomial
        degree: int, degree of multivariate polynomial
    Returns:
        jnp.array of shape (l1, ..., ln) power series coefficients
    """
    if isinstance(ab_coefficients[-1], tuple):
        ab_coefficient_tuple_lower = tuple(
            ab_coefficients[i] if i<(len(ab_coefficients) - 1) else ab_coefficients[i][0] for i in range(len(ab_coefficients))
        )
        ab_coefficient_tuple_upper = tuple(
            ab_coefficients[i] if i<(len(ab_coefficients) - 1) else ab_coefficients[i][1] for i in range(len(ab_coefficients))
        )
        if dimension == 1:
            power_series_coeff_matrix_lower = ab_to_ps_1d(ab_coefficient_tuple_lower, degree)
            power_series_coeff_matrix_upper = ab_to_ps_1d(ab_coefficient_tuple_upper, degree)
        elif dimension == 2:
            power_series_coeff_matrix_lower = ab_to_ps_2d(ab_coefficient_tuple_lower, degree)
            power_series_coeff_matrix_upper = ab_to_ps_2d(ab_coefficient_tuple_upper, degree)
        elif dimension == 3:
            power_series_coeff_matrix_lower = ab_to_ps_3d(ab_coefficient_tuple_lower, degree)
            power_series_coeff_matrix_upper = ab_to_ps_3d(ab_coefficient_tuple_upper, degree)
        else:
            raise ValueError("Unsupported dimension")
        
        return jnp.stack([power_series_coeff_matrix_lower, power_series_coeff_matrix_upper], axis=0)
    
    else:
        ab_coefficient_tuple = ab_coefficients

        if dimension == 1:
            power_series_coeff_matrix = ab_to_ps_1d(ab_coefficient_tuple, degree)
        elif dimension == 2:
            power_series_coeff_matrix = ab_to_ps_2d(ab_coefficient_tuple, degree)
        elif dimension == 3:
            power_series_coeff_matrix = ab_to_ps_3d(ab_coefficient_tuple, degree)
        else:
            raise ValueError("Unsupported dimension")
            
        return power_series_coeff_matrix