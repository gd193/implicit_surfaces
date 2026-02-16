import jax
import jax.numpy as jnp


def bionomial_coefficient(n: int, k: int):
    """Calculate the binomial coefficient C(n, k) = n! / (k! * (n - k)!)"""
    return jax.scipy.special.factorial(n) / (jax.scipy.special.factorial(k) * jax.scipy.special.factorial(n - k))


def generalized_bionomial_coefficient(n: jnp.array, k: jnp.array):
    """Calculate the generalized binomial coefficient for multi-indices n and k."""
    if n.shape != k.shape or n.ndim != 1 or k.ndim != 1:
        raise ValueError("n and k must have the same length")
    return jax.lax.scan(lambda carry, i: (carry * bionomial_coefficient(n[i], k[i]), None), jnp.array(1.), jnp.arange(len(n)))[0]



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

def l_B_i(l: jnp.array, i: jnp.array, x: jnp.array) -> jnp.array:
    """
    Bernstein basis polynomial of degree l at point i.
    Args:
        l: jnp.array of shape (n,) degree of the polynomial (multi-indexed)
        i: jnp.array of shape (n,) index of the polynomial (multi-indexed)
        x: jnp.array of shape (n,) point at which to evaluate the polynomial
    Returns:
        scalar jnp.array (,) evaluated polynomial at point x
    """
    bionomial_coeff = generalized_bionomial_coefficient(l, i)
    x_i = jnp.power(x, i)
    x_i_l = jnp.power(1 - x, l - i)
    return bionomial_coeff * jnp.prod(x_i * x_i_l)

def bernstein_polynomial(bernstein_coeff: jnp.array, x: jnp.array) -> jnp.array:
    """
    Bernstein polynomial at point x.
    Args: 
        bernstein_coeff: jnp.array (l1, ..., ln) of coefficients of the Bernstein polynomial
        x: jnp.array of shape (n,) point at which to evaluate the polynomial
    Returns:
        scalar jnp.array (,) evaluated polynomial at point x
    """
    dim = bernstein_coeff.ndim
    l_idx = jnp.array(bernstein_coeff.shape) - 1  # degree of the polynomial
    idxs = jnp.indices(bernstein_coeff.shape).reshape(dim, -1).T

    eval_points_in_basis = jax.vmap(l_B_i, in_axes=(None, 0, None))(l_idx, idxs, x)
    eval_points_with_coeff = jnp.where(bernstein_coeff.ravel() != 0, eval_points_in_basis * bernstein_coeff.ravel(), 0.0)

    return jnp.sum(eval_points_with_coeff)

def bernstein_l_b_i_from_power_series(power_series_coefficients, l, i):
    """
    Compute bernstein basis coeff l_b_i from power series coefficients via the inverse bionomial transform
    Equation (8) from https://link.springer.com/article/10.1007/s10898-007-9195-4 
    Args:
        power_series_coefficients: jnp.array (l1, ..., ln) coefficients
        l: jnp.array of shape (n,) degree of the polynomial (multi-indexed)
        i: jnp.array of shape (n,) index of the polynomial (multi-indexed)
    Returns:
    """
    dim = power_series_coefficients.ndim
    j_indices = jnp.indices(power_series_coefficients.shape).reshape(dim, -1).T
    valid_mask = jnp.all(j_indices <= i[None, :], axis=-1)

    bionomial_coeff = jax.vmap(generalized_bionomial_coefficient, in_axes=(None, 0))(i, j_indices) / \
                      jax.vmap(generalized_bionomial_coefficient, in_axes=(None, 0))(l, j_indices)
    bionomial_coeff = jnp.where(valid_mask, bionomial_coeff, 0.0)
    l_b_i_summands = bionomial_coeff * power_series_coefficients.ravel()

    return jnp.sum(l_b_i_summands, axis=0)


def transform_power_series_to_bernstein_basis(power_series_coefficients: jnp.array) -> jnp.array:
    """
    Transform power series basis coefficients to Bernstein basis coefficients on [0, 1]^2.
    Args:
        power_series_coefficients: jnp.array of shape (l1, ..., ln) power series coefficients
    Returns:
        jnp.array of shape (l1, ..., ln) Bernstein coefficients
    """
    dim = power_series_coefficients.ndim
    l_idx = jnp.array(power_series_coefficients.shape) - 1  # degree

    i_indices = jnp.indices(power_series_coefficients.shape).reshape(dim, -1).T
    bernstein_coeffs = jax.vmap(bernstein_l_b_i_from_power_series, in_axes=(None, None, 0))(power_series_coefficients, l_idx, i_indices)

    return bernstein_coeffs.reshape(power_series_coefficients.shape)


def bernstein_interval_l_b_i_from_power_series_summands(a_i: jnp.array, bbox: jnp.array, l: jnp.array, j: jnp.array) -> jnp.array:
    """
    Compute summands from k=j to l for generalized Bernstein basis transformation.
    Right-most term in equation (9) from https://link.springer.com/article/10.1007/s10898-007-9195-4

    Args:
        a_i: jnp.array of shape (l1, l2) power series coefficients
        bbox: jnp.array of shape (2, 2) bounding box of the polynomial
        l: jnp.array of shape (2,) degree of polynomial (multi-index)
        j: jnp.array of shape (2,) summation index (multi-index)
    Returns:
        scalar jnp.array (,) sum from k=j to l of weighted power series coefficients
    """
    dim = a_i.ndim
    k_indices = jnp.indices(a_i.shape).reshape(dim, -1).T
    mask = jnp.all(
        jnp.logical_and(k_indices >= j[None, :], k_indices <= l[None, :]), axis=-1
    )

    binomial_coeff = jax.vmap(generalized_bionomial_coefficient, in_axes=(0, None))(k_indices, j)
    binomial_coeff = jnp.where(mask, binomial_coeff, 0.0)
    x_k_j = jnp.pow(bbox[0, :], k_indices - j)
    x_k_j = jnp.where(mask, jnp.prod(x_k_j, axis=-1), 0.0)
    summands = binomial_coeff * a_i.ravel() * x_k_j
    return jnp.sum(summands)


def bernstein_interval_l_b_i_from_power_series(power_series_coeffs: jnp.array, bbox: jnp.array, l: jnp.array, i: jnp.array) -> jnp.array:
    """
    Compute bernstein basis coeff l_b_i from power series coefficients on the bbox interval.
    Equation (9) from https://link.springer.com/article/10.1007/s10898-007-9195-4
    Args:
        power_series_coeffs: jnp.array of shape (l1, ..., ln) power series coefficients
        bbox: jnp.array of shape (2, n) bounding box of the polynomial
        l: jnp.array of shape (n,) degree of the polynomial (multi-indexed)
        i: jnp.array of shape (n,) index of the polynomial (multi-indexed)
    Returns:
        scalar jnp.array (,) coeff l_b_i on the bbox interval
    """
    dim = power_series_coeffs.ndim
    j_indices = jnp.indices(power_series_coeffs.shape).reshape(dim, -1).T
    mask = jnp.all(jnp.logical_and(j_indices <= i[None, :], j_indices <= l[None, :]), axis=-1)

    bionomial_coeff = jax.vmap(generalized_bionomial_coefficient, in_axes=(None, 0))(i, j_indices) / \
                      jax.vmap(generalized_bionomial_coefficient, in_axes=(None, 0))(l, j_indices)
    bionomial_coeff = jnp.where(mask, bionomial_coeff, 0.0)
    delta_j = jnp.pow((bbox[1, :] - bbox[0, :]), j_indices)
    delta_j = jnp.prod(delta_j, axis=1)
    sum_contributions = jax.vmap(bernstein_interval_l_b_i_from_power_series_summands, in_axes=(None, None, None, 0))(
        power_series_coeffs, bbox, l, j_indices
    )
    return jnp.sum(bionomial_coeff * delta_j * sum_contributions)


@jax.jit
def transform_power_series_interval_to_bernstein(power_series_coeffs: jnp.array, bbox: jnp.array) -> jnp.array:
    """
    Gerneralized Bernstein basis transformation from power series coefficients on a bounding box [a_1, a_2]x[b_1, b_2].
    Args:
        power_series_coeffs: jnp.array of shape (l1, l2) power series coefficients
        bbox: jnp.array of shape (2, 2) bounding box of the polynomial
    Returns:
        jnp.array of shape (l1, l2) Bernstein coefficients on the bounding box
    """
    dim = power_series_coeffs.ndim
    l_idx = jnp.array(power_series_coeffs.shape) - 1  # degree of the polynomial
    i_indices = jnp.indices(power_series_coeffs.shape).reshape(dim, -1).T

    bernstein_coeffs = jax.vmap(bernstein_interval_l_b_i_from_power_series, in_axes=(None, None, None, 0))(
        power_series_coeffs, bbox, l_idx, i_indices
    )

    return bernstein_coeffs.reshape(power_series_coeffs.shape)