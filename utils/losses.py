import jax 
import jax.numpy as jnp
import operator
from typing import Callable, Optional, Any
from functools import partial
from utils.tree_ops import binary_tree_op


def shape_functional(loss_fn: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray], 
                     mask: Optional[jnp.ndarray], 
                     solution_u: jnp.ndarray,
                     solution_target: jnp.ndarray) -> jnp.ndarray:
    """Shape functional loss L over domain Omega, where mask is 1 in Omega and 0 outside."""
    if mask is not None:
        loss = jnp.sum(mask * loss_fn(solution_u, solution_target)) / jnp.sum(mask)
    else:
        loss = jnp.mean(loss_fn(solution_u, solution_target))
    return loss


def volume_integral_loss_grad(dL_dparams_field: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray],
                              mask: jnp.ndarray,
                              volume: jnp.ndarray,
                              normalization_factor: Optional[jnp.ndarray],
                              solution_u: jnp.ndarray,
                              del_solution_del_theta: jnp.ndarray,
                              solution_target: jnp.ndarray) -> jnp.ndarray:
    """
    volume integral of the gradient of the loss with respect to theta over domain Omega, 
    where mask is 1 in Omega and 0 outside.
    (See Reynolds transport theorem: https://en.wikipedia.org/wiki/Reynolds_transport_theorem)
    """
    if normalization_factor is None:
        normalization_factor = 1.0
    ### scalar loss field to apply to all points in the domain
    sensitivity_field = dL_dparams_field(solution_u, solution_target)
    del_L_del_theta = binary_tree_op(operator.mul, del_solution_del_theta, sensitivity_field)
    dV = (volume / (mask.sum() + 1e-12))
    return jax.tree.map(lambda leaf: jnp.sum(leaf[mask], axis=0) * dV * (1/normalization_factor), del_L_del_theta)


def surface_integral_loss_grad(loss_fn: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray],
                               volume: jnp.ndarray,
                               epsilon: jnp.ndarray,
                               solution_u: jnp.ndarray,
                               solution_target: jnp.ndarray,
                               normal_velocity_field: jnp.ndarray) -> jnp.ndarray:
    """
    Surface integral of the gradient of the loss with respect to theta over the boundary of domain Omega.
    Approximate the surface integral as a volume integral over a thin layer of thickness 2 epsilon around the boundary.
    (See Reynolds transport theorem: https://en.wikipedia.org/wiki/Reynolds_transport_theorem)
    """
    L_field = loss_fn(solution_u, solution_target)
    dV = 1 / (2 * epsilon) * (volume/ (L_field.size + 1e-12))  # Volume of the thin layer per point 
    grad_field = binary_tree_op(operator.mul, normal_velocity_field, L_field)
    return jax.tree.map(lambda leaf: jnp.sum(leaf, axis=0) * dV, grad_field)


def surface_integral_loss_grad_normalized(loss_fn: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray],
                                          mask: jnp.ndarray,
                                          epsilon_volume: jnp.ndarray,
                                          epsilon: jnp.ndarray,
                                          normalization_factor: jnp.ndarray,
                                          unnormalized_energy: jnp.ndarray,
                                          solution_u: jnp.ndarray,
                                          solution_target: jnp.ndarray,
                                          normal_velocity_field: jnp.ndarray) -> jnp.ndarray:
    """
    Normalized surface integral of the gradient of the loss with respect to theta over the boundary of domain Omega.
    1/(Vol_tot) * int_{del Omega} V_n(x) * (L(x) - J_energy / Vol_tot) dA
    where del Omega is approximated as vol(del Omega +/- epsilon)
    """
    L_field = loss_fn(solution_u, solution_target)
    dV_normalized = 1 / (2 * epsilon) * 1/(normalization_factor + 1e-12) * (epsilon_volume/ (mask.sum() + 1e-12))  # Volume of the thin layer per point normalized by total volume
    grad_field = binary_tree_op(operator.mul, normal_velocity_field, (L_field - unnormalized_energy / (normalization_factor + 1e-12)))
    return jax.tree.map(lambda leaf: jnp.sum(leaf[mask], axis=0) * dV_normalized, grad_field)

def l2_loss(solution_u: jnp.ndarray, solution_target: jnp.ndarray) -> jnp.ndarray:
    """L2 loss between solution_u and solution_target."""
    return 0.5 * jnp.power(solution_u - solution_target, 2)


def dL2_du(solution_u: jnp.ndarray, solution_target: jnp.ndarray) -> jnp.ndarray:
    """Gradient of L2 loss with respect to solution_u."""
    return solution_u - solution_target




