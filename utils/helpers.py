import jax.numpy as jnp


def update_trust_region_size(trust_region_size: jnp.ndarray, t_step: jnp.ndarray, in_tr: jnp.ndarray) -> jnp.ndarray:
    # Conservative default: shrink on unsafe step; grow only when the step was unusually
    # small relative to tr (rare; mostly serves as a safety hedge). NIQ's `cast_rays_iter`
    # uses a more aggressive grow-on-safe rule which suits its classify-and-step semantics
    # but destabilises autobound's root-find (slop scales as tr², so growing tr loosens
    # the polynomial enclosure → spurious roots → oscillation).
    new = jnp.where(~in_tr, trust_region_size * 0.5, trust_region_size)
    new = jnp.where((t_step < 0.5 * trust_region_size) & in_tr, trust_region_size * 2.0, new)
    return new


def spatial_termination_criterion(t_step: jnp.ndarray, fn_val: jnp.ndarray, tr_size: jnp.ndarray, level_set_threshold: float, epsilon: float) -> jnp.ndarray:
    return jnp.logical_and(t_step < epsilon, tr_size < 2*epsilon)
spatial_termination_criterion.needs_grad = False


def level_set_termination_criterion(t_step: jnp.ndarray, fn_val: jnp.ndarray, tr_size: jnp.ndarray, level_set_threshold: float, epsilon: float) -> jnp.ndarray:
    """|f − level_set| < ε. Pure value-based; bound-tightness-independent.
    Stops when the function value at the current position is within ε of the
    level set, without consulting the gradient."""
    return jnp.abs(fn_val - level_set_threshold) < epsilon
level_set_termination_criterion.needs_grad = False


def geometric_termination_criterion(t_step: jnp.ndarray, fn_val: jnp.ndarray, tr_size: jnp.ndarray, level_set_threshold: float, epsilon: float, *, grad_norm: jnp.ndarray) -> jnp.ndarray:
    """|f − level_set| / |∇f| < ε. First-order Euclidean-distance-to-surface
    estimate (one Newton step). Equivalent to declaring hit when a single
    gradient-step would land on the surface within ε of the current point.
    Tighter than `level_set_termination_criterion` for shallow gradients
    (|∇f| < 1) and stricter for steep ones."""
    return jnp.abs(fn_val - level_set_threshold) / (grad_norm + 1e-12) < epsilon
geometric_termination_criterion.needs_grad = True