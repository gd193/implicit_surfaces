import jax.numpy as jnp


def mesh_2d(res, bbox: tuple[tuple[float, float], tuple[float, float]]):
    """
    Create a 2D mesh grid.
    Args:
        res (int): The resolution of the mesh grid.
        bbox (tuple[tuple[float, float], tuple[float, float]]): The bounding box for the mesh grid
        in the format ((x_min, x_max), (y_min, y_max)).
    """
    x, y = bbox
    x_min, x_max = x
    y_min, y_max = y
    x = jnp.linspace(x_min, x_max, res)
    y = jnp.linspace(y_min, y_max, res)
    mesh_x, mesh_y = jnp.meshgrid(x, y)
    return mesh_x, mesh_y


def mesh_to_vec_2d(mesh_x, mesh_y):
    """
    Convert 2D mesh grid coordinates to a 2D vector.
    """
    return jnp.stack([mesh_x.flatten(), mesh_y.flatten()], axis=-1)


def update_trust_region_size(trust_region_size: jnp.ndarray, t_step: jnp.ndarray, in_tr: jnp.ndarray) -> jnp.ndarray:
    # Conservative default: shrink on unsafe step; grow only when the step was unusually
    # small relative to tr (rare; mostly serves as a safety hedge). NIQ's `cast_rays_iter`
    # uses a more aggressive grow-on-safe rule which suits its classify-and-step semantics
    # but destabilises autobound's root-find (slop scales as tr², so growing tr loosens
    # the polynomial enclosure → spurious roots → oscillation). For NIQ-style behavior
    # use `update_trust_region_size_niq_grow` below.
    new = jnp.where(~in_tr, trust_region_size * 0.5, trust_region_size)
    new = jnp.where((t_step < 0.5 * trust_region_size) & in_tr, trust_region_size * 2.0, new)
    return new


def update_trust_region_size_niq_grow(trust_region_size: jnp.ndarray, t_step: jnp.ndarray,
                                      in_tr: jnp.ndarray,
                                      max_tr: float = float('inf'),
                                      min_tr: float = 1e-3) -> jnp.ndarray:
    # NIQ-style adaptive update (cast_rays_iter, queries.py): grow ×1.5 on every safe step,
    # shrink ×0.5 on every unsafe step. Suited to NIQ's classify-and-step ray_step
    # (classified → full safe step → grow; UNKNOWN → no step → shrink).
    # Defaults mirror NIQ: no upper cap, lower clip at hit_eps (=1e-3).
    new_tr = jnp.where(in_tr, trust_region_size * 1.5, trust_region_size * 0.5)
    return jnp.clip(new_tr, min_tr, max_tr)


def update_trust_region_size_autobound(trust_region_size: jnp.ndarray, t_step: jnp.ndarray,
                                       in_tr: jnp.ndarray,
                                       max_tr: float = 0.15, min_tr: float = 1e-6) -> jnp.ndarray:
    # Autobound-suited adaptive update:
    #   * Ray passed through trust region (no root → in_tr=True): grow ×2.
    #   * Polynomial root in TR with small step (< 0.5·tr): shrink ×0.5.
    #   * Polynomial root in TR with big step (≥ 0.5·tr): keep.
    # Decouples "found a root" from "shrink": only the *magnitude* of the step decides
    # shrink, avoiding the grow/shrink oscillation that pure NIQ-style update produces
    # under autobound's tr²-scaling polynomial slop.
    pass_through = in_tr                                                      # in_box = False
    small_step = (~in_tr) & (t_step < 0.5 * trust_region_size)                # in_box = True, partial step
    new_tr = jnp.where(pass_through, trust_region_size * 1.5,
              jnp.where(small_step,    trust_region_size * 0.5,
                                       trust_region_size))
    return jnp.clip(new_tr, min_tr, max_tr)



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