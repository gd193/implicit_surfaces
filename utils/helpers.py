import jax.numpy as jnp

# def tea(v0: UInt, v1: UInt, rounds=4):
#         """
#         Tiny Encryption Algorithm (TEA)
#         This function implements the TEA algorithm, which is a simple and fast block cipher.
#         """
#         sum = UInt(0)
#         for i in range(rounds):
#             sum += 0x9e3779b9
#             v0 += ((v1 << 4) + 0xa341316c) ^ (v1 + sum) ^ ((v1>>5) + 0xc8013ea4)
#             v1 += ((v0 << 4) + 0xad90777d) ^ (v0 + sum) ^ ((v0>>5) + 0x7e95761e)
#         return v0, v1


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


def mesh_3d(res, bbox: tuple[tuple[float, float], tuple[float, float], tuple[float, float]]):
    """
    Create a 3D mesh grid.
    Args:
        res (int): The resolution of the mesh grid.
        bbox (tuple[float, float, float, float, float, float]): The bounding box for the mesh grid 
        in the format (x_min, x_max, y_min, y_max, z_min, z_max).
    """
    x, y, z = bbox
    x_min, x_max = x
    y_min, y_max = y
    z_min, z_max = z
    x = jnp.linspace(x_min, x_max, res)
    y = jnp.linspace(y_min, y_max, res)
    z = jnp.linspace(z_min, z_max, res)
    mesh_x, mesh_y, mesh_z = jnp.meshgrid(x, y, z)
    return mesh_x, mesh_y, mesh_z


def mesh_to_vec_3d(mesh_x, mesh_y, mesh_z):
    """
    Convert 3D mesh grid coordinates to a 3D vector.
    """
    return jnp.stack([mesh_x.flatten(), mesh_y.flatten(), mesh_z.flatten()], axis=-1)


def split_vector(vec: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    Split vectors along the axis with the maximum norm.
    Args:
        vec (jnp.ndarray): Input vector of shape (d,) where d is the dimension.
    Returns:
        tuple[jnp.ndarray, jnp.ndarray]: A tuple containing the split vectors and the axis index used for splitting.
    """
    split_axis_id = jnp.argmax(jnp.abs(vec))
    sub_vec_new = vec.at[split_axis_id].multiply(0.5)
    return sub_vec_new, split_axis_id

def new_subdivide_box(aa: jnp.ndarray, sub_vec: jnp.ndarray, split_axis_id: int) -> jnp.ndarray:
    """
    Create new sub-boxes by subdividing the original box.
    Args:
        box_center (jnp.ndarray): The center of the original box.
        sub_vec (jnp.ndarray): The vectors used for subdivision.
        split_axis_id (int): The axis index used for splitting.
    Returns:
        jnp.ndarray: The centers of the new sub-boxes.
    """
    add_vec = jnp.zeros_like(aa)
    add_vec = add_vec.at[split_axis_id].set(sub_vec[split_axis_id])
    return aa + jnp.stack([jnp.zeros_like(aa), add_vec], axis=0)


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
                                      max_tr: float = 2.0, min_tr: float = 1e-6) -> jnp.ndarray:
    # NIQ-style adaptive update (cast_rays_iter, queries.py): grow ×1.5 on every safe step,
    # shrink ×0.5 on every unsafe step. Suited to NIQ's classify-and-step ray_step
    # (classified → full safe step → grow; UNKNOWN → no step → shrink).
    new_tr = jnp.where(in_tr, trust_region_size * 1.5, trust_region_size * 0.5)
    return jnp.clip(new_tr, min_tr, max_tr)


def update_trust_region_size_autobound(trust_region_size: jnp.ndarray, t_step: jnp.ndarray,
                                       in_tr: jnp.ndarray,
                                       max_tr: float = 0.2, min_tr: float = 1e-6) -> jnp.ndarray:
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


def level_set_termination_criterion(t_step: jnp.ndarray, fn_val: jnp.ndarray, tr_size: jnp.ndarray, level_set_threshold: float, epsilon: float) -> jnp.ndarray:
    return jnp.abs(fn_val - level_set_threshold) < epsilon