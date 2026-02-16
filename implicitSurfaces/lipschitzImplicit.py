from typing import Tuple
from implicitSurfaces.implicit_surface import *


class lipschitzImplicit(implicitSurface):
    def __init__(self, lipschitz_bound: float, **kwargs):
        """
        Initialize the implicit Lipschitz function defined in jax.

        Args:
            fn (Callable): The signed distance function. Assumed to be a JAX function that takes parameters and a point.
            params (Optional[dict]): Additional parameters for the SDF function.
        """
        super().__init__(**kwargs)
        if lipschitz_bound is None:
            raise ValueError("lipschitz_bound must be provided.")
        self.lipschitz_bound = lipschitz_bound


    def classify_box(self, params, box_aa, box_bb):
        """
        Determine whether the function contains the level_set value within the box defined by box_aa and box_bb.
        args:
            box_aa (jnp.ndarray)(d): The lower corner of the axis-aligned bounding box
            box_bb (jnp.ndarray)(d): The upper corner of the axis-aligned bounding box
        Returns:
            fn_val (jnp.ndarray)(1): The value of the function at the center of the box
            box_class (jnp.ndarray)(1): The classification of the box based on the function value and

        """
        box_vecs = (box_bb - box_aa) * jnp.diag(jnp.ones(box_aa.shape[-1]))
        box_center = (box_aa + box_bb) / 2
        bounding_radius = jnp.linalg.norm(jnp.sum(box_vecs/2, axis=0), 2)

        fn_val = self(params, box_center)
        box_sign_is_stable = jnp.abs(fn_val - self.level_set) - bounding_radius * self.lipschitz_bound > 0.
        box_class = jax.lax.select(box_sign_is_stable, 
                                   jax.lax.select(fn_val > self.level_set, BC_CLASS[0], BC_CLASS[1]), BC_CLASS[2])
        return fn_val, box_class


    def empty_sphere(self, params, x: jnp.ndarray, trust_region_size: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """
        Compute the largest empty sphere around the given position within the trust region.
        We assume a global Lipschitz bound, hence the trust_region_size is inf
        args:
            x (jnp.ndarray)(dim,): The current position of the walker
            trust_region_size (jnp.ndarray)(1,): The current trust region size
        Returns:
            fn_val (jnp.ndarray)(1,): The value of the function at the current position
            radius (jnp.ndarray)(1,): The radius of the largest empty sphere
            new_trust_region_size (jnp.ndarray)(1,): The updated trust region size
        """
        trust_region_size = jnp.inf # assume global Lipschitz bound
        fn_val = self(params, x)
        distance_to_level_set = jnp.abs(fn_val - self.level_set) / self.lipschitz_bound
        return fn_val, distance_to_level_set, trust_region_size


class sdfImplicit(lipschitzImplicit):
    """
    A jax-based implementation of a signed distance function (SDF). Inherits from lipschitzImplicit with
    lipschitz_bound set to 1 and level_set set to 0.
    """
    def __init__(self, **kwargs):
        super().__init__(lipschitz_bound=1.0, level_set=0.0, **kwargs)