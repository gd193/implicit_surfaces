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


    def empty_sphere(self, params, x, trust_region_size, upper_lower_flag = None):
        """
        Compute the largest empty sphere around the given position within the trust region.
        We assume a global Lipschitz bound, hence the trust_region_size is inf
        and is the surface is always inside the trust region, so the trust_region_empty_flag is always False.
        """
        trust_region_size = jnp.array(jnp.inf) # assume global Lipschitz bound
        fn_val = self(params, x)
        distance_to_level_set = jnp.abs(fn_val - self.level_set) / self.lipschitz_bound
        return fn_val, distance_to_level_set, jnp.array(False)
    

    def ray_step(self, params, ray_origin, ray_direction, trust_region_size, upper_lower_flag = None):
        """
        Compute the ray step from the current position along the ray direction.
        We assume a global Lipschitz bound, hence the trust_region_size is inf
        and is the surface is always inside the trust region, so the trust_region_empty_flag is always False.
        """
        trust_region_size = jnp.array(jnp.inf) # assume global Lipschitz bound
        fn_val = self(params, ray_origin)
        distance_to_level_set = jnp.abs(fn_val - self.level_set) / self.lipschitz_bound
        return fn_val, distance_to_level_set, jnp.array(False)


class sdfImplicit(lipschitzImplicit):
    """
    A jax-based implementation of a signed distance function (SDF). Inherits from lipschitzImplicit with
    lipschitz_bound set to 1 and level_set set to 0.
    """
    def __init__(self, **kwargs):
        super().__init__(lipschitz_bound=1.0, level_set=0.0, **kwargs)