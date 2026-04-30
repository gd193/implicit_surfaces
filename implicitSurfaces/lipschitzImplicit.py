import jax.numpy as jnp

from implicitSurfaces.implicit_surface import implicitSurface


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
        """Lipschitz-derived empty sphere: r = |f(x) - level_set| / L is the
        smallest distance to the level set. Returns (fn_val_minus_level_set,
        radius, next_tr). With a global Lipschitz constant the TR is irrelevant
        for safety, so we pass the radius itself as next_tr — WoSIntegrator
        will then sample within that ball next iteration."""
        fn_val = self(params, x) - self.level_set
        radius = jnp.abs(fn_val) / self.lipschitz_bound
        return fn_val, radius, radius
    

    def box_bounds(self, params, lower, upper):
        """
        For Lipschitz f, |f(x) - f(c)| <= L * ||x - c|| <= L * half_diag for x in the box,
        so [f(c) - L*half_diag, f(c) + L*half_diag] is a provably-valid enclosure.
        """
        center = 0.5 * (lower + upper)
        half_diag = 0.5 * jnp.linalg.norm(upper - lower)
        fn_c = self(params, center)
        slack = self.lipschitz_bound * half_diag
        return fn_c - slack, fn_c + slack


    def ray_step(self, params, ray_origin, ray_direction, trust_region_size, upper_lower_flag = None):
        """Lipschitz sphere-trace step: t_step = |f - level_set| / L.
        Third return is `in_tr_complement` (True ⇔ no level-set crossing inside
        the step we just bounded). For Lipschitz, the step is the *largest
        guaranteed-empty distance*, so we step into a region that's provably
        away from the surface — return True (no crossing in this step)."""
        fn_val = self(params, ray_origin) - self.level_set
        t_step = jnp.abs(fn_val) / self.lipschitz_bound
        return fn_val, t_step, jnp.asarray(True)


class sdfImplicit(lipschitzImplicit):
    """
    A jax-based implementation of a signed distance function (SDF). Inherits from lipschitzImplicit with
    lipschitz_bound set to 1 and level_set set to 0.
    """
    def __init__(self, **kwargs):
        super().__init__(lipschitz_bound=1.0, level_set=0.0, **kwargs)