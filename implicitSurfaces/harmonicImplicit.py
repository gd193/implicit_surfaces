from typing import Tuple, Optional

import jax
import jax.numpy as jnp

from implicitSurfaces.implicit_surface import implicitSurface
from utils.helpers import mesh_2d, mesh_to_vec_2d


class harmonicImplicit(implicitSurface):
    """
    A jax-based implementation of a harmonic function f, i.e., a function that satisfies Laplace's equation \nabla^2 f = 0.
    Inherits from implicitSurface.
    """
    
    def __init__(self, alpha, **kwargs):
        """
        Initialize the implicit harmonic function defined in jax.

        Args:
            fn (Callable): The harmonic function. Assumed to be a JAX function that takes parameters and a point.
            params (Optional[dict]): Additional parameters for the harmonic function.
        """
        super().__init__(**kwargs)
        self.alpha = alpha

    def c_bound(self, params, x, R) -> jnp.ndarray:
        """
        Compute the C bound for the harmonic function at point x.
        
        Args:
            x (jnp.ndarray): The point at which to compute the C bound.
            R (jnp.ndarray): The bounding radius.
        Returns:
            c_bound (jnp.ndarray): The C bound value.
        """
        raise NotImplementedError("C bound computation is implemented for specific harmonic functions.")

    def harnack_bound(self, params, x, f_val, R) -> jnp.ndarray:
        """
        Compute the Harnack bound for the harmonic function at point x.
        
        Args:
            x (jnp.ndarray): The point at which to compute the Harnack bound.
            f_val (jnp.ndarray): The value of the harmonic function at point x.
            R (jnp.ndarray): The bounding radius.
        Returns:
            harnack_bound (jnp.ndarray): The Harnack bound value.
        """
        raise NotImplementedError("Harnack bound computation is implemented for specific harmonic functions.")
    
    
    def empty_sphere(self, params: jnp.ndarray, x: jnp.ndarray, trust_region_size: jnp.ndarray, upper_lower_flag: Optional[str]=None) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Largest provably-empty sphere around `x` within the trust region.

        Returns (fn_val − level_set, radius, next_tr) — matches NIQ /
        jnpboundedImplicit. Adaptive TR: grow when the Harnack ball reaches
        the TR boundary, shrink otherwise.
        """
        fn_val = self(params, x) - self.level_set
        # TODO: revisit. Currently the alpha * min-distance-to-monopole bound is computed
        # and then immediately discarded in favor of the trust-region-based R. Intent unclear:
        # should these be combined (jnp.minimum), or is one of them the right choice?
        r = jnp.linalg.norm(x[None, :] - params['p'], axis=-1)
        R = self.alpha * jnp.min(r, axis=-1)
        R = trust_region_size * 0.5
        h_bound = self.harnack_bound(params, x, fn_val, R)
        # NIQ-style adaptive TR: grow if the Harnack ball reached the TR
        # boundary, otherwise shrink. Use ×1.5 / ×0.5 to mirror NIQ.
        outside_tr = h_bound >= trust_region_size * 0.5
        next_tr = jnp.where(outside_tr, trust_region_size * 1.5, trust_region_size * 0.5)
        return fn_val, h_bound, next_tr

    def ray_step(self, params: jnp.ndarray, ray_origin: jnp.ndarray,
                 ray_direction: jnp.ndarray, trust_region_size: jnp.ndarray,
                 upper_lower_flag: Optional[str]=None) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Sphere-trace step. Uses Harnack ball radius as safe forward step.
        Returns (fn_val − level_set, t_step, in_tr_complement) where the third
        return is True iff the step stayed within an empty sphere (no level-set
        crossing inside it)."""
        fn_val, radius, _next_tr = self.empty_sphere(params, ray_origin, trust_region_size, upper_lower_flag)
        # The Harnack ball is provably empty by construction.
        return fn_val, radius, jnp.asarray(True)


class poissonSurface(harmonicImplicit):
    """
    A jax-based implementation of a Poisson surface, i.e. a surface represented as level set of a field defined as sum of monopoles.
    f(x) = c + \sum_i a_i * (1 / ||x - p_i||)
    where p_i are the monopole positions and
    a_i are the monopole strengths/associated areas.
    c is a constant offset 
    """
    def __init__(self, scale: float, offset: float, level_set:float=1.0, alpha:float=1.25, dim:int=2):
        """
        Initialize the Poisson surface.
        Args:
            level_set (float): The level set value for the implicit surface.
            alpha (float): The Harnack alpha.
            
        """
        super().__init__(fn=self.monopole_field, level_set=level_set, alpha=alpha, dim=dim)
        self.a = scale
        self.c = offset
    

    @staticmethod
    def init_monopole_field(key, bbox, num_points):
        """
        key (jax.random.PRNGKey): Random key for JAX operations.
        bbox_monopoles (tuple[tuple[float, float], tuple[float, float]]): The bounding box for the monopoles in the format ((x_min, x_max), (y_min, y_max)).
        num_points (int): The number of monopoles to generate.
        """
        key, subkey = jax.random.split(key)

        x_mesh, y_mesh = mesh_2d(int(jnp.floor(jnp.sqrt(num_points))), bbox)
        p = mesh_to_vec_2d(x_mesh, y_mesh) 
        p += jax.random.normal(subkey, shape=p.shape) * jnp.linalg.norm(p[0, :] - p[-1, :])/(10*num_points)
        return {'p': p}


    def monopole_field(self, params, x):
        p = params['p']
        a = self.a
        c = self.c
        d = jnp.linalg.norm(x - p, axis=-1)
        return jnp.sum(a * (1. / (d + 1e-9)), axis=-1) + c


    def c_bound(self, params, x, R):
        p = params['p']
        a = self.a
        c = self.c
        d = jnp.linalg.norm(x - p, axis=-1)
        t = (1. - R) * R
        #c_bound = jnp.sum(a * (1. - R) / (d + 1e-9 + jnp.sign(t)*t), axis=-1) + c
        c_bound_axis = (a) / (jnp.linalg.norm(x - p, axis=-1) + 1e-9 + jnp.sign(a-R))
        c_bound = jnp.sum(c_bound_axis, axis=-1) + c
        return c_bound


    def harnack_bound(self, params, x, f_val, R) -> jnp.ndarray:
        """
        Compute the Harnack bound for the Poisson surface at point x.
        
        Args:
            x (jnp.ndarray): The point at which to compute the Harnack bound.
            f_val (jnp.ndarray): The value of the Poisson function at point x.
            R (jnp.ndarray): The bounding radius.

        Returns:
            harnack_bound (jnp.ndarray): The Harnack bound value.
        """
        R = jnp.linalg.norm(x[None, :] - params['p'], axis=-1)
        R = jnp.min(R) * self.alpha
        #c_bound = self.c_bound(params, x, R)
        harnack_a = f_val / (self.level_set + 1e-9)
        #harnack_a = jnp.abs(f_val / (self.level_set))
        harnack_bound = R / 2. * jnp.abs(harnack_a + 2. - jnp.sqrt(jnp.power(harnack_a, 2.) + 8. * harnack_a))

        return harnack_bound
