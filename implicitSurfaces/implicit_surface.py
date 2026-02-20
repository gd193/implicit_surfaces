import jax
import jax.numpy as jnp
import jax.random as jr
from typing import Callable, Optional, Tuple, Any
import abc

from .sdf_primitives import safe_norm


class implicitSurface(abc.ABC):
    def __init__(self, fn: Callable[[Any, jnp.ndarray], jnp.ndarray], level_set: float, dim: int):
        self.fn = fn
        self.level_set = level_set
        self.dim = dim


    def __call__(self, params: Any, x: jnp.ndarray) -> jnp.ndarray:
        """
        evaluate the implicit surface function at point x.
        """
        return self.fn(params, x).squeeze()
    
    
    @abc.abstractmethod
    def empty_sphere(self, params: jnp.ndarray, x: jnp.ndarray, trust_region_size: jnp.ndarray, 
                     upper_lower_flag: Optional[str]=None) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """
        Compute the largest empty sphere around the given position within the trust region.
        args:
            params (jnp.ndarray): The parameters of the implicit surface function.
            x (jnp.ndarray)(dim,): The current position of the walker
            trust_region_size (jnp.ndarray)(,): The current trust region size
            upper_lower_flag (Optional[str]): If 'upper', use upper bounds for computing the empty sphere. 
            If 'lower', use lower bounds. If None, use both and take the minimum radius to not switch level set sides.
        Returns:
            fn_val (jnp.ndarray)(,): The value of the function at the current position
            radius (jnp.ndarray)(,): The radius of the largest empty sphere
            trust_region_empty_flag (jnp.ndarray)(,): A boolean flag indicating whether the entire trust region is empty (True) or not (False)
        """
        raise NotImplementedError("This method should be implemented by subclasses.")
    
    @abc.abstractmethod
    def ray_step(self, params: jnp.ndarray, ray_origin: jnp.ndarray, 
                 ray_direction: jnp.ndarray, trust_region_size: jnp.ndarray, 
                 upper_lower_flag: Optional[str]=None) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """
        Compute the ray step from the current position along the ray direction.
        args:
            params (jnp.ndarray): The parameters of the implicit surface function.
            ray_origin (jnp.ndarray)(dim,): The current position of the walker
            ray_direction (jnp.ndarray)(dim,): The current ray direction
            trust_region_size (jnp.ndarray)(,): The current trust region size
            upper_lower_flag (Optional[str]): If 'upper', use upper bounds for computing the ray step. 
            If 'lower', use lower bounds. If None, use both and take the minimum step size to not switch level set sides.
        Returns:
            fn_val (jnp.ndarray)(,): The value of the function at the current position
            step_size (jnp.ndarray)(,): The computed step size along the ray direction
            trust_region_empty_flag (jnp.ndarray)(,): A boolean flag indicating whether the entire trust region is empty (True) or not (False)
        """
        raise NotImplementedError("This method should be implemented by subclasses.")

    def parameter_derivative(self, params: Any, x: jnp.ndarray) -> jnp.ndarray:
        """
        Compute the derivative of the implicit surface function with respect to its parameters at point x.
        """
        return jax.grad(self.__call__, argnums=0)(params, x)


    def spatial_derivative(self, params: Any, x: jnp.ndarray) -> jnp.ndarray:
        """
        Compute the spatial derivative of the implicit surface function at point x.
        """
        return jax.grad(self.__call__, argnums=1)(params, x)
    

    def normal_at(self, params: Any, x: jnp.ndarray) -> jnp.ndarray:
        """
        Compute the normal vector at point x on the implicit surface.
        """
        grad = self.spatial_derivative(params, x)
        norm = safe_norm(grad, ord=2)
        return grad / norm
    

    def normal_surface_velocity(self, params: Any, x: jnp.ndarray) -> jnp.ndarray:
        """
        Compute the normal surface velocity at point x on the implicit surface gamma(t, x).
        V_n(x) = - (del_gamma/del_t) / || del_gamma/del_x ||
        """
        spatial_grad = self.spatial_derivative(params, x)
        parameter_grad = self.parameter_derivative(params, x)
        return jax.tree.map(lambda del_gamma_del_t: (-1.0) * del_gamma_del_t / safe_norm(spatial_grad, ord=2), parameter_grad)


    def dirichlet_boundary_differential(self, params: jnp.ndarray,  x: jnp.ndarray, g: Callable[[jnp.ndarray], jnp.ndarray]) -> jnp.ndarray:
        """
        Compute the differential of the Dirichlet boundary condition g at point x.
        
        Args:
            params (jnp.ndarray): The parameters of the implicit surface function.
            x (jnp.ndarray): The point at which to compute the differential.
            g (Callable): The Dirichlet boundary condition function.
        Returns:
            differential (jnp.ndarray): The differential of g at point x.
        """
        V_n = self.normal_surface_velocity(params, x)
        normal_vec = self.normal_at(params, x)
        del_g_del_n = jax.jvp(g, (x,), (normal_vec,))[1] # del g / del n normal derivative of g
        return jax.tree.map(lambda V_n_i: del_g_del_n * V_n_i, V_n)


    ### Helper function, prevents code duplication/exessive lambda usage in sampling functions
    def is_above_level_set(self, fn_val: jnp.ndarray) -> jnp.ndarray:
        return fn_val.squeeze() > self.level_set
    

    def is_below_level_set(self, fn_val: jnp.ndarray) -> jnp.ndarray:
        return fn_val.squeeze() < self.level_set