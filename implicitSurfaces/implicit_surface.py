import jax
import jax.numpy as jnp
import jax.random as jr
from typing import Callable, Tuple

from .sdf_primitives import safe_norm

#Classes: 1 fn > val safe, -1 fn < val safe, 0 unknown
BOX_CLASSIFICATION_CLASSES = (jnp.array(1), jnp.array(-1), jnp.array(0))
BC_CLASS = BOX_CLASSIFICATION_CLASSES


class implicitSurface(object):
    def __init__(self, fn: Callable=None, params: jnp.ndarray=None, level_set: float=None, dim: int=None):

        if fn is None or level_set is None or dim is None:
            raise ValueError("fn and level_set and dim must be provided.")
        self.fn = fn
        self.level_set = level_set
        self.dim = dim


    def __call__(self, params, x) -> jnp.ndarray:
        """
        evaluate the implicit surface function at point x.
        """
        return self.fn(params, x).squeeze()
    

    def classify_box(self, box_aa, box_bb, fn_threshold):
        """
        Determine whether the function contains the fn_threshold value within the box defined by box_aa and box_bb.
        """
        raise NotImplementedError("This method should be implemented by subclasses.")
    
    def empty_sphere(self, params: jnp.ndarray, x: jnp.ndarray, trust_region_size: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """
        Compute the largest empty sphere around the given position within the trust region.
        args:
            x (jnp.ndarray)(dim,): The current position of the walker
            trust_region_size (jnp.ndarray)(,): The current trust region size
        Returns:
            fn_val (jnp.ndarray)(,): The value of the function at the current position
            radius (jnp.ndarray)(,): The radius of the largest empty sphere
            new_trust_region_size (jnp.ndarray)(,): The updated trust region size
        """
        raise NotImplementedError("This method should be implemented by subclasses.")
    

    def parameter_derivative(self, params, x):
        """
        Compute the derivative of the implicit surface function with respect to its parameters at point x.
        """
        return jax.grad(self.__call__, argnums=0)(params, x)


    def spatial_derivative(self, params, x):
        """
        Compute the spatial derivative of the implicit surface function at point x.
        """
        return jax.grad(self.__call__, argnums=1)(params, x)
    

    def normal_at(self, params, x):
        """
        Compute the normal vector at point x on the implicit surface.
        """
        grad = self.spatial_derivative(params, x)
        norm = safe_norm(grad, ord=2)
        return grad / norm
    

    def normal_surface_velocity(self, params, x):
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
    

    def sample_points(self, params: jnp.ndarray, key: jnp.ndarray, aa: jnp.ndarray, bb: jnp.ndarray, num_points: int, in_out_flag: str, max_iters: int=1000) -> jnp.ndarray:
        """
        Sample points uniformly inside/outside the implicit surface defined by the function.
        
        Args:
            params (jnp.ndarray): The parameters of the implicit surface function.
            key (jnp.ndarray): The random key for sampling.
            aa (jnp.ndarray): The lower corner of the bounding box for sampling.
            bb (jnp.ndarray): The upper corner of the bounding box for sampling.
            num_points (int): The number of points to sample.
            max_iters (int): The maximum number of iterations to attempt sampling.
            in_out_flag (str): 'inside' to sample points inside the surface, 'outside' to sample points outside the surface.
        
        Returns:
            points_inside (jnp.ndarray): An array of shape (num_points, dim) containing points inside/outside the implicit surface.
        """
        dim = aa.shape[0]
        # State
        # points: The buffer we want to fill
        # mask: Boolean array tracking which slots are successfully filled
        # key: Current RNG key
        # iter_num: Counter
        init_points = jnp.zeros((num_points, dim))
        init_mask = jnp.zeros((num_points,), dtype=bool)
        init_state = (key, init_points, init_mask, 0)

        def cond_fun(state):
            _, _, mask, i = state
            return jnp.logical_and(
                jnp.logical_not(jnp.all(mask)), 
                i < max_iters
            )

        def body_fun(state):
            k, current_points, current_mask, i = state
            k_sub, k_next = jax.random.split(k)
            candidates = jax.random.uniform(k_sub, shape=(num_points, dim), 
                                            minval=aa, maxval=bb)
            
            dist = jax.vmap(self.__call__, in_axes=(None, 0))(params, candidates)
            if in_out_flag == 'inside':
                correct_side = dist < self.level_set
            else:
                correct_side = dist > self.level_set
            should_update = correct_side & (~current_mask)
            
            # Apply updates
            new_points = jnp.where(should_update[:, None], candidates, current_points)
            new_mask = jnp.logical_or(current_mask, should_update)
            
            return (k_next, new_points, new_mask, i + 1)

        # Run the loop
        final_key, final_points, final_mask, final_iters = jax.lax.while_loop(cond_fun, body_fun, init_state)
        
        return final_points, final_mask
    

    def sample_points_inside(self, params: jnp.ndarray, key: jnp.ndarray, aa: jnp.ndarray, bb: jnp.ndarray, num_points: int, max_iters: int=1000) -> jnp.ndarray:
        """
        Sample points uniformly inside the implicit surface defined by the function.
        """
        return self.sample_points(params, key, aa, bb, num_points, in_out_flag='inside', max_iters=max_iters)
    

    def sample_points_outside(self, params: jnp.ndarray, key: jnp.ndarray, aa: jnp.ndarray, bb: jnp.ndarray, num_points: int, max_iters: int=1000) -> jnp.ndarray:
        """
        Sample points uniformly outside the implicit surface defined by the function.
        """
        return self.sample_points(params, key, aa, bb, num_points, in_out_flag='outside', max_iters=max_iters)