import jax
import jax.numpy as jnp
import jax.random as jr
from typing import Callable, Optional, Tuple, Any
import abc

from .sdf_primitives import safe_norm


class implicitSurface(abc.ABC):
    # Whether `classify_subcubes` is meaningfully tighter/cheaper than N independent
    # `classify_box` calls. Surfaces with a local model (autobound polynomial) override
    # this to True; the tree builder uses it to pick the right driver (parent-classifies-
    # children vs classify-then-split). Class-level default; subclasses may flip per-instance.
    use_classify_subcubes: bool = False

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
            fn_val (jnp.ndarray)(,): f(x) − level_set at the current position
            radius (jnp.ndarray)(,): radius of the largest provably-empty sphere
            next_trust_region_size (jnp.ndarray)(,): adaptive TR size for the *next* step.
                Convention: grow when the empty sphere reached the TR boundary, shrink
                otherwise; clipped to surface-specific [min_tr, max_tr]. Matches NIQ
                cast_rays_iter step-size update so WoSIntegrator can consume it directly.
        """
        raise NotImplementedError("This method should be implemented by subclasses.")
    
    def classify_box(self, params: Any, lower: jnp.ndarray, upper: jnp.ndarray) -> jnp.ndarray:
        """
        Three-valued sign of (f - level_set) over the axis-aligned box [lower, upper].
        Returns int8 in {-1, 0, +1}: -1 strictly below, +1 strictly above, 0 unknown.

        Default: reduce to empty_sphere over the circumscribed sphere of the box. Conservative,
        but correct. Subclasses with a tighter native AABB bound should override.
        """
        center = 0.5 * (lower + upper)
        half_diag = 0.5 * jnp.linalg.norm(upper - lower)
        # Convention: trust_region_size = 2 * half_diag — empty_sphere uses 0.5 * tr internally
        # in jnpbounded/harmonic/ni, so this matches the circumscribed-sphere radius.
        fn_val, radius, _ = self.empty_sphere(params, center, 2.0 * half_diag)
        sign = jnp.sign(fn_val - self.level_set).astype(jnp.int8)
        # Tolerance absorbs floating clamp at trust_region_size/2 in jnpbounded.
        classified = radius >= half_diag * (1.0 - 1e-6)
        return jnp.where(classified, sign, jnp.int8(0))


    def box_bounds(self, params: Any, lower: jnp.ndarray, upper: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Provably-valid (lo, hi) interval of f over the axis-aligned box [lower, upper].
        Strictly more informative than classify_box; not every surface family supports it.
        Default: NotImplementedError; surfaces with native interval bounds override.
        """
        raise NotImplementedError(f"{type(self).__name__} does not implement box_bounds.")


    def classify_subcubes(self,
                          params: Any,
                          parent_lower: jnp.ndarray,
                          parent_upper: jnp.ndarray,
                          sub_lowers: jnp.ndarray,
                          sub_uppers: jnp.ndarray) -> jnp.ndarray:
        """
        Classify N axis-aligned sub-boxes inside the parent box [parent_lower, parent_upper].

        Args:
            parent_lower, parent_upper:    (3,) parent AABB
            sub_lowers, sub_uppers:        (N, 3) sub-box AABBs (must lie inside parent)

        Returns:
            (N,) int8 in {-1, 0, +1} — same convention as classify_box.

        Default: independent classify_box on each sub-box (N forward passes).
        Surfaces backed by a local model (e.g. autobound polynomial) should override and
        compute one parent-level model, then evaluate it over each sub-box. That collapses
        N forward passes to one.
        """
        return jax.vmap(lambda lo, hi: self.classify_box(params, lo, hi))(sub_lowers, sub_uppers)


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
            fn_val (jnp.ndarray)(,): f(ray_origin) − level_set at the current position
            step_size (jnp.ndarray)(,): forward step distance along the ray
            in_tr_complement (jnp.ndarray)(,): bool. True iff the polynomial / classifier
                proved no level-set crossing inside the trust region (= safe, full step
                taken, integrator should *grow* tr next iteration). False when a crossing
                was found inside tr (= step stopped at root, integrator should *shrink* tr).
                Naming reflects the integrator's tr-update logic in
                `update_trust_region_size_*` in `utils/helpers.py`.
        """
        raise NotImplementedError("This method should be implemented by subclasses.")

    def ray_step_with_grad(self, params: Any, ray_origin: jnp.ndarray,
                           ray_direction: jnp.ndarray, trust_region_size: jnp.ndarray,
                           upper_lower_flag: Optional[str] = None
                           ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """
        Same as `ray_step` but also returns ∇f(ray_origin) — shape (dim,).
        Default: call `ray_step`, then `jax.grad(self.__call__)`. Subclasses with a
        cheaper grad path (e.g. autobound's polynomial) override this to avoid the
        extra autograd pass. SimpleIntegrator picks this method at construction
        only when the chosen termination criterion declares `needs_grad = True`.
        """
        fn_val, t_step, in_tr = self.ray_step(params, ray_origin, ray_direction,
                                               trust_region_size, upper_lower_flag)
        grad = self.spatial_derivative(params, ray_origin)
        return fn_val, t_step, in_tr, grad

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