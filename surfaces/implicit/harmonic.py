from typing import Tuple, Optional

import jax.numpy as jnp

from .base import ImplicitSurface


class harmonicImplicit(ImplicitSurface):
    r"""
    A jax-based implementation of a harmonic function f, i.e., a function that satisfies Laplace's equation \nabla^2 f = 0.
    Inherits from ImplicitSurface.
    """

    @staticmethod
    def update_ray_trust_region(trust_region, distance, backend_flag):
        # The Harnack radius is set by the field's singularities, not by a local trust region.
        return trust_region

    def __init__(self, alpha, **kwargs):
        """
        Initialize the implicit harmonic function defined in jax.

        Args:
            alpha (float): Fraction of the singularity-free radius used as the Harnack ball.
            **kwargs: forwarded to ImplicitSurface (fn, level_set, dim).
        """
        super().__init__(**kwargs)
        self.alpha = alpha

    def c_bound(self, params, x, R) -> jnp.ndarray:
        """
        Lower bound of the harmonic function on the ball B_R(x).

        Args:
            x (jnp.ndarray): Center of the ball.
            R (jnp.ndarray): The ball radius.
        Returns:
            c_bound (jnp.ndarray): A value c with f(y) >= c for all y in B_R(x).
        """
        raise NotImplementedError("C bound computation is implemented for specific harmonic functions.")

    def harnack_bound(self, params, x, f_val, R) -> jnp.ndarray:
        """
        Compute the Harnack bound for the harmonic function at point x.

        Args:
            x (jnp.ndarray): The point at which to compute the Harnack bound.
            f_val (jnp.ndarray): f(x) − level_set.
            R (jnp.ndarray): The trust-region radius hint.
        Returns:
            harnack_bound (jnp.ndarray): Radius of a ball around x free of the level set.
        """
        raise NotImplementedError("Harnack bound computation is implemented for specific harmonic functions.")


    def empty_sphere(self, params: jnp.ndarray, x: jnp.ndarray, trust_region_size: jnp.ndarray, upper_lower_flag: Optional[jnp.ndarray]=None) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Largest provably-empty sphere around `x` from the Harnack bound.

        Returns (fn_val − level_set, radius, next_tr). The Harnack radius
        does not depend on the trust region, so the next TR is the radius
        itself; shrinking the TR would stall multi-step walks.
        """
        fn_val = self(params, x) - self.level_set
        h_bound = self.harnack_bound(params, x, fn_val, trust_region_size * 0.5)
        return fn_val, h_bound, h_bound

    def ray_step(self, params: jnp.ndarray, ray_origin: jnp.ndarray,
                 ray_direction: jnp.ndarray, trust_region_size: jnp.ndarray,
                 upper_lower_flag: Optional[jnp.ndarray]=None) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Sphere-trace step. Uses Harnack ball radius as safe forward step.
        Returns (fn_val − level_set, t_step, in_tr_complement) where the third
        return is True iff the step stayed within an empty sphere (no level-set
        crossing inside it)."""
        fn_val, radius, _next_tr = self.empty_sphere(params, ray_origin, trust_region_size, upper_lower_flag)
        # The Harnack ball is provably empty by construction.
        return fn_val, radius, jnp.asarray(True)


class poissonSurface(harmonicImplicit):
    r"""
    A jax-based implementation of a Poisson surface, i.e. a surface represented as level set of a field defined as sum of monopoles.
    f(x) = c + \sum_i a * (1 / ||x - p_i||)
    where p_i are the monopole positions,
    a > 0 is the common monopole strength and
    c is a constant offset.

    The 1/r kernel is harmonic only in 3D, so `dim` must be 3. The empty
    sphere follows the Harnack tracing bound of Gillespie et al. 2024
    (§2 Eq. 7, Algorithm 1).
    """
    def __init__(self, scale: float, offset: float, level_set: float = 1.0, alpha: float = 1.0, dim: int = 3):
        """
        Initialize the Poisson surface.
        Args:
            scale (float): Monopole strength a > 0.
            offset (float): Constant offset c.
            level_set (float): The level set value for the implicit surface.
            alpha (float): Fraction in (0, 1] of the distance to the nearest
                monopole used as the Harnack ball radius.
            dim (int): Spatial dimension; must be 3.
        """
        if dim != 3:
            raise ValueError(f"poissonSurface requires dim=3 (1/r is harmonic only in 3D), got dim={dim}")
        if not scale > 0:
            raise ValueError(f"poissonSurface requires scale > 0, got scale={scale}")
        if not 0.0 < alpha <= 1.0:
            raise ValueError(f"poissonSurface requires 0 < alpha <= 1, got alpha={alpha}")
        super().__init__(fn=self.monopole_field, level_set=level_set, alpha=alpha, dim=dim)
        self.a = scale
        self.c = offset


    def monopole_field(self, params, x):
        p = params['p']
        d = jnp.linalg.norm(x - p, axis=-1)
        return jnp.sum(self.a * (1. / (d + 1e-9)), axis=-1) + self.c


    def c_bound(self, params, x, R):
        """Exact lower bound of f on B_R(x) for 0 < R <= min_i ||x - p_i||.

        Each monopole a / ||y - p_i|| is minimized on the ball at distance
        ||x - p_i|| + R, which is strictly tighter than the offset alone.
        """
        d = jnp.linalg.norm(x[None, :] - params['p'], axis=-1)
        return self.c + jnp.sum(self.a / (d + R + 1e-9), axis=-1)


    def harnack_bound(self, params, x, f_val, R) -> jnp.ndarray:
        """
        Harnack-safe radius around x (Gillespie et al. 2024, Algorithm 1 lines 9–13).

        Args:
            x (jnp.ndarray): The point at which to compute the Harnack bound.
            f_val (jnp.ndarray): f(x) − level_set.
            R (jnp.ndarray): Trust-region hint; unused because the Harnack ball
                is set by the singularity distance.

        Returns:
            harnack_bound (jnp.ndarray): Radius of a ball around x free of the level set.
        """
        R_sing = jnp.min(jnp.linalg.norm(x[None, :] - params['p'], axis=-1))
        R_ball = self.alpha * R_sing
        c = self.c_bound(params, x, R_ball)
        f_raw = f_val + self.level_set
        # The level set cannot enter the ball when it lies at or below the lower bound.
        unreachable = self.level_set <= c
        denom = jnp.where(self.level_set > c, self.level_set - c, 1.0)
        a = jnp.maximum((f_raw - c) / denom, 0.0)
        rho = 0.5 * R_ball * jnp.abs(a + 2.0 - jnp.sqrt(a * a + 8.0 * a))
        return jnp.where(unreachable, R_ball, rho)
