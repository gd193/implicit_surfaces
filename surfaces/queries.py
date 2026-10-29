"""Geometry query adapters shared by algorithms, independent of their state.

Backend ray_step callbacks return (residual, distance, backend_flag). Only
this adapter interprets the flag; algorithms consume the next trust region.
Distances are world-space and ray directions must be unit length. Residuals
are already f - level_set, on either side of the surface. A zero distance
can mean an undecided bound, not a hit.
"""
from typing import Any, Callable, NamedTuple

import jax
import jax.numpy as jnp

from utils.helpers import update_trust_region_size


class RayStep(NamedTuple):
    residual: jax.Array
    distance: jax.Array
    next_trust_region: jax.Array
    gradient: jax.Array | None = None


class EmptySphere(NamedTuple):
    residual: jax.Array
    radius: jax.Array
    next_trust_region: jax.Array


class StarRadius(NamedTuple):
    radius: jax.Array
    status: jax.Array  # utils.types.Termination: EXIT means a valid radius


class SurfaceHit(NamedTuple):
    distance: jax.Array
    position: jax.Array
    normal: jax.Array  # meaningful only for HIT
    status: jax.Array  # HIT, EXIT (clear segment), or explicit failure


class RayQuery:
    """Adapt a backend's query and chosen TR policy once, outside JIT.

    `certified=False` identifies approximate fixed-step reference marching.
    This metadata describes the backend's contract, not a numerical proof.
    """
    def __init__(self, step: Callable, *, update_tr: Callable | None = None,
                 step_with_grad: Callable | None = None, needs_grad: bool = False,
                 certified: bool = True):
        if needs_grad and step_with_grad is None:
            raise ValueError("gradient termination requires step_with_grad")
        self.step = step_with_grad if needs_grad else step
        owner = getattr(step, '__self__', None)
        self.update_tr = update_tr if update_tr is not None else getattr(
            owner, 'update_ray_trust_region', update_trust_region_size)
        self.needs_grad = needs_grad
        self.certified = certified

    @classmethod
    def from_surface(cls, surface, **kwargs):
        return cls(surface.ray_step, step_with_grad=getattr(surface, 'ray_step_with_grad', None),
                   certified=getattr(surface, 'certified_ray_steps', True), **kwargs)

    def __call__(self, params: Any, position, direction, trust_region) -> RayStep:
        values = self.step(params, position, direction, trust_region)
        residual, distance, flag = (jnp.asarray(v).squeeze() for v in values[:3])
        gradient = values[3] if self.needs_grad else None
        return RayStep(residual, distance, self.update_tr(trust_region, distance, flag), gradient)


class EmptySphereQuery:
    """Normalize backend empty-sphere results without changing walker behavior."""
    def __init__(self, query: Callable):
        self.query = query

    def __call__(self, params, position, trust_region) -> EmptySphere:
        return EmptySphere(*self.query(params, position, trust_region))
