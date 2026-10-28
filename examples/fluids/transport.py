"""Trilinear sampling and semi-Lagrangian Euler, RK2/BFECC, RK3 or RK4 transport."""
from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax.scipy.ndimage import map_coordinates

from examples.fluids.grid import sample_vector_grid


def make_vortex_stretch(sample_velocity, clearance_fn, spacing, dt):
    """Rioux-Lavoie et al. (2022), Eqs.21/22: stretch at the departure point.

    The previous vorticity has already been sampled there by advection. Use
    cached velocities at the ends of a segment aligned with that vorticity;
    its length is at most one grid spacing and twice the boundary clearance.
    """
    @jax.jit
    def stretch(advected_vorticity, velocity, departures):
        omega = advected_vorticity.reshape((-1, 3))
        magnitude = jnp.linalg.norm(omega, axis=-1)
        direction = omega / jnp.where(magnitude > 0, magnitude, 1.)[:, None]
        clearance = jax.vmap(clearance_fn)(departures)
        length = jnp.minimum(jnp.min(jnp.asarray(spacing)), 2*jnp.maximum(clearance, 0.))
        offset = .5*length[:, None]*direction
        plus = jax.vmap(lambda x: sample_velocity(velocity, x))(departures+offset)
        minus = jax.vmap(lambda x: sample_velocity(velocity, x))(departures-offset)
        derivative = (magnitude/jnp.where(length > 0, length, 1.))[:, None]*(plus-minus)
        return (omega+dt*derivative).reshape(advected_vorticity.shape)
    return stretch


class TransportResult(NamedTuple):
    values: jax.Array
    departures: jax.Array


def make_transport(points, lower, spacing, dt, *, method="bfecc"):
    """Return sample_vec(grid,x), advect(scalar_or_vector_grid,velocity_grid).

    Samples outside the grid clamp to its edge; boundary masks/inflow values
    remain the caller's responsibility. points must be the flattened grid.
    """
    if method not in ("euler", "bfecc", "rk3", "rk4"):
        raise ValueError("transport method must be euler, bfecc, rk3 or rk4")
    def sample(field, x):
        idx = (x - jnp.asarray(lower, x.dtype)) / jnp.asarray(spacing, x.dtype)
        return map_coordinates(field, list(idx), order=1, mode="nearest")

    def sample_vec(field, x):
        return sample_vector_grid(field, x, lower, spacing)

    @partial(jax.jit, static_argnames=('return_details',))
    def advect(field, vel, time_step=dt, *, return_details=False):
        lookup = sample_vec if field.ndim == 4 else sample
        if method == "euler":
            # Rioux-Lavoie et al. (2022), Eq. (3): sample the previous cache
            # at x - dt*u(x), with no trajectory rejection or density retention.
            back = points-time_step*vel.reshape((-1, vel.shape[-1]))
            result = jax.vmap(lambda x: lookup(field, x))(back).reshape(field.shape)
            return TransportResult(result, back) if return_details else result
        def departure(x, step):
            return x - step * sample_vec(vel, x - .5 * step * sample_vec(vel, x))
        if method in ("rk3", "rk4"):
            def backtrace(x):
                k1 = sample_vec(vel, x)
                k2 = sample_vec(vel, x-.5*time_step*k1)
                match method:
                    case "rk3":
                        k3 = sample_vec(vel, x-.75*time_step*k2)
                        return x-time_step*(2*k1+3*k2+4*k3)/9
                    case "rk4":
                        k3 = sample_vec(vel, x-.5*time_step*k2)
                        k4 = sample_vec(vel, x-time_step*k3)
                        return x-time_step*(k1+2*k2+2*k3+k4)/6
            back = jax.vmap(backtrace)(points)
            result = jax.vmap(lambda x: lookup(field, x))(back).reshape(field.shape)
            return TransportResult(result, back) if return_details else result
        back = jax.vmap(lambda x: departure(x, time_step))(points)
        forward = jax.vmap(lambda x: departure(x, -time_step))(points)
        def pull(a, p):
            return jax.vmap(lambda x: lookup(a, x))(p).reshape(field.shape)
        plain = pull(field, back)
        corrected = field + .5 * (field - pull(plain, forward))
        shape = (3, 3, 3, 1) if field.ndim == 4 else (3, 3, 3)
        strides = (1,) * field.ndim
        low = jax.lax.reduce_window(field, jnp.inf, jax.lax.min, shape, strides, "SAME")
        high = jax.lax.reduce_window(field, -jnp.inf, jax.lax.max, shape, strides, "SAME")
        result = jnp.clip(pull(corrected, back), pull(low, back), pull(high, back))
        return TransportResult(result, back) if return_details else result

    return sample_vec, advect
