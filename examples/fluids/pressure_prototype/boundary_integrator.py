"""2024 velocity-only pressure estimator on the existing MC integrator driver.

One walker follows one boundary path. Completed samples are shared across
queries; there is no branching or separate traversal driver. The truncated Neumann series
uses the corrected signs and terminal half weights in the authors' 2026 errata:
https://rsugimoto.net/VelMCFluidsProject/ . Finite depth introduces bias.
"""
from typing import NamedTuple

import jax
import jax.numpy as jnp

from sphereWalker.integrator import BaseIntegrator, finish_step, get_scatter_fn
from sphereWalker.types import WalkerTermination as WT
from examples.fluids.pressure_prototype.boundary_sampling import (
    BoxBoundary, SphereBoundary, ExteriorSphereBoundary, ExteriorTorusBoundary,
)


def volume_sample_pairs(geometry, x, key, samples):
    """Original radial draws, shared by the JAX and fused grid evaluators."""
    from utils.sampling_primitives import sample_uniform_2sphere
    kd, kr = jax.random.split(key)
    directions = sample_uniform_2sphere(kd, x.dtype, shape=(samples//2,))
    radius = geometry.volume_radius(x)
    distances = radius*jax.random.uniform(kr, (samples//2, 1), dtype=x.dtype,
                                         minval=jnp.finfo(x.dtype).eps)
    return directions, distances, radius


class BoundaryWalkerState(NamedTuple):
    position: jax.Array  # root sampling reference; inherited driver indexes it
    nSteps: jax.Array
    w: jax.Array
    trust_region_size: jax.Array
    rng_key: jax.Array
    done_flag: jax.Array
    status: jax.Array
    boundary_position: jax.Array
    normal: jax.Array
    root_normal: jax.Array
    root_velocity: jax.Array
    a: jax.Array
    b: jax.Array
    weight: jax.Array


class WoBPressureIntegrator(BaseIntegrator):
    """All-Neumann boundary samples; geometry supplies scalar JAX queries.

    Geometry: contains(x), volume_radius(x), sample_uniform_boundary(key,dtype), and
    next_boundary(x,normal,key) -> (point,normal,weight,valid). A transition
    selects ONE intersection and includes its signed inverse probability.
    A valid zero-weight transition ends the path successfully; its point and
    normal must remain finite (the previous contact is a valid placeholder).
    Integration/refill/compaction/diagnostics are inherited from BaseIntegrator.
    """
    _supported_boundaries = (BoxBoundary, SphereBoundary)

    def __init__(self, geometry, velocity, *, path_length=4, volume_samples=16,
                 boundary_volume_samples=16, regularization=0.):
        if type(geometry) not in self._supported_boundaries:
            supported = ', '.join(cls.__name__ for cls in self._supported_boundaries)
            raise NotImplementedError(
                f'pressure prototype requires an explicit sampler: {supported}; '
                f'got {type(geometry).__name__}')
        if path_length < 3 or min(volume_samples, boundary_volume_samples) < 2:
            raise ValueError('length >= 3 and volume budgets >= 2 required')
        if volume_samples % 2 or boundary_volume_samples % 2 or regularization < 0:
            raise ValueError('even volume budgets and nonnegative regularization required')
        super().__init__(lambda p, x, tr: (0., 0., tr), lambda x: jnp.zeros(3, x.dtype),
                         lambda x: jnp.zeros(3, x.dtype), dim=3, max_steps=path_length+1)
        self.geometry, self.velocity = geometry, velocity
        self.path_length = path_length
        self.volume_samples, self.boundary_volume_samples = volume_samples, boundary_volume_samples
        self.regularization = regularization

    def _volume(self, params, x, key, samples):
        d, r, radius = volume_sample_pairs(self.geometry, x, key, samples)
        d, r = jnp.concatenate([d, -d]), jnp.concatenate([r, r])
        u = self.velocity(params, x)
        def integrand(pair):
            direction, distance = pair
            y = x+distance*direction
            delta = self.velocity(params, y)-u
            value = radius/distance*(3*direction*jnp.dot(direction, delta)-delta)
            return jnp.where(self.geometry.contains(y), value, 0.)
        return jnp.mean(jax.lax.map(integrand, (d, r), batch_size=256), axis=0)

    def _source(self, params, x, key, *, direct):
        samples = self.volume_samples if direct else self.boundary_volume_samples
        return self._volume(params, x, key, samples)

    def _green(self, x, y):
        r = y-x
        r2 = jnp.dot(r, r)
        return r/jnp.maximum(4*jnp.pi*jnp.sqrt(r2)*jnp.maximum(r2, self.regularization),
                             jnp.finfo(x.dtype).tiny)

    def _start(self, params, s):
        key, kb, kv = jax.random.split(s.rng_key, 3)
        y, normal, area, valid = self.geometry.sample_uniform_boundary(kb, s.position.dtype)
        u0 = self.velocity(params, y)
        v0 = self._source(params, y, kv, direct=False)
        w = s.w.at[0, :3].set(y).at[0, 3].set(area*jnp.dot(normal, u0+2*v0))
        w = w.at[0, 4:].set(area*normal)
        return s._replace(nSteps=s.nSteps+1, w=w, rng_key=key, boundary_position=y,
                          normal=normal, root_normal=normal, root_velocity=u0,
                          weight=area, b=2*area*jnp.dot(normal, v0+u0),
                          done_flag=~valid | (area == 0),
                          status=jnp.where(~valid, WT.INVALID,
                                           jnp.where(area == 0, WT.BOUNDARY_HIT, WT.ACTIVE)).astype(s.status.dtype))

    def _hop(self, params, s):
        key, kh = jax.random.split(s.rng_key)
        y, normal, factor, valid = self.geometry.next_boundary(s.boundary_position, s.normal, kh)
        a = jnp.where(s.nSteps == 1,
                      -s.weight*jnp.dot(s.root_normal, s.root_velocity-self.velocity(params, y)), s.a)*factor
        b = s.b*factor
        b = jnp.where(s.nSteps == self.path_length-1, .5*b, b)
        a = jnp.where(s.nSteps == self.path_length, .5*a, a)
        b = jnp.where(s.nSteps == self.path_length, 0., b)
        done = (s.nSteps == self.path_length) | ~valid | (factor == 0)
        return s._replace(nSteps=s.nSteps+1, rng_key=key, boundary_position=y, normal=normal,
                          a=a, b=b, w=s.w.at[s.nSteps, :3].set(y).at[s.nSteps, 3].set(a+b), done_flag=done,
                          status=jnp.where(~valid, WT.INVALID, jnp.where(done, WT.BOUNDARY_HIT, WT.ACTIVE)).astype(s.status.dtype))

    def step(self, params, state):
        updated = jax.lax.cond(state.nSteps == 0, lambda s: self._start(params, s),
                               lambda s: self._hop(params, s), state)
        return finish_step(state, updated)

    def _step_block(self, params, states, steps, unroll):
        # Initialization samples volume integrals once, not on every hop under
        # a vmapped condition (which would evaluate both branches).
        def initialize(s):
            fresh = jax.vmap(self._start, (None, 0))(params, s)
            return jax.tree.map(lambda a, b: jnp.where(((s.nSteps == 0) & ~s.done_flag).reshape(
                (len(s.nSteps),)+(1,)*(a.ndim-1)), b, a), s, fresh)
        states = jax.lax.cond(jnp.any((states.nSteps == 0) & ~states.done_flag), initialize, lambda s: s, states)
        def advance(_, s):
            new = jax.vmap(self._hop, (None, 0))(params, s)
            return jax.vmap(finish_step)(s, new)
        return jax.lax.fori_loop(0, steps, advance, states, unroll=unroll)

    def _get_initial_state(self, positions, keys, init_tr, buffer_size):
        z = jnp.zeros(buffer_size, positions.dtype)
        v = jnp.zeros_like(positions)
        return BoundaryWalkerState(positions, jnp.zeros(buffer_size, jnp.int32),
                                   jnp.zeros((buffer_size, self.path_length+1, 7), positions.dtype),
                                   z+init_tr, keys, jnp.zeros(buffer_size, bool),
                                   jnp.zeros(buffer_size, jnp.int32), v, v, v, v, z, z, z)

    def _transition_state_phases(self, params, states, init_trust_region_size):
        return states

    def _check_if_done(self, state, parents):
        return state.done_flag & (parents != -1)

    def _compute_accumulator_update(self, params, state):
        return state.w

    def _refill_state(self, states, valid, indices, positions, keys, init_tr, total):
        scatter = get_scatter_fn(indices, valid, total)
        fresh = self._get_initial_state(positions, keys, init_tr, len(positions))
        return jax.tree.map(lambda old, new: scatter(old, new, jnp.zeros(old.shape[1:], old.dtype)), states, fresh)


class ExteriorWoBPressureIntegrator(WoBPressureIntegrator):
    """Authors' unbounded-domain source: volume plus pseudo-boundary integral.

    VelMCFluids 4f777dd, velocity_fluids_optix.cu: estimate_pseudo_boundary_term.
    Both path-root and query evaluations need this correction. No new driver.
    """
    _supported_boundaries = (ExteriorSphereBoundary, ExteriorTorusBoundary)

    def __init__(self, geometry, velocity, *, pseudo_samples=100000,
                 boundary_pseudo_samples=10, **options):
        super().__init__(geometry, velocity, **options)
        if min(pseudo_samples, boundary_pseudo_samples) < 1:
            raise ValueError('positive pseudo-boundary budgets required')
        self.pseudo_samples = pseudo_samples
        self.boundary_pseudo_samples = boundary_pseudo_samples

    def _pseudo(self, params, x, key, samples):
        keys = jax.random.split(key, samples)
        u = self.velocity(params, x)
        def integrand(k):
            y, normal, area = self.geometry.sample_pseudo_boundary(k, x.dtype)
            return self._green(x, y)*(area*jnp.dot(normal, self.velocity(params, y)-u))
        return jnp.mean(jax.lax.map(integrand, keys, batch_size=256), axis=0)

    def pseudo_records(self, params, key, dtype):
        """Unnormalized (position, a, b) sources for the existing gather.

        One fresh set is shared by all projection queries. No target-by-source
        samples or interpolated velocities are stored.
        """
        keys = jax.random.split(key, self.pseudo_samples)
        y, normal, area = jax.vmap(self.geometry.sample_pseudo_boundary,
                                  (0, None))(keys, dtype)
        velocity = jax.vmap(self.velocity, (None, 0))(params, y)
        a = area*jnp.sum(normal*velocity, axis=-1)
        return jnp.concatenate((y, a[:, None], -area[:, None]*normal), axis=-1)

    def _source(self, params, x, key, *, direct):
        kv, kp = jax.random.split(key)
        volume = self.volume_samples if direct else self.boundary_volume_samples
        pseudo = self.pseudo_samples if direct else self.boundary_pseudo_samples
        return self._volume(params, x, kv, volume)+self._pseudo(params, x, kp, pseudo)
