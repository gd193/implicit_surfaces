"""Vector-potential velocity reconstruction through the existing WoS integrators.

The paired Dirichlet solver uses a separate first-ball source estimate and
ordinary WoS continuations. Geometry/source callbacks and the bounded batch
driver remain owned by the integrator. No boundary cache or temporal history.
"""
from collections.abc import Callable
from functools import partial
from numbers import Integral
from typing import NamedTuple

import jax
import jax.numpy as jnp
import jax.random as jr
from jax.extend.random import threefry2x32_p
from sphereWalker.integrator import WoSIntegrator, get_scatter_fn
from sphereWalker.types import WalkerDiagnostics, WalkerState
from sphereWalker.types import WalkerTermination as WT
from utils.sampling_primitives import sample_uniform_2sphere, unit_interval, sample_green_2ball as _green_ball


def antithetic_source_samples(total_samples: int, walks: int) -> int:
    """First-ball draws per pair for a total budget of distinct source samples.

    Both halves reuse the local source integral. Continuation source samples
    have a separate budget.
    """
    if (isinstance(walks, bool) or not isinstance(walks, Integral) or walks < 2 or walks % 2
            or isinstance(total_samples, bool) or not isinstance(total_samples, Integral)
            or total_samples < walks//2 or total_samples % (walks//2)):
        raise ValueError('source samples must be a positive multiple of the number of walk pairs')
    return total_samples//(walks//2)


class VectorPotentialResult(NamedTuple):
    velocity: jax.Array
    diagnostics: WalkerDiagnostics

class VelocitySolver(NamedTuple):
    """Prepare static query geometry once; solve changing source fields per step."""
    prepare_queries: Callable
    solve: Callable

def curl_from_jacobian(jac):
    return jnp.stack((jac[..., 2, 1]-jac[..., 1, 2],
                      jac[..., 0, 2]-jac[..., 2, 0],
                      jac[..., 1, 0]-jac[..., 0, 1]), axis=-1)

def make_gradient_velocity_solve(gradient_factory, *args, **kwargs):
    """Adapt an existing gradient integrator to the velocity solver contract."""
    integrate = gradient_factory(*args, **kwargs)

    @jax.jit
    def velocity_result(details):
        jac = details.contribution_sum/jnp.maximum(details.successful_samples, 1)[:, None, None]
        return VectorPotentialResult(curl_from_jacobian(jac), details)

    def solve(params, points, key, **options):
        details = integrate(params, points, key, return_details=True, **options)
        return velocity_result(details)

    return VelocitySolver(lambda points: points, solve)

def _reduce_parent_runs(accumulator: jax.Array, parents: jax.Array,
                        updates: jax.Array) -> jax.Array:
    """Reduce adjacent equal parents with a segmented scan, then add run totals.

    Refilling can split or reorder parents; separate runs still scatter-add.
    Drop indices for unfinished slots simply split runs. Storage is bounded by
    the active buffer; no per-query expansion or sorted-parent assumption.
    """
    starts = jnp.concatenate((jnp.ones_like(parents[:1], bool), parents[1:] != parents[:-1]))

    def combine(left, right):
        start_a, a = left
        start_b, b = right
        return start_a | start_b, jnp.where(start_b[..., None, None], b, a+b)

    _, totals = jax.lax.associative_scan(combine, (starts, updates))
    ends = jnp.concatenate((starts[1:], jnp.ones_like(starts[:1])))
    destinations = jnp.where(ends, parents, len(accumulator))
    return accumulator.at[destinations].add(totals, mode='drop')


class _PairState(NamedTuple):
    walkers: WalkerState
    weight: jax.Array
    nSteps: jax.Array
    done_flag: jax.Array
    status: jax.Array

    @property
    def position(self):
        return self.walkers.position


class _FluidPairs(WoSIntegrator):
    """One scheduled sample is a pair of ordinary WoS continuations.

    The batch driver's parent is the original query, so contributions reduce
    directly to that query. Only active pairs have expanded positions/state.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.continuation = WoSIntegrator(*args, **kwargs)

    def _prepare_positions(self, seeds, *options):
        # Private seed layout: query xyz followed by its precomputed radius.
        positions = super()._prepare_positions(seeds[:, :3], *options)
        return jnp.concatenate((positions, seeds[:, 3:4]), axis=1)

    @partial(jax.jit, static_argnums=(0,))
    def _get_accumulator_initial(self, params, seeds):
        n = len(seeds)
        return (jnp.zeros((n, 3, 3), seeds.dtype),
                jnp.zeros((n, len(WT)), jnp.int32), jnp.zeros(n, jnp.int32))

    @partial(jax.jit, static_argnums=(0, 4))
    def _get_initial_state(self, seeds, keys, trust_radius, size):
        split = jax.vmap(lambda k: jr.split(k, 4))(keys)
        directions = jax.vmap(lambda k: sample_uniform_2sphere(k, seeds.dtype))(split[:, 0])
        radius = seeds[:, 3]
        delta = radius[:, None]*directions
        positions = seeds[:, None, :3]+jnp.stack((delta, -delta), axis=1)
        # Both legs use common continuation randomness, independent of the
        # direction and the two keys used by the local source estimator.
        walk_keys = jnp.broadcast_to(split[:, None, 3], (size, 2, 2))
        walkers = super()._get_initial_state(
            positions.reshape(-1, 3), walk_keys.reshape(-1, 2), trust_radius, 2*size)
        walkers = walkers._replace(w=jnp.zeros((2*size, 3), seeds.dtype))
        walkers = jax.tree.map(lambda a: a.reshape((size, 2)+a.shape[1:]), walkers)
        weight = (3/jnp.where(radius > 0, radius, 1.))[:, None]*directions
        return _PairState(walkers, weight, jnp.zeros(size, jnp.int32),
                          jnp.zeros(size, bool), jnp.zeros(size, jnp.int32))

    def step(self, params, state):
        walkers = jax.vmap(super().step, in_axes=(None, 0))(params, state.walkers)
        done = jnp.all(walkers.done_flag)
        # The driver schedules pairs. Actual per-leg termination reasons and
        # accepted sample counts are carried by the contribution below.
        return _PairState(walkers, state.weight, walkers.nSteps.sum(), done,
                          jnp.where(done, WT.MAX_STEPS, WT.ACTIVE).astype(jnp.int32))

    def _step_block(self, params, states, steps, unroll):
        # Pairing affects initialization and reduction only. Keep the existing
        # walk loop flat, with precisely the same ordinary WalkerState layout.
        size = len(states.done_flag)
        flat = jax.tree.map(lambda a: a.reshape((2*size,)+a.shape[2:]), states.walkers)
        flat = self.continuation._step_block(params, flat, steps, unroll)
        walkers = jax.tree.map(lambda a: a.reshape((size, 2)+a.shape[1:]), flat)
        done = jnp.all(walkers.done_flag, axis=1)
        return _PairState(walkers, states.weight, walkers.nSteps.sum(1), done,
                          jnp.where(done, WT.MAX_STEPS, WT.ACTIVE).astype(jnp.int32))

    def _contributing_status(self, status):
        return status == WT.MAX_STEPS

    def _accumulate(self, sums, parents, updates):
        gradient = _reduce_parent_runs(sums[0], parents, updates[0])
        return (gradient, *super()._accumulate(sums[1:], parents, updates[1:]))

    def _compute_accumulator_update(self, params, state):
        walkers = state.walkers
        hit = walkers.status == WT.BOUNDARY_HIT
        accepted = hit | (walkers.status == WT.MAX_STEPS)
        boundary = jax.vmap(self.boundary_condition_fn)(walkers.position)
        values = walkers.w+jnp.where(hit[:, None], boundary, 0.)
        finite = jnp.all(jnp.isfinite(values), axis=-1)
        status = jnp.where(accepted & ~finite, WT.INVALID, walkers.status)
        accepted &= finite
        values = jnp.where(accepted[:, None], values, 0.)
        gradient = (values[0]-values[1])[:, None]*state.weight
        counts = (status[:, None] == jnp.arange(len(WT))).sum(0)
        return gradient, counts, accepted.sum()

    def _refill_state(self, states, valid, indices, seeds, keys, trust_radius, total):
        fresh = self._get_initial_state(seeds, keys, trust_radius, len(seeds))
        scatter = get_scatter_fn(indices, valid, total)
        return jax.tree.map(lambda old, new: scatter(old, new, jnp.zeros_like(old[0])),
                            states, fresh)


def _identity_source_params(params):
    return params


def _local_source_keys(
    key: jax.Array, index: jax.Array, draws: int,
) -> tuple[tuple[jax.Array, ...], tuple[jax.Array, ...]]:
    """Original local sample keys, with key words kept separate for XLA fusion.

    Partitionable Threefry split/fold_in both hash (0, counter). Calling JAX's
    primitive directly avoids packing/unpacking key arrays between hashes; it
    does not change the generator or counters. Other configurations retain the
    ordinary random API. Branches resolve while tracing.
    """
    if jax.config.jax_threefry_partitionable and jr.key_impl(key) == 'threefry2x32':
        def counter(words, value):
            value = jnp.asarray(value, jnp.uint32)
            # Preserve batched key shapes for Threefry's CPU loop.
            zero = value ^ value
            return tuple(threefry2x32_p.bind(
                *(w+zero for w in words), zero, value))

        initial = counter(tuple(jr.key_data(key)), index)
        radial = counter(initial, 1)
        if draws == 1:
            return jax.tree.map(lambda a: a[None], (radial, counter(initial, 2)))

        def draw(j):
            base = counter(radial, j)
            return counter(base, 0), counter(base, 1)

        return jax.vmap(draw)(jnp.arange(draws))

    _, radial, direction, _ = jr.split(jr.fold_in(key, index), 4)
    if draws == 1:
        radial, direction = radial[None], direction[None]
    else:
        keys = jax.vmap(lambda j: jr.split(jr.fold_in(radial, j)))(jnp.arange(draws))
        radial, direction = keys[:, 0], keys[:, 1]
    return (tuple(jnp.moveaxis(jr.key_data(radial), -1, 0)),
            tuple(jnp.moveaxis(jr.key_data(direction), -1, 0)))


def make_dirichlet_gradient_solve(empty_sphere, boundary, source, *, dim,
                                  gradient_source_samples,
                                  prepare_local_source=_identity_source_params,
                                  local_gradient_cuda=None,
                                  **configuration):
    """Same gradient factory contract; homogeneous or nonzero Dirichlet data.

    gradient_source_samples is the number of distinct local draws per pair.
    JAX batches contain at most 512 queries and target at most 512K source
    samples, independently of grid size (one query is the minimum batch).
    The existing WoS batch driver, geometry queries and PRNG are reused.
    prepare_local_source may change the local source's parameter layout, but
    must preserve its values at every point. Walks receive the original params.
    An optional local_source callback may evaluate that same source with a
    different expression; it follows source_takes_params and must be callable.
    local_gradient_cuda replaces the local CUDA integral or returns NotImplemented.
    """
    if dim != 3:
        raise ValueError('the fluid adapter requires three dimensions')
    if not callable(prepare_local_source):
        raise TypeError('prepare_local_source must be callable')
    if local_gradient_cuda is not None and not callable(local_gradient_cuda):
        raise TypeError('local_gradient_cuda must be callable')
    configuration = dict(configuration)
    local_source = configuration.pop('local_source', source)
    if not callable(local_source):
        raise TypeError('local_source must be callable')
    evaluate_local = (local_source if configuration.get('source_takes_params', False)
                      else lambda params, x: local_source(x))
    max_steps = configuration.pop('max_steps', 1000)
    control = configuration.pop('control_fn', None)
    if control is not None:
        raise ValueError('paired Dirichlet fluid gradients use their own antithetic control')
    if max_steps < 1:
        raise ValueError('max_steps must be positive')
    paired = _FluidPairs(empty_sphere, boundary, source, dim=dim,
                         max_steps=max_steps-1, **configuration)
    # Existing sampler extension points: Green radial PDF absorbs G, yielding
    # bounded weight R^2/6. Continuations delegate to the existing WoS step.
    for walker in (paired, paired.continuation):
        walker.ball_sampling_fn = _green_ball
        walker.weighted_greens_fn = lambda r, R: R**2/6
    draws = gradient_source_samples
    if isinstance(draws, bool) or not isinstance(draws, int) or draws < 1:
        raise ValueError('gradient_source_samples must be a positive integer')

    @partial(jax.jit, static_argnames=('pairs',))
    def local_gradient(params, points, radii, key, *, pairs):
        key_dtype = jr.key_impl(key)

        def at_point(item):
            index, point, radius = item

            def pair(i):
                def sample(radial_key, direction_key):
                    radial_key = jr.wrap_key_data(jnp.stack(radial_key), dtype=key_dtype)
                    direction_key = jr.wrap_key_data(jnp.stack(direction_key), dtype=key_dtype)
                    t = unit_interval(radial_key, points.dtype)
                    direction = sample_uniform_2sphere(direction_key, points.dtype)
                    # Avoid recomputing trig for position and gradient in XLA.
                    # Share components individually to avoid a vector transpose.
                    direction = jnp.stack(jax.lax.optimization_barrier(tuple(direction)))
                    value = evaluate_local(params, point+(radius*t)*direction)
                    # Keep components separate through the sample reduction;
                    # packing a 3x3 array here generates shared-memory transposes.
                    return tuple(value[i]*(radius*(1-t**3))*direction[j]
                                 for i in range(3) for j in range(3))

                radial, direction = _local_source_keys(key, index*pairs+i, draws)
                return jax.vmap(sample)(radial, direction)

            samples = jax.vmap(pair)(jnp.arange(pairs))
            # One variadic reduction shares the sample traversal across all nine
            # components, without packing a per-sample 3x3 intermediate.
            totals = jax.lax.reduce(samples, tuple(jnp.zeros((), a.dtype) for a in samples),
                lambda a, b: tuple(x+y for x, y in zip(a, b)), (0, 1))
            return jnp.stack(totals).reshape(3, 3)/(pairs*draws)

        batch = min(512, max(1, len(points)), max(1, (512*1024)//(pairs*draws)))
        return jax.lax.map(at_point, (jnp.arange(len(points)), points, radii), batch_size=batch)

    if local_gradient_cuda is not None:
        reference = local_gradient

        @partial(jax.jit, static_argnames=('pairs',))
        def local_gradient(params, points, radii, key, *, pairs):
            def cuda(*args):
                result = local_gradient_cuda(*args, pairs=pairs)
                return reference(*args, pairs=pairs) if result is NotImplemented else result

            return jax.lax.platform_dependent(params, points, radii, key,
                cuda=cuda,
                default=lambda *args: reference(*args, pairs=pairs))

    @jax.jit
    def prepare(params, points, trust_radius):
        residual, radii, _ = jax.vmap(
            lambda x: paired.empty_sphere(params, x, trust_radius))(points)
        valid = (jnp.abs(residual) > paired.eps) & jnp.isfinite(radii) & (radii > 0)
        return jnp.concatenate((points, radii[:, None]), axis=1), jnp.where(valid, radii, 0.)

    @jax.jit
    def finish(points, details, local):
        gradient, counts, successful = details.contribution_sum
        return WalkerDiagnostics(points, gradient+successful[:, None, None]*local,
                                 successful, counts, 1+details.nSteps/2)

    def integrate(params, points, key, init_trust_region_size, n_samples, **options):
        antithetic_source_samples((n_samples//2)*draws, n_samples)
        if not options.pop('return_details', False):
            raise ValueError('fluid gradient solves require return_details=True to expose truncated walks')
        pairs = n_samples//2
        options['steps_between_refill'] = min(options.get('steps_between_refill', 16), max_steps)
        active = options.get('max_active_states', 1000**2)
        if isinstance(active, bool) or not isinstance(active, Integral) or active < 2:
            raise ValueError('max_active_states must be an integer >= 2 for paired walks')
        # The public limit counts individual walks, not pairs.
        options['max_active_states'] = active//2
        seeds, radii = prepare(params, points, init_trust_region_size)
        details = paired.integrate_batch(params, seeds, key, init_trust_region_size,
                                         n_samples=pairs, return_details=True, **options)
        local = local_gradient(prepare_local_source(params), points, radii, key, pairs=pairs)
        return finish(points, details, local)

    return integrate
