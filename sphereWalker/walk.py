"""Tier 2: shared PDE base for interior random-walk solvers of -Lap u = f."""

import abc
from typing import Any, Callable, Tuple

import jax
import jax.numpy as jnp
import jax.random as jr

from sphereWalker.driver import MCIntegrator
from sphereWalker.ops import (
    average_draws,
    ball_sampling_fn_select,
    finish_step,
    sphere_sampling_fn_select,
    walker_status,
    weight_greens_fn_select,
)
from sphereWalker.types import StateT
from sphereWalker.types import WalkerTermination as WT
from surfaces.queries import EmptySphere, EmptySphereQuery
from utils.validation import check_int


def check_broadcast(shape, target, what):
    """Raise unless `shape` broadcasts to exactly `target`."""
    try:
        ok = jnp.broadcast_shapes(shape, target) == target
    except ValueError:
        ok = False
    if not ok:
        raise ValueError(f"{what} shape {shape} must broadcast to boundary value shape {target}")


class WalkIntegrator(MCIntegrator[StateT]):
    """Interior walks for -Lap u = f with Dirichlet data g.

    Boundary values declare the component shape (at least (1,)); sources may
    be scalar or broadcast to it, and all components share one walk.
    source_takes_params calls source_term_fn(params, x) instead of
    source_term_fn(x), so a source that changes between solves (e.g. a
    vorticity grid in params) does not retrace the jitted walk.
    source_samples averages this many volume draws per ball in fixed chunks.
    truncate=True accepts capped paths with accumulated sources and zero
    unknown solution tail; its finite-depth bias is explicit in
    termination_counts.
    """

    def __init__(self,
                 empty_sphere_fn: Callable[[Any, jnp.ndarray, jnp.ndarray], Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]],
                 boundary_condition_fn: Callable[[jnp.ndarray], jnp.ndarray],
                 source_term_fn: Callable[..., jnp.ndarray],
                 dim: int,
                 *,
                 max_steps: int = 1000,
                 eps: float = 1e-3,
                 print_progress: bool = False,
                 source_takes_params: bool = False,
                 source_samples: int = 1,
                 truncate: bool = False):
        if dim not in (2, 3):
            raise ValueError('walk integrators support dimensions 2 and 3')
        check_int('source_samples', source_samples)
        super().__init__(dim, max_steps=max_steps, print_progress=print_progress)
        self.empty_sphere = EmptySphereQuery(empty_sphere_fn)
        self.boundary_condition_fn = boundary_condition_fn
        self.source_term = source_term_fn
        self.eps = eps
        self.source_takes_params = source_takes_params
        self.source_samples = int(source_samples)
        self.truncate = bool(truncate)
        self.sphere_sampling_fn = sphere_sampling_fn_select[dim]
        self.ball_sampling_fn = ball_sampling_fn_select[dim]
        self.weighted_greens_fn = weight_greens_fn_select[dim]

    # ----------------------------------------------------------- callbacks
    def _problem_params(self, params: Any) -> Any:
        """Parameters for the Dirichlet query and the source callback."""
        return params

    def _value_shape(self, dtype) -> Tuple[int, ...]:
        point = jax.ShapeDtypeStruct((self.dim,), dtype)
        return jnp.broadcast_shapes((1,), jax.eval_shape(self.boundary_condition_fn, point).shape)

    def _source(self, problem_params: Any, x: jnp.ndarray) -> jnp.ndarray:
        return self.source_term(problem_params, x) if self.source_takes_params else self.source_term(x)

    def _source_components(self, problem_params, x, value_shape):
        """Source at x broadcast to the value shape, in the callback's precision.

        Callers multiply by their Green's (or flux) weight in the promoted
        precision and only then cast the weighted contribution to the walk
        dtype: a large source with a small weight must not overflow early.
        """
        raw = self._source(problem_params, x)
        # Strip weak typing on Python scalars so x64 arithmetic promotes before casting.
        source = jnp.asarray(raw, dtype=jnp.result_type(raw))
        check_broadcast(source.shape, tuple(value_shape), 'Source')
        return jnp.broadcast_to(source, value_shape)

    def _run_plan(self, problem_params, state, point, finish):
        """Complete a (point, finish) step plan with its own source call."""
        return finish(None if point is None
                      else self._source_components(problem_params, point, state.w.shape))

    # ------------------------------------------------------ estimator hooks
    def _contributing_status(self, status):
        return (status == WT.BOUNDARY_HIT) | (self.truncate & (status == WT.MAX_STEPS))

    def _terminal_value(self, state):
        # At a cap retain accumulated sources; only the unresolved solution
        # tail is zero. Never use boundary data as a value at an interior cap.
        boundary = jnp.where(state.status == WT.BOUNDARY_HIT,
                             self.boundary_condition_fn(state.position), 0.)
        return state.w + boundary

    def _compute_accumulator_update(self, params: Any, state: StateT) -> Any:
        return self._terminal_value(state)

    def _first_ball_clearance(self, params: Any, state: StateT, query: EmptySphere):
        """(radius, valid, next_tr) of a ball around the start that may hold a gradient.

        valid is checked before any clipping, so an invalid (e.g. +inf)
        clearance cannot become a finite radius. next_tr is the trust region
        for a zero-radius retry.
        """
        radius = query.radius
        return radius, jnp.isfinite(radius) & (radius >= 0), query.next_trust_region

    @abc.abstractmethod
    def _continuation_step(self, params: Any, state: StateT, query: EmptySphere) -> StateT:
        """One ordinary walk step given the Dirichlet query at state.position."""

    def _continuation_plan(self, params: Any, state: StateT, query: EmptySphere):
        """`_continuation_step` split around its source call: (point, finish).

        point is the step's single source draw position, or None when the
        step evaluates its sources itself; finish(source) -> state takes the
        source components at point (None when point is None). A masked
        first-ball step uses this to call the source once for both branches.
        """
        return None, lambda source: self._continuation_step(params, state, query)

    def step(self, params: Any, state: StateT) -> StateT:
        query = self.empty_sphere(self._problem_params(params), state.position, state.trust_region_size)
        return self._continuation_step(params, state, query)

    # ------------------------------------------------------------ WoS step
    def _wos_step_plan(self, problem_params: Any, state: StateT, query: EmptySphere, *,
                       accumulate_source=True):
        """One Dirichlet walk-on-spheres ball step as a (point, finish) plan.

        See `_continuation_plan`. accumulate_source may be a Python or traced
        bool; False (Python) skips the source integral entirely.
        """
        residual, radius, next_tr = query
        status = walker_status(state, residual, radius, next_tr, self.eps, self.max_steps)
        active = status == WT.ACTIVE
        dtype = state.position.dtype
        sphere_key, ball_key, next_key = jr.split(state.rng_key, 3)
        direction = self.sphere_sampling_fn(sphere_key, dtype=dtype)
        # A zero radius (undecided query) has no ball to integrate over:
        # the contribution is 0, never weighted_greens(0, 0) = NaN.
        has_ball = active & (radius > 0)
        safe_radius = jnp.where(has_ball, radius, 1.)

        def draw(key):
            unit_r, ball_direction = self.ball_sampling_fn(key, dtype=dtype)
            r = unit_r * safe_radius
            # Do not evaluate the source away from a live, nonempty ball.
            x = jnp.where(has_ball, state.position + r * ball_direction, state.position)
            return x, self.weighted_greens_fn(r=r, R=safe_radius)

        def weighted(source, green):
            return jnp.where(has_ball, source * green, 0.)

        single = accumulate_source is not False and self.source_samples == 1
        point, green = draw(ball_key) if single else (None, None)

        def finish(source):
            w = state.w
            if accumulate_source is not False:
                if single:
                    contribution = weighted(source, green).astype(state.w.dtype)
                else:
                    def sample(key):
                        x, weight = draw(key)
                        return weighted(self._source_components(problem_params, x, state.w.shape), weight)
                    contribution = average_draws(
                        ball_key, sample, count=self.source_samples).astype(state.w.dtype)
                w = state.w + (contribution if accumulate_source is True
                               else jnp.where(accumulate_source, contribution, 0.))
            updated = state._replace(
                position=jnp.where(active, state.position + radius * direction, state.position),
                nSteps=state.nSteps + active.astype(state.nSteps.dtype),
                w=jnp.where(active, w, state.w),
                trust_region_size=jnp.where(active, next_tr, state.trust_region_size),
                rng_key=next_key,
                done_flag=~active,
                status=status,
            )
            return finish_step(state, updated)

        return point, finish
