"""First-ball spatial gradient estimator, mixable into any WalkIntegrator."""

from numbers import Integral

import jax
import jax.numpy as jnp
import jax.random as jr

from sphereWalker.ops import average_draws, finish_step, first_ball_source_weight, walker_status
from sphereWalker.types import WalkerTermination as WT
from sphereWalker.walk import check_broadcast
from utils.sampling_primitives import unit_interval


class FirstBallGradientMixin:
    """Spatial gradient of u (-Lap u = f, u = g) from the first ball only.

    Differentiates the ball mean-value formula (Sawhney & Crane 2020, Eq. 14):
        grad u(x) = E[(d/R) (u(y) - c(x)) n] + E[f(z) grad_x G(x, z) / p(z)],
    with y uniform on the first sphere, n = (y - x)/R, and u(y) the ordinary
    continuation estimate from y. In the first ball the radius of z is
    uniform in [0, R], so the source weight R (1 - (r/R)^d) f(z) stays bounded.
    The ball formula holds for any ball inside the domain, so conservative
    radii stay unbiased. The first radius is the host integrator's
    `_first_ball_clearance`; starts on a boundary are invalid. Zero-clearance
    retries count toward max_steps without consuming the first ball. The
    output adds a trailing (dim,) axis to the value shape.

    control_fn(x) (or control_fn(problem_params, x) with control_takes_params)
    is a fixed local value c subtracted from the continuation; it must
    broadcast to the boundary value shape. Since E[n] = 0 any c keeps the
    estimate unbiased; a stochastic pilot must be independent of the current
    directions. mirror_first_sphere negates the first-sphere draw, so two
    solves with the same key form an antithetic pair sharing the local source
    integral. gradient_source_samples counts first-ball source draws per walk;
    source_samples independently counts continuation draws.

    Subclasses set `_gradient_state_cls` to the base state plus
    (grad_weight, grad_source) fields.
    """

    _gradient_state_cls: type

    def __init__(self, *args, control_fn=None, control_takes_params=False,
                 mirror_first_sphere=False, gradient_source_samples: int = 1, **kwargs):
        super().__init__(*args, **kwargs)
        if (isinstance(gradient_source_samples, bool)
                or not isinstance(gradient_source_samples, Integral) or gradient_source_samples < 1):
            raise ValueError('gradient_source_samples must be a positive integer')
        self.gradient_source_samples = int(gradient_source_samples)
        self.control_fn = control_fn
        # A dynamic pilot travels with problem parameters, never a mutable JIT closure.
        self.control_takes_params = control_takes_params
        self.mirror_first_sphere = mirror_first_sphere

    def _get_initial_state(self, positions, keys, init_tr, buffer_size):
        base = super()._get_initial_state(positions, keys, init_tr, buffer_size)
        dtype = positions.dtype
        shape = base.w.shape[1:]
        if self.control_fn is not None and not self.control_takes_params:
            point = jax.ShapeDtypeStruct((self.dim,), dtype)
            check_broadcast(jax.eval_shape(self.control_fn, point).shape, shape, 'Control')
        return self._gradient_state_cls(
            *base,
            grad_weight=jnp.zeros((buffer_size, self.dim), dtype),
            grad_source=jnp.zeros((buffer_size,) + shape + (self.dim,), dtype),
        )

    def _first_sphere_direction(self, key, dtype):
        """Continuation direction; a paired estimator may negate only this draw."""
        direction = self.sphere_sampling_fn(key, dtype=dtype)
        return -direction if self.mirror_first_sphere else direction

    def _control(self, params, state):
        if self.control_fn is None:
            return jnp.zeros((), state.w.dtype)
        if self.control_takes_params:
            control = self.control_fn(self._problem_params(params), state.position)
        else:
            control = self.control_fn(state.position)
        control = jnp.asarray(control, state.w.dtype)
        check_broadcast(control.shape, state.w.shape, 'Control')
        return control

    def _first_step_plan(self, params, state, query):
        """The first-ball step as a (point, finish) plan; see `_continuation_plan`."""
        radius, clearance_valid, next_tr = self._first_ball_clearance(params, state, query)
        status = walker_status(state, query.residual, radius, next_tr, self.eps, self.max_steps)
        # The gradient needs one ball around the start point.
        status = jnp.where(~clearance_valid | (status == WT.BOUNDARY_HIT), WT.INVALID, status)
        active = status == WT.ACTIVE
        advance = active & (radius > 0)
        step_radius = jnp.where(advance, radius, 0.0)
        dtype = state.position.dtype
        problem = self._problem_params(params)
        sphere_key, source_key, direction_key, next_key = jr.split(state.rng_key, 4)
        direction = self._first_sphere_direction(sphere_key, dtype)

        def draw(kr, kd):
            t = unit_interval(kr, dtype)
            d = self.sphere_sampling_fn(kd, dtype=dtype)
            # Weight in the promoted precision; callers cast after averaging.
            def weigh(source):
                return first_ball_source_weight(self.dim, step_radius, t, d, source)
            return state.position + step_radius * t * d, weigh

        single = self.gradient_source_samples == 1
        point, weigh = draw(source_key, direction_key) if single else (None, None)

        def finish(source):
            if single:
                source_weight = weigh(source).astype(state.w.dtype)
            else:
                def sample(key):
                    x, weigh_draw = draw(*jr.split(key))
                    return weigh_draw(self._source_components(problem, x, state.w.shape))
                source_weight = average_draws(
                    source_key, sample, count=self.gradient_source_samples).astype(state.w.dtype)
            weight = (self.dim / jnp.where(advance, step_radius, 1.)) * direction
            control = self._control(params, state)
            updated = state._replace(
                position=state.position + step_radius * direction,
                nSteps=state.nSteps + active.astype(state.nSteps.dtype),
                w=jnp.where(advance, state.w - control, state.w),
                trust_region_size=jnp.where(active, next_tr, state.trust_region_size),
                rng_key=next_key,
                grad_weight=jnp.where(advance, weight, state.grad_weight),
                grad_source=jnp.where(advance, source_weight, state.grad_source),
                status=status,
                done_flag=status != WT.ACTIVE,
            )
            return finish_step(state, updated)

        return point, finish

    def step(self, params, state):
        # A sampled unit direction always gives a nonzero gradient weight.
        # nSteps cannot identify initialization: zero-clearance retries count too.
        first = ~jnp.any(state.grad_weight != 0)
        # Masks, not lax.cond: under vmap cond runs both branches, which would
        # repeat the (possibly expensive) Dirichlet query.
        problem = self._problem_params(params)
        query = self.empty_sphere(problem, state.position, state.trust_region_size)
        first_point, first_finish = self._first_step_plan(params, state, query)
        walk_point, walk_finish = self._continuation_plan(params, state, query)
        if first_point is not None and walk_point is not None:
            # Each lane needs only its own branch's single source draw, so the
            # source (e.g. a learned MLP) runs once per masked step, not twice.
            source = self._source_components(
                problem, jnp.where(first, first_point, walk_point), state.w.shape)
            initial, walked = first_finish(source), walk_finish(source)
        else:
            initial = self._run_plan(problem, state, first_point, first_finish)
            walked = self._run_plan(problem, state, walk_point, walk_finish)
        return jax.tree.map(lambda a, b: jnp.where(first, a, b), initial, walked)

    def _step_block(self, params, states, steps, unroll):
        # The masked step computes the first-ball branch for every lane. After
        # one step only stalled first balls still need it, and no lane can
        # newly need it before the next refill. Masked steps run only while
        # such a lane remains; the rest of the block runs continuation-only
        # steps (the same states as the masked step).
        step = jax.vmap(self.step, in_axes=(None, 0))
        continuation = jax.vmap(super().step, in_axes=(None, 0))
        states = step(params, states)
        if steps == 1:
            return states

        def pending(s):
            return jnp.any(~jnp.any(s.grad_weight != 0, axis=-1) & ~s.done_flag)

        def masked(s):
            def cond(carry):
                i, t = carry
                return (i < steps) & pending(t)

            def body(carry):
                i, t = carry
                return i + jnp.int32(1), step(params, t)

            i, s = jax.lax.while_loop(cond, body, (jnp.int32(1), s))
            return jax.lax.fori_loop(i, steps, lambda _, t: continuation(params, t), s)

        def fast(s):
            return jax.lax.fori_loop(1, steps, lambda _, t: continuation(params, t), s, unroll=unroll)

        return jax.lax.cond(pending(states), masked, fast, states)

    def _compute_accumulator_update(self, params, state):
        return state.grad_source + self._terminal_value(state)[..., None] * state.grad_weight
