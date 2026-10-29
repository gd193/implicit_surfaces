"""Walk on Stars with an implicit Dirichlet query and a Neumann enclosure.

Solves -Delta u = f, u = g on the Dirichlet surface, du/dn = 0 on
Neumann surfaces supporting star_radius and intersect_ray. Nonzero outward
flux is supported by a specialized analytic-box sampling estimator.
Geometric derivatives are deliberately unsupported.
"""

import math
from numbers import Integral

import jax
import jax.numpy as jnp
import jax.random as jr

from sphereWalker.integrator import (
    BaseIntegrator,
    finish_step,
    first_ball_gradient_weights,
    get_scatter_fn,
    walker_status,
)
from sphereWalker.types import GradientStarWalkerState, StarWalkerState, WoStParameters
from sphereWalker.types import WalkerTermination as WT
from surfaces.implicit.base import ImplicitSurface
from utils.sampling_primitives import (
    unit_interval, sample_green_radius, sample_box_boundary_in_ball,
    weighted_greens_fn_box_boundary,
)
from surfaces.implicit.lipschitz import sdfImplicit
from surfaces.implicit.primitives.sdfnd import bbox_sdf, box_contact_count
from utils.types import Termination as T


class WoStIntegrator(BaseIntegrator[StarWalkerState]):
    """Same batch driver and Dirichlet/source callbacks as WoS.

    Boundary values declare the component shape. Sources may be scalar or
    broadcast to that shape; all components share the same geometric walk.
    neumann_condition_fn(problem_params, position, outward_normal) supplies
    outward derivative data, scalar or broadcastable to the boundary shape.
    Omit it for homogeneous data. Nonzero flux currently supports analytic boxes.

    source_samples averages this many volume draws per sphere in fixed chunks.
    truncate=True accepts capped paths with accumulated sources and zero unknown
    solution tail; its finite-depth bias is explicit in termination_counts.

    Pass neumann_params explicitly to integrate_batch/pre_warm_compaction_buckets.
    Initial points may be interior or on supported Neumann contacts (including
    box edges/corners). Unresolved queries fail explicitly. eps retains the
    current WoS field-residual termination convention, not a distance certificate.
    """

    def __init__(
        self,
        empty_sphere_fn,
        boundary_condition_fn,
        source_term_fn,
        dim,
        *,
        neumann_surface,
        max_radius=math.inf,
        neumann_condition_fn=None,
        source_samples: int = 1,
        truncate: bool = False,
        **kwargs,
    ):
        if dim not in (2, 3):
            raise ValueError("WoSt supports dimensions 2 and 3")
        if neumann_surface.dim != dim:
            raise ValueError("Neumann surface dimension must match the integrator")
        for name in ("star_radius", "intersect_ray"):
            if getattr(type(neumann_surface), name, None) is getattr(ImplicitSurface, name):
                raise NotImplementedError(
                    f"{type(neumann_surface).__name__} does not implement {name}"
                )
        if (neumann_condition_fn is not None
                and not (type(neumann_surface) is sdfImplicit and neumann_surface.fn is bbox_sdf)):
            raise NotImplementedError(
                "nonzero Neumann data currently requires an analytic box sampling estimator"
            )
        if math.isnan(max_radius) or max_radius <= 0:
            raise ValueError("max_radius must be positive")
        if isinstance(source_samples, bool) or not isinstance(source_samples, Integral) or source_samples < 1:
            raise ValueError('source_samples must be a positive integer')
        super().__init__(
            empty_sphere_fn, boundary_condition_fn, source_term_fn, dim, **kwargs
        )
        self.neumann_surface = neumann_surface
        self.max_radius = max_radius
        self.neumann_condition_fn = neumann_condition_fn
        self.source_samples = source_samples
        self.truncate = truncate

    def _contributing_status(self, status):
        return (status == WT.BOUNDARY_HIT) | (self.truncate & (status == WT.MAX_STEPS))

    def _terminal_value(self, state):
        # At a cap retain accumulated sources; only the unresolved solution
        # tail is zero. Never use boundary data as a value at an interior cap.
        boundary = jnp.where(state.status == WT.BOUNDARY_HIT,
                             self.boundary_condition_fn(state.position), 0.)
        return state.w + boundary

    def _average_sources(self, key, sample, *, count: int):
        """Average source draws in chunks of 32; storage is independent of budget."""
        width = min(count, 32)
        indices = jnp.arange(width)
        zeros = jax.tree.map(jnp.zeros_like, sample(jr.fold_in(key, 0)))

        def add_block(i, total):
            offsets = i*width+indices
            keys = jax.vmap(lambda n: jr.fold_in(key, n))(offsets)
            values = jax.vmap(sample)(keys)
            def add(a, b):
                mask = (offsets < count).reshape((width,)+(1,)*(b.ndim-1))
                return a+jnp.sum(jnp.where(mask, b, 0.), axis=0)
            return jax.tree.map(add, total, values)

        total = jax.lax.fori_loop(0, (count+width-1)//width, add_block, zeros)
        return jax.tree.map(lambda x: x/count, total)

    def integrate_batch(
        self,
        params,
        initial_position_batch,
        rng_key,
        init_trust_region_size,
        n_samples,
        *args,
        neumann_params,
        **kwargs,
    ):
        return super().integrate_batch(
            WoStParameters(params, neumann_params),
            initial_position_batch,
            rng_key,
            init_trust_region_size,
            n_samples,
            *args,
            **kwargs,
        )

    def pre_warm_compaction_buckets(
        self,
        params,
        initial_position_batch,
        rng_key,
        init_trust_region_size,
        n_samples,
        *args,
        neumann_params,
        **kwargs,
    ):
        return super().pre_warm_compaction_buckets(
            WoStParameters(params, neumann_params),
            initial_position_batch,
            rng_key,
            init_trust_region_size,
            n_samples,
            *args,
            **kwargs,
        )

    def step(self, params: WoStParameters, state: StarWalkerState) -> StarWalkerState:
        query = self.empty_sphere(params.problem, state.position, state.trust_region_size)
        return self._star_step(params, state, query)

    def _source_components(self, params, position, value_shape: tuple[int, ...]):
        source = jnp.asarray(self._source(params, position))
        if jnp.broadcast_shapes(source.shape, value_shape) != value_shape:
            raise ValueError(
                f"Source shape {source.shape} must broadcast to boundary value shape {value_shape}"
            )
        return jnp.broadcast_to(source, value_shape)

    def _box_flux_contribution(self, params, state, radius):
        """Nonzero Neumann estimator for the supported analytic box."""
        # Independent boundary sampling (WoSt 2023, Eq. 18/Algorithm 1).
        # Positive Green function for -Lap u=f and OUTWARD derivative h
        # gives +G*h/(alpha*pdf); the paper uses the opposite h convention.
        # fold_in leaves homogeneous walk directions/source samples intact.
        sample = sample_box_boundary_in_ball(
            jr.fold_in(state.rng_key, 0x4E4555), state.position, radius,
            params.neumann['min_corner'], params.neumann['max_corner'])
        # alpha is the fraction of the sphere inside the reflecting box.
        contacts = box_contact_count(params.neumann, state.position)
        weight = weighted_greens_fn_box_boundary(sample, radius) * jnp.exp2(
            contacts.astype(state.position.dtype))
        flux = jnp.asarray(self.neumann_condition_fn(params.problem, sample.position, sample.normal))
        if jnp.broadcast_shapes(flux.shape, state.w.shape) != state.w.shape:
            raise ValueError("Neumann flux must broadcast to the boundary value shape")
        return jnp.where(weight > 0, weight * flux, 0.)

    def _star_step(self, params, state, query):
        residual, clearance, next_tr = query
        status = walker_status(
            state, residual, clearance, next_tr, self.eps, self.max_steps
        )
        active = status == WT.ACTIVE
        # Recognize exact wall points every step, not only reported hits: starts,
        # tangent first-ball samples, and clear rays whose endpoint rounds onto
        # the wall (~1e-8 per step, enough to fail long solves).
        value = self.neumann_surface.signed_value(params.neumann, state.position)
        on_boundary = state.on_neumann | (value == 0)
        initial_normal = self.neumann_surface.normal_at(params.neumann, state.position)
        normal = jnp.where(state.on_neumann, state.normal, initial_normal)
        normal_valid = jnp.all(jnp.isfinite(normal)) & (
            jnp.abs(jnp.linalg.norm(normal) - 1) < 1e-4
        )
        cap = jnp.minimum(clearance, self.max_radius)
        # Zero clearance is an undecided Dirichlet query: like WoS, stay put
        # and retry with the next trust region.
        stalled = active & (cap == 0)
        query_cap = jnp.where(active & ~stalled, cap, 1.0)
        star = self.neumann_surface.star_radius(
            params.neumann,
            state.position,
            query_cap,
            on_boundary=on_boundary,
            normal=normal,
        )
        geometry_valid = (
            (star.status == T.EXIT)
            & jnp.isfinite(star.radius)
            & (star.radius > 0)
            & (star.radius <= query_cap)
            & (~on_boundary | normal_valid)
        )
        # Containment applies even to terminal Dirichlet and stalled points.
        status = jnp.where(~geometry_valid, WT.INVALID, status)
        moving = (status == WT.ACTIVE) & ~stalled
        radius = jnp.where(moving, star.radius, 1.0)
        direction_key, source_key, next_key = jr.split(state.rng_key, 3)
        direction = self.sphere_sampling_fn(direction_key, dtype=state.position.dtype)
        inward = self.neumann_surface.inward_direction(
            params.neumann, state.position, direction, normal
        )
        direction = jnp.where(on_boundary, inward, direction)
        hit = self.neumann_surface.intersect_ray(
            params.neumann, state.position, direction, radius, on_boundary=on_boundary
        )
        hit_valid = (hit.status == T.HIT) | (hit.status == T.EXIT)
        hit_valid &= (
            jnp.isfinite(hit.distance) & (hit.distance > 0) & (hit.distance <= radius)
        )
        hit_valid &= jnp.all(jnp.isfinite(hit.position))
        hit_valid &= (hit.status != T.HIT) | (
            jnp.all(jnp.isfinite(hit.normal))
            & (jnp.abs(jnp.linalg.norm(hit.normal) - 1) < 1e-4)
        )
        status = jnp.where(moving & ~hit_valid, WT.INVALID, status)
        active = status == WT.ACTIVE
        advance = active & ~stalled
        source_distance = radius * sample_green_radius(
            source_key, self.dim, state.position.dtype
        )
        source_position = state.position + source_distance * direction
        # Don't evaluate f outside its domain, even when its weight would be zero.
        inside_star = source_distance < hit.distance
        source_position = jnp.where(
            inside_star & advance, source_position, state.position
        )
        if self.source_samples == 1:
            source_value = self._source_components(params.problem, source_position, state.w.shape)
            contribution = jnp.where(inside_star, source_value * radius**2 / (2 * self.dim), 0.0)
        else:
            def sample_source(key):
                kr, kd = jr.split(key)
                d = self.sphere_sampling_fn(kd, dtype=state.position.dtype)
                d = jnp.where(on_boundary, self.neumann_surface.inward_direction(
                    params.neumann, state.position, d, normal), d)
                source_hit = self.neumann_surface.intersect_ray(
                    params.neumann, state.position, d, radius, on_boundary=on_boundary)
                distance = radius*sample_green_radius(kr, self.dim, state.position.dtype)
                inside = distance < source_hit.distance
                x = jnp.where(inside & advance, state.position+distance*d, state.position)
                value = self._source_components(params.problem, x, state.w.shape)
                valid = ((source_hit.status == T.HIT) | (source_hit.status == T.EXIT)) & jnp.isfinite(source_hit.distance)
                return jnp.where(valid, jnp.where(inside, value*radius**2/(2*self.dim), 0.), jnp.nan)
            contribution = self._average_sources(source_key, sample_source, count=self.source_samples)
        if self.neumann_condition_fn is not None:
            contribution += self._box_flux_contribution(params, state, radius)
        updated = state._replace(
            position=jnp.where(advance, hit.position, state.position),
            # Stalls count toward max_steps, as in WoS.
            nSteps=state.nSteps + active.astype(jnp.int32),
            w=jnp.where(advance, state.w + contribution, state.w),
            trust_region_size=jnp.where(active, next_tr, state.trust_region_size),
            rng_key=next_key,
            on_neumann=jnp.where(advance, hit.status == T.HIT, state.on_neumann),
            normal=jnp.where(advance, hit.normal, state.normal),
            status=jnp.asarray(status, jnp.int32),
            done_flag=status != WT.ACTIVE,
        )
        return finish_step(state, updated)

    def _get_initial_state(self, init_position_batch, rng_keys, init_tr, buffer_size):
        dtype = init_position_batch.dtype
        point = jax.ShapeDtypeStruct((self.dim,), dtype)
        shape = jnp.broadcast_shapes((1,), jax.eval_shape(self.boundary_condition_fn, point).shape)
        return StarWalkerState(
            position=init_position_batch,
            nSteps=jnp.zeros(buffer_size, jnp.int32),
            w=jnp.zeros((buffer_size,) + shape, dtype),
            trust_region_size=jnp.full((buffer_size,), init_tr, dtype),
            rng_key=rng_keys,
            done_flag=jnp.zeros(buffer_size, bool),
            status=jnp.zeros(buffer_size, jnp.int32),
            on_neumann=jnp.zeros(buffer_size, bool),
            normal=jnp.zeros_like(init_position_batch),
        )

    def _transition_state_phases(self, params, states, init_trust_region_size):
        return states

    def _compute_accumulator_update(self, params, state):
        return self._terminal_value(state)

    def _check_if_done(self, state, parents):
        return state.done_flag & (parents != -1)

    def _refill_state(
        self,
        current_states,
        is_valid_mask,
        free_indices,
        new_positions,
        new_keys,
        init_trust_region_size,
        N_total_samples,
    ):
        scatter = get_scatter_fn(free_indices, is_valid_mask, N_total_samples)
        fresh = self._get_initial_state(
            new_positions, new_keys, init_trust_region_size, new_positions.shape[0]
        )
        return jax.tree.map(
            lambda old, new: scatter(old, new, jnp.zeros(old.shape[1:], old.dtype)),
            current_states,
            fresh,
        )


class WoStGradientIntegrator(WoStIntegrator):
    """Spatial gradient from an interior first ball, then ordinary WoSt paths.

    The first radius is bounded by BOTH Dirichlet and Neumann clearance.
    Starts on either boundary are invalid. Zero-clearance retries count toward
    max_steps without consuming the first-ball estimator. Output shape follows
    WoSGradientIntegrator: (points, value components, dim).

    gradient_source_samples counts first-ball draws per walk, shared by an
    antithetic pair. source_samples independently counts continuation draws.

    control_fn(x) subtracts a fixed local value from the continuation estimate.
    With control_takes_params=True, call control_fn(problem_params, x); its
    output must broadcast to the boundary value shape. A stochastic pilot must
    be independent of the current gradient directions. This removes offset
    variance, not all continuation/source variance at small first radii.
    """

    def __init__(self, *args, control_fn=None, control_takes_params=False,
                 mirror_first_sphere=False, gradient_source_samples: int = 1, **kwargs):
        super().__init__(*args, **kwargs)
        if (isinstance(gradient_source_samples, bool)
                or not isinstance(gradient_source_samples, Integral) or gradient_source_samples < 1):
            raise ValueError('gradient_source_samples must be a positive integer')
        # Independent budgets for the two terms of Sawhney/Crane (2020), Eq.14.
        # Opposite first-sphere walks share this local source integral.
        self.gradient_source_samples = gradient_source_samples
        self.control_fn = control_fn
        self.mirror_first_sphere = mirror_first_sphere
        # A dynamic pilot travels with problem parameters, never a mutable JIT closure.
        # Parameterized controls must broadcast to the boundary value shape.
        self.control_takes_params = control_takes_params

    def _get_initial_state(self, positions, keys, init_tr, buffer_size):
        base = super()._get_initial_state(positions, keys, init_tr, buffer_size)
        dtype = positions.dtype
        point = jax.ShapeDtypeStruct((self.dim,), dtype)
        shape = base.w.shape[1:]
        if self.control_fn is not None and not self.control_takes_params:
            if jnp.broadcast_shapes(shape, jax.eval_shape(self.control_fn, point).shape) != shape:
                raise ValueError("Control must broadcast to the boundary value shape")
        return GradientStarWalkerState(
            *base,
            grad_weight=jnp.zeros((buffer_size, self.dim), dtype),
            grad_source=jnp.zeros((buffer_size,) + shape + (self.dim,), dtype),
        )

    def step(self, params, state):
        # A sampled unit direction always gives a nonzero gradient weight.
        # nSteps cannot identify initialization: zero-clearance retries count too.
        first = ~jnp.any(state.grad_weight != 0)
        # Masks, not lax.cond: under vmap cond runs both branches, which would
        # repeat the (possibly expensive) Dirichlet query.
        query = self.empty_sphere(params.problem, state.position, state.trust_region_size)
        initial = self._first_step(params, state, query)
        walked = self._star_step(params, state, query)
        return jax.tree.map(lambda a, b: jnp.where(first, a, b), initial, walked)

    def _step_block(self, params, states, steps, unroll):
        # The masked step computes the first-ball branch for every lane. After
        # one step, only stalled first balls still need it, and no lane can
        # newly need it before the next refill, so a block with none left runs
        # continuation-only steps (same states as the masked step).
        step = jax.vmap(self.step, in_axes=(None, 0))
        states = step(params, states)
        if steps == 1:
            return states
        pending = jnp.any(~jnp.any(states.grad_weight != 0, axis=-1) & ~states.done_flag)

        def continuation(state):
            query = self.empty_sphere(params.problem, state.position, state.trust_region_size)
            return self._star_step(params, state, query)

        def run(advance):
            return lambda s: jax.lax.fori_loop(1, steps, lambda i, t: advance(t), s, unroll=unroll)

        return jax.lax.cond(
            pending,
            run(lambda s: step(params, s)),
            run(jax.vmap(continuation)),
            states,
        )

    def _first_sphere_direction(self, key, dtype):
        """Continuation direction; a paired estimator may negate only this draw."""
        direction = self.sphere_sampling_fn(key, dtype=dtype)
        return -direction if self.mirror_first_sphere else direction

    def _first_ball_source_gradient(self, params, state, radius, source_key, direction_key):
        def sample(kr, kd):
            t = unit_interval(kr, state.position.dtype)
            d = self.sphere_sampling_fn(kd, dtype=state.position.dtype)
            f = self._source_components(params.problem, state.position+radius*t*d, state.w.shape)
            # Uniform radial sampling, including its PDF: Sawhney/Crane Eq. (14).
            _, gradient = first_ball_gradient_weights(self.dim, radius, t, d, d, f)
            return gradient

        if self.gradient_source_samples == 1:
            return sample(source_key, direction_key)
        return self._average_sources(source_key, lambda key: sample(*jr.split(key)),
                                     count=self.gradient_source_samples)

    def _first_step(self, params, state, query):
        residual, clearance, next_tr = query
        nvalue, nclearance, _ = self.neumann_surface.empty_sphere(
            params.neumann, state.position, state.trust_region_size
        )
        radius = jnp.minimum(jnp.minimum(clearance, nclearance), self.max_radius)
        status = walker_status(state, residual, radius, next_tr, self.eps, self.max_steps)
        valid = jnp.isfinite(nvalue) & (nvalue < 0) & jnp.isfinite(nclearance) & (nclearance >= 0)
        status = jnp.where(~valid | (status == WT.BOUNDARY_HIT), WT.INVALID, status)
        active = status == WT.ACTIVE
        advance = active & (radius > 0)
        radius = jnp.where(advance, radius, 0.0)
        sphere_key, source_key, direction_key, next_key = jr.split(state.rng_key, 4)
        direction = self._first_sphere_direction(sphere_key, state.position.dtype)
        source_weight = self._first_ball_source_gradient(params, state, radius, source_key, direction_key)
        weight = (self.dim/jnp.where(radius > 0, radius, 1.))*direction
        if self.control_fn is None:
            control = 0.0
        elif self.control_takes_params:
            control = self.control_fn(params.problem, state.position)
            if jnp.broadcast_shapes(jnp.shape(control), state.w.shape) != state.w.shape:
                raise ValueError("Parameterized control must broadcast to the boundary value shape")
        else:
            control = self.control_fn(state.position)
        updated = state._replace(
            position=state.position + radius * direction,
            nSteps=state.nSteps + active.astype(jnp.int32),
            w=jnp.where(advance, state.w - control, state.w),
            trust_region_size=jnp.where(active, next_tr, state.trust_region_size),
            rng_key=next_key,
            grad_weight=jnp.where(advance, weight, state.grad_weight),
            grad_source=jnp.where(advance, source_weight, state.grad_source),
            status=jnp.asarray(status, jnp.int32),
            done_flag=status != WT.ACTIVE,
        )
        return finish_step(state, updated)

    def _compute_accumulator_update(self, params, state):
        value = self._terminal_value(state)
        return state.grad_source + value[..., None] * state.grad_weight
