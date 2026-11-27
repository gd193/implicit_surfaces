"""Walk on Spheres: value, first-ball gradient, and differential (shape) estimators."""

import math
from typing import Any, Callable, Tuple

import jax
import jax.numpy as jnp

from sphereWalker.gradient import FirstBallGradientMixin
from sphereWalker.ops import mask_leaf, run_chunked
from sphereWalker.types import (DifferentialWalkerState, GradientWalkerState, WalkerState,
                                WalkerTermination as WT)
from sphereWalker.walk import WalkIntegrator
from utils.validation import check_int


class WoSIntegrator(WalkIntegrator[WalkerState]):
    """Walk on Spheres for -Lap u = f, u = g (scalar or vector g)."""

    def _get_initial_state(self, init_position_batch, rng_keys, init_tr, buffer_size) -> WalkerState:
        dtype = init_position_batch.dtype
        return WalkerState(
            position=init_position_batch,
            nSteps=jnp.zeros((buffer_size,), dtype=jnp.int32),
            w=jnp.zeros((buffer_size,) + self._value_shape(dtype), dtype=dtype),
            trust_region_size=jnp.full((buffer_size,), init_tr, dtype=dtype),
            rng_key=rng_keys,
            done_flag=jnp.zeros((buffer_size,), dtype=bool),
            status=jnp.zeros((buffer_size,), dtype=jnp.int32),
        )

    def _continuation_plan(self, params, state, query):
        return self._wos_step_plan(self._problem_params(params), state, query)

    def _continuation_step(self, params, state, query):
        return self._run_plan(self._problem_params(params), state,
                              *self._continuation_plan(params, state, query))


class WoSGradientIntegrator(FirstBallGradientMixin, WoSIntegrator):
    """WoS estimate of grad u; see FirstBallGradientMixin. Output (N, *value_shape, dim)."""

    _gradient_state_cls = GradientWalkerState


class DifferentialWoSIntegrator(WoSIntegrator):
    """Differential WoS: derivative of u with respect to shape parameters.

    Phase 0 walks to the boundary; the walker is offset inward along the
    normal and phase 1 estimates u there. Keeps a scalar w per walker so the
    boundary derivative can combine it with arbitrary PyTree parameters.
    """

    def __init__(self,
                 ### normal_fn: (params, position) -> normal_vector_at_position
                 normal_fn: Callable[[Any, jnp.ndarray], jnp.ndarray],
                 ### Df: (params, position, dirichlet_bc) -> dirichlet_boundary_differential
                 Df: Callable[[Any, jnp.ndarray, Callable[[jnp.ndarray], jnp.ndarray]], jnp.ndarray],
                 ### V_n: (params, position) -> params_derivative_at_position in normal direction
                 V_n: Callable[[Any, jnp.ndarray], Any],
                 offset_l: float,
                 offset_segment_fn: Callable[[Any, jnp.ndarray, jnp.ndarray], jax.Array] | None = None,
                 transition_bucket_size: int = 256,
                 **kwargs):
        super().__init__(**kwargs)
        if self.truncate:
            raise ValueError('differential WoS requires both boundary hits; truncate is unsupported')
        self.normal_fn = normal_fn
        self.Df = Df
        self.V_n = V_n
        self.offset_l = offset_l
        if not math.isfinite(self.eps) or self.eps <= 0:
            raise ValueError("eps must be finite and positive for offset selection")
        if not math.isfinite(self.offset_l) or self.offset_l <= self.eps:
            raise ValueError("offset_l must be finite and greater than eps")
        self.offset_segment_fn = offset_segment_fn
        # Walkers switching phase, and completed walkers' Df/V_n, are
        # evaluated in chunks of this size, so the cost scales with the number
        # of such walkers, not the buffer size.
        self.transition_bucket_size = check_int('transition_bucket_size', transition_bucket_size)

    def _get_initial_state(self, init_position_batch, rng_keys, init_tr, buffer_size):
        dtype = init_position_batch.dtype
        return DifferentialWalkerState(
            position=init_position_batch,
            boundary_hit_position=jnp.zeros((buffer_size, self.dim), dtype=dtype),
            nSteps=jnp.zeros((buffer_size,), dtype=jnp.int32),
            w=jnp.zeros((buffer_size,), dtype=dtype),
            trust_region_size=jnp.full((buffer_size,), init_tr, dtype=dtype),
            rng_key=rng_keys,
            done_flag=jnp.zeros((buffer_size,), dtype=bool),
            status=jnp.zeros((buffer_size,), dtype=jnp.int32),
            offset_length=jnp.zeros((buffer_size,), dtype=dtype),
            phase=jnp.zeros((buffer_size,), dtype=jnp.int32),  # start in phase 0
        )

    def _continuation_plan(self, params, state, query):
        """Phase 0 walks to the boundary; only phase 1 accumulates sources."""
        return self._wos_step_plan(params, state, query, accumulate_source=state.phase == 1)

    def _transition_walker(self, params, state: DifferentialWalkerState, init_trust_region_size,
                           active=True) -> DifferentialWalkerState:
        """Offset the position inward and switch to phase 1."""
        dtype = state.position.dtype
        normal = jnp.asarray(self.normal_fn(params, state.position), dtype=dtype)
        norm = jnp.linalg.norm(normal)
        normal_valid = jnp.isfinite(norm) & (norm > 0)
        normal = normal / jnp.where(normal_valid, norm, 1.)
        initial_tr = jnp.asarray(init_trust_region_size, dtype=dtype)
        active = jnp.asarray(active)

        def candidate_valid(trial_length):
            candidate = state.position - trial_length * normal
            residual, radius, next_tr = self.empty_sphere(params, candidate, initial_tr)
            # F < 0 is the interior convention. Radius is a geometric lower
            # distance bound: radius > eps proves clearance, a small radius
            # is only inconclusive. Also exclude immediate raw-residual hits
            # until the geometric termination TODO in walker_status is done.
            valid = (normal_valid & (trial_length > self.eps)
                     & jnp.all(jnp.isfinite(candidate)) & jnp.isfinite(residual)
                     & (residual < -self.eps) & jnp.isfinite(radius)
                     & (radius > self.eps) & jnp.isfinite(next_tr) & (next_tr > 0))
            if self.offset_segment_fn is not None:
                valid &= self.offset_segment_fn(params, state.position, candidate)
            return valid, candidate

        def cond(carry):
            attempt, accepted, _, _ = carry
            return active & ~accepted & (attempt < 4)

        def body(carry):
            attempt, _, point, length = carry
            trial_length = jnp.asarray(self.offset_l, dtype=dtype) * (
                jnp.asarray(0.5, dtype=dtype) ** attempt.astype(dtype))
            valid, candidate = candidate_valid(trial_length)
            return (attempt + jnp.int32(1), valid, jnp.where(valid, candidate, point),
                    jnp.where(valid, trial_length, length))

        _, accepted, point, length = jax.lax.while_loop(
            cond, body, (jnp.int32(0), jnp.asarray(False), state.position, jnp.asarray(0., dtype=dtype)))
        return state._replace(
            position=point,
            boundary_hit_position=state.position,
            offset_length=length,
            # Recompute the safe radius at the offset in the next walk step.
            trust_region_size=initial_tr,
            status=jnp.asarray(jnp.where(accepted, WT.ACTIVE, WT.INVALID), dtype=state.status.dtype),
            done_flag=~accepted,
            phase=jnp.asarray(jnp.where(accepted, 1, state.phase), dtype=state.phase.dtype))

    def _transition_state_phases(self, params, states, init_trust_region_size):
        """Offset walkers that just hit the boundary in phase 0."""
        needs = (states.status == WT.BOUNDARY_HIT) & (states.phase == 0)
        transition = jax.vmap(self._transition_walker, in_axes=(None, 0, None, 0))

        def body(idx, safe_idx, active, current):
            # Selected lanes are disjoint, so gather from the input states; the
            # tail repeats a real boundary hit, never a zero-filled walker.
            picked = jax.tree.map(lambda a: a[safe_idx], states)
            moved = transition(params, picked, init_trust_region_size, active)
            return jax.tree.map(lambda a, b: a.at[idx].set(b, mode='drop'), current, moved)

        return run_chunked(needs, self.transition_bucket_size, body, states)

    def _check_if_done(self, state, parents):
        return state.done_flag & ((state.phase == 1) | (state.status != WT.BOUNDARY_HIT)) & (parents != -1)

    def compute_boundary_derivative(self, params, boundary_hit_position: jnp.ndarray,
                                    solution_estimate: jnp.ndarray, offset_length: jnp.ndarray):
        """Boundary derivative contribution D(x) - V_n(x) (g - u) / l.

        Differential WoS Algorithm 2 line 11. Returns the params PyTree.
        """
        D_vec = self.Df(params, boundary_hit_position, self.boundary_condition_fn)
        Vn_vec = self.V_n(params, boundary_hit_position)
        g_val = self.boundary_condition_fn(boundary_hit_position)
        return jax.tree.map(lambda D, Vn: D - Vn * (g_val - solution_estimate) / offset_length, D_vec, Vn_vec)

    def _compute_accumulator_update(self, params, state: DifferentialWalkerState):
        return self.compute_boundary_derivative(
            params,
            state.boundary_hit_position,
            self._terminal_value(state),
            # Guard the division; only completed contributing walkers reach here.
            jnp.where(state.offset_length > 0, state.offset_length, 1.))

    def _accumulate_completed(self, params, current_states, current_parents, is_done,
                              sums, counts, drop_idx) -> Tuple[Any, jnp.ndarray]:
        """Evaluate Df and V_n only on completed contributing walkers in chunks."""
        successful = is_done & self._contributing_status(current_states.status)
        failed_done = is_done & ~successful
        failed_parents = jnp.where(failed_done, current_parents, drop_idx)
        counts = counts.at[failed_parents, current_states.status].add(
            failed_done.astype(counts.dtype), mode='drop')
        evaluate = jax.vmap(self._compute_accumulator_update, in_axes=(None, 0))

        def body(idx, safe_idx, active, carry):
            cur_sums, cur_counts = carry
            picked = jax.tree.map(lambda a: a[safe_idx], current_states)
            values = evaluate(params, picked)
            finite = active
            for leaf in jax.tree.leaves(values):
                finite &= jnp.all(jnp.isfinite(leaf).reshape((leaf.shape[0], -1)), axis=1)
            status = jnp.where(finite, picked.status, WT.INVALID)
            parents = jnp.where(active, current_parents[safe_idx], drop_idx)
            cur_sums = jax.tree.map(lambda acc, add: acc.at[parents].add(add, mode='drop'),
                                    cur_sums, jax.tree.map(lambda leaf: mask_leaf(leaf, finite), values))
            cur_counts = cur_counts.at[parents, status].add(active.astype(cur_counts.dtype), mode='drop')
            return cur_sums, cur_counts

        return run_chunked(successful, self.transition_bucket_size, body, (sums, counts))
