import jax 
import jax.numpy as jnp
import jax.random as jr

import time
import timeit
import math
from numbers import Integral
from typing import Callable, Tuple, TypeVar, Generic, Optional, Any, Union, cast
from functools import partial
import abc

from surfaces.queries import EmptySphereQuery
from sphereWalker.types import (WalkerState, WalkerOutput, DifferentialWalkerState,
                                WalkerTermination as WT, WalkerDiagnostics)
from utils.jnp_array_compaction import (
    uniform_size_generator,
    should_compact,
    compact_tree,
)
from utils.sampling_primitives import (
    sample_uniform_1sphere,
    sample_uniform_2sphere,
    sample_uniform_1ball,
    sample_uniform_2ball,
    weighted_greens_fn_1ball,
    weighted_greens_fn_2ball,
)


sphere_sampling_fn_select = {
    2: sample_uniform_1sphere,
    3: sample_uniform_2sphere,
}

ball_sampling_fn_select = {
    2: sample_uniform_1ball,
    3: sample_uniform_2ball,
}

weight_greens_fn_select = {
    2: weighted_greens_fn_1ball,
    3: weighted_greens_fn_2ball,
}

StateT = TypeVar('StateT', bound=Union[WalkerState, DifferentialWalkerState])


@partial(jax.jit, static_argnames=('new_size',))
def _compact_step(states, parents, is_active, new_size):
    """One full compaction transition jit'd as a single kernel.

    Replaces an eager sequence of `compact_tree(...)` + `parents.at[ti].get(...)` +
    `jnp.where(...)` whose individual ops were re-tracing per (input_size, new_size)
    pair on the WoS hot path. Bundled together they share one cache entry per
    `new_size`. Module-level so the JIT cache survives across BaseIntegrator
    instances.
    """
    new_states, valid_mask, target_indices = compact_tree(states, is_active, new_size)
    new_parents = parents.at[target_indices].get(mode='drop')
    new_parents = jnp.where(valid_mask, new_parents,
                             jnp.array(-1, dtype=new_parents.dtype))
    return new_states, new_parents


def walker_status(state, residual, radius, next_tr, eps, max_steps):
    """A terminal condition is not automatically a boundary hit."""
    # TODO: Replace residual-based termination with a bounded closest-point
    # query: minimize ||y - x||^2 subject to F(y) = 0 using interval bounds
    # and branch-and-bound. See Huang, "Geometric Queries on Closed Implicit
    # Surfaces for Walk on Stars", Sec. 3.1, Eq. (1):
    # algo_paper/3757376.3771378.pdf (doi:10.1145/3757376.3771378).
    # Retain cheap empty-sphere queries for walking; return a boundary point,
    # outward normal, distance lower/upper bounds, and convergence status.
    # A lower bound supports safe steps; an upper bound <= eps proves proximity.
    # Use the located boundary point for g, D, V_n, and the differential offset,
    # checking that the offset stays inside the domain. Projection failure must
    # remain explicit. Only tighten the pruning upper bound with a feasible
    # point or certified root-containing box; an UNKNOWN box is insufficient.
    hit = jnp.abs(residual) <= eps
    exhausted = state.nSteps >= max_steps
    valid = (jnp.isfinite(residual) & jnp.isfinite(radius) & (radius >= 0)
             & jnp.all(jnp.isfinite(state.position)) & jnp.all(jnp.isfinite(state.w)))
    valid &= hit | exhausted | (jnp.isfinite(next_tr) & (next_tr > 0))
    return jnp.asarray(jnp.where(~valid, WT.INVALID,
                       jnp.where(hit, WT.BOUNDARY_HIT,
                       jnp.where(exhausted, WT.MAX_STEPS, WT.ACTIVE))), dtype=jnp.int32)


def finish_step(state, updated):
    """Preserve terminal states until they are reduced or change phase."""
    # Loop-carried state keeps the input geometry precision even under x64.
    updated = jax.tree.map(lambda old, new: jnp.asarray(new, dtype=old.dtype), state, updated)
    finite = jnp.all(jnp.isfinite(updated.position)) & jnp.all(jnp.isfinite(updated.w))
    updated = updated._replace(status=jnp.where(finite, updated.status, WT.INVALID),
                               done_flag=updated.done_flag | ~finite)
    return jax.tree.map(lambda old, new: jnp.where(state.done_flag, old, new), state, updated)


class BaseIntegrator(Generic[StateT], abc.ABC):
    """
    Abstract base class for Walk-on-Spheres integrators.
    """
    def __init__(self,
                 empty_sphere_fn: Callable[[Any, jnp.ndarray, jnp.ndarray], Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]],
                 boundary_condition_fn: Callable[[jnp.ndarray], jnp.ndarray],
                 source_term_fn: Callable[[jnp.ndarray], jnp.ndarray],
                 dim: int,
                 level_set: float = 0.0,
                 max_steps: int = 1000,
                 eps: float = 1e-3,
                 print_progress: bool = False,):
        self.empty_sphere = EmptySphereQuery(empty_sphere_fn)
        self.boundary_condition_fn = boundary_condition_fn
        self.source_term = source_term_fn
        self.level_set = level_set
        self.dim = dim
        self.max_steps = max_steps
        self.eps = eps
        self.sphere_sampling_fn = sphere_sampling_fn_select[dim]
        self.ball_sampling_fn = ball_sampling_fn_select[dim]
        self.weighted_greens_fn = weight_greens_fn_select[dim]
        self.print_progress = print_progress

    @abc.abstractmethod
    def step(self, params: Any, state: StateT) -> StateT:
        pass

    @abc.abstractmethod
    def _get_initial_state(self, init_position_batch: jnp.ndarray, rng_keys: jnp.ndarray, init_tr: float, buffer_size: int) -> StateT:
        pass

    def _get_accumulator_initial(self, params: Any, positions: jnp.ndarray) -> Any:
        # Infer array/PyTree output structure without indexing an actual point
        # or evaluating a boundary callback at an invented location.
        dummy_state = self._get_initial_state(
            jnp.zeros((1, self.dim), dtype=positions.dtype), jr.PRNGKey(0)[None, :], 1., 1)
        shape = jax.eval_shape(
            jax.vmap(self._compute_accumulator_update, in_axes=(None, 0)), params, dummy_state)
        return jax.tree.map(lambda leaf: create_zero_buffer(leaf, positions.shape[0]), shape)

    def _prepare_positions(self, positions, n_samples, steps_between_refill,
                           max_active_states, jit_buck_size, n_compactions):
        for name, value in (('n_samples', n_samples), ('steps_between_refill', steps_between_refill),
                            ('max_active_states', max_active_states), ('jit_buck_size', jit_buck_size)):
            if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
                raise ValueError(f'{name} must be a positive integer')
        if isinstance(n_compactions, bool) or not isinstance(n_compactions, Integral) or n_compactions < 0:
            raise ValueError('n_compactions must be a nonnegative integer')
        positions = jnp.asarray(positions, dtype=jnp.result_type(positions, jnp.float32))
        if positions.ndim != 2 or positions.shape[1] != self.dim:
            raise ValueError('positions must have shape (N, dim)')
        return positions

    @abc.abstractmethod
    def _transition_state_phases(self, params: Any, states: StateT, init_trust_region_size: float) -> StateT:
        pass

    @abc.abstractmethod
    def _compute_accumulator_update(self, params: Any, state: StateT) -> Any:
        pass

    @abc.abstractmethod
    def _check_if_done(self, state: StateT, parents: jnp.ndarray) -> jnp.ndarray:
        pass

    @abc.abstractmethod
    def _refill_state(self, current_states: StateT, is_valid_mask: jnp.ndarray,
                      free_indices: jnp.ndarray, new_positions: jnp.ndarray, new_keys: jnp.ndarray,
                      init_trust_region_size: float, N_total_samples: int) -> StateT:
        pass

    @partial(jax.jit, static_argnames=['self','active_buffer_size', 'N_total_samples', 'n_samples', 'steps_between_refill', 'unroll_loop'])
    def _process_active_states(self,
                               # dynamic state
                               current_states: StateT,
                               current_parents: jnp.ndarray,
                               current_accumulated_w: Any,
                               current_accumulated_nSteps: jnp.ndarray,
                               next_task_index: jnp.ndarray,
                               # dynamic context
                               params: Any,
                               initial_position_batch: jnp.ndarray,
                               rng_key: jnp.ndarray,
                               # Static configuration
                               init_trust_region_size: float,
                               active_buffer_size: int,
                               N_total_samples: int,
                               n_samples: int,
                               steps_between_refill: int,
                               unroll_loop: bool,
                               ):
        if self.print_progress:
            # Use jax.debug.print to see this during execution, or print() to see during trace
            print(f"🔄 [JIT TRACING] _process_active_states called with N_total_samples={N_total_samples}")
            
        current_states = self._transition_state_phases(params, current_states, init_trust_region_size)
        is_done = self._check_if_done(current_states, current_parents)

        sums, counts = current_accumulated_w
        to_accumulate = jax.vmap(self._compute_accumulator_update, in_axes=(None, 0))(params, current_states)
        successful = is_done & (current_states.status == WT.BOUNDARY_HIT)
        finite = jnp.ones_like(successful)
        for leaf in jax.tree.leaves(to_accumulate):
            finite &= jnp.all(jnp.isfinite(leaf).reshape((leaf.shape[0], -1)), axis=1)
        terminal_status = jnp.where(successful & ~finite, WT.INVALID, current_states.status)
        successful &= finite
        
        # Accumulate results
        # Use N_total_samples + 1 as safe drop index for logic, 
        # but for accumulation into buffers of size N_positions, we need a safe index relative to N_positions.
        # We use current_accumulated_nSteps.shape[0] as the safe out-of-bounds index.
        drop_idx = current_accumulated_nSteps.shape[0]
        
        done_parents = jnp.where(is_done, current_parents, drop_idx)
        done_nSteps = jnp.where(is_done, current_states.nSteps, 0)
        done_w = jax.tree.map(lambda leaf: mask_leaf(leaf, successful), to_accumulate)

        updated_accumulated_w = jax.tree.map(
            lambda acc_leaf, to_add_leaf: acc_leaf.at[done_parents].add(to_add_leaf, mode='drop'),
            sums,
            done_w
        )
        updated_accumulated_nSteps = current_accumulated_nSteps.at[done_parents].add(done_nSteps, mode='drop')

        updated_counts = counts.at[done_parents, terminal_status].add(
            is_done.astype(counts.dtype), mode='drop')
        updated_accumulated_w = (updated_accumulated_w, updated_counts)

        # Refill active states
        free_indices_mask = jnp.nonzero(is_done, size=active_buffer_size, fill_value=active_buffer_size)[0]
        n_free = jnp.sum(is_done, dtype=next_task_index.dtype)

        offsets = jnp.arange(active_buffer_size, dtype=next_task_index.dtype)
        candidate_indices = next_task_index + offsets
        valid_candidates_mask = jnp.logical_and(candidate_indices < N_total_samples, offsets < n_free)

        # Prepare new tasks
        new_parents_val = candidate_indices // n_samples
        safe_new_parents_read = jnp.where(valid_candidates_mask, new_parents_val, 0)
        new_positions_val = initial_position_batch[safe_new_parents_read, :]
        new_keys_val = jax.vmap(lambda i: jr.fold_in(rng_key, i))(candidate_indices)

        # Update parents array
        next_parents = current_parents.at[free_indices_mask].set(-1, mode='drop')
        target_indices = jnp.where(valid_candidates_mask, free_indices_mask, active_buffer_size)
        next_parents = next_parents.at[target_indices].set(new_parents_val, mode='drop')

        # Refill states (Delegate to concrete class)
        refilled_states = self._refill_state(
            current_states,
            valid_candidates_mask,
            free_indices_mask,
            new_positions_val,
            new_keys_val,
            init_trust_region_size,
            N_total_samples
        )

        # Step loop
        def loop_body(i, states):
           return jax.vmap(self.step, in_axes=(None, 0))(params, states)
        
        final_states = jax.lax.fori_loop(0, steps_between_refill, loop_body, refilled_states, unroll=unroll_loop)

        return final_states, next_parents, updated_accumulated_w, updated_accumulated_nSteps, n_free

    def pre_warm_compaction_buckets(self,
                                    params: Any,
                                    initial_position_batch: jnp.ndarray,
                                    rng_key: jnp.ndarray,
                                    init_trust_region_size: float,
                                    n_samples: int,
                                    steps_between_refill: int = 50,
                                    unroll_loop: bool = False,
                                    max_active_states: int = 1000**2,
                                    jit_buck_size: int = 1000,
                                    n_compactions: int = 20) -> None:
        """Force one JIT compile per compaction-bucket size.

        `_process_active_states` is jitted with `active_buffer_size` as a
        static_argname, so each unique buffer size triggers a fresh compile.
        Random-key warmup runs visit only the buckets the walker absorbing-
        time distribution happens to hit, leaving the others uncached. This
        method invokes `_process_active_states` once per bucket size with
        synthetic state, guaranteeing the JIT cache covers all of them
        before timed runs.
        """
        initial_position_batch = self._prepare_positions(
            initial_position_batch, n_samples, steps_between_refill,
            max_active_states, jit_buck_size, n_compactions)
        if initial_position_batch.shape[0] == 0:
            return
        positions_padded, N_original = _pad_to_bucket(initial_position_batch, bucket_size=jit_buck_size)
        N_positions = positions_padded.shape[0]
        # Padding only stabilizes storage shapes; it is not extra work.
        N_total_samples = N_original * n_samples
        active_buffer_size = min(N_total_samples, max_active_states)
        compaction_sizes = uniform_size_generator(active_buffer_size, n_compactions)

        accumulated_w = (self._get_accumulator_initial(params, positions_padded),
                         jnp.zeros((N_positions, len(WT)), dtype=jnp.int32))
        accumulated_nSteps = jnp.zeros((N_positions,), dtype=jnp.int32)
        next_task_index_dummy = jnp.array(N_total_samples, dtype=jnp.int32)
        # All-valid parents so the kernel takes its main branch (some processing
        # happens, no refills since next_task_index ≥ N_total).
        for buf in compaction_sizes:
            if buf == 0:
                continue
            dummy_keys = jax.vmap(lambda i: jr.fold_in(rng_key, i))(jnp.arange(buf))
            dummy_pos = jnp.broadcast_to(positions_padded[0:1], (buf, positions_padded.shape[1]))
            dummy_state = self._get_initial_state(
                dummy_pos, dummy_keys, init_trust_region_size, buf,
            )
            dummy_parents = jnp.zeros((buf,), dtype=jnp.int32)
            out = self._process_active_states(
                dummy_state, dummy_parents,
                accumulated_w, accumulated_nSteps,
                next_task_index_dummy,
                params, positions_padded, rng_key,
                init_trust_region_size,
                int(buf),                     # active_buffer_size — STATIC
                N_total_samples, n_samples,
                steps_between_refill, unroll_loop,
            )
            jax.block_until_ready(out[0].position)

        # `compact_tree` is jit-cached on `new_size` only (fill_index_value was
        # dropped — was unused). Pre-compile one entry per bucket size on a
        # state of the largest buffer; downstream transitions from any larger
        # buffer to this `new_size` reuse the same compile.
        nonzero_sizes = [b for b in compaction_sizes if b > 0]

        def _wait_all(state_out, parents_out):
            jax.tree_util.tree_map(jax.block_until_ready, state_out)
            jax.block_until_ready(parents_out)

        if nonzero_sizes:
            big = nonzero_sizes[0]
            big_keys = jax.vmap(lambda i: jr.fold_in(rng_key, 100 + i))(jnp.arange(big))
            big_pos = jnp.broadcast_to(positions_padded[0:1], (big, positions_padded.shape[1]))
            big_state = self._get_initial_state(big_pos, big_keys, init_trust_region_size, big)
            big_parents = jnp.zeros((big,), dtype=jnp.int32)
            is_active_big = jnp.ones((big,), dtype=jnp.bool_)
            for new in nonzero_sizes[1:]:
                _wait_all(*_compact_step(big_state, big_parents, is_active_big, int(new)))
            # Pre-warm transitions from each non-`big` size — JIT cache key
            # includes input array shapes which scale with the source buffer.
            for old in nonzero_sizes[1:-1]:
                old_keys = jax.vmap(lambda i: jr.fold_in(rng_key, 200 + i))(jnp.arange(old))
                old_pos = jnp.broadcast_to(positions_padded[0:1], (old, positions_padded.shape[1]))
                old_state = self._get_initial_state(old_pos, old_keys, init_trust_region_size, old)
                old_parents = jnp.zeros((old,), dtype=jnp.int32)
                is_active_old = jnp.ones((old,), dtype=jnp.bool_)
                idx_old = nonzero_sizes.index(old)
                for new in nonzero_sizes[idx_old + 1:]:
                    _wait_all(*_compact_step(old_state, old_parents, is_active_old, int(new)))


    def integrate_batch(self,
                        params: jnp.ndarray,
                        initial_position_batch: jnp.ndarray,
                        rng_key: jnp.ndarray,
                        init_trust_region_size: float,
                        n_samples: int,
                        steps_between_refill: int=50,
                        unroll_loop: bool=False,
                        max_active_states: int=1000**2,
                        jit_buck_size: int=1000,
                        n_compactions: int=0,
                        diag_log: Optional[list]=None, *,
                        return_details: bool=False) -> WalkerOutput | WalkerDiagnostics:
        """Estimate only from completed walks; raise on any incomplete sample.

        return_details=True returns successful contribution sums and per-position
        termination counts instead of an estimate. It does not average a biased
        subset of successful walks. max_steps is a total per-sample budget,
        shared by both phases of differential WoS.

        diag_log: if a list is passed, each outer iteration appends a dict
            with `{outer_idx, buffer_size, n_active, ms_total, ms_compact,
            ms_process, ms_sync}`. The caller can use this to break the wall
            time into per-step kernel work vs Python/host overhead and per-
            bucket-size cost. Pass None to skip instrumentation entirely.
        """

        initial_position_batch = self._prepare_positions(
            initial_position_batch, n_samples, steps_between_refill,
            max_active_states, jit_buck_size, n_compactions)
        if initial_position_batch.shape[0] == 0:
            sums = self._get_accumulator_initial(params, initial_position_batch)
            steps = jnp.zeros(0, dtype=initial_position_batch.dtype)
            if return_details:
                return WalkerDiagnostics(initial_position_batch, sums, jnp.zeros(0, jnp.int32),
                                         jnp.zeros((0, len(WT)), jnp.int32), steps)
            return WalkerOutput(initial_position_batch, steps, sums)

        # 1. Pad Positions
        positions_padded, N_original = _pad_to_bucket(initial_position_batch, bucket_size=jit_buck_size)

        N_positions = positions_padded.shape[0]
        # Padding only stabilizes storage shapes; it is not extra work.
        N_total_samples = N_original * n_samples
        active_buffer_size = min(N_total_samples, max_active_states)

        # 3. Init Global Output
        accumulated_w = (self._get_accumulator_initial(params, positions_padded),
                         jnp.zeros((N_positions, len(WT)), dtype=jnp.int32))
        accumulated_nSteps = jnp.zeros((N_positions,), dtype=jnp.int32)

        init_task_indices = jnp.arange(active_buffer_size, dtype=jnp.int32)
        active_parent_indices = init_task_indices // n_samples
        
        # Force strong type for loop variable
        next_task_index = jnp.array(active_buffer_size, dtype=jnp.int32)

        rng_keys_init = jax.vmap(lambda i: jr.fold_in(rng_key, i))(init_task_indices)
        active_states = self._get_initial_state(
            positions_padded[active_parent_indices, :],
            rng_keys_init,
            init_trust_region_size,
            active_buffer_size
        )

        # Compaction schedule. WoS path lengths are highly uneven (random walks
        # have a heavy tail), so during the drain phase a few stragglers can pin
        # a wide buffer. `compaction_sizes` is a descending list of buffer
        # widths; we walk down it as the active set fits each smaller bucket.
        # `n_compactions=0` yields a single-size schedule = current behavior.
        compaction_sizes = uniform_size_generator(active_buffer_size, n_compactions)
        current_compaction_index = 0
        current_buffer_size = compaction_sizes[current_compaction_index]

        process_calls = 0
        while True:
            t_iter = time.perf_counter() if diag_log is not None else None
            # Reading active_parent_indices != -1 forces a host sync; we
            # measure that cost as part of `ms_sync` for diagnostics.
            t_sync_start = time.perf_counter() if diag_log is not None else None
            active_count_int = int(jnp.sum(active_parent_indices != -1))
            ms_sync = (time.perf_counter() - t_sync_start) * 1000 if diag_log is not None else 0.0
            if active_count_int == 0 and next_task_index >= N_total_samples:
                break

            # Drain-phase compaction: only kick in once dispatch is done (pending
            # tasks exhausted). Until then, completed slots are refilled with
            # fresh tasks and the buffer should stay full.
            t_compact_start = time.perf_counter() if diag_log is not None else None
            ms_compact = 0.0
            if (n_compactions > 0
                    and next_task_index >= N_total_samples
                    and current_compaction_index < len(compaction_sizes) - 1):
                will_compact, new_idx, new_size = should_compact(
                    active_count_int, current_compaction_index, compaction_sizes,
                )
                if will_compact and new_size > 0:
                    is_active_mask = active_parent_indices != -1
                    active_states, active_parent_indices = _compact_step(
                        active_states, active_parent_indices, is_active_mask, int(new_size),
                    )
                    current_compaction_index = new_idx
                    current_buffer_size = new_size
                    if self.print_progress:
                        print(f"  → compacted to buffer size {current_buffer_size}")
            if diag_log is not None:
                # block_until_ready on the compacted result so compaction wall
                # time is attributed to ms_compact, not to ms_process below.
                jax.block_until_ready(active_parent_indices)
                ms_compact = (time.perf_counter() - t_compact_start) * 1000

            t_process_start = time.perf_counter() if diag_log is not None else None
            active_states, active_parent_indices, accumulated_w, accumulated_nSteps, n_filled = self._process_active_states(
                active_states,
                active_parent_indices,
                accumulated_w,
                accumulated_nSteps,
                next_task_index,
                params,
                positions_padded,
                rng_key,
                init_trust_region_size,
                current_buffer_size,
                N_total_samples,
                n_samples,
                steps_between_refill,
                unroll_loop,
            )
            if diag_log is not None:
                jax.block_until_ready(active_states.position)
                ms_process = (time.perf_counter() - t_process_start) * 1000
                ms_total = (time.perf_counter() - t_iter) * 1000
                diag_log.append(dict(
                    outer_idx=process_calls,
                    buffer_size=int(current_buffer_size),
                    n_active=int(active_count_int),
                    next_task_index=int(next_task_index),
                    ms_total=ms_total,
                    ms_compact=ms_compact,
                    ms_process=ms_process,
                    ms_sync=ms_sync,
                ))

            process_calls += 1
            next_task_index += n_filled
            if self.print_progress:
                print(f"Remaining Active States: {jnp.sum(active_parent_indices != -1)} | Next Task Index: {next_task_index}/{N_total_samples}  (buffer={current_buffer_size})")
                print(f"Process called so far. {process_calls}")

        # Counts include all requested samples; failed samples never enter sums.
        sums, counts = accumulated_w
        sums = jax.tree.map(lambda leaf: leaf[:N_original], sums)
        counts = counts[:N_original]
        final_nSteps = accumulated_nSteps[:N_original] / n_samples
        final_positions = positions_padded[:N_original, :]
        details = WalkerDiagnostics(final_positions, sums, counts[:, WT.BOUNDARY_HIT],
                                    counts, final_nSteps)
        if return_details:
            return details
        if not bool(jnp.all(details.complete)):
            exhausted = int(jnp.sum(counts[:, WT.MAX_STEPS]))
            invalid = int(jnp.sum(counts[:, WT.INVALID]))
            raise RuntimeError(f'incomplete WoS sampling: {exhausted} exhausted, {invalid} invalid; '
                               'use return_details=True to inspect partial sums and counts')
        final_w = jax.tree.map(lambda leaf: leaf / n_samples, sums)
        return WalkerOutput(position=final_positions, nSteps=final_nSteps, w=final_w)


class WoSIntegrator(BaseIntegrator[WalkerState]):
    """
    Walk on Spheres integrator for solving boundary value problems.
    """
    def step(self, params, state: WalkerState) -> WalkerState:
        """
        Perform a single WoS step for the walker given its current state.
        """
        ### compute empty sphere
        fn_val, radius, new_trust_region_size = self.empty_sphere(params, state.position, state.trust_region_size)

        ### sample new position on sphere and inside ball
        sphere_key, ball_key, new_key = jr.split(state.rng_key, 3)
        sample_sphere = self.sphere_sampling_fn(sphere_key, dtype=state.position.dtype) * radius
        sample_ball_radius, sample_ball_direction = self.ball_sampling_fn(ball_key, dtype=state.position.dtype)
        sample_ball_radius *= radius
        sample_ball_position = state.position + (sample_ball_radius * sample_ball_direction)

        ### update state
        status = walker_status(state, fn_val, radius, new_trust_region_size, self.eps, self.max_steps)
        is_done = status != WT.ACTIVE
        new_nSteps = jnp.where(is_done, state.nSteps, state.nSteps + 1)
        new_position = jnp.where(is_done, state.position,
                                 state.position + sample_sphere)
        # Guard the source-Greens accumulation against radius=0 (empty-sphere classifier
        # failed → no ball to integrate over → contribution is 0, not NaN). Without this,
        # weighted_greens(0, 0) → NaN poisons state.w forever.
        source_contrib = jnp.where(radius > 0,
                                   self.source_term(sample_ball_position) *
                                       self.weighted_greens_fn(r=sample_ball_radius, R=radius),
                                   jnp.asarray(0.0, dtype=state.w.dtype))
        new_w = jnp.where(is_done, state.w, state.w + source_contrib)
        new_nSteps, new_position, new_w = cast(jnp.ndarray, new_nSteps), cast(jnp.ndarray, new_position), cast(jnp.ndarray, new_w) # type hints b.c. of jnp.where
        
        
        updated = WalkerState(position=new_position,
                           nSteps=new_nSteps, 
                           w=new_w, 
                           trust_region_size=new_trust_region_size, 
                           rng_key=new_key,
                           done_flag=is_done, status=status)
        return finish_step(state, updated)
    
    def _get_initial_state(self, init_position_batch, rng_keys, init_tr, buffer_size) -> WalkerState:
        return WalkerState(
            position=init_position_batch,
            nSteps=jnp.zeros((buffer_size,), dtype=jnp.int32),
            w=jnp.zeros((buffer_size, 1), dtype=init_position_batch.dtype),
            trust_region_size=jnp.full((buffer_size,), init_tr, dtype=init_position_batch.dtype),
            rng_key=rng_keys,
            status=jnp.zeros((buffer_size,), dtype=jnp.int32),
            done_flag=jnp.zeros((buffer_size,), dtype=bool)
        )
    
    def _transition_state_phases(self, params: Any, states: WalkerState, init_trust_region_size: float) -> WalkerState:
        """
        No transition in standard WoS
        """
        return states  # No transition in standard WoS
    
    def _compute_accumulator_update(self, params: Any, state: WalkerState) -> jnp.ndarray:
        """
        Compute the accumulator update upon walker completion.
        """
        return state.w + self.boundary_condition_fn(state.position)
    
    def _check_if_done(self, state: WalkerState, parents: jnp.ndarray) -> jnp.ndarray:
        """
        Standard Wos: is done if done_flag is True
        """
        return jnp.logical_and(state.done_flag, parents != -1)
    
    
    def _refill_state(self,
                      current_states: WalkerState,
                      is_valid_mask: jnp.ndarray,
                      free_indices: jnp.ndarray,
                      new_positions: jnp.ndarray,
                      new_keys: jnp.ndarray,
                      init_trust_region_size: float,
                      N_total_samples: int) -> WalkerState:
        """
        Refill the walker state buffer with new tasks.
        """
        scatter = get_scatter_fn(free_indices, is_valid_mask, N_total_samples)
        
        next_positions = scatter(current_states.position, new_positions, jnp.zeros((self.dim,), dtype=current_states.position.dtype))
        next_nSteps = scatter(current_states.nSteps, 0, reset_value=0)
        next_w = scatter(current_states.w, 0.0, reset_value=jnp.zeros(current_states.w.shape[1:], dtype=current_states.w.dtype))
        next_trust_region_size = scatter(current_states.trust_region_size, init_trust_region_size, reset_value=init_trust_region_size)
        next_rng_keys = scatter(current_states.rng_key, new_keys, reset_value=jr.PRNGKey(0))
        next_done = current_states.done_flag.at[jnp.where(is_valid_mask, free_indices, N_total_samples + 1)].set(False, mode='drop')
        
        return WalkerState(
            position=next_positions,
            nSteps=next_nSteps,
            w=next_w,
            trust_region_size=next_trust_region_size,
            rng_key=next_rng_keys,
            status=current_states.status.at[jnp.where(is_valid_mask, free_indices, N_total_samples + 1)].set(WT.ACTIVE, mode='drop'),
            done_flag=next_done
        )


class DifferentialWoSIntegrator(BaseIntegrator[DifferentialWalkerState]):
    """
    Differential Walk on Spheres integrator for solving boundary value problems with differential Wos.
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
        self.normal_fn = normal_fn
        self.Df = Df
        self.V_n = V_n
        self.offset_l = offset_l
        if not math.isfinite(self.eps) or self.eps <= 0:
            raise ValueError("eps must be finite and positive for offset selection")
        if not math.isfinite(self.offset_l) or self.offset_l <= self.eps:
            raise ValueError("offset_l must be finite and greater than eps")
        self.offset_segment_fn = offset_segment_fn
        if (isinstance(transition_bucket_size, bool) or not isinstance(transition_bucket_size, Integral)
                or transition_bucket_size < 1):
            raise ValueError("transition_bucket_size must be a positive integer")
        # Walkers switching phase are offset in chunks of this size, so the
        # cost scales with the number of switches, not the buffer size.
        self.transition_bucket_size = int(transition_bucket_size)

    def step(self, params, state: DifferentialWalkerState) -> DifferentialWalkerState:
        """
        Perform differential WoS step for the walker given its current state.
        Phase 0: Walk -> Hit Boundary -> Offset Position -> Switch Walker to Phase 1
        Phase 1: Walk -> Hit Boundary -> Done
        """
        # The geometry backend owns trust-region adaptation and its limits.
        fn_val, radius, new_trust_region_size = self.empty_sphere(
            params, state.position, state.trust_region_size)

        ### sample new position on sphere and inside ball
        sphere_key, ball_key, new_key = jr.split(state.rng_key, 3)
        sample_sphere = self.sphere_sampling_fn(sphere_key, dtype=state.position.dtype) * radius
        sample_ball_radius, sample_ball_direction = self.ball_sampling_fn(ball_key, dtype=state.position.dtype)
        sample_ball_radius *= radius
        sample_ball_position = state.position + (sample_ball_radius * sample_ball_direction)

        status = walker_status(state, fn_val, radius, new_trust_region_size, self.eps, self.max_steps)
        is_done = status != WT.ACTIVE
        new_nSteps = jnp.where(is_done, state.nSteps, state.nSteps + 1)
        new_position = jnp.where(is_done, state.position,
                                 state.position + sample_sphere)
        new_position, new_nSteps = cast(jnp.ndarray, new_position), cast(jnp.ndarray, new_nSteps) # type hints b.c. of jnp.where
        
        ### only update weight in phase 1 and if not done
        new_w = jnp.where(jnp.logical_or(is_done, state.phase == 0), state.w,
                    state.w + self.source_term(sample_ball_position) * self.weighted_greens_fn(r=sample_ball_radius, R=radius))
        
        updated = DifferentialWalkerState(
            position=new_position,
            boundary_hit_position=state.boundary_hit_position,
            nSteps=new_nSteps, 
            w=new_w, 
            trust_region_size=new_trust_region_size, 
            rng_key=new_key,
            done_flag=is_done, status=status,
            offset_length=state.offset_length,
            phase=state.phase
        )
        return finish_step(state, updated)
    
    
    def _get_initial_state(self, init_position_batch, rng_keys, init_tr, buffer_size):
        return DifferentialWalkerState(
            position=init_position_batch,
            boundary_hit_position=jnp.zeros((buffer_size, self.dim), dtype=init_position_batch.dtype),
            nSteps=jnp.zeros((buffer_size,), dtype=jnp.int32),
            w=jnp.zeros((buffer_size,), dtype=init_position_batch.dtype),
            trust_region_size=jnp.full((buffer_size,), init_tr, dtype=init_position_batch.dtype),
            rng_key=rng_keys,
            status=jnp.zeros((buffer_size,), dtype=jnp.int32),
            done_flag=jnp.zeros((buffer_size,), dtype=bool),
            offset_length=jnp.zeros((buffer_size,), dtype=init_position_batch.dtype),
            phase=jnp.zeros((buffer_size,), dtype=jnp.int32) # Start in phase 0
        )


    def _transition_walker(self, params, state: DifferentialWalkerState, init_trust_region_size,
                           active=True) -> DifferentialWalkerState:
        """
        Offset the position and switch to phase 1.
        """
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
            # until the geometric termination TODO above is implemented.
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
        """
        Offset walkers that just hit the boundary in phase 0.        """
        needs = (states.status == WT.BOUNDARY_HIT) & (states.phase == 0)
        n_states = needs.shape[0]
        bucket = min(self.transition_bucket_size, n_states)
        count = jnp.sum(needs, dtype=jnp.int32)
        # Pad so every chunk slice stays in bounds; padding points past the
        # buffer, which gathers fill and scatters drop.
        indices = jnp.nonzero(needs, size=n_states, fill_value=n_states)[0].astype(jnp.int32)
        indices = jnp.concatenate([indices, jnp.full((bucket,), n_states, jnp.int32)])
        transition = jax.vmap(self._transition_walker, in_axes=(None, 0, None, 0))

        def cond(carry):
            chunk, _ = carry
            return chunk * bucket < count

        def body(carry):
            chunk, current = carry
            idx = jax.lax.dynamic_slice(indices, (chunk * bucket,), (bucket,))
            active = idx < n_states
            picked = jax.tree.map(lambda a: a.at[idx].get(mode='fill', fill_value=0), current)
            moved = transition(params, picked, init_trust_region_size, active)
            current = jax.tree.map(lambda a, b: a.at[idx].set(b, mode='drop'), current, moved)
            return chunk + jnp.int32(1), current

        _, states = jax.lax.while_loop(cond, body, (jnp.int32(0), states))
        return states
    
    def _check_if_done(self, state, parents):
        return state.done_flag & ((state.phase == 1) | (state.status != WT.BOUNDARY_HIT)) & (parents != -1)
    
    def _refill_state(self, current_states, is_valid_mask, free_indices, new_positions, new_keys, init_trust_region_size, N_total_samples):
        
        scatter = get_scatter_fn(free_indices, is_valid_mask, N_total_samples)
        ftype = current_states.position.dtype
        
        next_positions = scatter(current_states.position, new_positions, jnp.zeros((self.dim,), dtype=ftype))
        next_boundary_hit_positions = scatter(current_states.boundary_hit_position, jnp.zeros((self.dim,), dtype=ftype), jnp.zeros((self.dim,), dtype=ftype))
        next_nSteps = scatter(current_states.nSteps, 0, reset_value=0)
        next_w = scatter(current_states.w, 0.0, reset_value=0.0)
        next_trust_region_size = scatter(current_states.trust_region_size, init_trust_region_size, reset_value=init_trust_region_size)
        next_rng_keys = scatter(current_states.rng_key, new_keys, reset_value=jr.PRNGKey(0))
        next_phase = scatter(current_states.phase, jnp.array(0, dtype=current_states.phase.dtype), reset_value=jnp.array(0, dtype=current_states.phase.dtype))
        next_done = current_states.done_flag.at[jnp.where(is_valid_mask, free_indices, N_total_samples + 1)].set(False, mode='drop')
        
        return DifferentialWalkerState(
            position=next_positions,
            boundary_hit_position=next_boundary_hit_positions,
            nSteps=next_nSteps,
            w=next_w,
            trust_region_size=next_trust_region_size,
            rng_key=next_rng_keys,
            status=current_states.status.at[jnp.where(is_valid_mask, free_indices, N_total_samples + 1)].set(WT.ACTIVE, mode='drop'),
            done_flag=next_done,
            offset_length=scatter(current_states.offset_length, 0.0, reset_value=0.0),
            phase=next_phase
        )
    

    def compute_boundary_derivative(self, params: jnp.ndarray, boundary_hit_position: jnp.ndarray, solution_estimate: jnp.ndarray,
                                    offset_length: jnp.ndarray) -> jnp.ndarray:
        """
        Compute the boundary derivative contributions D(x) and Vn(x) at the given position.
        Differential WoS Algorithm 2 line 11
        Args:
            boundary_hit_position: jnp.ndarray of shape (dim, ) position on the boundary
            solution_estimate: jnp.ndarray of shape (, ) estimated solution at offset position
            offset_length: accepted per-walker inward offset length
        Returns:
            del_u_del_theta: jnp.ndarray of shape (N_params, ) boundary derivative contribution
        """
        D_vec = self.Df(params, boundary_hit_position, self.boundary_condition_fn)
        Vn_vec = self.V_n(params, boundary_hit_position)
        g_val = self.boundary_condition_fn(boundary_hit_position)
        return jax.tree.map(lambda D, Vn: D - Vn * (g_val - solution_estimate) / offset_length, D_vec, Vn_vec)
    

    def _compute_accumulator_update(self, params: jnp.ndarray, state: DifferentialWalkerState) -> jnp.ndarray:
        return self.compute_boundary_derivative(
            params,
            state.boundary_hit_position,
            state.w + self.boundary_condition_fn(state.position),
            # Inactive/failed slots are evaluated too, but never accumulated.
            jnp.where(state.offset_length > 0, state.offset_length, 1.)
        )
    

def get_scatter_fn(free_indices: jnp.ndarray, is_valid_mask: jnp.ndarray, N_total_samples: int):
    """
    Create a scatter function for refilling states.
    """
    def scatter(arr, val, reset_value):
        arr = arr.at[free_indices].set(reset_value, mode='drop')
        target_indices = jnp.where(is_valid_mask, free_indices, N_total_samples + 1)
        return arr.at[target_indices].set(val, mode='drop')
    return scatter
    

def create_zero_buffer(leaf_array: jnp.ndarray, buffer_size: int) -> jnp.ndarray:
    """
    Create a zero buffer with the leading dimension of size buffer_size and the remaining dimensions matching leaf_array.
    leaf_array.shape (1, *D) -> output.shape (buffer_size, *D)
    """
    return jnp.zeros((buffer_size,) + leaf_array.shape[1:], dtype=leaf_array.dtype)


def mask_leaf(leaf_array: jnp.ndarray, mask: jnp.ndarray) -> jnp.ndarray:
    """
    Mask the leaf array with the given boolean mask.
    leaf_array.shape (N, *D), mask.shape (N,) -> output.shape (N, *D) with masked entries set to zero
    """
    ### make sure mask is broadcastable to leaf_array
    broadcast_shape = (leaf_array.shape[0],) + (1,) * (leaf_array.ndim - 1)
    mask = jnp.reshape(mask, broadcast_shape)
    return jnp.where(mask, leaf_array, jnp.array(0, dtype=leaf_array.dtype))


def _pad_to_bucket(array: jnp.ndarray, bucket_size: int = 1024):
    """
    Pads an array to the next multiple of bucket_size
    Uses the first element for padding to ensure valid evaluation.
    Returns:
        padded_array: The array with shape (N_padded, ...)
        original_size: The original length N, used for slicing later.
    """
    N_original = array.shape[0]
    
    # Calculate target size (next multiple of bucket_size)
    N_padded = ((N_original - 1) // bucket_size + 1) * bucket_size
    pad_len = N_padded - N_original

    if pad_len > 0:
        # Use the first element to pad. This is safer than zeros because of potential singularities
        # We use array[0:1] to keep dimensions for concatenation.
        padding_val = array[0:1]
        
        # Tile the padding value to match the required length
        # tile shape: (pad_len, 1, 1...) depending on rank
        tile_shape = (pad_len,) + (1,) * (array.ndim - 1)
        padding = jnp.tile(padding_val, tile_shape)
        
        padded_array = jnp.concatenate([array, padding], axis=0)
        return padded_array, N_original
    
    return array, N_original
