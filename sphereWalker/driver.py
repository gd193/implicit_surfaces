"""Tier 1: the Monte Carlo batch driver, free of PDE assumptions."""

import abc
import time
from functools import partial
from typing import Any, Generic, Optional, Tuple

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np

from sphereWalker.ops import (
    _compact_step,
    compaction_schedule,
    create_zero_buffer,
    get_scatter_fn,
    mask_leaf,
    pad_to_bucket,
)
from sphereWalker.types import StateT, WalkerDiagnostics, WalkerOutput
from sphereWalker.types import WalkerTermination as WT
from utils.jnp_array_compaction import should_compact
from utils.validation import check_int


def _validate_rng_key(rng_key):
    """Reject counter-free RBG keys.

    A finished lane freezes its key while vmapped live lanes keep splitting;
    with `rbg`/`unsafe_rbg`, batched split/fold_in does not give independent
    streams per lane, so live lanes can repeat directions.
    """
    if jax.dtypes.issubdtype(jnp.result_type(rng_key), jax.dtypes.prng_key):
        impl = str(jr.key_impl(rng_key))
    else:
        impl = str(jax.config.jax_default_prng_impl)
    if 'rbg' in impl:
        raise ValueError(f'MCIntegrator does not support the {impl!r} PRNG; use threefry2x32')


class MCIntegrator(Generic[StateT], abc.ABC):
    """Refill/compaction batch driver for per-sample Monte Carlo state machines.

    A state is a NamedTuple of arrays with a leading buffer axis. The driver
    reads `position`, `nSteps`, `done_flag` (sample finished, possibly only a
    phase) and `status` (`WalkerTermination`). Each sample's
    `_compute_accumulator_update` (any array/PyTree) is summed per input
    position when it finishes with a contributing status.

    Subclasses implement `step`, `_get_initial_state` and
    `_compute_accumulator_update`; the other hooks have generic defaults.
    The driver steps the whole buffer, including lanes that already finished,
    so `step` must leave a terminal (done) state unchanged. `finish_step`
    does this and expects `position` and `w` leaves (it also invalidates
    nonfinite ones). Keys may be raw or typed, but not `rbg`/`unsafe_rbg`.
    """

    def __init__(self, dim: int, *, max_steps: int = 1000, print_progress: bool = False):
        self.dim = check_int('dim', dim)
        # Zero is allowed: a walk then terminates (MAX_STEPS) at its first step.
        self.max_steps = check_int('max_steps', max_steps, minimum=0)
        self.print_progress = print_progress

    # ----------------------------------------------------------------- hooks
    @abc.abstractmethod
    def step(self, params: Any, state: StateT) -> StateT:
        """Advance one (unbatched) sample by one step."""

    @abc.abstractmethod
    def _get_initial_state(self, init_position_batch: jnp.ndarray, rng_keys: jnp.ndarray,
                           init_tr: float, buffer_size: int) -> StateT:
        """Fresh batched states at the given positions."""

    @abc.abstractmethod
    def _compute_accumulator_update(self, params: Any, state: StateT) -> Any:
        """Contribution of one finished sample."""

    def _refill_state(self, current_states: StateT, is_valid_mask: jnp.ndarray,
                      free_indices: jnp.ndarray, new_positions: jnp.ndarray, new_keys: jnp.ndarray,
                      init_trust_region_size: float) -> StateT:
        """Reset freed slots and write fresh initial states into valid ones."""
        scatter = get_scatter_fn(free_indices, is_valid_mask)
        fresh = self._get_initial_state(
            new_positions, new_keys, init_trust_region_size, new_positions.shape[0])

        def refill(old, new):
            # Typed PRNG keys have no zero; any valid key is a placeholder.
            reset = (new[0] if jax.dtypes.issubdtype(old.dtype, jax.dtypes.prng_key)
                     else jnp.zeros(old.shape[1:], old.dtype))
            return scatter(old, new, reset)
        return jax.tree.map(refill, current_states, fresh)

    def _transition_state_phases(self, params: Any, states: StateT,
                                 init_trust_region_size: float) -> StateT:
        """Multi-phase estimators move finished phases on here."""
        return states

    def _check_if_done(self, state: StateT, parents: jnp.ndarray) -> jnp.ndarray:
        return state.done_flag & (parents != -1)

    def _contributing_status(self, status):
        """Strict by default; explicit truncated estimators can accept the cap."""
        return status == WT.BOUNDARY_HIT

    def _step_block(self, params: Any, states: StateT, steps: int, unroll: bool) -> StateT:
        """Advance the whole buffer `steps` times."""
        step = jax.vmap(self.step, in_axes=(None, 0))
        return jax.lax.fori_loop(0, steps, lambda i, s: step(params, s), states, unroll=unroll)

    def _accumulate_completed(self, params: Any, current_states: StateT,
                              current_parents: jnp.ndarray, is_done: jnp.ndarray,
                              sums: Any, counts: jnp.ndarray, drop_idx: int) -> Tuple[Any, jnp.ndarray]:
        to_accumulate = jax.vmap(self._compute_accumulator_update, in_axes=(None, 0))(params, current_states)
        successful = is_done & self._contributing_status(current_states.status)
        finite = jnp.ones_like(successful)
        for leaf in jax.tree.leaves(to_accumulate):
            finite &= jnp.all(jnp.isfinite(leaf).reshape((leaf.shape[0], -1)), axis=1)
        terminal_status = jnp.where(successful & ~finite, WT.INVALID, current_states.status)
        successful &= finite

        done_parents = jnp.where(is_done, current_parents, drop_idx)
        done_w = jax.tree.map(lambda leaf: mask_leaf(leaf, successful), to_accumulate)
        updated_sums = jax.tree.map(
            lambda acc_leaf, to_add_leaf: acc_leaf.at[done_parents].add(to_add_leaf, mode='drop'),
            sums, done_w)
        updated_counts = counts.at[done_parents, terminal_status].add(
            is_done.astype(counts.dtype), mode='drop')
        return updated_sums, updated_counts

    # ---------------------------------------------------------------- driver
    def _get_accumulator_initial(self, params: Any, positions: jnp.ndarray,
                                 rng_key: jnp.ndarray) -> Any:
        # Infer array/PyTree output structure without indexing an actual point
        # or evaluating a callback at an invented location. A dummy key derived
        # from rng_key keeps its kind (typed or raw) for _get_initial_state.
        dummy_key = jr.fold_in(rng_key, 0)[None]
        dummy_state = self._get_initial_state(
            jnp.zeros((1, self.dim), dtype=positions.dtype), dummy_key, 1., 1)
        shape = jax.eval_shape(
            jax.vmap(self._compute_accumulator_update, in_axes=(None, 0)), params, dummy_state)
        return jax.tree.map(lambda leaf: create_zero_buffer(leaf, positions.shape[0]), shape)

    def _prepare_positions(self, positions, n_samples, steps_between_refill,
                           max_active_states, jit_buck_size, n_compactions):
        for name, value in (('n_samples', n_samples), ('steps_between_refill', steps_between_refill),
                            ('max_active_states', max_active_states), ('jit_buck_size', jit_buck_size)):
            check_int(name, value)
        check_int('n_compactions', n_compactions, minimum=0)
        positions = jnp.asarray(positions, dtype=jnp.result_type(positions, jnp.float32))
        if positions.ndim != 2 or positions.shape[1] != self.dim:
            raise ValueError('positions must have shape (N, dim)')
        return positions

    @partial(jax.jit, static_argnames=['self', 'active_buffer_size', 'n_samples',
                                       'steps_between_refill', 'unroll_loop'])
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
                               init_trust_region_size: float,
                               # static configuration, except the dynamic
                               # np.int32 N_total_samples: problems of one
                               # padded size reuse the compiled kernel.
                               active_buffer_size: int,
                               N_total_samples: jnp.ndarray,
                               n_samples: int,
                               steps_between_refill: int,
                               unroll_loop: bool):
        if self.print_progress:
            print(f"🔄 [JIT TRACING] _process_active_states called with active_buffer_size={active_buffer_size}")

        current_states = self._transition_state_phases(params, current_states, init_trust_region_size)
        is_done = self._check_if_done(current_states, current_parents)

        sums, counts = current_accumulated_w
        # Out-of-bounds index for buffers of size N_positions.
        drop_idx = current_accumulated_nSteps.shape[0]
        updated_sums, updated_counts = self._accumulate_completed(
            params, current_states, current_parents, is_done, sums, counts, drop_idx)
        done_parents = jnp.where(is_done, current_parents, drop_idx)
        done_nSteps = jnp.where(is_done, current_states.nSteps, 0)
        updated_accumulated_nSteps = current_accumulated_nSteps.at[done_parents].add(done_nSteps, mode='drop')
        updated_accumulated_w = (updated_sums, updated_counts)

        # Refill active states
        free_indices_mask = jnp.nonzero(is_done, size=active_buffer_size, fill_value=active_buffer_size)[0]
        n_free = jnp.sum(is_done, dtype=next_task_index.dtype)

        offsets = jnp.arange(active_buffer_size, dtype=next_task_index.dtype)
        candidate_indices = next_task_index + offsets
        valid_candidates_mask = jnp.logical_and(candidate_indices < N_total_samples, offsets < n_free)

        new_parents_val = candidate_indices // n_samples
        safe_new_parents_read = jnp.where(valid_candidates_mask, new_parents_val, 0)
        new_positions_val = initial_position_batch[safe_new_parents_read, :]
        new_keys_val = jax.vmap(lambda i: jr.fold_in(rng_key, i))(candidate_indices)

        next_parents = current_parents.at[free_indices_mask].set(-1, mode='drop')
        target_indices = jnp.where(valid_candidates_mask, free_indices_mask, active_buffer_size)
        next_parents = next_parents.at[target_indices].set(new_parents_val, mode='drop')

        refilled_states = self._refill_state(
            current_states, valid_candidates_mask, free_indices_mask, new_positions_val,
            new_keys_val, init_trust_region_size)

        # Lanes without a sample are dead: frozen and terminal. A phase
        # transition selects on status, so it cannot revive them either.
        dead = next_parents == -1
        refilled_states = refilled_states._replace(
            done_flag=refilled_states.done_flag | dead,
            status=jnp.where(dead, WT.INVALID, refilled_states.status).astype(refilled_states.status.dtype))

        # No lax.cond to skip a dead block: on GPU its predicate is copied to
        # the host every round, and the skipped work (one block per solve) is
        # not measurable.
        final_states = self._step_block(params, refilled_states, steps_between_refill, unroll_loop)

        # One array, so the host reads both counters with a single copy.
        counters = jnp.stack([n_free, jnp.sum(next_parents != -1, dtype=n_free.dtype)])
        return final_states, next_parents, updated_accumulated_w, updated_accumulated_nSteps, counters

    @partial(jax.jit, static_argnames=['self', 'active_buffer_size',
                                       'n_samples', 'steps_between_refill', 'unroll_loop', 'rounds'])
    def _process_refill_rounds(self, states: StateT, parents: jnp.ndarray,
                               accumulated_w: Any, accumulated_nSteps: jnp.ndarray,
                               next_task_index: jnp.ndarray, params: Any,
                               initial_position_batch: jnp.ndarray, rng_key: jnp.ndarray,
                               init_trust_region_size: float, active_buffer_size: int,
                               N_total_samples: jnp.ndarray, n_samples: int, steps_between_refill: int,
                               unroll_loop: bool, rounds: int):
        """Dispatch existing refill rounds together, preserving each round's order."""
        initial_task_index = next_task_index

        # Keep one copy of the walk/source body in the executable, even for
        # multi-sample source integrals. Grouping still needs no host counters.
        def advance(_, carry):
            states, parents, accumulated_w, accumulated_nSteps, next_task_index, _ = carry
            states, parents, accumulated_w, accumulated_nSteps, counters = self._process_active_states(
                states, parents, accumulated_w, accumulated_nSteps, next_task_index,
                params, initial_position_batch, rng_key, init_trust_region_size,
                active_buffer_size, N_total_samples, n_samples, steps_between_refill, unroll_loop)
            return states, parents, accumulated_w, accumulated_nSteps, next_task_index + counters[0], counters

        states, parents, accumulated_w, accumulated_nSteps, next_task_index, counters = jax.lax.fori_loop(
            0, rounds, advance, (states, parents, accumulated_w, accumulated_nSteps,
                                 next_task_index, jnp.zeros(2, jnp.int32)))
        counters = jnp.stack([next_task_index - initial_task_index, counters[1]])
        return states, parents, accumulated_w, accumulated_nSteps, counters

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
                                    n_compactions: int = 20,
                                    compaction_mode: str = 'uniform',
                                    *,
                                    refill_rounds: int = 1) -> None:
        """Force one JIT compile per compaction-bucket size (and grouped refill rounds).

        `_process_active_states` is jitted with `active_buffer_size` as a
        static_argname, so each unique buffer size triggers a fresh compile.
        When `refill_rounds > 1`, `_process_refill_rounds` is also warmed for
        the initial active buffer size used during the refill phase.
        """
        _validate_rng_key(rng_key)
        initial_position_batch = self._prepare_positions(
            initial_position_batch, n_samples, steps_between_refill,
            max_active_states, jit_buck_size, n_compactions)
        check_int('refill_rounds', refill_rounds)
        if initial_position_batch.shape[0] == 0:
            return
        positions_padded, N_original = pad_to_bucket(initial_position_batch, bucket_size=jit_buck_size)
        N_positions = positions_padded.shape[0]
        # Padding only stabilizes storage shapes; it is not extra work.
        N_total_samples = N_original * n_samples
        active_buffer_size = min(N_total_samples, max_active_states)
        compaction_sizes = compaction_schedule(active_buffer_size, n_compactions, compaction_mode)

        accumulated_w = (self._get_accumulator_initial(params, positions_padded, rng_key),
                         jnp.zeros((N_positions, len(WT)), dtype=jnp.int32))
        accumulated_nSteps = jnp.zeros((N_positions,), dtype=jnp.int32)
        next_task_index_dummy = np.int32(N_total_samples)  # same arg type as integrate_batch

        def dummy(size):
            keys = jax.vmap(lambda i: jr.fold_in(rng_key, i))(jnp.arange(size))
            pos = jnp.broadcast_to(positions_padded[0:1], (size, positions_padded.shape[1]))
            return (self._get_initial_state(pos, keys, init_trust_region_size, size),
                    jnp.zeros((size,), dtype=jnp.int32))

        # All-valid parents so the kernel takes its main branch (some processing
        # happens, no refills since next_task_index >= N_total).
        for buf in compaction_sizes:
            if buf == 0:
                continue
            out = self._process_active_states(
                *dummy(buf), accumulated_w, accumulated_nSteps, next_task_index_dummy,
                params, positions_padded, rng_key, init_trust_region_size,
                int(buf), np.int32(N_total_samples), n_samples, steps_between_refill, unroll_loop)
            jax.block_until_ready(out[0].position)

        if refill_rounds > 1 and active_buffer_size > 0:
            buf = int(active_buffer_size)
            out = self._process_refill_rounds(
                *dummy(buf), accumulated_w, accumulated_nSteps, next_task_index_dummy,
                params, positions_padded, rng_key, init_trust_region_size,
                buf, np.int32(N_total_samples), n_samples, steps_between_refill, unroll_loop,
                rounds=int(refill_rounds))
            jax.block_until_ready(out[0].position)

        # `_compact_step` is jit-cached on `new_size` and input shapes. Warm one
        # entry per (source size, smaller bucket) pair of the schedule.
        nonzero_sizes = [b for b in compaction_sizes if b > 0]
        for i, old in enumerate(nonzero_sizes[:-1]):
            old_state, old_parents = dummy(old)
            is_active = jnp.ones((old,), dtype=jnp.bool_)
            for new in nonzero_sizes[i + 1:]:
                state_out, parents_out = _compact_step(old_state, old_parents, is_active, int(new))
                jax.tree.map(jax.block_until_ready, state_out)
                jax.block_until_ready(parents_out)

    def integrate_batch(self,
                        params: Any,
                        initial_position_batch: jnp.ndarray,
                        rng_key: jnp.ndarray,
                        init_trust_region_size: float,
                        n_samples: int,
                        steps_between_refill: int = 50,
                        unroll_loop: bool = False,
                        max_active_states: int = 1000**2,
                        jit_buck_size: int = 1000,
                        n_compactions: int = 0,
                        diag_log: Optional[list] = None, *,
                        return_details: bool = False,
                        refill_rounds: int = 1,
                        compaction_mode: str = 'uniform') -> WalkerOutput | WalkerDiagnostics:
        """Estimate from accepted terminations; strict boundary hits by default.

        return_details=True returns successful contribution sums and per-position
        termination counts instead of an estimate. An explicit truncation policy
        may retain source contributions at max_steps, with zero remaining tail.
        max_steps is a total per-sample budget, shared by both phases of
        differential WoS.

        refill_rounds groups up to this many existing refill rounds into one
        device dispatch while pending tasks can keep the buffer full. Drain
        rounds retain their usual host checks and compaction schedule.

        diag_log: if a list is passed, each outer iteration appends a dict
            with `{outer_idx, buffer_size, n_active, ms_total, ms_compact,
            ms_process, ms_sync}`. The caller can use this to break the wall
            time into per-step kernel work vs Python/host overhead and per-
            bucket-size cost. Pass None to skip instrumentation entirely.
        """
        _validate_rng_key(rng_key)
        initial_position_batch = self._prepare_positions(
            initial_position_batch, n_samples, steps_between_refill,
            max_active_states, jit_buck_size, n_compactions)
        check_int('refill_rounds', refill_rounds)
        if compaction_mode not in ('uniform', 'geometric'):
            raise ValueError('compaction_mode must be uniform or geometric')
        if initial_position_batch.shape[0] == 0:
            sums = self._get_accumulator_initial(params, initial_position_batch, rng_key)
            steps = jnp.zeros(0, dtype=initial_position_batch.dtype)
            if return_details:
                return WalkerDiagnostics(initial_position_batch, sums, jnp.zeros(0, jnp.int32),
                                         jnp.zeros((0, len(WT)), jnp.int32), steps)
            return WalkerOutput(initial_position_batch, steps, sums)

        positions_padded, N_original = pad_to_bucket(initial_position_batch, bucket_size=jit_buck_size)
        N_positions = positions_padded.shape[0]
        # Padding only stabilizes storage shapes; it is not extra work.
        N_total_samples = N_original * n_samples
        active_buffer_size = min(N_total_samples, max_active_states)

        accumulated_w = (self._get_accumulator_initial(params, positions_padded, rng_key),
                         jnp.zeros((N_positions, len(WT)), dtype=jnp.int32))
        accumulated_nSteps = jnp.zeros((N_positions,), dtype=jnp.int32)

        init_task_indices = jnp.arange(active_buffer_size, dtype=jnp.int32)
        active_parent_indices = init_task_indices // n_samples

        # Host-side task cursor and live count: one device read per round.
        next_task_index = active_buffer_size
        active_count_int = active_buffer_size

        rng_keys_init = jax.vmap(lambda i: jr.fold_in(rng_key, i))(init_task_indices)
        active_states = self._get_initial_state(
            positions_padded[active_parent_indices, :], rng_keys_init,
            init_trust_region_size, active_buffer_size)

        # Compaction schedule. Walk lengths are highly uneven (random walks
        # have a heavy tail), so during the drain phase a few stragglers can pin
        # a wide buffer. `compaction_sizes` is a descending list of buffer
        # widths; we walk down it as the active set fits each smaller bucket.
        # `n_compactions=0` yields a single-size schedule.
        compaction_sizes = compaction_schedule(active_buffer_size, n_compactions, compaction_mode)
        current_compaction_index = 0
        current_buffer_size = compaction_sizes[current_compaction_index]

        process_calls = 0
        while True:
            t_iter = time.perf_counter() if diag_log is not None else None
            if active_count_int == 0 and next_task_index >= N_total_samples:
                break
            n_active_before = active_count_int

            # Drain-phase compaction: only once dispatch is done (pending tasks
            # exhausted). Until then, completed slots are refilled with fresh
            # tasks and the buffer should stay full.
            t_compact_start = time.perf_counter() if diag_log is not None else None
            ms_compact = 0.0
            if (n_compactions > 0
                    and next_task_index >= N_total_samples
                    and current_compaction_index < len(compaction_sizes) - 1):
                will_compact, new_idx, new_size = should_compact(
                    active_count_int, current_compaction_index, compaction_sizes)
                if will_compact and new_size > 0:
                    is_active_mask = active_parent_indices != -1
                    active_states, active_parent_indices = _compact_step(
                        active_states, active_parent_indices, is_active_mask, int(new_size))
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
            # Even if every lane finishes in every round, this group cannot
            # exhaust pending tasks before its final round. Thus no intermediate
            # compaction opportunity or task assignment changes.
            rounds = (refill_rounds
                      if next_task_index + refill_rounds * current_buffer_size <= N_total_samples else 1)
            process = self._process_refill_rounds if rounds > 1 else self._process_active_states
            round_options = {'rounds': rounds} if rounds > 1 else {}
            (active_states, active_parent_indices, accumulated_w, accumulated_nSteps,
             counters) = process(
                active_states, active_parent_indices, accumulated_w, accumulated_nSteps,
                np.int32(next_task_index), params, positions_padded, rng_key,
                init_trust_region_size, current_buffer_size, np.int32(N_total_samples), n_samples,
                steps_between_refill, unroll_loop, **round_options)
            if diag_log is not None:
                jax.block_until_ready(active_states.position)
                ms_process = (time.perf_counter() - t_process_start) * 1000
            # The round's only host sync: the next loop test needs both values.
            t_sync_start = time.perf_counter() if diag_log is not None else None
            n_filled, active_count_int = (int(v) for v in jax.device_get(counters))
            if diag_log is not None:
                ms_sync = (time.perf_counter() - t_sync_start) * 1000
                ms_total = (time.perf_counter() - t_iter) * 1000
                diag_log.append(dict(
                    outer_idx=process_calls,
                    buffer_size=int(current_buffer_size),
                    n_active=int(n_active_before),
                    next_task_index=int(next_task_index),
                    refill_rounds=int(rounds),
                    ms_total=ms_total,
                    ms_compact=ms_compact,
                    ms_process=ms_process,
                    ms_sync=ms_sync,
                ))

            process_calls += 1
            next_task_index += n_filled
            if self.print_progress:
                print(f"Remaining Active States: {active_count_int} | "
                      f"Next Task Index: {next_task_index}/{N_total_samples}  (buffer={current_buffer_size})")
                print(f"Process called so far. {process_calls}")

        # Counts retain the actual termination reason even for accepted caps.
        sums, counts = accumulated_w
        sums = jax.tree.map(lambda leaf: leaf[:N_original], sums)
        counts = counts[:N_original]
        final_nSteps = accumulated_nSteps[:N_original] / n_samples
        final_positions = positions_padded[:N_original, :]
        contributing = jnp.sum(jnp.where(self._contributing_status(jnp.arange(len(WT))), counts, 0), axis=-1)
        details = WalkerDiagnostics(final_positions, sums, contributing, counts, final_nSteps)
        if return_details:
            return details
        if not bool(jnp.all(details.successful_samples == n_samples)):
            exhausted = int(jnp.sum(counts[:, WT.MAX_STEPS]))
            invalid = int(jnp.sum(counts[:, WT.INVALID]))
            raise RuntimeError(f'incomplete sampling: {exhausted} exhausted, {invalid} invalid; '
                               'use return_details=True to inspect partial sums and counts')
        final_w = jax.tree.map(lambda leaf: leaf / n_samples, sums)
        return WalkerOutput(position=final_positions, nSteps=final_nSteps, w=final_w)
