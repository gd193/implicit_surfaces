import jax 
import jax.numpy as jnp
import jax.random as jr

import time
import timeit
from typing import Callable, Tuple, TypeVar, Generic, Optional, Any, Union, cast
from functools import partial
import abc

from sphereWalker.types import WalkerState, WalkerOutput, DifferentialWalkerState
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
        self.empty_sphere = empty_sphere_fn
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

    @abc.abstractmethod
    def _get_accumulator_initial(self, params: Any, positions: jnp.ndarray) -> Any:
        pass

    @abc.abstractmethod
    def _transition_state_phases(self, params: Any, states: StateT) -> StateT:
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
            
        current_states = self._transition_state_phases(params, current_states)
        is_done = self._check_if_done(current_states, current_parents)

        to_accumulate = jax.vmap(self._compute_accumulator_update, in_axes=(None, 0))(params, current_states)
        
        # Accumulate results
        # Use N_total_samples + 1 as safe drop index for logic, 
        # but for accumulation into buffers of size N_positions, we need a safe index relative to N_positions.
        # We use current_accumulated_nSteps.shape[0] as the safe out-of-bounds index.
        drop_idx = current_accumulated_nSteps.shape[0]
        
        done_parents = jnp.where(is_done, current_parents, drop_idx)
        done_nSteps = jnp.where(is_done, current_states.nSteps, 0)
        done_w = jax.tree.map(lambda leaf: mask_leaf(leaf, is_done), to_accumulate)

        updated_accumulated_w = jax.tree.map(
            lambda acc_leaf, to_add_leaf: acc_leaf.at[done_parents].add(to_add_leaf, mode='drop'),
            current_accumulated_w,
            done_w
        )
        updated_accumulated_nSteps = current_accumulated_nSteps.at[done_parents].add(done_nSteps, mode='drop')

        # Refill active states
        free_indices_mask = jnp.nonzero(is_done, size=active_buffer_size, fill_value=active_buffer_size)[0]
        n_free = jnp.sum(is_done) 

        offsets = jnp.arange(active_buffer_size)
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
        positions_padded, _ = _pad_to_bucket(initial_position_batch, bucket_size=jit_buck_size)
        N_positions = positions_padded.shape[0]
        N_total_samples = N_positions * n_samples
        active_buffer_size = min(N_total_samples, max_active_states)
        compaction_sizes = uniform_size_generator(active_buffer_size, n_compactions)

        accumulated_w = self._get_accumulator_initial(params, positions_padded)
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
                        diag_log: Optional[list]=None) -> WalkerOutput:
        """Run WoS to completion across `initial_position_batch`.

        diag_log: if a list is passed, each outer iteration appends a dict
            with `{outer_idx, buffer_size, n_active, ms_total, ms_compact,
            ms_process, ms_sync}`. The caller can use this to break the wall
            time into per-step kernel work vs Python/host overhead and per-
            bucket-size cost. Pass None to skip instrumentation entirely.
        """

        # 1. Pad Positions
        positions_padded, N_original = _pad_to_bucket(initial_position_batch, bucket_size=jit_buck_size)

        N_positions = positions_padded.shape[0]
        N_total_samples = N_positions * n_samples
        active_buffer_size = min(N_total_samples, max_active_states)

        # 3. Init Global Output
        accumulated_w = self._get_accumulator_initial(params, positions_padded)
        accumulated_nSteps = jnp.zeros((N_positions,), dtype=jnp.int32)

        init_task_indices = jnp.arange(active_buffer_size)
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

        # 4. Final Slicing (Remove Padding)
        final_w = jax.tree.map(lambda leaf: leaf[:N_original] / n_samples, accumulated_w)
        final_nSteps = accumulated_nSteps[:N_original] / n_samples
        final_positions = positions_padded[:N_original, :]

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
        # surfaces return fn_val = f(x) - level_set already; compare to 0, not to level_set again.
        # Force-terminate walkers that exhaust max_steps without reaching the surface;
        # without this, ill-conditioned walkers (very small empty-sphere radius forever)
        # trap the integrator's Python while-loop in `integrate_batch`.
        reached_surface = jnp.abs(fn_val) <= self.eps
        exhausted = state.nSteps >= self.max_steps
        is_done = jnp.logical_or(state.done_flag, jnp.logical_or(reached_surface, exhausted))
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
        
        
        return WalkerState(position=new_position, 
                           nSteps=new_nSteps, 
                           w=new_w, 
                           trust_region_size=new_trust_region_size, 
                           rng_key=new_key,
                           done_flag=is_done)
    
    def _get_initial_state(self, init_position_batch, rng_keys, init_tr, buffer_size) -> WalkerState:
        return WalkerState(
            position=init_position_batch,
            nSteps=jnp.zeros((buffer_size,), dtype=jnp.int32),
            w=jnp.zeros((buffer_size, 1)),
            trust_region_size=jnp.full((buffer_size,), init_tr, dtype=init_position_batch.dtype),
            rng_key=rng_keys,
            done_flag=jnp.zeros((buffer_size,), dtype=bool)
        )
    
    def _get_accumulator_initial(self, params, positions) -> Any:
        N_positions = positions.shape[0]
        dummy_pos = positions[0]
        dummy_state = self._get_initial_state(
            dummy_pos[None, :],
            jr.PRNGKey(0)[None, :],
            1.0,
            1
        )
        sample_accumulator_output = jax.vmap(self._compute_accumulator_update, in_axes=(None, 0))(params, dummy_state)
        accumulator = jax.tree.map(lambda acuum_leaf: create_zero_buffer(acuum_leaf, N_positions), sample_accumulator_output)
        return accumulator
    

    def _transition_state_phases(self, params: Any, states: WalkerState) -> WalkerState:
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
        
        next_positions = scatter(current_states.position, new_positions, jnp.zeros((self.dim,)))
        next_nSteps = scatter(current_states.nSteps, 0, reset_value=0)
        next_w = scatter(current_states.w, 0.0, reset_value=jnp.zeros(current_states.w.shape[1:]))
        next_trust_region_size = scatter(current_states.trust_region_size, init_trust_region_size, reset_value=init_trust_region_size)
        next_rng_keys = scatter(current_states.rng_key, new_keys, reset_value=jr.PRNGKey(0))
        next_done = current_states.done_flag.at[jnp.where(is_valid_mask, free_indices, N_total_samples + 1)].set(False, mode='drop')
        
        return WalkerState(
            position=next_positions,
            nSteps=next_nSteps,
            w=next_w,
            trust_region_size=next_trust_region_size,
            rng_key=next_rng_keys,
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
                 **kwargs):
        super().__init__(**kwargs)
        self.normal_fn = normal_fn
        self.Df = Df
        self.V_n = V_n
        self.offset_l = offset_l

    def step(self, params, state: DifferentialWalkerState) -> DifferentialWalkerState:
        """
        Perform differential WoS step for the walker given its current state.
        Phase 0: Walk -> Hit Boundary -> Offset Position -> Switch Walker to Phase 1
        Phase 1: Walk -> Hit Boundary -> Done
        """
        # `empty_sphere` now returns the surface's recommended next trust-region
        # size (matching NIQ + WoSIntegrator). Adopt it directly, and keep the
        # safety net: if the empty-sphere radius collapsed to a tiny fraction of
        # tr the polynomial got stuck → shrink further regardless of the
        # surface's recommendation.
        fn_val, radius, next_tr_from_es = self.empty_sphere(params, state.position, state.trust_region_size)
        new_trust_region_size = jnp.where(radius < state.trust_region_size * 0.05,
                                          state.trust_region_size * 0.5,
                                          next_tr_from_es)

        ### sample new position on sphere and inside ball
        sphere_key, ball_key, new_key = jr.split(state.rng_key, 3)
        sample_sphere = self.sphere_sampling_fn(sphere_key, dtype=state.position.dtype) * radius
        sample_ball_radius, sample_ball_direction = self.ball_sampling_fn(ball_key, dtype=state.position.dtype)
        sample_ball_radius *= radius
        sample_ball_position = state.position + (sample_ball_radius * sample_ball_direction)

        ### update state — surfaces return fn_val = f(x) - level_set already; compare to 0.
        # Same max_steps force-termination as WoSIntegrator (avoid loop deadlock).
        reached_surface = jnp.abs(fn_val) <= self.eps
        exhausted = state.nSteps >= self.max_steps
        is_done = jnp.logical_or(state.done_flag, jnp.logical_or(reached_surface, exhausted))
        new_nSteps = jnp.where(is_done, state.nSteps, state.nSteps + 1)
        new_position = jnp.where(is_done, state.position,
                                 state.position + sample_sphere)
        new_position, new_nSteps = cast(jnp.ndarray, new_position), cast(jnp.ndarray, new_nSteps) # type hints b.c. of jnp.where
        
        ### only update weight in phase 1 and if not done
        new_w = jnp.where(jnp.logical_or(is_done, state.phase == 0), state.w,
                    state.w + self.source_term(sample_ball_position) * self.weighted_greens_fn(r=sample_ball_radius, R=radius))
        
        return DifferentialWalkerState(
            position=new_position,
            boundary_hit_position=state.boundary_hit_position,
            nSteps=new_nSteps, 
            w=new_w, 
            trust_region_size=new_trust_region_size, 
            rng_key=new_key,
            done_flag=is_done,
            phase=state.phase
        )
    
    
    def _get_initial_state(self, init_position_batch, rng_keys, init_tr, buffer_size):
        return DifferentialWalkerState(
            position=init_position_batch,
            boundary_hit_position=jnp.zeros((buffer_size, self.dim)),
            nSteps=jnp.zeros((buffer_size,), dtype=jnp.int32),
            w=jnp.zeros((buffer_size,)),
            trust_region_size=jnp.full((buffer_size,), init_tr, dtype=init_position_batch.dtype),
            rng_key=rng_keys,
            done_flag=jnp.zeros((buffer_size,), dtype=bool),
            phase=jnp.zeros((buffer_size,), dtype=jnp.int32) # Start in phase 0
        )


    def _get_accumulator_initial(self, params, positions):
        N_positions = positions.shape[0]
        dummy_pos = positions[0]
        dummy_state = self._get_initial_state(
            dummy_pos[None, :],
            jr.PRNGKey(0)[None, :],
            1.0,
            1
        )
        sample_accumulator_output = jax.vmap(self._compute_accumulator_update, in_axes=(None, 0))(params, dummy_state)
        accumulator = jax.tree.map(lambda acuum_leaf: create_zero_buffer(acuum_leaf, N_positions), sample_accumulator_output)
        return accumulator
    
    
    def _transition_walker(self, params, state: DifferentialWalkerState) -> DifferentialWalkerState:
        """
        Offset the position and switch to phase 1.
        """
        normal = self.normal_fn(params, state.position) # Compute normal at boundary
        offset_position = state.position - self.offset_l * normal # Offset inward along normal
        new_phase = jnp.array(1, dtype=state.phase.dtype) # Switch to phase 1
        not_done = jnp.array(False, dtype=state.done_flag.dtype) # Reset done flag for phase 1


        return DifferentialWalkerState(
            position=offset_position,
            boundary_hit_position=state.position,
            nSteps=state.nSteps,
            w=state.w,
            trust_region_size=state.trust_region_size,
            rng_key=state.rng_key,
            done_flag=not_done,
            phase=new_phase  
        )

    def _transition_state_phases(self, params, states):
        """
        Transition the walker states upon hitting the boundary.
        """        
        needs_transition = jnp.logical_and(states.done_flag, states.phase == 0)
        any_transition = jnp.any(needs_transition)
        def transition_fn(_params, _states):
            "vmap over states (arg 1) and keep params (arg 0) fixed"
            return jax.vmap(
                lambda _p, _st: jax.lax.cond(
                    jnp.logical_and(_st.done_flag, _st.phase==0), # condition
                    self._transition_walker,                      # true branch
                    lambda _, st: st,                             # false branch
                    _p, _st                                       # arguments
                ),
                in_axes=(None, 0))(_params, _states)
        return jax.lax.cond(any_transition, transition_fn, lambda _, s: s, *(params, states))
    
    def _check_if_done(self, state, parents):
        return jnp.logical_and(jnp.logical_and(state.done_flag, state.phase == 1), parents != -1)
    
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
            done_flag=next_done,
            phase=next_phase
        )
    

    def compute_boundary_derivative(self, params: jnp.ndarray, boundary_hit_position: jnp.ndarray, solution_estimate: jnp.ndarray) -> jnp.ndarray:
        """
        Compute the boundary derivative contributions D(x) and Vn(x) at the given position.
        Differential WoS Algorithm 2 line 11
        Args:
            boundary_hit_position: jnp.ndarray of shape (dim, ) position on the boundary
            solution_estimate: jnp.ndarray of shape (, ) estimated solution at offset position
        Returns:
            del_u_del_theta: jnp.ndarray of shape (N_params, ) boundary derivative contribution
        """
        D_vec = self.Df(params, boundary_hit_position, self.boundary_condition_fn)
        Vn_vec = self.V_n(params, boundary_hit_position)
        g_val = self.boundary_condition_fn(boundary_hit_position)
        return jax.tree.map(lambda D, Vn: D - Vn * (g_val - solution_estimate) / self.offset_l, D_vec, Vn_vec)
    

    def _compute_accumulator_update(self, params: jnp.ndarray, state: DifferentialWalkerState) -> jnp.ndarray:
        return self.compute_boundary_derivative(
            params,
            state.boundary_hit_position,
            state.w + self.boundary_condition_fn(state.position)
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
