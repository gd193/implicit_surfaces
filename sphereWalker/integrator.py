import jax 
import jax.numpy as jnp
import jax.random as jr

import timeit
from typing import Callable, List, TypeVar, Generic, Optional
from functools import partial

from sphereWalker.types import WalkerState, WalkerOutput, DifferentialWalkerState 
from sphereWalker.sampling import *

def analytical_solution(R, x):
    r = jnp.linalg.norm(x)
    return 1 - (r / R)**2

DEBUG_FLAG = True

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

StateT = TypeVar('StateT', bound=WalkerState)

class WoSIntegrator:
    """
    Walk on Spheres integrator for solving boundary value problems.
    """
    def __init__(self,
                 empty_sphere_fn: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray],
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
        is_done = jnp.logical_or(state.done_flag, jnp.linalg.norm(fn_val - self.level_set) <= self.eps)
        new_nSteps = jnp.where(is_done, state.nSteps, state.nSteps + 1)
        new_position = jnp.where(is_done, state.position, 
                                 state.position + sample_sphere)
        new_w = jnp.where(is_done, state.w,
                          state.w + self.source_term(sample_ball_position) * self.weighted_greens_fn(r=sample_ball_radius, R=radius))
        
        
        return WalkerState(position=new_position, 
                           nSteps=new_nSteps, 
                           w=new_w, 
                           trust_region_size=new_trust_region_size, 
                           rng_key=new_key,
                           done_flag=is_done)
    
    def _get_initial_state(self,
                           init_position_batch: jnp.ndarray,
                           rng_keys: jnp.ndarray,
                           init_tr: jnp.ndarray,
                           buffer_size: int) -> WalkerState:
        """
        Initialize the walker state for a batch of initial positions.
        """
        return WalkerState(
            position=init_position_batch,
            nSteps=jnp.zeros((buffer_size,), dtype=jnp.int32),
            w=jnp.zeros((buffer_size, 1)),
            trust_region_size=jnp.full((buffer_size,), init_tr),
            rng_key=rng_keys,
            done_flag=jnp.zeros((buffer_size,), dtype=bool)
        )
    
    def _get_accumulator(self, params: jnp.ndarray, position_batch: jnp.ndarray, dtype: Optional[jnp.dtype] = None) -> jnp.ndarray:
        """
        Initialize the global accumulator for the walker outputs.
        """
        accum_dtype = dtype if dtype is not None else position_batch.dtype
        N_positions = position_batch.shape[0]
        dummy_pos = position_batch[0]
        dummy_state = self._get_initial_state(
            dummy_pos[None, :],
            jr.PRNGKey(0)[None, :],
            jnp.array([1.0]),
            1
        )
        sample_accumulator_output = jax.vmap(self._compute_accumulator_update, in_axes=(None, 0))(params, dummy_state)
        accumulator = jax.tree.map(lambda acuum_leaf: create_zero_buffer(acuum_leaf, N_positions), sample_accumulator_output)
        return accumulator
    

    def _transition_state_phases(self, params: jnp.ndarray, state: WalkerState) -> WalkerState:
        """
        Transition the walker state upon hitting the boundary.
        """
        return state  # No transition in standard WoS
    
    def _compute_accumulator_update(self, params: jnp.ndarray, state: WalkerState) -> jnp.ndarray:
        """
        Compute the accumulator update upon walker completion.
        """
        return state.w + self.boundary_condition_fn(state.position)
    
    def _check_if_done(self, state: WalkerState, parents: jnp.ndarray) -> jnp.ndarray:
        """
        Check if the state fully completed.
        """
        return jnp.logical_and(state.done_flag, parents != -1)
    
    def _get_scatter_fn(self, free_indices: jnp.ndarray, is_valid_mask: jnp.ndarray, N_total_samples: int):
        """
        Create a scatter function for refilling states.
        """
        def scatter(arr, val, reset_value):
            arr = arr.at[free_indices].set(reset_value, mode='drop')
            target_indices = jnp.where(is_valid_mask, free_indices, N_total_samples + 1)
            return arr.at[target_indices].set(val, mode='drop')
        return scatter
    
    def _refill_state(self,
                      current_states: WalkerState,
                      is_valid_mask: jnp.ndarray,
                      free_indices: jnp.ndarray,
                      new_positions: jnp.ndarray,
                      new_keys: jnp.ndarray,
                      init_trust_region_size: jnp.ndarray,
                      N_total_samples: int) -> WalkerState:
        """
        Refill the walker state buffer with new tasks.
        """
        scatter = self._get_scatter_fn(free_indices, is_valid_mask, N_total_samples)
        
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

    
    @partial(jax.jit, static_argnames=['self','active_buffer_size', 'N_total_samples', 'n_samples', 'steps_between_refill', 'unroll_loop'])
    def _process_active_states(self,
                               # dynamic state
                               current_states: WalkerState,
                               current_parents: jnp.ndarray,
                               current_accumulated_w: jnp.ndarray,
                               current_accumulated_nSteps: jnp.ndarray,
                               next_task_index: int,
                               # dynamic context
                               params: jnp.ndarray,
                               initial_position_batch: jnp.ndarray,
                               rng_key: jnp.ndarray,
                               init_trust_region_size: jnp.ndarray,
                               # Static configuration
                               active_buffer_size: int,
                               N_total_samples: int,
                               n_samples: int,
                               steps_between_refill: int,
                               unroll_loop: bool,
                               ):
        """
        process the current active states and update the global accumulators
        """
        if self.print_progress:
            print(f"🔄 [JIT TRACING] _process_active_states called with N_total_samples={N_total_samples}")
        current_states = self._transition_state_phases(params, current_states)
        is_done = self._check_if_done(current_states, current_parents)

        to_accumulate = jax.vmap(self._compute_accumulator_update, in_axes=(None, 0))(params, current_states)
        ### scatter add results to global accumulators
        done_parents = jnp.where(is_done, current_parents, N_total_samples + 1)
        done_nSteps = jnp.where(is_done, current_states.nSteps, 0)
        done_w = jax.tree.map(lambda leaf: mask_leaf(leaf, is_done), to_accumulate)

        updated_accumulated_w = jax.tree.map(
            lambda acc_leaf, to_add_leaf: acc_leaf.at[done_parents].add(to_add_leaf, mode='drop'),
            current_accumulated_w,
            done_w
        )
        updated_accumulated_nSteps = current_accumulated_nSteps.at[done_parents].add(done_nSteps, mode='drop')

        ### refill active states: filter out done states and add new ones
        free_indices_mask = jnp.nonzero(is_done, size=active_buffer_size, fill_value=active_buffer_size)[0]
        n_free = jnp.sum(is_done) 

        offsets = jnp.arange(active_buffer_size)
        candidate_indices = next_task_index + offsets
        valid_candidates_mask = jnp.logical_and(candidate_indices < N_total_samples, offsets < n_free)

        ### Prepare new tasks
        new_parents_val = candidate_indices // n_samples
        safe_new_parents_read = jnp.where(valid_candidates_mask, new_parents_val, 0)
        new_positions_val = initial_position_batch[safe_new_parents_read, :]
        new_keys_val = jax.vmap(lambda i: jr.fold_in(rng_key, i))(candidate_indices)

        ### update parents array
        next_parents = current_parents.at[free_indices_mask].set(-1, mode='drop')
        target_indices = jnp.where(valid_candidates_mask, free_indices_mask, active_buffer_size)
        next_parents = next_parents.at[target_indices].set(new_parents_val, mode='drop')

        ### refill states
        refilled_states = self._refill_state(
            current_states,
            valid_candidates_mask,
            free_indices_mask,
            new_positions_val,
            new_keys_val,
            init_trust_region_size,
            N_total_samples
        )

        ## step the new active states
        def loop_body(i, states):
           return jax.vmap(self.step, in_axes=(None, 0))(params, states)
        final_states = jax.lax.fori_loop(0, steps_between_refill, loop_body, refilled_states, unroll=unroll_loop)

        return final_states, next_parents, updated_accumulated_w, updated_accumulated_nSteps, n_free
           

    def integrate_batch(self,
                        params: jnp.ndarray,
                        initial_position_batch: jnp.ndarray, 
                        rng_key: jnp.ndarray, 
                        init_trust_region_size: jnp.ndarray, 
                        n_samples: int,
                        steps_between_refill: int=50,
                        unroll_loop: bool=False,
                        max_active_states: int=1000**2,
                        jit_buck_size: int=1000) -> WalkerOutput:
        """
        Integrate the WoS process for a batch of initial positions with multiple samples per position.
        Args:
            initial_position_batch: jnp.ndarray of shape (batch_size, dim) initial positions
            rng_key: jnp.ndarray of shape (, ) PRNG key
            init_trust_region_size: jnp.ndarray of shape (, ) initial trust region size
            n_samples: int number of samples per initial position
            max_active_states: int maximum number of active states to process in one go to avoid OOM
            jit_buck_size: int bucket size for JIT compilation to balance overhead and memory usage
        Returns:
            WalkerOutput containing final positions, number of steps, and solution estimates
        """
        ### Pad initial arrays to fit into buckets to prevent recompilation
        positions_padded, N_original = self._pad_to_bucket(initial_position_batch, bucket_size=jit_buck_size)

        N_positions = positions_padded.shape[0]
        N_total_samples = N_positions * n_samples
        active_buffer_size = min(N_total_samples, max_active_states)

        ### Init global output buffers
        accumulated_w = self._get_accumulator(params, positions_padded)
        accumulated_nSteps = jnp.zeros((N_positions,), dtype=jnp.int32)

        ### Process in chunks to avoid OOM
        init_task_indices = jnp.arange(active_buffer_size)
        active_parent_indices = init_task_indices // n_samples
        next_task_index = jnp.array(active_buffer_size, dtype=jnp.int32)

        ### Initialize active states
        rng_keys_init = jax.vmap(lambda i: jr.fold_in(rng_key, i))(init_task_indices)
        active_states = self._get_initial_state(
            positions_padded[active_parent_indices, :],
            rng_keys_init,
            init_trust_region_size,
            active_buffer_size
        )
        

        process_calls = 0
        ### Main loop
        while True:
            active_count = jnp.sum(active_parent_indices != -1)
            if active_count == 0 and next_task_index >= N_total_samples:
                break

            active_states, active_parent_indices, accumulated_w, accumulated_nSteps, n_filled = self._process_active_states(
                # dynamic state
                active_states,
                active_parent_indices,
                accumulated_w,
                accumulated_nSteps,
                next_task_index,
                # dynamic context
                params,
                positions_padded,
                rng_key,
                init_trust_region_size,
                # Static configuration
                active_buffer_size,
                N_total_samples,
                n_samples,
                steps_between_refill,
                unroll_loop,
            )

            process_calls += 1
            next_task_index += n_filled
            if self.print_progress:
                print(f"Remaining Active States: {jnp.sum(active_parent_indices != -1)} | Next Task Index: {next_task_index}/{N_total_samples}")
                print(f"Process called so far. {process_calls}")
                print(f"Max point norm: {jnp.max(jnp.linalg.norm(active_states.position, axis=-1))}")
    
        #### Return final accumulated results averaged over samples
        final_w = jax.tree.map(lambda leaf: leaf[:N_original] / n_samples, accumulated_w)
        final_nSteps = accumulated_nSteps[:N_original] / n_samples
        final_positions = positions_padded[:N_original, :]

        return WalkerOutput(position=final_positions,
                            nSteps=final_nSteps,
                            w=final_w)
    

    def _pad_to_bucket(self, array: jnp.ndarray, bucket_size: int = 1024):
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
    

    def profile_overhead(integrator, 
                        initial_position_batch, 
                        rng_key, 
                        init_trust_region_size, 
                        n_samples, 
                        max_gpu_active_states=1024**2,
                        n_repeats=5):
        """
        Benchmarks overhead using the timeit module for higher precision.
        """
        
        print(f"--- Profiling Overhead (timeit) ---")
        print(f"Batch Size: {initial_position_batch.shape[0]} | N Samples: {n_samples}")

        # --- Setup ---
        N_total_tasks = initial_position_batch.shape[0] * n_samples
        buffer_size = min(N_total_tasks, max_gpu_active_states)
        
        # 1. Warmup / Compilation
        # We must compile before timing, or the first run will distort results.
        print("1. Compiling & Warming up...")
        warmup_pos = initial_position_batch[:min(10, initial_position_batch.shape[0])]
        _ = integrator.integrate_n_estimate(
            warmup_pos, 
            rng_key, 
            init_trust_region_size, 
            n_samples=1, 
            max_active_states=buffer_size
        ).w.block_until_ready()

        # --- 2. Benchmark Pure Physics ---
        print("2. Benchmarking Pure Physics Kernel...")
        
        # Prepare Dummy State
        dummy_keys = jr.split(rng_key, buffer_size)
        dummy_state = WalkerState(
            position=jnp.zeros((buffer_size, integrator.dim)),
            nSteps=jnp.zeros(buffer_size, dtype=jnp.int32),
            w=jnp.zeros(buffer_size),
            trust_region_size=jnp.full((buffer_size,), init_trust_region_size),
            rng_key=dummy_keys,
            done_flag=jnp.zeros(buffer_size, dtype=bool)
        )
        
        # Define the physics workload
        # We run K steps per call to dilute Python overhead
        K_benchmark_steps = 50
        
        def run_physics():
            # Crucial: We iterate Python side to mimic the management overhead structure,
            # but without the logic.
            state = dummy_state
            for _ in range(K_benchmark_steps):
                state = integrator.jit_vmap_step_fn(state)
            # Force synchronization so timeit captures the GPU work
            state.nSteps.block_until_ready()

        # Run timeit
        # number=1 means we run the function once per repeat (since the function itself does 50 steps)
        raw_times = timeit.repeat(run_physics, repeat=n_repeats, number=1)
        avg_raw_time = min(raw_times) # Min is often preferred to filter out system jitter
        
        total_raw_steps = buffer_size * K_benchmark_steps
        throughput_raw = total_raw_steps / avg_raw_time
        
        print(f"   -> Raw Physics Time: {avg_raw_time*1000:.2f} ms")
        print(f"   -> Raw Throughput:   {throughput_raw/1e6:.2f} M steps/sec")

        # --- 3. Benchmark System ---
        print("3. Benchmarking End-to-End System...")

        def run_system():
            result = integrator.integrate_n_estimate(
                initial_position_batch,
                rng_key,
                init_trust_region_size,
                n_samples=n_samples,
                max_active_states=max_gpu_active_states
            )
            # Force synchronization
            result.w.block_until_ready()
            return result

        # Run timeit
        sys_times = timeit.repeat(run_system, repeat=n_repeats, number=1)
        avg_sys_time = min(sys_times)
        
        # Calculate Total Actual Steps (we need to run it once to count them)
        # We assume steps are deterministic per seed, so we use one result to count
        final_result = run_system()
        total_actual_steps = jnp.sum(final_result.nSteps * n_samples).item()

        throughput_sys = total_actual_steps / avg_sys_time

        print(f"   -> System Time:      {avg_sys_time:.4f} s")
        print(f"   -> System Throughput:{throughput_sys/1e6:.2f} M steps/sec")

        # --- 4. Report ---
        overhead_ratio = 1.0 - (throughput_sys / throughput_raw)
        
        print("\n--- Final Report (timeit) ---")
        print(f"Raw Potential:    {throughput_raw/1e6:.2f} M steps/sec")
        print(f"Actual System:    {throughput_sys/1e6:.2f} M steps/sec")
        print(f"Efficiency:       {(1.0-overhead_ratio)*100:.2f}%")
        print(f"Overhead Loss:    {overhead_ratio*100:.2f}%")
        
        return overhead_ratio
    

class DifferentialWoSIntegrator(WoSIntegrator):
    """
    Differential Walk on Spheres integrator for solving boundary value problems with differential Wos.
    """
    def __init__(self,
                 normal_fn: Callable[[jnp.ndarray], jnp.ndarray],
                 Df: Callable[[jnp.ndarray], jnp.ndarray],
                 V_n: Callable[[jnp.ndarray], jnp.ndarray],
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
        fn_val, radius, new_trust_region_size = self.empty_sphere(params, state.position, state.trust_region_size)

        ### sample new position on sphere and inside ball
        sphere_key, ball_key, new_key = jr.split(state.rng_key, 3)
        sample_sphere = self.sphere_sampling_fn(sphere_key, dtype=state.position.dtype) * radius
        sample_ball_radius, sample_ball_direction = self.ball_sampling_fn(ball_key, dtype=state.position.dtype)
        sample_ball_radius *= radius
        sample_ball_position = state.position + (sample_ball_radius * sample_ball_direction)

        ### update state
        is_done = jnp.logical_or(state.done_flag, jnp.linalg.norm(fn_val - self.level_set) <= self.eps)
        new_nSteps = jnp.where(is_done, state.nSteps, state.nSteps + 1)
        new_position = jnp.where(is_done, state.position, 
                                 state.position + sample_sphere)
        
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
            trust_region_size=jnp.full((buffer_size,), init_tr),
            rng_key=rng_keys,
            done_flag=jnp.zeros((buffer_size,), dtype=bool),
            phase=jnp.zeros((buffer_size,), dtype=jnp.int32) # Start in phase 0
        )


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
        
        scatter = self._get_scatter_fn(free_indices, is_valid_mask, N_total_samples)
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
