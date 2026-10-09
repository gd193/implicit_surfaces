
from typing import Callable, Tuple, Any, Optional, cast

import jax
import jax.numpy as jnp
import jax.random as jr

from rayMarcher.camera import Camera
from rayMarcher.sampler import generate_antialiased_ray
from rayMarcher import utils
from utils.types import RayMarchState, RayMarchOutput
from utils.helpers import update_trust_region_size, level_set_termination_criterion
from utils.jnp_array_compaction import perform_compaction, should_compact, uniform_size_generator


def write_done_to_output_arrays(current_state: RayMarchState, output_state: RayMarchOutput, active_indices: jnp.ndarray) -> RayMarchOutput:
    """
    Scatters completed rays from the current local batch into the global output buffers.
    Only transfers fields present in RayMarchOutput.
    """
    new_out_niter = output_state.niter.at[active_indices].set(
        current_state.niter, mode='drop'
    )
    new_out_position = output_state.ray_position.at[active_indices].set(
        current_state.ray_origin + current_state.t[:, None] * current_state.ray_direction, mode='drop'
    )
    new_out_hit = output_state.hit_flag.at[active_indices].set(
        current_state.hit_flag, mode='drop'
    )

    return output_state._replace(
        niter=new_out_niter,
        ray_position=new_out_position,
        hit_flag=new_out_hit
    )

def update_ray_march_state(state: RayMarchState, valid_mask: jnp.ndarray) -> RayMarchState:
    forced_done_flags = jnp.logical_or(state.done_flag, ~valid_mask)
    return state._replace(done_flag=forced_done_flags)

class SimpleIntegrator:
    """
    A simple ray marching integrator that can be used to render implicit surfaces.

    All static marcher config (image size, step caps, level set, epsilon, n_compactions)
    is taken in __init__ so the JIT-compiled step kernel and post-process are built once
    and survive across render() calls. `camera` and `background_color` are mutable
    instance attributes — assign new values between calls to orbit/animate without
    triggering a re-trace.

    scene_bounds is a world-space (lower, upper) box, defaulting to [-2, 2]^3.
    Set it to None to disable scene clipping. Cameras may start outside it.
    """
    def __init__(self,
                 camera: Camera,
                 background_color: jnp.ndarray,
                 # ray_march_fn: (params, positions, directions, trust_region_size) -> (fn_val, t_step, inside_tr)
                 ray_march_fn: Callable[[Any, jnp.ndarray, jnp.ndarray, jnp.ndarray], tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]],
                 surface_normal_fn: Callable[[Any, jnp.ndarray], jnp.ndarray],
                 *,
                 image_height: int,
                 image_width: int,
                 max_ray_march_steps: int,
                 max_ray_distance: float,
                 level_set: float,
                 epsilon: float,
                 n_compactions: int,
                 termination_crit_fn: Callable[..., jnp.ndarray] = level_set_termination_criterion,
                 update_tr_fn: Callable[[jnp.ndarray, jnp.ndarray, jnp.ndarray], jnp.ndarray] = update_trust_region_size,
                 # Used only when `termination_crit_fn.needs_grad` is True: the
                 # marcher calls this in place of `ray_march_fn`. Pass
                 # `surface.ray_step_with_grad`. Default base-class implementation
                 # of that method is `ray_step + jax.grad(surface)`; subclasses
                 # with a cheap grad path (e.g. autobound) override it to use
                 # bookkeeping they already do.
                 ray_march_with_grad_fn: Optional[Callable[..., tuple]] = None,
                 scene_bounds: utils.SceneBounds | None = utils.DEFAULT_SCENE_BOUNDS):
        self.camera = camera
        self.background_color = background_color
        self.ray_march_fn = ray_march_fn
        self.surface_normal_fn = surface_normal_fn
        self.ray_march_with_grad_fn = ray_march_with_grad_fn
        self.scene_bounds = utils.validate_scene_bounds(scene_bounds)
        if getattr(termination_crit_fn, 'needs_grad', False) and ray_march_with_grad_fn is None:
            raise ValueError(
                f"termination_crit_fn={termination_crit_fn.__name__} declares needs_grad; "
                f"pass ray_march_with_grad_fn (e.g. surface.ray_step_with_grad)."
            )
        self.image_height = image_height
        self.image_width = image_width
        self.n_rays = image_height * image_width
        self.n_compactions = n_compactions

        # Built once: per-ray step kernel. JIT cache hits across render() calls
        # because nothing in this closure changes per call (ray_march_fn is the
        # bound surface method, captured at construction).
        step_fn = self.ray_march_state_fn(
            max_ray_march_steps,
            max_ray_distance,
            epsilon,
            level_set,
            termination_crit_fn,
            update_tr_fn,
        )
        self._step_states_fn = jax.jit(jax.vmap(step_fn, in_axes=(None, 0)))

        # Built once: post-process. ray directions and background_color are passed
        # as traced arguments (NOT closed over), so mutating self.camera does not
        # invalidate this cache.
        self._post_process_jit = jax.jit(
            self._post_process_impl,
            static_argnums=(2, 3),  # image_height, image_width
        )

    def sample_camera_rays(self, key: jax.Array, image_height: int, image_width: int):
        i_coords, j_coords = jnp.meshgrid(jnp.arange(image_width), jnp.arange(image_height))
        i_coords = i_coords.flatten()
        j_coords = j_coords.flatten()
        pixel_keys = jr.split(key, image_height * image_width)
        ray_origins, ray_directions = jax.vmap(generate_antialiased_ray, in_axes=(None, 0, 0, 0))(self.camera, i_coords, j_coords, pixel_keys)
        return ray_origins, ray_directions


    def ray_march_state_fn(self,
                        # static config
                        max_ray_march_steps: int,
                        max_ray_distance: float,
                        epsilon: float,
                        level_set_threshold: float,
                        termination_crit_fn: Callable[..., jnp.ndarray],
                        update_tr_fn: Callable[[jnp.ndarray, jnp.ndarray, jnp.ndarray], jnp.ndarray] = update_trust_region_size,
                        ) -> Callable[[Any, RayMarchState], RayMarchState]:
        needs_grad = getattr(termination_crit_fn, 'needs_grad', False)
        # Pick the per-ray step kernel ONCE at construction, based on whether the
        # criterion needs the gradient. No flag threaded through traced code.
        if needs_grad:
            ray_march_step = self.ray_march_with_grad_fn
        else:
            ray_march_step = self.ray_march_fn

        def step_fn(params: Any, state: RayMarchState) -> RayMarchState:
            entry, exit, intersects = utils.ray_scene_interval(
                state.ray_origin, state.ray_direction, self.scene_bounds)
            end_t = jnp.minimum(exit, max_ray_distance)
            intersects = intersects & (entry <= end_t)
            current_t = jnp.where(intersects & ~state.done_flag,
                                  jnp.maximum(state.t, entry), state.t)
            current_position = state.ray_origin + current_t * state.ray_direction
            if needs_grad:
                fn_val, t_step, in_tr, grad = ray_march_step(params, current_position, state.ray_direction, state.trust_region_size)
                grad_norm = jnp.linalg.norm(grad)
                hit_now = termination_crit_fn(t_step, fn_val, state.trust_region_size, 0.0, epsilon, grad_norm=grad_norm)
            else:
                fn_val, t_step, in_tr = ray_march_step(params, current_position, state.ray_direction, state.trust_region_size)
                hit_now = termination_crit_fn(t_step, fn_val, state.trust_region_size, 0.0, epsilon)
            hit_now = hit_now & intersects & (current_t <= end_t) & ~state.done_flag
            hit_flag_next = jnp.logical_or(state.hit_flag, hit_now)
            done_flag_next = jnp.logical_or(
                state.done_flag | ~intersects, jnp.logical_or(
                    hit_flag_next,
                    jnp.logical_or(
                        state.niter >= max_ray_march_steps,
                        current_t >= end_t)
                    )
                )
            niter_next = jnp.where(done_flag_next, state.niter, state.niter + 1)
            t_next = jnp.where(done_flag_next, current_t, jnp.minimum(current_t + t_step, end_t))
            t_next = cast(jnp.ndarray, t_next)
            trust_region_size = jnp.where(done_flag_next, state.trust_region_size, update_tr_fn(state.trust_region_size, t_step, in_tr))

            return RayMarchState(
                niter=niter_next,
                t=t_next,
                ray_origin=state.ray_origin,
                ray_direction=state.ray_direction,
                trust_region_size=trust_region_size,
                done_flag=done_flag_next,
                hit_flag=hit_flag_next,
            )
        return step_fn

    def _post_process_impl(self,
                           params: Any,
                           output_states: RayMarchOutput,
                           image_height: int,
                           image_width: int,
                           ray_directions: jnp.ndarray,
                           background_color: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        # Misses may terminate at t=0; use the actual sampled directions.
        view_dirs = ray_directions
        t = 0.5 * (view_dirs[:, 1] + 1.0)  # shape (N,)
        background_colors = (1.0 - t[:, None]) * jnp.array([1.0, 1.0, 1.0]) + t[:, None] * background_color
        hit_normals = jax.vmap(self.surface_normal_fn, in_axes=(None, 0))(params, output_states.ray_position)
        test_colors = jnp.clip(0.5 * hit_normals + 0.5, 0.0, 1.0)
        colors = jnp.where(output_states.hit_flag[:, None], test_colors, background_colors)
        image = colors.reshape((image_height, image_width, 3))  # type: ignore
        return image, output_states.niter.reshape((image_height, image_width))

    def trace_rays(self,
                   params: Any,
                   ray_origins: jnp.ndarray,
                   ray_directions: jnp.ndarray,
                   init_trust_region_size: float = 0.5,
                   active_mask: Optional[jnp.ndarray] = None) -> RayMarchOutput:
        """Intersect a batch of rays using the same adaptive, compacted marcher.

        Directions are normalized so steps and distance limits are world-space.
        Inactive or zero-direction rays return misses at their original positions.
        The Python compaction loop is intentionally outside JIT, as in render().
        """
        n_rays = ray_origins.shape[0]
        lengths = jnp.linalg.norm(ray_directions, axis=-1)
        valid = jnp.isfinite(lengths) & (lengths > 0)
        ray_directions = jnp.where(
            valid[:, None], ray_directions / jnp.where(valid, lengths, 1)[:, None], 0)
        active_mask = valid if active_mask is None else active_mask & valid
        compaction_sizes = uniform_size_generator(n_rays, self.n_compactions)
        invalid_index = n_rays + 1
        active_states = RayMarchState(
            niter=jnp.zeros(n_rays, dtype=jnp.int32),
            t=jnp.zeros(n_rays, dtype=ray_origins.dtype),
            ray_origin=ray_origins,
            ray_direction=ray_directions,
            trust_region_size=init_trust_region_size * jnp.ones(n_rays, dtype=ray_origins.dtype),
            done_flag=~active_mask,
            hit_flag=jnp.zeros(n_rays, dtype=jnp.bool),
        )
        output_states = RayMarchOutput(
            niter=jnp.zeros(n_rays, dtype=jnp.int32),
            ray_position=ray_origins,
            hit_flag=jnp.zeros(n_rays, dtype=jnp.bool),
        )
        active_indices = jnp.arange(n_rays)

        ### Python control loop. Cannot be jitted because of dynamic compaction sizes
        ### (XLA needs to know static shapes and step numbers).
        current_compaction_index = 0
        N_valid = int(jnp.sum(active_mask))
        while N_valid > 0:
            active_states = self._step_states_fn(params, active_states)
            N_valid = int(jnp.sum(~active_states.done_flag))

            should_compact_flag, new_compaction_index, new_compaction_size = should_compact(
                N_valid, current_compaction_index, compaction_sizes,
            )
            if should_compact_flag:
                current_compaction_index = new_compaction_index
                active_states, output_states, active_indices = perform_compaction(
                    active_states,
                    output_states,
                    active_indices,
                    active_states.done_flag,
                    fill_index_value=invalid_index,
                    new_compacted_size=new_compaction_size,
                    write_to_output_fn=write_done_to_output_arrays,
                    update_state_fn=update_ray_march_state,
                )
            if N_valid == 0:
                break

        # Final flush: write any state that hasn't been written to output_states
        # yet (rays that terminated since the last compaction, or all rays when
        # n_compactions=0). Without this, n_compactions=0 produces all-zero
        # output buffers because compaction is the only path that writes the
        # output. Always-on; cheap when most rays already flushed.
        output_states = write_done_to_output_arrays(
            active_states, output_states, active_indices,
        )

        return output_states

    def render(self,
               params: Any,
               key: jax.Array,
               init_trust_region_size: float = 0.5):
        ray_origins, ray_directions = self.sample_camera_rays(key, self.image_height, self.image_width)
        output_states = self.trace_rays(params, ray_origins, ray_directions, init_trust_region_size)

        ### Postprocessing to get the actual image
        image, niter = self._post_process_jit(
            params,
            output_states,
            self.image_height,
            self.image_width,
            ray_directions,
            self.background_color,
        )
        return image, niter
