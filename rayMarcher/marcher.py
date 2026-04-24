
from typing import Callable, Tuple, Any, cast

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
                 termination_crit_fn: Callable[[jnp.ndarray, jnp.ndarray, jnp.ndarray, float, float], jnp.ndarray] = level_set_termination_criterion):
        self.camera = camera
        self.background_color = background_color
        self.ray_march_fn = ray_march_fn
        self.surface_normal_fn = surface_normal_fn
        self.image_height = image_height
        self.image_width = image_width
        self.n_rays = image_height * image_width
        self._compaction_sizes = uniform_size_generator(self.n_rays, n_compactions=n_compactions)
        self._INVALID_IND = self.n_rays + 1

        # Built once: per-ray step kernel. JIT cache hits across render() calls
        # because nothing in this closure changes per call (ray_march_fn is the
        # bound surface method, captured at construction).
        step_fn = self.ray_march_state_fn(
            max_ray_march_steps,
            max_ray_distance,
            epsilon,
            level_set,
            termination_crit_fn,
        )
        self._step_states_fn = jax.jit(jax.vmap(step_fn, in_axes=(None, 0)))

        # Built once: post-process. camera_center and background_color are passed
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
                        termination_crit_fn: Callable[[jnp.ndarray, jnp.ndarray, jnp.ndarray, float, float], jnp.ndarray]
                        ) -> Callable[[Any, RayMarchState], RayMarchState]:
        def step_fn(params: Any, state: RayMarchState) -> RayMarchState:
            current_position = state.ray_origin + state.t * state.ray_direction
            fn_val, t_step, in_tr = self.ray_march_fn(params, current_position, state.ray_direction, state.trust_region_size)

            hit_now = termination_crit_fn(t_step, fn_val, state.trust_region_size, level_set_threshold, epsilon)
            hit_flag_next = jnp.logical_or(state.hit_flag, hit_now)
            done_flag_next = jnp.logical_or(
                state.done_flag, jnp.logical_or(
                    hit_flag_next,
                    jnp.logical_or(
                        state.niter >= max_ray_march_steps,
                        state.t >= max_ray_distance)
                    )
                )
            niter_next = jnp.where(done_flag_next, state.niter, state.niter + 1)
            t_next = jnp.where(done_flag_next, state.t, state.t + t_step)
            t_next = cast(jnp.ndarray, t_next)
            trust_region_size = jnp.where(done_flag_next, state.trust_region_size, update_trust_region_size(state.trust_region_size, t_step, in_tr))

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
                           camera_center: jnp.ndarray,
                           background_color: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        # Approximate view directions from the camera center to each final ray endpoint.
        view_dirs = jax.vmap(utils.normalize)(output_states.ray_position - camera_center)
        t = 0.5 * (view_dirs[:, 1] + 1.0)  # shape (N,)
        background_colors = (1.0 - t[:, None]) * jnp.array([1.0, 1.0, 1.0]) + t[:, None] * background_color
        hit_normals = jax.vmap(self.surface_normal_fn, in_axes=(None, 0))(params, output_states.ray_position)
        test_colors = jnp.clip(0.5 * hit_normals + 0.5, 0.0, 1.0)
        colors = jnp.where(output_states.hit_flag[:, None], test_colors, background_colors)
        image = colors.reshape((image_height, image_width, 3))  # type: ignore
        return image, output_states.niter.reshape((image_height, image_width))

    def render(self,
               params: Any,
               key: jax.Array,
               init_trust_region_size: float = 0.5):
        ### Initialization of ray marching state
        ray_origins, ray_directions = self.sample_camera_rays(key, self.image_height, self.image_width)
        active_states = RayMarchState(
            niter=jnp.zeros(self.n_rays, dtype=jnp.int32),
            t=jnp.zeros(self.n_rays, dtype=ray_origins.dtype),
            ray_origin=ray_origins,
            ray_direction=ray_directions,
            trust_region_size=init_trust_region_size * jnp.ones(self.n_rays),
            done_flag=jnp.zeros(self.n_rays, dtype=jnp.bool),
            hit_flag=jnp.zeros(self.n_rays, dtype=jnp.bool),
        )
        output_states = RayMarchOutput(
            niter=jnp.zeros(self.n_rays, dtype=jnp.int32),
            ray_position=jnp.zeros_like(ray_origins),
            hit_flag=jnp.zeros(self.n_rays, dtype=jnp.bool),
        )
        active_indices = jnp.arange(self.n_rays)

        ### Python control loop. Cannot be jitted because of dynamic compaction sizes
        ### (XLA needs to know static shapes and step numbers).
        current_compaction_index = 0
        while True:
            active_states = self._step_states_fn(params, active_states)
            N_valid = int(jnp.sum(~active_states.done_flag))

            should_compact_flag, new_compaction_index, new_compaction_size = should_compact(
                N_valid, current_compaction_index, self._compaction_sizes,
            )
            if should_compact_flag:
                current_compaction_index = new_compaction_index
                active_states, output_states, active_indices = perform_compaction(
                    active_states,
                    output_states,
                    active_indices,
                    active_states.done_flag,
                    fill_index_value=self._INVALID_IND,
                    new_compacted_size=new_compaction_size,
                    write_to_output_fn=write_done_to_output_arrays,
                    update_state_fn=update_ray_march_state,
                )
            if N_valid == 0:
                break

        ### Postprocessing to get the actual image
        image, niter = self._post_process_jit(
            params,
            output_states,
            self.image_height,
            self.image_width,
            self.camera.camera_center,
            self.background_color,
        )
        return image, niter
