

class SimpleIntegrator:
    """
    A simple ray marching integrator that can be used to render implicit surfaces.
    """
    def __init__(self, 
                 camera: Camera,
                 background_color: jnp.ndarray,
                 # ray_march_fn: (params, positions, directions, trust_region_size) -> (fn_val, t_step, inside_tr)
                 ray_march_fn: Callable[[Any, jnp.ndarray, jnp.ndarray, jnp.ndarray], tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]],
                 surface_normal_fn: Callable[[Any, jnp.ndarray], jnp.ndarray]):

         self.camera = camera
         self.background_color = background_color
         self.ray_march_fn = ray_march_fn
         self.surface_normal_fn = surface_normal_fn


    def sample_camera_rays(self, key: jax.Array, image_height: int, image_width: int):
        i_coords, j_coords = jnp.meshgrid(jnp.arange(image_width), jnp.arange(image_height))
        i_coords = i_coords.flatten()
        j_coords = j_coords.flatten()
        pixel_keys = jr.split(key, image_height * image_width)
        ray_origins, ray_directions = jax.vmap(generate_antialiased_ray, in_axes=(None, 0, 0, 0))(self.camera, i_coords, j_coords, pixel_keys)
        return ray_origins, ray_directions
    

    def ray_march_state_fn(self,
                        # statig config
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
            t_next = cast(t_next, jnp.ndarray)
            trust_region_size = jnp.where(done_flag_next, state.trust_region_size, update_trust_region_size(state.trust_region_size, t_step, in_tr))

            return RayMarchState(niter_next, t_next, state.ray_origin, state.ray_direction, trust_region_size, done_flag_next, hit_flag_next)
        return step_fn

    def post_process_output(self, params, output_states: RayMarchOutput, image_height: int, image_width: int) -> Tuple[jnp.ndarray, jnp.ndarray]:
        unit_direction_background = jax.vmap(utils.normalize)(output_states.ray_position)
        t = 0.5 * (unit_direction_background[1] + 1.0)
        background_colors = (1.0 - t) * jnp.array([1.0, 1.0, 1.0]) + t * self.background_color                           
        hit_normals = jax.vmap(self.surface_normal_fn, in_axes=(None, 0))(params, output_states.ray_position)
        test_colors = jnp.clip(0.5 * hit_normals + jnp.array([1., 1., 1.]), 0.0, 1.0)
        colors = jnp.where(output_states.hit_flag[:, None], test_colors, background_colors)
        image = colors.reshape((image_height, image_width, 3)) # type: ignore
        return image, output_states.niter.reshape((image_height, image_width))

    def render(self,
               params: Any,
               key: jax.Array,
               image_height: int,
               image_width: int,
               max_ray_march_steps: int, 
               max_ray_distance: float,
               level_set: float,
               epsilon: float,
               init_trust_region_size: float,
               n_compactions: int
               ):
        ### Initialization of ray marching state
        ray_origins, ray_directions = self.sample_camera_rays(key, image_height, image_width)
        active_states = RayMarchState(
            niter=jnp.zeros(ray_origins.shape[0], dtype=jnp.int32),
            t=jnp.zeros(ray_origins.shape[0], dtype=ray_origins.dtype),
            ray_origin=ray_origins,
            ray_direction=ray_directions,
            trust_region_size=init_trust_region_size * jnp.ones(ray_origins.shape[0]),
            done_flag=jnp.zeros(ray_origins.shape[0], dtype=jnp.bool),
            hit_flag=jnp.zeros(ray_origins.shape[0], dtype=jnp.bool)
        )
        output_states = RayMarchOutput(
            niter=jnp.zeros(ray_origins.shape[0], dtype=jnp.int32),
            ray_position=jnp.zeros_like(ray_origins),
            hit_flag=jnp.zeros(ray_origins.shape[0], dtype=jnp.bool)
        )
        active_indices = jnp.arange(ray_origins.shape[0])

        step_states_fn = jax.jit(jax.vmap(
            self.ray_march_state_fn(
                max_ray_march_steps, 
                max_ray_distance, 
                epsilon, level_set, 
                level_set_termination_criterion
            ),
            in_axes=(None, 0)
            )
        )
        post_process_fn = jax.jit(self.post_process_output, static_argnums=(3, 4))

        ### Python control loop variables
        iter = 0
        current_compaction_index = 0
        init_compaction_sizes(uniform_size_generator(ray_directions.shape[0], n_compactions=n_compactions))
        INVALID_IND = ray_directions.shape[0] + 1
        ### This cannot be jitted because of the dynamic compaction sizes (XLA needs to know static shapes and step numbers)
        while(True):
            active_states = step_states_fn(params, active_states)
            iter += 1
            N_valid = int(jnp.sum(~active_states.done_flag))


            should_compact_flag, new_compaction_index, new_compaction_size = should_compact(N_valid, current_compaction_index)
            if should_compact_flag:
                
                current_compaction_index = new_compaction_index

                active_states, output_states, active_indices = perform_compaction(
                    active_states,
                    output_states,
                    active_indices,
                    fill_index_value=INVALID_IND,
                    new_compacted_size=new_compaction_size
                )
            if N_valid == 0:
                break

        ### Postprocessing to get the actual image
        image, niter = post_process_fn(params, output_states, image_height, image_width)
        return image, niter


