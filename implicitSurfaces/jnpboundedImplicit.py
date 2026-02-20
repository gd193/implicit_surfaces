from typing import NamedTuple, Tuple, cast
from matplotlib.pyplot import box
from implicitSurfaces.implicit_surface import *
import jnp_bounds.autobounds as jb
from functools import partial
from utils.jnpbounds_roots import roots_2nd_order_tensor
from utils.jnpbound_empty_sphere import empty_sphere_2nd_order_tensor
from utils.jnp_array_compaction import perform_compaction, should_compact, uniform_size_generator
from rayMarcher.utils import ray_aabb_intersection_inside
from utils.types import RayMarchState, RayMarchOutput


class jnpboundedImplicit(implicitSurface):
    """
    A class for implicit surfaces that can bounded by the jnp autobound library.
    Current limitations of autobound:
        - Only JAX-traceable functions can be automatically bounded.
        - Many JAX library functions are not yet supported. 
          What is supported is bounding MLPs or CNNs that uses the jax.nn.sigmoid, jax.nn.softplus, or jax.nn.swish activations.
        - To compute accurate bounds for deeper neural networks, you may need to use float64 rather than float32.
        - Tracing times can be very long because the library uses tons of explicit for loops --> jax.jit unrolls them,
          which can lead to long compilation times. (1-10minutes)
    For details see https://github.com/google/autobound/
    """
    def __init__(self, 
                autobounds_max_degree: int,
                **kwargs
                ):
        """
        Initialize the autobounded implicit surface.

        Args:
            fn (Callable): A jnp autobound-able JAX function that defines the implicit surface.
            params (Optional[dict]): Additional parameters for the jnp autobound library.
            level_set (float): The level set value for the implicit surface.
        """
        super().__init__(**kwargs)
        self.autobounds_max_degree = autobounds_max_degree
        self.taylor_bounds = jb.taylor_bounds(self.fn, max_degree=self.autobounds_max_degree)


    def ray_step(self, params, ray_origin, ray_direction, trust_region_size, upper_lower_flag=None):

        auto_bound = self.taylor_bounds(ray_origin, x_trust_region=None, trust_region_size=trust_region_size)
        center_fn_val = auto_bound.polynomial_enclosure[0]
        taylor_enclosure = auto_bound.polynomial_enclosure

        if upper_lower_flag is None:
            upper_lower_flag = jnp.where(center_fn_val > self.level_set, jnp.array(-1), jnp.array(1)).squeeze() # -1 for lower, 1 for upper
            
        taylor_enclosure.real_coeffs = taylor_enclosure.real_coeffs.at[0].add(-self.level_set)
        hit, t_poly_intersection = roots_2nd_order_tensor(taylor_enclosure, ray_direction, upper_lower_flag, dim=ray_origin.shape[0])
        ### check if point is within bounds
        t_box_intersection = ray_aabb_intersection_inside(ray_origin, ray_direction, trust_region_size)
        in_box = jnp.logical_and(hit, t_poly_intersection < t_box_intersection)
        t = jnp.where(in_box, t_poly_intersection, t_box_intersection)

        center_fn_val = cast(jnp.ndarray, center_fn_val)
        t = cast(jnp.ndarray, t)

        return center_fn_val, t, jnp.logical_not(in_box)
    

    def empty_sphere(self, params: jnp.ndarray, x: jnp.ndarray, trust_region_size: jnp.ndarray, upper_lower_flag: Optional[str]=None) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """
        Compute the largest empty sphere around the given position within the trust region.
        We assume global bounds from the autobound library, hence the trust_region_size is inf
        args:
            x (jnp.ndarray)(dim,): The current position of the walker
            trust_region_size (jnp.ndarray)(1,): The current trust region size
        Returns:
            fn_val (jnp.ndarray)(1,): The value of the function at the current position
            radius (jnp.ndarray)(1,): The radius of the largest empty sphere
            trust_region_empty_flag (jnp.ndarray)(1,): A boolean flag indicating whether the entire trust region is empty (True) or not (False)
        """
        auto_bound = self.taylor_bounds(params, x_trust_region=None, trust_region_size=trust_region_size)
        taylor_enclosure = auto_bound.polynomial_enclosure
        fn_val = taylor_enclosure[0].squeeze()

        if upper_lower_flag is None:
            upper_lower_flag = jnp.where(fn_val.squeeze() > self.level_set, jnp.array(-1), jnp.array(1)) # stay on same side of level set
        radius = empty_sphere_2nd_order_tensor(taylor_enclosure, upper_lower_flag, dim=x.shape[0], isolevel=self.level_set).squeeze()
        outside_trust_region = radius > trust_region_size * 0.5
        safe_radius = jnp.where(outside_trust_region, trust_region_size * 0.5, radius)
        return fn_val, safe_radius, outside_trust_region


#     def ray_marching_step_fn(self, spatial_eps: float, level_set_eps: float, max_distance: float, upper_lower_flag: Optional[str]=None):

#         def step_fn(ray: RayMarchState):
#             in_box, hit_point, t, t_box_intersection, center_fn_val = self.ray_intersection_bounded(ray.ray_position, ray.ray_direction, ray.trust_region_size, upper_lower_flag)

#             ### Check termination criteria
#             is_hit = (t < spatial_eps) | (jnp.abs(center_fn_val - self.level_set) < level_set_eps)
#             distance_traveled = jnp.linalg.norm(hit_point - ray.ray_origin)
#             is_done = jnp.logical_or(is_hit, distance_traveled >= max_distance)
            
#             ### Update loop variables
#             new_niter = jnp.where(is_done, ray.niter, ray.niter + 1)
#             new_ray_position = jnp.where(is_done, ray.ray_position, hit_point)
#             new_trust_region_size = jax.lax.cond(
#                 is_done,
#                 no_update_trust_region, update_trust_region,
#                 *(ray.trust_region_size, t, t_box_intersection, in_box)
#             )

#             return RayMarchState(new_niter, new_ray_position, ray.ray_direction, new_trust_region_size, ray.ray_origin, is_done, is_hit)
#         return step_fn

#     def ray_surface_intersection_batch_fn(self, max_distance: float=1e4, spatial_eps: float=1e-5, level_set_eps: float=1e-5, max_iterations: int=100, init_interval: float=1.0, upper_lower_flag: Optional[str]=None):
#         """
#         Compute the ray intersection with the implicit surface for a batch of rays.
#         Args:
#             max_distance (float): The maximum distance to search for an intersection.
#             spatial_eps (float): The spatial epsilon termination criterion.
#             level_set_eps (float): The level set epsilon termination criterion.
#             max_iterations (int): The maximum number of iterations for the ray-surface intersection.
#             init_interval (float): The initial interval size for the bounding boxes.
#             upper_lower_flag (Optional[str]): Flag to indicate whether to use the upper or lower bound for the intersection.
#         Returns:
#             hit_flags (jnp.ndarray): (N,) A boolean array indicating if there is an intersection with the surface for each ray.
#             hit_points (jnp.ndarray): (N, 3) The coordinates of the intersection points for each ray.
#             n_iters (jnp.ndarray): (N,) The number of iterations taken for each ray.
#         """
#         # Pre-compile the step function (VMAP over the RayMarchState)
#         step_fn_jit_vmap = jax.jit(jax.vmap(self.ray_marching_step_fn(spatial_eps, level_set_eps, max_distance, upper_lower_flag)))
#         def batch_intersection_fn(ray_origins, ray_directions):
            
#             ### Python control loop variables
#             iter = 0
#             current_compaction_index = 0
#             init_compaction_sizes(uniform_size_generator(ray_directions.shape[0], n_compactions=10))
#             INVALID_IND = ray_directions.shape[0] + 1
#             ### JAX loop and output states
#             active_state, output_state, active_indices = _init_ray_marching_states(ray_directions.shape[0], 
#                                                                                                     init_interval, 
#                                                                                                     ray_origins, 
#                                                                                                     ray_directions)
#             ### Python control loop for ray marching with compaction
#             ### This cannot be jitted because of the dynamic compaction sizes (XLA needs to know static shapes and step numbers)
#             while(True):
#                 active_state = step_fn_jit_vmap(active_state)
#                 iter += 1
#                 N_valid = int(jnp.sum(~active_state.done_flag))


#                 should_compact_flag, new_compaction_index, new_compaction_size = should_compact(N_valid, current_compaction_index)
#                 if should_compact_flag:
                    
#                     current_compaction_index = new_compaction_index

#                     active_state, output_state, active_indices = perform_compaction(
#                         active_state,
#                         output_state,
#                         active_indices,
#                         fill_index_value=INVALID_IND,
#                         new_compacted_size=new_compaction_size
#                     )
#                 if N_valid == 0 or iter >= max_iterations:
#                     break

#             return output_state
#         return batch_intersection_fn


# def update_trust_region(trust_region_size, t, t_box_intersection, in_box):
#     trust_region_size = jnp.where(in_box, trust_region_size, trust_region_size * jnp.sqrt(2))

#     box_too_big = t < 0.5 * t_box_intersection

#     trust_region_size = jnp.where(box_too_big, trust_region_size * 1/jnp.sqrt(2), trust_region_size)
#     return trust_region_size.squeeze()

# def no_update_trust_region(trust_region_size, t, t_box_intersection, in_box):
#     return trust_region_size


# @partial(jax.jit, static_argnames=('N_rays', 'init_interval'))
# def _init_ray_marching_states(N_rays: int, init_interval: float, ray_origins: jnp.ndarray, ray_directions: jnp.ndarray) -> tuple[jnp.ndarray, ...]:
#     """
#     Initialize the ray marching loop variables and output arrays.
#     """
#     ### init loop variables
#     niters = jnp.zeros(N_rays, dtype=jnp.int32)
#     ray_origins = jnp.ones_like(ray_directions) * ray_origins

#     intervals = jnp.ones(N_rays) * init_interval
#     done_flags = jnp.zeros(N_rays, dtype=bool)
#     hit_flags = jnp.zeros(N_rays, dtype=bool)

#     ### output arrays
#     out_ray_origins = jnp.zeros_like(ray_origins)
#     out_hit_flags = jnp.zeros_like(hit_flags)
#     out_niters = jnp.zeros_like(niters)

#     ### active indices
#     active_indices = jnp.arange(N_rays)

#     return RayMarchState(niters, ray_origins, ray_directions, intervals, ray_origins, done_flags, hit_flags), RayMarchOutput(out_niters, out_ray_origins, out_hit_flags), active_indices