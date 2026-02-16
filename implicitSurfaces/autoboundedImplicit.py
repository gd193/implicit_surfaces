from implicitSurfaces.implicit_surface import *
import autobound.jax as ab
from functools import partial
from utils.ab_bounds_conversion import autobounds_coefficients_to_power_series
from utils.bernstein_polynomial_transform_nd import transform_power_series_interval_to_bernstein
from utils.autobounds_roots import transform_2nd_order_tensor, roots_2nd_order_tensor
from utils.jnp_array_compaction import perform_compaction, should_compact, init_compaction_sizes, uniform_size_generator
from rayMarcher.utils import ray_aabb_intersection_inside


class autoboundedImplicit(implicitSurface):
    """
    A class for implicit surfaces that can bounded by the autobound library.
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
                fn: Callable, 
                autobounds_max_degree: int, 
                level_set: float=None,
                classification_flag: str=None):
        """
        Initialize the autobounded implicit surface.

        Args:
            fn (Callable): A autobound-able JAX function that defines the implicit surface.
            params (Optional[dict]): Additional parameters for the autobound library.
            level_set (float): The level set value for the implicit surface.
        """
        super().__init__(fn, level_set)
        self.autobounds_max_degree = autobounds_max_degree
        self.taylor_bounds = jax.jit(ab.taylor_bounds(self.fn, max_degree=self.autobounds_max_degree))
        self.fn = jax.jit(self.fn)

        if classification_flag == 'bernstein':
            self.classify_box = self.classify_box_bernstein
        elif classification_flag == 'control_points':
            self.classify_box = self.classify_box_control_points
        else:
            raise ValueError("classification_flag must be either 'bernstein' or 'control_points'.")
        self.classification_flag = classification_flag


    def upper_bound_below_level_set_control_points(self, operands: tuple):
        """
        Check if the upper bound of the function is below the level set value.

        Args:
            operands (tuple): A tuple containing the lower bound values at the box edges

        Returns:
            result (() jnp.ndarray): A boolean scalar  indicating if the upper bound is below the level set.
        """
        bounds_lower, bounds_upper = operands
        return jnp.all(bounds_upper < self.level_set)
    
    def upper_bound_below_level_set_bernstein(self, operands: tuple):
        """
        Check if the upper bound of the function is below the level set value using Bernstein polynomial coefficients.

        Args:
            operands (tuple): A tuple containing the bernstein polynomial coefficients

        Returns:
            result (() jnp.ndarray): A boolean scalar indicating if the upper bound is below the level set.
        """
        bs_coeff_lower, bs_coeff_upper = operands
        return jnp.all(bs_coeff_upper < self.level_set)

    def lower_bound_above_level_set_control_points(self, operands: tuple):
        """
        Check if the lower bound of the function is above the level set value.

        Args:
            operands (tuple): A tuple containing the lower bound values at the box edges

        Returns:
            result (() jnp.ndarray): A boolean scalar indicating if the lower bound is above the level set.
        """
        bounds_lower, bounds_upper = operands
        return jnp.all(bounds_lower > self.level_set)
    
    def lower_bound_above_level_set_bernstein(self, operands: tuple):
        """
        Check if the lower bound of the function is above the level set value using Bernstein polynomial coefficients.

        Args:
            operands (tuple): A tuple containing the bernstein polynomial coefficients

        Returns:
            result (() jnp.ndarray): A boolean scalar indicating if the lower bound is above the level set.
        """
        bs_coeff_lower, bs_coeff_upper = operands
        return jnp.all(bs_coeff_lower > self.level_set)


    @partial(jax.jit, static_argnames=('self',))
    def classify_box_control_points(self, box_aa, box_bb):
        """
        Determine whether the function contains the level_set value within the box defined by box_aa and box_bb
        using control points as the box edges.

        Args:
            box_aa (jnp.ndarray): The lower corner of the axis-aligned bounding box.
            box_bb (jnp.ndarray): The upper corner of the axis-aligned bounding box.

        Returns:
            fn_val (jnp.ndarray): The value of the function at the center of the box.
            box_class (jnp.ndarray): The classification of the box based on the function value and level set.
        """
        print("recompile triggered")
        box_center = (box_aa + box_bb) / 2
        box_edges = jnp.array([
            box_aa,
            jnp.array([box_aa[0], box_bb[1]]),
            jnp.array([box_bb[0], box_aa[1]]),
            box_bb,
        ])
        auto_bound = self.taylor_bounds(box_center, (box_aa, box_bb))
        center_fn_val = self.fn(box_center).squeeze()
        bounds_lower = jax.vmap(auto_bound.lower)(box_edges)
        bounds_upper = jax.vmap(auto_bound.upper)(box_edges)

        box_level_stable = jax.lax.cond(center_fn_val > self.level_set,
                                        self.lower_bound_above_level_set_control_points,
                                        self.upper_bound_below_level_set_control_points,
                                        (bounds_lower, bounds_upper, box_edges))
        box_class = jax.lax.select(box_level_stable,
                                   jax.lax.select(center_fn_val > self.level_set, BC_CLASS[0], BC_CLASS[1]),
                                   BC_CLASS[2])

        return center_fn_val, box_class
    
    @partial(jax.jit, static_argnames=('self',))
    def classify_box_bernstein(self, box_aa, box_bb):
        """
        Determine whether the function contains the level_set value within the box defined by box_aa and box_bb
        using Bernstein polynomial coefficients as the polynomial bounds.

        Args:
            box_aa (jnp.ndarray): The lower corner of the axis-aligned bounding box.
            box_bb (jnp.ndarray): The upper corner of the axis-aligned bounding box.

        Returns:
            fn_val (jnp.ndarray): The value of the function at the center of the box.
            box_class (jnp.ndarray): The classification of the box based on the function value and level set.
        """
        box_center = (box_aa + box_bb) / 2
        auto_bound = self.taylor_bounds(box_center, (box_aa, box_bb))
        center_fn_val = auto_bound.lower(auto_bound.x0).squeeze()
        ps_coeff_lower = autobounds_coefficients_to_power_series(auto_bound.coefficients, 
                                                                 dimension=box_aa.shape[-1], 
                                                                 degree=self.autobounds_max_degree, 
                                                                 upper_lower_flag='lower')
    
        ps_coeff_upper = autobounds_coefficients_to_power_series(auto_bound.coefficients,
                                                                 dimension=box_aa.shape[-1],
                                                                 degree=self.autobounds_max_degree,
                                                                 upper_lower_flag='upper')

        bs_coeff_lower = transform_power_series_interval_to_bernstein(ps_coeff_lower, jnp.stack([box_aa - box_center, 
                                                                                                 box_bb - box_center]))
        bs_coeff_upper = transform_power_series_interval_to_bernstein(ps_coeff_upper, jnp.stack([box_aa - box_center,
                                                                                                 box_bb - box_center]))
        
        box_level_stable = jax.lax.cond(center_fn_val > self.level_set,
                                        self.lower_bound_above_level_set_control_points,
                                        self.upper_bound_below_level_set_control_points,
                                        (bs_coeff_lower, bs_coeff_upper))
        box_class = jax.lax.select(box_level_stable,
                                   jax.lax.select(center_fn_val > self.level_set, BC_CLASS[0], BC_CLASS[1]),
                                   BC_CLASS[2])

        return center_fn_val, box_class
    
    
    def ray_intersection_bounded(self, ray_origin: jnp.ndarray, ray_direction: jnp.ndarray, trust_region_size: jnp.ndarray, upper_lower_flag: Optional[str]=None):
        """
        Compute the ray intersection with the implicit surface defined by the power series coefficients inside the bounding box.
        Args:
            ray_origin (jnp.ndarray): The origin of the ray.
            ray_direction (jnp.ndarray): The direction of the ray.
            trust_region_size (jnp.ndarray): ( ) The length of the axis-aligned bounding box edges in each dimension. Same for all dimensions. Centered at ray_origin.
            upper_lower_flag (Optional[str]): Indicates whether to use the upper or lower bound.


        Returns:
            in_box (jnp.ndarray): A boolean indicating if there is an intersection in the bounding box.
            hit_point (jnp.ndarray): The coordinates of the intersection point.
            t (jnp.ndarray): The distance to the intersection point along the ray.
            t_box_intersection (jnp.ndarray): The distance to the intersection point with the bounding box.
        """
        bbox_aa = ray_origin - jnp.ones_like(ray_origin) * trust_region_size / 2
        bbox_bb = ray_origin + jnp.ones_like(ray_origin) * trust_region_size / 2
        auto_bound = self.taylor_bounds(ray_origin, (bbox_aa, bbox_bb))
        center_fn_val = auto_bound.coefficients[0][0]

        ps_coeff = autobounds_coefficients_to_power_series(
            auto_bound.coefficients,
            dimension=bbox_aa.shape[-1],
            degree=self.autobounds_max_degree,
        )
        if upper_lower_flag is None:
            upper_lower_flag = jnp.where(center_fn_val > self.level_set, jnp.array(-1), jnp.array(1))

        zero_element = (0,) * ps_coeff.ndim
        ps_coeff = ps_coeff.at[zero_element].add(-self.level_set)
        hit, t_poly_intersection = roots_2nd_order_tensor(ps_coeff, ray_direction, upper_lower_flag)
        ### check if point is within bounds
        t_box_intersection = ray_aabb_intersection_inside(ray_origin, ray_direction, bbox_aa, bbox_bb)
        in_box = jnp.logical_and(hit, t_poly_intersection < t_box_intersection)
        t = jnp.where(in_box, t_poly_intersection, t_box_intersection)
        hit_point = ray_origin + t * ray_direction

        return in_box, hit_point, t, t_box_intersection, center_fn_val
    

    #@partial(jax.jit, static_argnames=('self',))
    def ray_intersection_lower_bound(self, ray_origin: jnp.ndarray, ray_direction: jnp.ndarray, trust_region_size: jnp.ndarray):
        """
        Compute the ray intersection with the implicit surface within the axis-aligned bounding box using the lower bound.
        Use when the function value at the box center is above the level set.
        """
        return self.ray_intersection_bounded(ray_origin, ray_direction, trust_region_size, upper_lower_flag='lower')


    #@partial(jax.jit, static_argnames=('self',))
    def ray_intersection_upper_bound(self, ray_origin: jnp.ndarray, ray_direction: jnp.ndarray, trust_region_size: jnp.ndarray):
        """
        Compute the ray intersection with the implicit surface within the axis-aligned bounding box using the upper bound.
        Use when the function value at the box center is below the level set.
        """
        return self.ray_intersection_bounded(ray_origin, ray_direction, trust_region_size, upper_lower_flag='upper')

    def ray_marching_step_fn(self, spatial_eps: float, level_set_eps: float, max_distance: float, ray_origin_init: jnp.ndarray):

        def step_fn(
                niter: jnp.ndarray, 
                ray_origin: jnp.ndarray, 
                ray_direction: jnp.ndarray, 
                trust_region_size: jnp.ndarray, 
                done_flag: jnp.ndarray,
                hit_flag: jnp.ndarray,):
            in_box, hit_point, t, t_box_intersection, center_fn_val = self.ray_intersection_bounded(ray_origin, ray_direction, trust_region_size)
            hit_flag = (t < spatial_eps) | (jnp.abs(center_fn_val - self.level_set) < level_set_eps)
            done_flag = jnp.where(jnp.logical_or(hit_flag, jnp.linalg.norm(ray_origin - ray_origin_init) > max_distance), jnp.array(True), jnp.array(False))
            niter = jnp.where(done_flag, niter, niter + 1)
            ray_origin = jnp.where(done_flag, ray_origin, hit_point)
            trust_region_size = jax.lax.cond(done_flag, no_update_trust_region, update_trust_region,
                                                            *(trust_region_size, t, t_box_intersection, in_box))

            return niter, ray_origin, ray_direction, trust_region_size, done_flag, hit_flag
        return step_fn

    def ray_surface_intersection_single(self, ray_origin: jnp.ndarray, ray_direction: jnp.ndarray, max_distance: float=1e4, spatial_eps: float=1e-5, level_set_eps: float=1e-5, max_iterations: int=100, init_interval: float=1.0, unroll: bool=False):
        """
        Compute the ray intersection with the implicit surface using a bounding volume hierarchy.

        Args:
            ray_origin (jnp.ndarray): The origin of the ray.
            ray_direction (jnp.ndarray): The direction of the ray.
            max_distance (float): The maximum distance to search for an intersection.

        Returns:
            in_box (jnp.ndarray): A boolean indicating if there is an intersection with the surface.
            hit_point (jnp.ndarray): The coordinates of the intersection point.
            t (jnp.ndarray): The distance to the intersection point along the ray.
            t_box_intersection (jnp.ndarray): The distance to the intersection point with the bounding box.
        """
        ray_origin_init = ray_origin
        done_flag = jnp.array(False)
        hit_flag = jnp.array(False)
        interval = jnp.array(init_interval)
        step_fn = self.ray_marching_step_fn(spatial_eps, level_set_eps, max_distance, ray_origin_init)

        def loop_body_fn(i, state):
            return step_fn(*state)


        niter, hit_point, ray_direction, interval, done_flag, hit_flag = jax.lax.fori_loop(0, max_iterations, loop_body_fn, (jnp.array(0), ray_origin, ray_direction, interval, done_flag, hit_flag), unroll=unroll)
        return hit_flag, hit_point, niter
    
    def ray_surface_intersection_batch(self, ray_origin: jnp.ndarray, ray_directions: jnp.ndarray, max_distance: float=1e4, spatial_eps: float=1e-5, level_set_eps: float=1e-5, max_iterations: int=100, init_interval: float=1.0):
        """
        Compute the ray intersection with the implicit surface for a batch of rays.

        Args:
            ray_origins (jnp.ndarray): (3) The origin of the rays.
            ray_directions (jnp.ndarray): (N, 3) The directions of the rays.
            max_distance (float): The maximum distance to search for an intersection.
            spatial_eps (float): The spatial epsilon termination criterion.
            level_set_eps (float): The level set epsilon termination criterion.
            max_iterations (int): The maximum number of iterations for the ray-surface intersection.
            init_interval (float): The initial interval size for the bounding boxes.
            unroll (bool): Whether to unroll the for-loop for better performance.

        Returns:
            hit_flags (jnp.ndarray): (N,) A boolean array indicating if there is an intersection with the surface for each ray.
            hit_points (jnp.ndarray): (N, 3) The coordinates of the intersection points for each ray.
            n_iters (jnp.ndarray): (N,) The number of iterations taken for each ray.
        """
        ### step function and number of step function calls
        step_fn_jit_vmap = jax.jit(jax.vmap(self.ray_marching_step_fn(spatial_eps, level_set_eps, max_distance, ray_origin)))
        N_evals = 0
        ### Python control loop variables
        iter = 0
        current_compaction_index = 0
        init_compaction_sizes(uniform_size_generator(ray_directions.shape[0], n_compactions=10))
        INVALID_IND = ray_directions.shape[0] + 1
        ### JAX loop and output states
        ### These are tuples of jnp.ndarrays and they are compacted during the ray marching process
        jnp_device_loop_state, jnp_device_output_state, active_indices = _init_ray_marching_states(ray_directions.shape[0], 
                                                                                                   init_interval, 
                                                                                                   ray_origin, 
                                                                                                   ray_directions)

        ### Python control loop for ray marching with compaction
        ### This cannot be jitted because of the dynamic compaction sizes (XLA needs to know static shapes and step numbers)
        while(True):
            jnp_device_loop_state = step_fn_jit_vmap(*jnp_device_loop_state)
            N_evals += jnp_device_loop_state[0].shape[0]
            iter += 1
            N_valid = int(jnp.sum(~jnp_device_loop_state[4]))  # done_flags


            should_compact_flag, new_compaction_index, new_compaction_size = should_compact(N_valid, current_compaction_index)
            if should_compact_flag:
                
                current_compaction_index = new_compaction_index

                jnp_device_loop_state, jnp_device_output_state, active_indices = perform_compaction(
                    jnp_device_loop_state,
                    jnp_device_output_state,
                    active_indices,
                    fill_index_value=INVALID_IND,
                    new_compacted_size=new_compaction_size
                )
            if N_valid == 0 or iter >= max_iterations:
                break

        return jnp_device_output_state


def update_trust_region(trust_region_size, t, t_box_intersection, in_box):
    trust_region_size = jnp.where(in_box, trust_region_size, trust_region_size * jnp.sqrt(2))

    box_too_big = t < 0.5 * t_box_intersection

    trust_region_size = jnp.where(box_too_big, trust_region_size * 1/jnp.sqrt(2), trust_region_size)
    return trust_region_size

def no_update_trust_region(trust_region_size, t, t_box_intersection, in_box):
    return trust_region_size


@partial(jax.jit, static_argnames=('N_rays', 'init_interval'))
def _init_ray_marching_states(N_rays: int, init_interval: float, ray_origin: jnp.ndarray, ray_directions: jnp.ndarray) -> tuple[jnp.ndarray, ...]:
    """
    Initialize the ray marching loop variables and output arrays.
    """
    ### init loop variables
    niters = jnp.zeros(N_rays, dtype=jnp.int32)
    ray_origins = jnp.ones_like(ray_directions) * ray_origin

    intervals = jnp.ones(N_rays) * init_interval
    done_flags = jnp.zeros(N_rays, dtype=bool)
    hit_flags = jnp.zeros(N_rays, dtype=bool)

    ### output arrays
    out_ray_origins = jnp.zeros_like(ray_origins)
    out_hit_flags = jnp.zeros_like(hit_flags)
    out_niters = jnp.zeros_like(niters)

    ### active indices
    active_indices = jnp.arange(N_rays)

    return (niters, ray_origins, ray_directions, intervals, done_flags, hit_flags), (out_niters, out_ray_origins, out_hit_flags), active_indices