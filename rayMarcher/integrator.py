### This is a simple path tracer implementation in Jax
### It uses a basic Monte Carlo integration to estimate the color of each pixel
### The rendering logic is taken from Ray Tracing in One Weekend by Peter Shirley 
### https://raytracing.github.io/books/RayTracingInOneWeekend.html
### The code structure roughly follows the outline provided in the DrJiT talk
### https://www.youtube.com/watch?v=rs5V8ZG7JeA&t=150s
###
### Main feature: An arbitrary implicit surface from /implicitSurfaces can be rendered
### provided it provides a dist function and a normal function
### While rendering SDFs (or lipschitz continuous functions more generally) can be done 
### easily with the well known sphere tracing
### this repo enable rendering of arbitrary (autoboundable) implicit surfaces,
### a much more general function class.

import jax 
import jax.numpy as jnp
import jax.random as jr

from typing import Callable, List, cast, Any, Tuple

from rayMarcher.materials import Material
from rayMarcher.camera import Camera
from rayMarcher.sampler import generate_antialiased_ray, generate_random_unit_ray
from rayMarcher import utils
from utils.types import RayMarchState, RayMarchOutput
from utils.jnp_array_compaction import perform_compaction, should_compact, uniform_size_generator
from utils.helpers import spatial_termination_criterion, level_set_termination_criterion, update_trust_region_size

class Integrator:

    def __init__(self,
                 camera: Camera,
                 materials: Material,
                 background_color: jnp.array,
                 scene_intersection_fn: Callable[[jnp.array, jnp.array], tuple[jnp.array, jnp.array, jnp.array]],
                 scene_intersection_batch_fn: Callable[[jnp.array, jnp.array], tuple[jnp.array, jnp.array, jnp.array]],
                 scene_dist_fns: List[Callable[[jnp.array], jnp.array]], 
                 scene_normal_fns: List[Callable[[jnp.array], jnp.array]],
                 samples_per_pixel: int,
                 max_depth: int,
                 epsilon: float = 1e-3,
                 max_sphere_trace_dist: float = 100,):        
        
        self.camera = camera
        self.materials = materials
        self.background_color = background_color
        self.samples_per_pixel = samples_per_pixel
        self.max_depth = max_depth
        self.scene_intersection_fn = scene_intersection_fn
        self.scene_intersection_batch_fn = scene_intersection_batch_fn
        self.scene_dist_fns = scene_dist_fns
        self.scene_normal_fns = scene_normal_fns
        self.epsilon = epsilon
        self.max_sphere_trace_dist = max_sphere_trace_dist

    def get_scene_dist(self, point: jnp.array) -> tuple[jnp.array, jnp.array]:
        # This helper is clean and reusable
        dist_list = jnp.stack([dist_fn(point) for dist_fn in self.scene_dist_fns])
        min_dist = jnp.min(dist_list)
        material_idx = jnp.argmin(dist_list)
        return min_dist, material_idx
    
    def scene_intersection(self, ray_origin: jnp.array, ray_direction: jnp.array) -> tuple[jnp.array, jnp.array, jnp.array]:
        hits, ts, hit_points, normals = self.scene_intersection_fn(ray_origin, ray_direction)

        valid_intersections = jnp.where(hits, ts, jnp.inf)
        closest_idx = jnp.argmin(valid_intersections)

        t_closest = valid_intersections[closest_idx]
        hit_closest = hit_points[closest_idx]
        normal_closest = normals[closest_idx]
        material_idx = closest_idx 
        hit = jnp.isfinite(t_closest)
        return hit, hit_closest, t_closest, normal_closest, material_idx

    def compute_scatter(self, material_idx: int, current_direction: jnp.array, hit_normal: jnp.array, hit_point: jnp.array, key: jr.PRNGKey):
        albedo = self.materials.albedo[material_idx]
        fuzz = self.materials.fuzz[material_idx]
        ref_idx = self.materials.ref_idx[material_idx]

        # Split into more keys to avoid reuse
        key_diffuse, key_metal, key_glass_reflect, key_glass_random, key_recursive = jr.split(key, 5)

        # Diffuse scattering
        diffuse_scatter = hit_normal + generate_random_unit_ray(key_diffuse)
        near_zero = jnp.linalg.norm(diffuse_scatter) < 1e-8
        diffuse_scatter = jnp.where(near_zero, hit_normal, diffuse_scatter)

        # Metal reflection
        reflected = utils.reflect(utils.normalize(current_direction), hit_normal)
        metal_scatter = reflected + fuzz * generate_random_unit_ray(key_metal)

        # Glass refraction/reflection
        front_face = jnp.dot(current_direction, hit_normal) < 0
        outward_normal = jnp.where(front_face, hit_normal, -hit_normal)
        eta_ration = jnp.where(front_face, 1.0 / ref_idx, ref_idx)

        unit_direction = utils.normalize(current_direction)
        cos_theta = jnp.minimum(-jnp.dot(unit_direction, outward_normal), 1.0)
        sin_theta = jnp.sqrt(1.0 - cos_theta * cos_theta)
        
        cannot_refract = eta_ration * sin_theta > 1.0
        should_reflect = cannot_refract | (jr.uniform(key_glass_random) < utils.reflectance(cos_theta, eta_ration))

        refracted_direction = utils.refract(unit_direction, outward_normal, eta_ration)
        reflected_direction = utils.reflect(unit_direction, outward_normal)

        glass_scatter = jnp.where(should_reflect, reflected_direction, refracted_direction)

        # Choose scatter direction based on material
        scatter_direction = jax.lax.switch(
            self.materials.type[material_idx],
            [
                lambda: diffuse_scatter, #material.type == 0
                lambda: metal_scatter,   #material.type == 1
                lambda: glass_scatter    #material.type == 2
            ]
        )
        metal_absorbed = (self.materials.type[material_idx] == 1) & (jnp.dot(metal_scatter, hit_normal) <= 0)

        return scatter_direction, outward_normal, albedo, metal_absorbed, should_reflect, key_recursive

    def sphere_trace(self, ray_origin: jnp.array, ray_direction: jnp.array):

        current_origin = ray_origin
        current_dist = 1.0 + self.epsilon

        def sphere_trace_body_fn(state):
            current_position, ray_direction, current_dist, min_dist_material_idx = state
            min_dist, min_dist_material_idx = self.get_scene_dist(current_position)
            new_position = current_position + min_dist * ray_direction
            current_dist = min_dist
            return (new_position, ray_direction, current_dist, min_dist_material_idx)

        def sphere_trace_cond_fn(state):
            min_dist = state[2]
            return (min_dist > self.epsilon) & (min_dist < self.max_sphere_trace_dist)
        
        init_state = (current_origin, ray_direction, current_dist, -1)
        final_state = jax.lax.while_loop(sphere_trace_cond_fn, sphere_trace_body_fn, init_state)
        final_origin, final_direction, final_dist, final_material_idx = final_state
        hit = final_dist < self.epsilon
        hit_point = final_origin
        hit_normal = jax.lax.switch(
            final_material_idx,
            self.scene_normal_fns,
            hit_point
        )
        hit_normal = jnp.where(hit, hit_normal, jnp.array([0.0, 0.0, 0.0]))
        return hit, hit_point, final_material_idx, hit_normal

    
    def compute_background_contribution(self, done_flag: jnp.array, hit_flag: jnp.array, ray_direction: jnp.array) -> jnp.array:
        unit_direction_background = utils.normalize(ray_direction)
        t = 0.5 * (unit_direction_background[1] + 1.0)
        # Use the provided background_color in the gradient
        background_color = (1.0 - t) * jnp.array([1.0, 1.0, 1.0]) + t * self.background_color
        
        # Contribution should only be non-zero if the ray is NOT done AND NOT hit (i.e., it missed the scene).
        should_contribute = (~done_flag) & (~hit_flag)
        
        return jnp.where(should_contribute,
                         background_color,
                         jnp.zeros_like(background_color)) # Return zero if already done OR if it hit
    
    
    def trace_ray_recursive(self, ray_origin: jnp.array, ray_direction: jnp.array, key: jr.PRNGKey, depth: int) -> jnp.array:
        """
        Trace a ray through the scene and compute the color recursively.
        Args:
            ray_origin: jnp.array of shape (3,) ray origin
            ray_direction: jnp.array of shape (3,) ray direction
            key: jax.random.PRNGKey
            depth: int, current recursion depth
        Returns:
            jnp.array of shape (3,) color
        """
        if depth >= self.max_depth:
            return jnp.array([0.0, 0.0, 0.0])
        
        if self.scene_intersection_fn is not None:
            hit, hit_point, t, hit_normal, material_idx = self.scene_intersection(ray_origin, ray_direction)
        else:
            hit, hit_point, material_idx, hit_normal = self.sphere_trace(ray_origin, ray_direction)

        scatter_direction, outward_normal, albedo, metal_absorbed, should_reflect, key_recursive = self.compute_scatter(material_idx, ray_direction, hit_normal, hit_point, key)
        ray_offset_normal = jnp.where((self.materials.type[material_idx] == 2) & (~should_reflect), -outward_normal, outward_normal)

        bounced_color = self.trace_ray_recursive(hit_point + 0.001 * ray_offset_normal, scatter_direction, key_recursive, depth + 1)
        background_color = self.compute_background_contribution(jnp.array(False), hit, ray_direction)

        color = albedo * bounced_color
        color = jnp.where(metal_absorbed, jnp.array([0.0, 0.0, 0.0]), color)
        return jnp.where(hit, color, background_color)

        
    def trace_ray(self, ray_origin: jnp.array, ray_direction: jnp.array, key: jr.PRNGKey) -> jnp.array:
        """
        Trace a ray through the scene and compute the color.
        Args:
            ray_origin: jnp.array of shape (3,) ray origin
            ray_direction: jnp.array of shape (3,) ray direction
            key: jax.random.PRNGKey
        Returns:
            jnp.array of shape (3,) color
        """
        color = jnp.array([0.0, 0.0, 0.0])
        attenuation = jnp.array([1.0, 1.0, 1.0])
        current_origin = ray_origin
        current_direction = ray_direction
        key_recursive = key
        done_flag = jnp.array(False)

        for i in range(0, self.max_depth):
            if self.scene_intersection_fn is not None:
                hit, hit_point, t, hit_normal, material_idx = self.scene_intersection(current_origin, current_direction)
            else:
                hit, hit_point, material_idx, hit_normal = self.sphere_trace(current_origin, current_direction)

            background_contribution = self.compute_background_contribution(done_flag, hit, current_direction)
            color += background_contribution * attenuation
            done_flag = done_flag | (~hit)

            scatter_direction, outward_normal, albedo, metal_absorbed, should_reflect, key_recursive = self.compute_scatter(material_idx, current_direction, hit_normal, hit_point, key_recursive)
            ray_offset_normal = jnp.where((self.materials.type[material_idx] == 2) & (~should_reflect), -outward_normal, outward_normal)

            # Update ray for next bounce
            current_origin = jnp.where(done_flag, current_origin, hit_point + self.epsilon * ray_offset_normal)
            current_direction = jnp.where(done_flag, current_direction, scatter_direction)

            # Update attenuation
            next_attenuation = attenuation * albedo
            next_attenuation = jnp.where(metal_absorbed, jnp.array([0.0, 0.0, 0.0]), next_attenuation)
            attenuation = jnp.where(done_flag, attenuation, next_attenuation)
        
        return color


    def render_pixel(self, pixel_i: jnp.array, pixel_j: jnp.array, key: jr.PRNGKey, samples_per_pixel: int) -> jnp.array:
        """
        Render a single pixel by averaging multiple samples.
        Args:
            pixel_i: int, pixel x-coordinate
            pixel_j: int, pixel y-coordinate
            key: jax.random.PRNGKey
        Returns:
            jnp.array of shape (3,) color
        """
        def render_fn(key):
            key_ray, key_camera = jr.split(key, 2)
            ray_origin, ray_direction = generate_antialiased_ray(self.camera, pixel_i, pixel_j, key_camera)
            return self.trace_ray(ray_origin, ray_direction, key_ray)

        keys = jr.split(key, samples_per_pixel)
        final_color = jax.vmap(render_fn, in_axes=(0,))(keys)
        final_color = jnp.mean(final_color, axis=0)
        # key_ray, key_camera, carry_key_next = jr.split(key, 3)
        # ray_origin, ray_direction = generate_antialiased_ray(self.camera, pixel_i, pixel_j, key_camera)
        
        # unit_direction_background = utils.normalize(ray_direction)
        # t = 0.5 * (unit_direction_background[1] + 1.0)
        # background_color = (1.0 - t) * jnp.array([1.0, 1.0, 1.0]) + t * self.background_color

        # hit, hit_point, t, hit_normal, material_idx = self.scene_intersection(ray_origin, ray_direction)
        # test_color = jnp.clip(0.5*hit_normal + jnp.array([1., 1., 1.]), 0.0, 1.0)
        # return jnp.where(hit, test_color, background_color)
        
        return final_color
    
    def render_pixel_simple(self, pixel_i: jnp.array, pixel_j: jnp.array, key: jr.PRNGKey) -> tuple[jnp.array, jnp.array]:
        """
        Render a single pixel using a simple ray marching algorithm. Also returns niter actually needed
        for intersections for debugging/diagnostic purposes.
        Args:
            pixel_i: int, pixel x-coordinate
            pixel_j: int, pixel y-coordinate
            key: jax.random.PRNGKey
        Returns:
            color: jnp.array of shape (3,)
            niter: jnp.array, number of marching iterations taken
        """
        key_ray, key_camera, carry_key_next = jr.split(key, 3)
        ray_origin, ray_direction = generate_antialiased_ray(self.camera, pixel_i, pixel_j, key_camera)

        unit_direction_background = utils.normalize(ray_direction)
        t = 0.5 * (unit_direction_background[1] + 1.0)
        background_color = (1.0 - t) * jnp.array([1.0, 1.0, 1.0]) + t * self.background_color
        hit, hit_point, niter = self.scene_intersection_fn(ray_origin, ray_direction)
        hit_normal = self.scene_normal_fns[0](hit_point)
        test_color = jnp.clip(0.5 * hit_normal + jnp.array([1., 1., 1.]), 0.0, 1.0)
        return jnp.where(hit, test_color, background_color), niter
    

    def render_pixels_simple(self, pixel_i_coords: jnp.array, pixel_j_coords: jnp.array, keys: jnp.array) -> tuple[jnp.array, jnp.array]:
        """
        Render batch of pixels using a simple ray marching algorithm. Also returns niter actually needed
        """
        ray_origins, ray_directions = jax.vmap(generate_antialiased_ray, in_axes=(None, 0, 0, 0))(self.camera, pixel_i_coords, pixel_j_coords, keys)
        unit_directions_background = utils.normalize(ray_directions)
        ts = 0.5 * (unit_directions_background[:, 1] + 1.0)
        background_colors = (1.0 - ts[:, None]) * jnp.array([1.0, 1.0, 1.0]) + ts[:, None] * self.background_color
        hits, hit_points, niters = self.scene_intersection_batch_fn(ray_origins, ray_directions)
        hit_normals = jax.vmap(self.scene_normal_fns[0])(hit_points)
        test_colors = jnp.clip(0.5 * hit_normals + jnp.array([1., 1., 1.]), 0.0, 1.0)
        colors = jnp.where(hits[:, None], test_colors, background_colors)
        return colors, niters

    

    def render_sphere_trace(self, key: jr.PRNGKey) -> jnp.array:
        H, W = self.camera.image_height, self.camera.image_width
        i_coords, j_coords = jnp.meshgrid(jnp.arange(W), jnp.arange(H))
        i_coords = i_coords.flatten()
        j_coords = j_coords.flatten()
        pixel_keys = jr.split(key, H * W)

        colors_flat, niter_flat = self.render_pixels_simple(i_coords, j_coords, pixel_keys)
        image = colors_flat.reshape((H, W, 3))
        niter = niter_flat.reshape((H, W))
        return image, niter


    def render(self, key: jr.PRNGKey) -> jnp.array:
        """
        Render the entire image by rendering each pixel.
        Args:
            key: jax.random.PRNGKey
        Returns:
            jnp.array of shape (H, W, 3) image
        """
        H, W = self.camera.image_height, self.camera.image_width
        i_coords, j_coords = jnp.meshgrid(jnp.arange(W), jnp.arange(H))
        i_coords = i_coords.flatten()
        j_coords = j_coords.flatten()
        pixel_keys = jr.split(key, H * W)

        colors_flat = jax.vmap(self.render_pixel, in_axes=(0, 0, 0, None))(i_coords, j_coords, pixel_keys, self.samples_per_pixel)
        image = colors_flat.reshape((H, W, 3))
        return image
