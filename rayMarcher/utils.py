import jax.numpy as jnp

def normalize(v: jnp.array) -> jnp.array:
    return v / jnp.linalg.norm(v)

def reflect(v: jnp.array, n: jnp.array) -> jnp.array:
    return v - 2 * jnp.dot(v, n) * n

def refract(v: jnp.array, n: jnp.array, eta: float) -> jnp.array:
    """
    Compute the refraction of a vector v through a surface with normal n
    using Snell's law.
    """
    cos_theta = jnp.minimum(jnp.dot(-v, n), 1.0)
    r_out_perp = eta * (v + cos_theta * n)
    r_out_parallel = -jnp.sqrt(jnp.abs(1.0 - jnp.linalg.norm(r_out_perp)**2)) * n
    return r_out_perp + r_out_parallel

def reflectance(cosine: jnp.array, ref_idx: float) -> jnp.array:
    """
    Compute the reflectance using Schlick's approximation.
    """
    r0 = (1 - ref_idx) / (1 + ref_idx)
    r0 = r0 * r0
    return r0 + (1 - r0) * jnp.power((1 - cosine), 5)


def ray_aabb_intersection_inside(ray_origin: jnp.array, ray_direction: jnp.array, trust_region_size: jnp.array) -> tuple[jnp.array, jnp.array]:
    """
    Computes the intersection of a ray with an AABB, assuming the ray_origin is 
    strictly inside the box. Finds the single exit point.

    Args:
        ray_origin: jnp.array of shape (3,) ray origin (inside the box).
        ray_direction: jnp.array of shape (3,) ray direction.
        trust_region_size: jnp.array of shape () size of the trust region (box).

    Returns:
        tuple[jnp.array, jnp.array]: 
        (t_hit: float, hit_point: jnp.array)
    """

    # --- 1. Calculate intersection times (t) for all 6 planes ---
    
    # Pre-calculate the inverse of the direction for stability and optimization
    inv_direction = 1.0 / ray_direction
    
    # Intersection distances (t) for the three pairs of planes (X, Y, Z axes).
    # t_min_planes: Time to hit the box_min planes (e.g., -X, -Y, -Z face)
    # t_max_planes: Time to hit the box_max planes (e.g., +X, +Y, +Z face)
    #box_min = ray_origin - (trust_region_size / 2)
    #box_max = ray_origin + (trust_region_size / 2)
    t_min_planes = - (trust_region_size / 2) * inv_direction
    t_max_planes = (trust_region_size / 2) * inv_direction
    #t_min_planes = (box_min - ray_origin) * inv_direction
    #t_max_planes = (box_max - ray_origin) * inv_direction

    # --- 2. Determine the single exit time (t_exit) ---
    
    # Since the ray starts inside, the exit time for any axis is the maximum
    # of the two plane intersection times for that axis.
    t_exit_planes = jnp.maximum(t_min_planes, t_max_planes)

    # The ray exits the AABB when it exits the *first* of the three planes.
    # We must ensure t_exit is positive (t > 0) since the ray is traveling forward.
    t_exit = jnp.max(jnp.array([0.0, jnp.min(t_exit_planes)]))

    return t_exit