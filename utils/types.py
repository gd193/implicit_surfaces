import jax.numpy as jnp
from typing import NamedTuple

class RayMarchState(NamedTuple):
    ### Ray marching loop state
    niter: jnp.ndarray          # (N_rays,) Number of iterations taken so far
    ray_position: jnp.ndarray    # (N_rays, 3) Current ray origins
    ray_direction: jnp.ndarray # (N_rays, 3) Ray directions
    trust_region_size: jnp.ndarray # (N_rays,) Current trust region sizes
    ray_origin: jnp.ndarray   # (N_rays, 3) Initial ray origins
    done_flag: jnp.ndarray     # (N_rays,) Boolean flags indicating if the ray marching is done for each ray
    hit_flag: jnp.ndarray      # (N_rays,) Boolean flags indicating if the ray has hit the surface

class RayMarchOutput(NamedTuple):
    niter: jnp.ndarray          # (N_rays,) Number of iterations taken
    ray_position: jnp.ndarray    # (N_rays, 3) Final ray positions
    hit_flag: jnp.ndarray      # (N_rays,) Boolean flags indicating if the ray has hit the surface