import jax.numpy as jnp
from typing import NamedTuple

class RayMarchState(NamedTuple):
    ### Ray marching loop state
    niter: jnp.ndarray          # (N_rays,) Number of iterations taken so far
    t: jnp.ndarray    # (N_rays,) Current distance traveled along the ray
    ray_direction: jnp.ndarray # (N_rays, 3) Ray directions
    ray_origin: jnp.ndarray   # (N_rays, 3) ray origins
    trust_region_size: jnp.ndarray # (N_rays,) Current trust region sizes
    done_flag: jnp.ndarray     # (N_rays,) Boolean flags indicating if the ray marching is done for each ray
    hit_flag: jnp.ndarray      # (N_rays,) Boolean flags indicating if the ray has hit the surface


class RayMarchOutput(NamedTuple):
    niter: jnp.ndarray          # (N_rays,) Number of iterations taken
    ray_position: jnp.ndarray    # (N_rays, 3) Final ray positions
    hit_flag: jnp.ndarray      # (N_rays,) Boolean flags indicating if the ray has hit the surface


class RayBuffer(NamedTuple):
    """Container for the successfully sampled rays in bbox."""
    ray_origin: jnp.ndarray
    ray_direction: jnp.ndarray
    max_ts: jnp.ndarray           # (N,) max distance before outside bbox

class MultiHitRayState(NamedTuple):
    niter: jnp.ndarray          # (N_rays,) Number of iterations taken so far
    t: jnp.ndarray    # (N_rays,) Current distance traveled along the ray
    t_hits: jnp.ndarray # (N_rays, max_hits) Distances to each hit along the ray
    t_max: jnp.ndarray # (N_rays, ) the max dist the ray can travel before done
    hit_counts: jnp.ndarray # (N_rays,) Number of hits recorded so far for each ray
    inside_surface: jnp.ndarray # (N_rays,) Boolean flags indicating if the ray is currently inside the surface
    ray_origin: jnp.ndarray   # (N_rays, 3) ray origins
    ray_direction: jnp.ndarray # (N_rays, 3) Ray directions
    trust_region_size: jnp.ndarray # (N_rays,) Current trust region sizes
    done_flag: jnp.ndarray     # (N_rays,) Boolean flags indicating if this ray has finished marching

class MultiHitRayOutput(NamedTuple):
    niter: jnp.ndarray         # (N_rays,) Number of iterations taken
    ray_origin: jnp.ndarray  # (N_rays, 3) ray origins
    ray_direction: jnp.ndarray # (N_rays, 3) Ray directions
    t_hits: jnp.ndarray # (N_rays, max_hits) Distances to each hit along the ray
    hit_counts: jnp.ndarray # (N_rays,) Number of hits recorded for each ray