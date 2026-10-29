"""Ray and sampling state/results. All ray distances are world-space."""
from enum import IntEnum
from typing import NamedTuple
import jax.numpy as jnp


class Termination(IntEnum):
    ACTIVE = 0
    HIT = 1
    EXIT = 2             # reached the configured ray interval's end
    MAX_STEPS = 3       # incomplete query, not a proven miss
    HIT_CAPACITY = 4    # sampler stopped because storage filled
    INVALID = 5
    INACTIVE = 6


class RayMarchState(NamedTuple):
    niter: jnp.ndarray
    t: jnp.ndarray
    ray_origin: jnp.ndarray
    ray_direction: jnp.ndarray
    trust_region_size: jnp.ndarray
    done_flag: jnp.ndarray
    hit_flag: jnp.ndarray
    t_max: jnp.ndarray
    status: jnp.ndarray


class RayMarchOutput(NamedTuple):
    niter: jnp.ndarray
    ray_position: jnp.ndarray
    hit_flag: jnp.ndarray
    status: jnp.ndarray


class RayBuffer(NamedTuple):
    """Unit rays starting on/in the sampling box, with its exit distances."""
    ray_origin: jnp.ndarray
    ray_direction: jnp.ndarray
    max_ts: jnp.ndarray


class MultiHitRayState(NamedTuple):
    niter: jnp.ndarray
    t: jnp.ndarray
    t_hits: jnp.ndarray
    t_max: jnp.ndarray
    hit_counts: jnp.ndarray
    in_hit_band: jnp.ndarray  # epsilon-band membership, NOT object interior
    last_value: jnp.ndarray  # last nonzero residual, for crossing-mode sampling
    last_t: jnp.ndarray
    ray_origin: jnp.ndarray
    ray_direction: jnp.ndarray
    trust_region_size: jnp.ndarray
    done_flag: jnp.ndarray
    status: jnp.ndarray


class MultiHitRayOutput(NamedTuple):
    niter: jnp.ndarray
    ray_origin: jnp.ndarray
    ray_direction: jnp.ndarray
    t_hits: jnp.ndarray
    hit_counts: jnp.ndarray
    status: jnp.ndarray

    @property
    def complete(self):
        return self.status == Termination.EXIT


class SurfaceSamples(NamedTuple):
    points: jnp.ndarray
    rays: MultiHitRayOutput


class BoxBoundarySample(NamedTuple):
    """Sample from a uniformly selected face plane intersected with a ball.

    Rejection outside the finite face retains zero mass; do not renormalize.
    Density factors are retained for stable importance-weight cancellation.
    position is clamped to the face when invalid, so callbacks remain defined.
    No PDE kernel or solid-angle normalization is included.
    """
    position: jnp.ndarray
    normal: jnp.ndarray
    distance: jnp.ndarray
    face_radius: jnp.ndarray
    radial_fraction: jnp.ndarray
    valid: jnp.ndarray

    @property
    def inverse_pdf(self):
        """Inverse surface-area/length PDF, including the selected-face probability.

        The 2D endpoint t=1 has zero density and infinite inverse PDF. Green
        weights must use the cancelled expression below, not inf times zero.
        """
        dim = self.position.shape[0]
        a, t = self.face_radius, self.radial_fraction
        if dim == 3:
            return (2*dim)*(2*jnp.pi*a*(a*t))
        return (2*dim)*(2*a)/(-jnp.log(t))
