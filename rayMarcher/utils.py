import jax.numpy as jnp
import numpy as np

SceneBounds = tuple[tuple[float, ...], tuple[float, ...]]
DEFAULT_SCENE_BOUNDS: SceneBounds = ((-2.0, -2.0, -2.0), (2.0, 2.0, 2.0))


def sky_background(ray_directions: jnp.ndarray, background_color: jnp.ndarray) -> jnp.ndarray:
    """Blend white at the bottom of the sky into the background color at the top."""
    t = 0.5 * (ray_directions[:, 1] + 1.0)
    white = jnp.ones(3, dtype=ray_directions.dtype)
    return (1.0 - t[:, None]) * white + t[:, None] * background_color


def validate_scene_bounds(bounds: SceneBounds | None) -> SceneBounds | None:
    """Validate static world-space bounds at integrator construction."""
    if bounds is None:
        return None
    values = np.asarray(bounds, dtype=float)
    if (values.ndim != 2 or values.shape[0] != 2 or values.shape[1] not in (2, 3) or not np.all(np.isfinite(values))
            or not np.all(values[0] < values[1])):
        raise ValueError("scene_bounds must contain finite lower < upper corners of shape (2, dim), with dim 2 or 3")
    return tuple(map(tuple, values))


def ray_scene_interval(origin: jnp.ndarray, direction: jnp.ndarray,
                       bounds: SceneBounds | None) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Forward ray/AABB interval; handles parallel rays, including box faces."""
    if bounds is None:
        return jnp.asarray(0.0), jnp.asarray(jnp.inf), jnp.asarray(True)
    lower, upper = jnp.asarray(bounds)
    parallel = direction == 0
    divisor = jnp.where(parallel, 1.0, direction)
    t0, t1 = (lower - origin) / divisor, (upper - origin) / divisor
    entry = jnp.maximum(0.0, jnp.max(jnp.where(parallel, -jnp.inf, jnp.minimum(t0, t1))))
    exit = jnp.min(jnp.where(parallel, jnp.inf, jnp.maximum(t0, t1)))
    valid = ((entry <= exit) & jnp.any(~parallel)
             & ~jnp.any(parallel & ((origin < lower) | (origin > upper))))
    return entry, exit, valid

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
    r_out_parallel = -jnp.sqrt(jnp.abs(1.0 - jnp.sum(r_out_perp**2))) * n
    return r_out_perp + r_out_parallel

def reflectance(cosine: jnp.array, ref_idx: float) -> jnp.array:
    """
    Compute the reflectance using Schlick's approximation.
    """
    r0 = (1 - ref_idx) / (1 + ref_idx)
    r0 = r0 * r0
    return r0 + (1 - r0) * jnp.power((1 - cosine), 5)
