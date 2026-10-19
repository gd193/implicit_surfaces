"""Primitives for any dimension (sphere_sdf is a circle in 2D). The *_sdf
suffix means exact signed distance, negative inside.
Source: Inigo Quilez (https://iquilezles.org/articles/distfunctions2d/) and (https://iquilezles.org/articles/distfunctions/)
"""
import jax
import jax.numpy as jnp

from surfaces.queries import StarRadius, SurfaceHit
from utils.types import Termination as T


def safe_norm(x, ord=2, axis=None, keepdims=False, eps=1e-12):
    """
    Compute the norm of x while ensuring numerical stability by adding a small epsilon to prevent division by zero.
    """
    return jnp.sqrt(jnp.sum(jax.lax.integer_pow(jnp.abs(x), ord), axis=axis, keepdims=keepdims) + eps)


def _box(params, x):
    lo, hi = (
        jnp.asarray(params["min_corner"], x.dtype),
        jnp.asarray(params["max_corner"], x.dtype),
    )
    if lo.shape != x.shape or hi.shape != x.shape:
        raise ValueError("box corners must match the query dimension")
    return lo, hi


def _sphere(params, x):
    c, r = (
        jnp.asarray(params["center"], x.dtype),
        jnp.asarray(params["radius"], x.dtype),
    )
    if c.shape != x.shape or r.shape != ():
        raise ValueError(
            "sphere center must match the query dimension and radius must be scalar"
        )
    return c, r


def sphere_sdf(params, x):
    """
    A simple signed distance function for a sphere defined by its center and radius.
    params: a dictionary containing:
        - 'center' (jnp.ndarray)
        - 'radius' (float)
    x: the point at which to evaluate the SDF
    """
    center, radius = _sphere(params, x)
    squared = jnp.sum((x - center) ** 2)
    # Preserve exact distance and use a finite zero subgradient at the center.
    norm = jnp.sqrt(jnp.where(squared > 0, squared, 1))
    return jnp.where(squared > 0, norm, 0) - radius


def bbox_sdf(params, x):
    """
    A simple signed distance function for an axis-aligned bounding box defined by its min and max corners.
    params: a dictionary containing :
        - 'min_corner' (jnp.ndarray)
        - 'max_corner' (jnp.ndarray)
    x: the point at which to evaluate the SDF
    """
    lo, hi = _box(params, x)
    q = jnp.maximum(lo - x, x - hi)
    squared = jnp.sum(jnp.maximum(q, 0) ** 2)
    # Exact zeros on faces, without a norm(0) derivative poisoning the SDF.
    outside = jnp.sqrt(jnp.where(squared > 0, squared, 1))
    return jnp.where(squared > 0, outside, jnp.max(q))


def rounded_box_sdf(params, x):
    """
    Signed distance to an axis-aligned box with rounded edges, any dimension.
    params: a dictionary containing:
        - 'center' (jnp.ndarray)
        - 'half_extents' (jnp.ndarray): half size including the rounding
        - 'radius' (float): edge rounding radius, at most min(half_extents)
    x: the point at which to evaluate the SDF
    """
    q = jnp.abs(x - params['center']) - params['half_extents'] + params['radius']
    return safe_norm(jnp.maximum(q, 0)) + jnp.minimum(jnp.max(q), 0) - params['radius']


def capsule_sdf(params, x):
    """
    Signed distance to a capsule (segment a-b swept by a ball), any dimension.
    params: a dictionary containing:
        - 'a', 'b' (jnp.ndarray): segment end points
        - 'radius' (float)
    x: the point at which to evaluate the SDF
    """
    pa, ba = x - params['a'], params['b'] - params['a']
    h = jnp.clip(jnp.dot(pa, ba) / jnp.dot(ba, ba), 0., 1.)
    return safe_norm(pa - ba * h) - params['radius']


def _ray_valid(origin, direction, cap):
    tol = 32 * jnp.finfo(origin.dtype).eps
    return (
        jnp.all(jnp.isfinite(origin))
        & jnp.all(jnp.isfinite(direction))
        & (jnp.abs(jnp.linalg.norm(direction) - 1) <= tol)
        & jnp.isfinite(cap)
        & (cap > 0)
    )


def _result(origin, direction, cap, t, normal, valid):
    hit = t <= cap
    distance = jnp.where(hit, t, cap)
    valid &= jnp.isfinite(t) & (t > 0)
    valid &= ~hit | jnp.all(jnp.isfinite(normal))
    status = jnp.asarray(
        jnp.where(~valid, T.INVALID, jnp.where(hit, T.HIT, T.EXIT)), jnp.int32
    )
    return SurfaceHit(
        distance,
        origin + distance * direction,
        jnp.where(hit, normal, jnp.zeros_like(origin)),
        status,
    )


def box_normal_at(params, x):
    lo, hi = _box(params, x)
    distances = jnp.concatenate((jnp.abs(x - lo), jnp.abs(x - hi)))
    nearest = jnp.argmin(distances)
    # At an edge/corner this is a deterministic representative face normal,
    # not a unique geometric normal. inward_direction uses all active faces.
    n = jax.nn.one_hot(nearest % x.shape[0], x.shape[0], dtype=x.dtype)
    n *= jnp.where(nearest < x.shape[0], -1, 1)
    return n


def box_inward_direction(params, position, direction, normal) -> jax.Array:
    """Uniform hemisphere/quadrant/octant folding; no position offset.

    Exact contact coordinates are retained by intersect_ray. Non-contact
    components keep their signs, including at points arbitrarily near an edge.
    """
    lo, hi = _box(params, position)
    direction = jnp.where(position == lo, jnp.abs(direction), direction)
    return jnp.where(position == hi, -jnp.abs(direction), direction)


def box_star_radius(
    params, position, radius_cap, *, on_boundary=False, normal=None
) -> StarRadius:
    lo, hi = _box(params, position)
    valid = (
        jnp.all(jnp.isfinite(lo))
        & jnp.all(jnp.isfinite(hi))
        & jnp.all(hi > lo)
        & jnp.all(jnp.isfinite(position))
        & jnp.all(position >= lo)
        & jnp.all(position <= hi)
        & jnp.isfinite(radius_cap)
        & (radius_cap > 0)
    )
    return StarRadius(
        jnp.asarray(radius_cap, position.dtype),
        jnp.asarray(jnp.where(valid, T.EXIT, T.INVALID), jnp.int32),
    )


def box_intersect_ray(
    params, origin, direction, max_distance, *, on_boundary=False
) -> SurfaceHit:
    lo, hi = _box(params, origin)
    bound = box_star_radius(params, origin, max_distance)
    moving = direction != 0
    wall = jnp.where(direction > 0, hi, lo)
    times = jnp.where(
        moving, (wall - origin) / jnp.where(moving, direction, 1), jnp.inf
    )
    axis = jnp.argmin(times)
    t = times[axis]
    normal = jax.nn.one_hot(axis, origin.shape[0], dtype=origin.dtype) * jnp.sign(
        direction[axis]
    )
    result = _result(
        origin,
        direction,
        max_distance,
        t,
        normal,
        (bound.status == T.EXIT) & _ray_valid(origin, direction, max_distance),
    )
    # Snap all simultaneous contacts, so the next step sees every active face.
    position = jnp.where(
        (result.status == T.HIT) & (times == t), wall, result.position
    )
    return result._replace(position=position)


def sphere_normal_at(params, x):
    c, _ = _sphere(params, x)
    delta = x - c
    return delta / jnp.linalg.norm(delta)


def sphere_star_radius(
    params, position, radius_cap, *, on_boundary=False, normal=None
) -> StarRadius:
    c, r = _sphere(params, position)
    distance = jnp.linalg.norm(position - c)
    # Permit only roundoff around a known boundary contact.
    slack = jnp.where(on_boundary, 8 * jnp.finfo(position.dtype).eps * r, 0)
    valid = (
        jnp.all(jnp.isfinite(c))
        & jnp.isfinite(r)
        & (r > 0)
        & jnp.isfinite(distance)
        & (distance <= r + slack)
        & jnp.isfinite(radius_cap)
        & (radius_cap > 0)
    )
    return StarRadius(
        jnp.asarray(radius_cap, position.dtype),
        jnp.asarray(jnp.where(valid, T.EXIT, T.INVALID), jnp.int32),
    )


def sphere_intersect_ray(
    params, origin, direction, max_distance, *, on_boundary=False
) -> SurfaceHit:
    center, radius = _sphere(params, origin)
    delta = origin - center
    b = jnp.dot(delta, direction)
    c = jnp.where(on_boundary, 0, jnp.dot(delta, delta) - radius * radius)
    root = jnp.sqrt(jnp.maximum(b * b - c, 0))
    # Stable positive root when close to the boundary and pointing outward.
    t = jnp.where(b > 0, -c / jnp.where(b + root > 0, b + root, 1), -b + root)
    point = origin + t * direction
    normal = (point - center) / jnp.linalg.norm(point - center)
    bound = sphere_star_radius(params, origin, max_distance, on_boundary=on_boundary)
    result = _result(
        origin,
        direction,
        max_distance,
        t,
        normal,
        (bound.status == T.EXIT) & _ray_valid(origin, direction, max_distance),
    )
    return result._replace(
        position=jnp.where(
            result.status == T.HIT, center + radius * normal, result.position
        )
    )
