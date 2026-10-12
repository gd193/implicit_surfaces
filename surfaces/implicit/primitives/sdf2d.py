"""2D primitives: points x of shape (2,). The *_sdf suffix means exact
signed distance, negative inside; *_udf is unsigned.
Source: Inigo Quilez (https://iquilezles.org/articles/distfunctions2d/)
"""
import jax.numpy as jnp

from .sdfnd import safe_norm


def arc2d(params, x):
    """
    Computes the signed distance to a 2D Arc.
    source: https://www.shadertoy.com/view/wl23RK
    
    Args:
        params: A dictionary containing:
            - 'sc': jnp.ndarray of shape (2,): sin and cos of the arc's aperture (half-angle).
            - 'ra': float, Radius of the arc center line.
            - 'rb': float, Thickness (radius) of the arc border.
        x:  (2,) coordinates of the point.
        
    Returns:
        Scalar signed distance.
    """
    if x.shape != (2,):
        raise ValueError(f"Point x must be 2-dimensional (got shape {x.shape})")
    sc = params['sc']  # sin and cos of the arc's aperture
    ra = params['ra']  # radius of the arc center line
    rb = params['rb']  # thickness (radius) of the arc border
    # Reflect p.x to symmetry (abs(p.x))
    px = jnp.abs(x[0])
    py = x[1]
    
    # Condition: sc.y * p.x > sc.x * p.y
    cond = (sc[1] * px) > (sc[0] * py)
    
    # Branch 1: Distance to the endpoints (tips of the arc)
    diff_x = px - sc[0] * ra
    diff_y = py - sc[1] * ra
    dist_endpoint = jnp.sqrt(diff_x**2 + diff_y**2)
    
    # Branch 2: Distance to the main arc body
    len_p = jnp.sqrt(px**2 + py**2)
    dist_body = jnp.abs(len_p - ra)
        
    return jnp.where(cond, dist_endpoint, dist_body) - rb


def star2d_sdf(params, x, n: int = 5):
    """
    Signed distance to a regular 2D star with n points (n is static).
    source: https://iquilezles.org/articles/distfunctions2d/ (sdStar)
    params: a dictionary containing:
        - 'center' (jnp.ndarray) of shape (2,)
        - 'radius' (float): tip radius
        - 'm' (float): sharpness in [2, n]; 2 is spikiest, n gives a regular polygon
    x: (2,) point
    """
    if x.shape != (2,):
        raise ValueError(f"Point x must be 2-dimensional (got shape {x.shape})")
    r, m = params['radius'], params['m']
    an, en = jnp.pi / n, jnp.pi / m
    acs = jnp.array([jnp.cos(an), jnp.sin(an)])
    ecs = jnp.stack([jnp.cos(en), jnp.sin(en)])
    p = x - params['center']
    p = jnp.stack([jnp.abs(p[0]), p[1]])
    # Reduce to the first sector, then distance to one edge segment.
    bn = jnp.mod(jnp.arctan2(p[0], p[1]), 2. * an) - an
    p = safe_norm(p) * jnp.stack([jnp.cos(bn), jnp.abs(jnp.sin(bn))])
    p = p - r * acs
    p = p + ecs * jnp.clip(-jnp.dot(p, ecs), 0., r * acs[1] / ecs[1])
    return safe_norm(p) * jnp.sign(p[0])


def _circle_wave_arcs(params, x):
    """Distances to the two arcs of one circle-wave period and the side of each.
    The wave alternates arcs of circles centered at (0, 0) and (2 co.x, 2 co.y)
    (after folding x into one half period); arc 2 is mirrored onto arc 1."""
    tb = jnp.pi * 5.0 / 6.0 * jnp.maximum(params['thickness'], 0.0001)
    ra = params['radius']
    co = ra * jnp.stack([jnp.sin(tb), jnp.cos(tb)])
    p = x - params['center']
    px = jnp.abs(jnp.mod(p[0], co[0] * 4.0) - co[0] * 2.0)
    p1 = jnp.stack([px, p[1]])
    p2 = jnp.stack([jnp.abs(px - 2.0 * co[0]), -p[1] + 2.0 * co[1]])
    d1 = jnp.where(co[1] * p1[0] > co[0] * p1[1], safe_norm(p1 - co), jnp.abs(safe_norm(p1) - ra))
    d2 = jnp.where(co[1] * p2[0] > co[0] * p2[1], safe_norm(p2 - co), jnp.abs(safe_norm(p2) - ra))
    # Above the wave: outside circle 1, inside (mirrored) circle 2.
    return d1, d2, safe_norm(p1) - ra, ra - safe_norm(p2)


def circle_wave2d_udf(params, x):
    """
    Exact unsigned distance to a periodic wave of circular arcs along x.
    source: https://www.shadertoy.com/view/stGyzt (sdCircleWave, Inigo Quilez)
    params: a dictionary containing:
        - 'center' (jnp.ndarray) of shape (2,): a trough lies below center at x = center[0]
        - 'thickness' (float) in (0, 1]: arc opening, 1 gives 150 degree arcs
        - 'radius' (float): arc radius; period 4 radius sin(5 pi thickness / 6)
    x: (2,) point
    """
    d1, d2, _, _ = _circle_wave_arcs(params, x)
    return jnp.minimum(d1, d2)


def circle_wave2d_sdf(params, x):
    """
    Signed circle wave: the exact distance of circle_wave2d_udf, positive above
    the wave. The arcs join with a common tangent, so the side of the nearest
    arc is the side of the curve. Same params as circle_wave2d_udf.
    """
    d1, d2, s1, s2 = _circle_wave_arcs(params, x)
    return jnp.where(d1 < d2, jnp.sign(s1) * d1, jnp.sign(s2) * d2)
