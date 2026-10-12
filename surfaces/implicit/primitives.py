import jax 
import jax.numpy as jnp

### Utility functions for SDF primitives
### Source: Inigo Quilez (https://iquilezles.org/articles/distfunctions2d/) and (https://iquilezles.org/articles/distfunctions/)


def safe_norm(x, ord=2, axis=None, keepdims=False, eps=1e-12):
    """
    Compute the norm of x while ensuring numerical stability by adding a small epsilon to prevent division by zero.
    """
    return jnp.sqrt(jnp.sum(jax.lax.integer_pow(jnp.abs(x), ord), axis=axis, keepdims=keepdims) + eps)


def sphere_sdf(params, x):
    """
    A simple signed distance function for a sphere defined by its center and radius.
    params: a dictionary containing:
        - 'center' (jnp.ndarray)
        - 'radius' (float)
    x: the point at which to evaluate the SDF
    """
    center = params['center']
    radius = params['radius']
    if center.shape != x.shape:
        raise ValueError(f"Center and point must have the same dimensions (got {center.shape} and {x.shape})")
    if isinstance(radius, jnp.ndarray) and radius.shape != ():
        raise ValueError(f"Radius must be a scalar (got shape {radius.shape})")
    return safe_norm(x - center, 2) - radius


def bbox_sdf(params, x):
    """
    A simple signed distance function for an axis-aligned bounding box defined by its min and max corners.
    params: a dictionary containing :
        - 'min_corner' (jnp.ndarray)
        - 'max_corner' (jnp.ndarray)
    x: the point at which to evaluate the SDF
    """
    min_corner = params['min_corner']
    max_corner = params['max_corner']
    if min_corner.shape != x.shape:
        raise ValueError(f"Min corner and point must have the same dimensions (got {min_corner.shape} and {x.shape})")
    if max_corner.shape != x.shape:
        raise ValueError(f"Max corner and point must have the same dimensions (got {max_corner.shape} and {x.shape})")
    
    center = (min_corner + max_corner) * 0.5
    half_extents = (max_corner - min_corner) * 0.5
    q = jnp.abs(x - center) - half_extents
    outside_dist = safe_norm(jnp.maximum(q, 0), 2)
    inside_dist = jnp.minimum(jnp.max(q), 0)
    return outside_dist + inside_dist


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

### Exact SDFs (the *_sdf suffix means exact signed distance, negative inside).


def torus_sdf(params, x):
    """
    Signed distance to a torus around the z axis.
    params: a dictionary containing:
        - 'center' (jnp.ndarray) of shape (3,)
        - 'major' (float): distance from the center to the tube's center line
        - 'minor' (float): tube radius
    x: (3,) point
    """
    p = x - params['center']
    q = jnp.stack([safe_norm(p[:2]) - params['major'], p[2]])
    return safe_norm(q) - params['minor']


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


### Combinators. They act on SDF values and return 1-Lipschitz fields, not
### exact SDFs: |f(x)| <= distance to {f = 0} still holds, so empty-sphere
### radii stay safe (use lipschitz_bound=1), but steps can be shorter.


def union(a, b):
    """min(a, b). Exact outside, underestimates distance inside."""
    return jnp.minimum(a, b)


def intersection(a, b):
    """max(a, b). Exact inside, underestimates distance outside."""
    return jnp.maximum(a, b)


def subtraction(a, b):
    """a with b removed: max(a, -b). Underestimates distance."""
    return jnp.maximum(a, -b)


def smooth_union(a, b, k):
    """
    Polynomial smooth minimum with blend radius k > 0 (adds material in the
    blend). Its gradient is a convex combination of the inputs' gradients,
    so it stays 1-Lipschitz.
    source: https://iquilezles.org/articles/smin/
    """
    h = jnp.maximum(k - jnp.abs(a - b), 0.) / k
    return jnp.minimum(a, b) - h * h * k * 0.25
