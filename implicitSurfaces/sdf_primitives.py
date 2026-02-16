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
    assert center.shape == x.shape, "Center and point must have the same dimensions"
    assert radius.shape == () if isinstance(radius, jnp.ndarray) else True, "Radius must be a scalar"
    return  safe_norm(x - center, 2) - radius


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
    assert min_corner.shape == x.shape, "Min corner and point must have the same dimensions"
    assert max_corner.shape == x.shape, "Max corner and point must have the same dimensions"
    
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
    assert x.shape == (2,), "Point x must be 2-dimensional."
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