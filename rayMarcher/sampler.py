import jax
import jax.numpy as jnp
import jax.random as jr
from rayMarcher.camera import Camera
from rayMarcher.utils import normalize

def sample_uniform_1disk(key: jr.PRNGKey) -> jnp.array:
    """
    Sample a point uniformly in the unit disk.
    Args:
        key: jax.random.PRNGKey
    Returns:
        jnp.array of shape (2,) sampled point in the unit disk
    """
    key1, key2 = jr.split(key)
    phi = jr.uniform(key1, minval=0.0, maxval=2 * jnp.pi)
    r = jnp.sqrt(jr.uniform(key2, minval=0.0, maxval=1.0))
    return jnp.array([r * jnp.cos(phi), r * jnp.sin(phi)])

def sample_uniform_1disk_3d(key: jr.PRNGKey) -> jnp.array:
    """
    Sample a point uniformly in the unit disk in 3D (z=0).
    Args:
        key: jax.random.PRNGKey
    Returns:
        jnp.array of shape (3,) sampled point in the unit disk in 3D
    """
    disk_2d = sample_uniform_1disk(key)
    return jnp.array([disk_2d[0], disk_2d[1], 0.0])

def defocus_disk_sample(camera: Camera, key: jr.PRNGKey) -> jnp.array:
    """
    Sample a point on the defocus disk of the camera.
    Args:
        camera: Camera object
        key: jax.random.PRNGKey
    Returns:
        jnp.array of shape (3,) sampled point on the defocus disk in 3D
    """
    disk_sample_2d = sample_uniform_1disk_3d(key)
    return camera.camera_center + disk_sample_2d[0] * camera.defocus_disk_u + disk_sample_2d[1] * camera.defocus_disk_v

def generate_antialiased_ray(camera: Camera, pixel_i: int, pixel_j: int, key: jr.PRNGKey) -> tuple[jnp.array, jnp.array]:
    """
    Generate an antialiased ray from the camera through the pixel (pixel_i, pixel_j).
    Args:
        camera: Camera object
        pixel_i: int, pixel x-coordinate
        pixel_j: int, pixel y-coordinate
        key: jax.random.PRNGKey
    Returns:
        tuple of (ray_origin, ray_direction)
    """
    key1, key2, key3 = jr.split(key, 3)

    offset_u = jr.uniform(key1, minval=-0.5, maxval=0.5)
    offset_v = jr.uniform(key2, minval=-0.5, maxval=0.5)
    pixel_center = camera.pixel_origin + ((pixel_i + offset_u) * camera.pixel_du) + ((pixel_j + offset_v) * camera.pixel_dv)

    if camera.defocus_angle > 0.0:
        ray_origin = defocus_disk_sample(camera, key3)
    else:
        ray_origin = camera.camera_center
    ray_direction = normalize(pixel_center - ray_origin)

    return ray_origin, ray_direction

def generate_random_unit_ray(key: jr.PRNGKey) -> jnp.array:
    """
    Generate a random ray direction uniformly sampled in the unit sphere.
    """
    u = jr.normal(key, shape=(3,))
    return normalize(u)