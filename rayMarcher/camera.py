import jax
import jax.numpy as jnp
import jax.random as jr

from typing import NamedTuple

from rayMarcher.utils import normalize



class Camera(NamedTuple):

    image_width: int
    image_height: int
    aspect_ratio: float

    camera_center: jnp.array
    pixel_origin: jnp.array
    pixel_du: jnp.array
    pixel_dv: jnp.array

    defocus_disk_u: jnp.array
    defocus_disk_v: jnp.array
    defocus_angle: jnp.array

    def setup_Camera( 
                 image_width: int, 
                 image_height: int, 
                 vfov: float, 
                 lookfrom: jnp.array, 
                 lookat: jnp.array,
                 vup: jnp.array,
                 defocus_angle: float = 0.0, 
                 focus_distance: float = 1.0):

        aspect_ratio = image_width / image_height
        focal_length = float(jnp.linalg.norm(lookfrom - lookat))


        theta = jnp.deg2rad(vfov)
        h = jnp.tan(theta / 2)
        viewport_height = 2.0 * h * focus_distance
        viewport_width = aspect_ratio * viewport_height

        camera_center = lookfrom
        w = normalize(lookfrom - lookat)
        u = normalize(jnp.cross(vup, w))
        v = jnp.cross(w, u)

        viewport_u = viewport_width * u
        viewport_v = viewport_height * -v
        viewport_upper_left = (camera_center - focus_distance * w - viewport_u / 2 - viewport_v / 2)

        pixel_du = viewport_u / image_width
        pixel_dv = viewport_v / image_height
        pixel_origin = viewport_upper_left + 0.5 * (pixel_du + pixel_dv)

        defocus_radius = focus_distance * jnp.tan(jnp.deg2rad(defocus_angle) / 2)
        defocus_disk_u = defocus_radius * u
        defocus_disk_v = defocus_radius * v
        defocus_angle = defocus_angle
        return Camera(
            image_width=image_width,
            image_height=image_height,
            aspect_ratio=aspect_ratio,
            camera_center=camera_center,
            pixel_origin=pixel_origin,
            pixel_du=pixel_du,
            pixel_dv=pixel_dv,
            defocus_disk_u=defocus_disk_u,
            defocus_disk_v=defocus_disk_v,
            defocus_angle=defocus_angle,
        )
