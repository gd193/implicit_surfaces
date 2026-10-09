import jax
import jax.numpy as jnp
from typing import NamedTuple
import jax.random as jr

from rayMarcher import utils
from rayMarcher.sampler import generate_random_unit_ray

class Material(NamedTuple):
    type: jax.Array  # (surfaces,) 0: diffuse, 1: metal, 2: glass
    albedo: jax.Array  # (surfaces, 3) RGB
    fuzz: jax.Array  # (surfaces,)
    ref_idx: jax.Array  # (surfaces,)


def scatter(materials: Material, material_idx: int, current_direction: jnp.ndarray,
            hit_normal: jnp.ndarray, hit_point: jnp.ndarray, key: jax.Array
            ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    """Shared diffuse, metal, and dielectric scattering for both renderers."""
    albedo = materials.albedo[material_idx]
    fuzz = materials.fuzz[material_idx]
    ref_idx = materials.ref_idx[material_idx]

    # Split into more keys to avoid reuse
    key_diffuse, key_metal, key_glass_reflect, key_glass_random, key_recursive = jr.split(key, 5)

    # Diffuse scattering
    diffuse_scatter = hit_normal + generate_random_unit_ray(key_diffuse)
    near_zero = jnp.linalg.norm(diffuse_scatter) < 1e-8
    diffuse_scatter = jnp.where(near_zero, hit_normal, diffuse_scatter)

    # Metal reflection
    reflected = utils.reflect(utils.normalize(current_direction), hit_normal)
    metal_scatter = reflected + fuzz * generate_random_unit_ray(key_metal)

    # Glass refraction/reflection
    front_face = jnp.dot(current_direction, hit_normal) < 0
    outward_normal = jnp.where(front_face, hit_normal, -hit_normal)
    eta_ration = jnp.where(front_face, 1.0 / ref_idx, ref_idx)

    unit_direction = utils.normalize(current_direction)
    cos_theta = jnp.minimum(-jnp.dot(unit_direction, outward_normal), 1.0)
    sin_theta = jnp.sqrt(1.0 - cos_theta * cos_theta)

    cannot_refract = eta_ration * sin_theta > 1.0
    should_reflect = cannot_refract | (jr.uniform(key_glass_random) < utils.reflectance(cos_theta, eta_ration))

    refracted_direction = utils.refract(unit_direction, outward_normal, eta_ration)
    reflected_direction = utils.reflect(unit_direction, outward_normal)

    glass_scatter = jnp.where(should_reflect, reflected_direction, refracted_direction)

    # Choose scatter direction based on material
    scatter_direction = jax.lax.switch(
        materials.type[material_idx],
        [
            lambda: diffuse_scatter, #material.type == 0
            lambda: metal_scatter,   #material.type == 1
            lambda: glass_scatter    #material.type == 2
        ]
    )
    metal_absorbed = (materials.type[material_idx] == 1) & (jnp.dot(metal_scatter, hit_normal) <= 0)

    return scatter_direction, outward_normal, albedo, metal_absorbed, should_reflect, key_recursive
