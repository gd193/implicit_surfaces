import jax
import jax.numpy as jnp
from typing import NamedTuple

class Material(NamedTuple):
    type: jnp.array # (,) 0: diffuse, 1: metal, 2: glass
    albedo: jnp.array # (3,) RGB
    fuzz: jnp.array # (,) scalar
    ref_idx: jnp.array # (,) scalar