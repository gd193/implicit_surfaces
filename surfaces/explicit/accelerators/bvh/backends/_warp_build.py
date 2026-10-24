"""Warp BVH build validation."""
from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from surfaces.explicit.accelerators.protocol import Config
from surfaces.explicit.primitives import Primitives, TriMeshPrimitives

from ._warp_specs import STACK_CAPACITY


def validate_warp_mesh_and_config(
    primitives: Primitives, config: Config, backend_name: str,
) -> tuple[np.ndarray, np.ndarray]:
    if not isinstance(primitives, TriMeshPrimitives):
        raise TypeError(f"{backend_name} requires TriMeshPrimitives")
    if config.max_stack_depth < 1 or config.max_stack_depth > STACK_CAPACITY:
        raise ValueError(
            f"max_stack_depth must be in [1, {STACK_CAPACITY}]"
        )
    if config.leaf_size_threshold < 1:
        raise ValueError("leaf_size_threshold must be >= 1")

    vertices = np.asarray(primitives.mesh.vertices, dtype=np.float32)
    faces = np.asarray(primitives.mesh.faces, dtype=np.int32)
    if faces.ndim != 2 or faces.shape[1:] != (3,):
        raise ValueError("Warp BVH mesh faces must have shape (F, 3)")
    if vertices.ndim != 2 or vertices.shape[1:] != (3,):
        raise ValueError("Warp BVH mesh vertices must have shape (V, 3)")
    if faces.size and (np.any(faces < 0) or np.any(faces >= len(vertices))):
        raise ValueError("mesh faces contain an invalid vertex index")
    if faces.shape[0] and not np.isfinite(vertices[faces]).all():
        raise ValueError("Warp BVH requires finite referenced mesh vertices")
    return vertices, faces


def finite_primitive_bounds(
    primitives: TriMeshPrimitives,
) -> tuple[np.ndarray, np.ndarray]:
    indices = jnp.arange(primitives.n_primitives, dtype=jnp.int32)
    lower, upper = primitives.primitive_aabb(indices)
    lower = np.asarray(lower, dtype=np.float32)
    upper = np.asarray(upper, dtype=np.float32)
    expected = (primitives.n_primitives, 3)
    if lower.shape != expected or upper.shape != expected:
        raise ValueError("Warp BVH primitive bounds must have shape (F, 3)")
    if not np.isfinite(lower).all() or not np.isfinite(upper).all():
        raise ValueError("Warp BVH requires finite primitive bounds")
    return lower, upper
