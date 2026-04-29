"""Top-level mesh extraction pipeline."""
from __future__ import annotations

from typing import Any, Tuple

import jax.numpy as jnp

from meshExtractor.base import ImplicitTree, Leaves, MeshExtractor


def extract_mesh(surface,
                 params: Any,
                 bounds: Tuple[jnp.ndarray, jnp.ndarray],
                 tree: ImplicitTree,
                 extractor: MeshExtractor) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Build a leaf cover with `tree` and run `extractor` on it.

    Returns (vertices [V, 3], triangles [T, 3]).
    """
    if tree.leaf_uniformity not in extractor.accepts_leaf_uniformity:
        raise ValueError(
            f"{type(extractor).__name__} accepts {extractor.accepts_leaf_uniformity} leaves, "
            f"but {type(tree).__name__} produces {tree.leaf_uniformity!r}."
        )

    for method in extractor.required_surface_methods:
        if method == "__call__":
            continue  # always present
        if not hasattr(surface, method):
            raise TypeError(
                f"{type(extractor).__name__} requires surface.{method}, "
                f"but {type(surface).__name__} has none."
            )

    lower, upper = bounds
    leaves: Leaves = tree.build(surface, params, jnp.asarray(lower), jnp.asarray(upper))
    return extractor.extract(surface, params, leaves)
