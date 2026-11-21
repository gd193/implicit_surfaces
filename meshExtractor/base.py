"""Abstract bases for mesh extraction.

A mesh extraction pipeline has three orthogonal pieces:

  surface (implicitSurface)  --  exposes classify_box / box_bounds / __call__ / spatial_derivative
  tree    (ImplicitTree)     --  spatial subdivision producing a Leaves bundle
  extractor (MeshExtractor)  --  consumes Leaves, calls surface methods, emits triangles

The pipeline (`pipeline.extract_mesh`) wires them and checks tree↔extractor compatibility.

Builders that implement `build_with_topology` also return the subdivision
nodes and the mapping from surviving leaves to those nodes.
"""
from __future__ import annotations

import abc
from typing import Any, NamedTuple, Tuple, TYPE_CHECKING

import jax.numpy as jnp

if TYPE_CHECKING:
    from surfaces.spatial_tree import SpatialBuildResult


class Leaves(NamedTuple):
    """Set of axis-aligned leaf cells emitted by a tree.

    valid: (N,) bool        — which entries are real (rest are padding for batched JAX).
    lower: (N, 3) float     — per-cell AABB lower corner.
    upper: (N, 3) float     — per-cell AABB upper corner.
    uniformity: str         — "uniform" if all valid leaves have the same edge length, else "graded".
    """
    valid: jnp.ndarray
    lower: jnp.ndarray
    upper: jnp.ndarray
    uniformity: str


class ImplicitTree(abc.ABC):
    """Subdivides a root box into a leaf cover of the level-set neighbourhood.

    Implementations call surface.classify_box on each candidate cell and keep UNKNOWN cells.
    """

    leaf_uniformity: str = "uniform"

    @abc.abstractmethod
    def build(self,
              surface,
              params: Any,
              root_lower: jnp.ndarray,
              root_upper: jnp.ndarray) -> Leaves:
        ...

    def build_with_topology(self,
                            surface,
                            params: Any,
                            root_lower: jnp.ndarray,
                            root_upper: jnp.ndarray) -> "SpatialBuildResult":
        """Build the leaf cover and record full subdivision topology.

        Octree and k-d tree builders implement this for consumers of the
        full subdivision topology.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not support build_with_topology"
        )


class MeshExtractor(abc.ABC):
    """Turns a Leaves bundle into (vertices, triangles)."""

    accepts_leaf_uniformity: Tuple[str, ...] = ("uniform",)
    required_surface_methods: Tuple[str, ...] = ("__call__",)

    @abc.abstractmethod
    def extract(self,
                surface,
                params: Any,
                leaves: Leaves) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Returns (vertices [V, 3], triangles [T, 3] of vertex indices)."""
        ...
