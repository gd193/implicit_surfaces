"""Indexed meshes with a single element topology in 2D or 3D."""
from dataclasses import dataclass
from typing import ClassVar

import jax
import numpy as np


@jax.tree_util.register_dataclass
@dataclass(frozen=True, eq=False)
class Mesh:
    """Vertex positions and zero-based connectivity for uniform-width elements."""

    vertices: jax.Array | np.ndarray
    elements: jax.Array | np.ndarray
    arity: ClassVar[int | None] = None

    def __post_init__(self):
        if self.vertices.ndim != 2 or self.dim not in (2, 3):
            raise ValueError("vertices must have shape (V, 2) or (V, 3)")
        if self.elements.ndim != 2 or self.elements.shape[1] < 2:
            raise ValueError("elements must have shape (N, K) with K >= 2")
        if self.arity is not None and self.elements.shape[1] != self.arity:
            raise ValueError(f"{type(self).__name__} requires {self.arity} indices per element")
        if not np.issubdtype(self.elements.dtype, np.integer):
            raise TypeError("element indices must have integer dtype")

    @property
    def dim(self) -> int:
        return self.vertices.shape[-1]


@jax.tree_util.register_dataclass
@dataclass(frozen=True, eq=False)
class SegmentMesh(Mesh):
    """Indexed line segments."""
    arity = 2


@jax.tree_util.register_dataclass
@dataclass(frozen=True, eq=False)
class TriMesh(Mesh):
    """Indexed triangles in boundary order."""
    arity = 3


@jax.tree_util.register_dataclass
@dataclass(frozen=True, eq=False)
class QuadMesh(Mesh):
    """Indexed quadrilaterals in boundary order."""
    arity = 4
