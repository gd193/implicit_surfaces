"""Extraction outputs that carry diagnostics alongside geometry."""
from typing import NamedTuple

import jax
import numpy as np

from surfaces.meshes import Mesh, TriMesh


class AdaptiveMeshResult(NamedTuple):
    mesh: Mesh
    unresolved_boxes: jax.Array  # (N, 2, dim), world-space lower and upper corners


class CapacityOverflow(RuntimeError):
    """The requested mesh exceeds a resident buffer budget."""

    def __init__(self, counts):
        self.counts = tuple(map(int, counts))
        super().__init__(f'dual contouring capacity exceeded; C/E/T/status/F/K/U={self.counts}')


class DualContouringBuffers(NamedTuple):
    vertices: jax.Array
    faces: jax.Array
    counts: jax.Array  # C, E, T, overflow bits, frontier, raw corners, unique corners

    def compact(self) -> TriMesh:
        """Read the counts once and return the valid vertex and triangle prefixes."""
        counts = np.asarray(jax.device_get(self.counts))
        if counts[3]:
            raise CapacityOverflow(counts)
        return TriMesh(self.vertices[:int(counts[0])], self.faces[:int(counts[2])])
