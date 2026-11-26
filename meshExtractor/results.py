"""Extraction outputs that carry diagnostics alongside geometry."""
from typing import NamedTuple

import jax

from surfaces.meshes import Mesh


class AdaptiveMeshResult(NamedTuple):
    mesh: Mesh
    unresolved_boxes: jax.Array  # (N, 2, dim), world-space lower and upper corners
