"""Wavefront OBJ mesh export."""
from pathlib import Path

import numpy as np

from surfaces.meshes import Mesh


def write_obj(path: str | Path, mesh: Mesh) -> None:
    """Write segments as lines and polygons as faces, embedding 2D vertices at z=0."""
    vertices = np.asarray(mesh.vertices)
    if mesh.dim == 2:
        vertices = np.pad(vertices, ((0, 0), (0, 1)))
    record = "l" if mesh.elements.shape[1] == 2 else "f"
    with Path(path).open("w") as stream:
        for x, y, z in vertices:
            stream.write(f"v {x:.6f} {y:.6f} {z:.6f}\n")
        for element in np.asarray(mesh.elements):
            indices = " ".join(str(int(i) + 1) for i in element)
            stream.write(f"{record} {indices}\n")
