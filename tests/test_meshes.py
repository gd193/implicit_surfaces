import jax
import numpy as np
import pytest

from surfaces.meshes import Mesh, QuadMesh, SegmentMesh, TriMesh
from meshExtractor import write_obj


@pytest.mark.parametrize("dim", [2, 3])
@pytest.mark.parametrize(("kind", "width", "record"), [
    (SegmentMesh, 2, "l"), (TriMesh, 3, "f"), (QuadMesh, 4, "f"), (Mesh, 5, "f"),
])
def test_mesh_topology_and_obj(tmp_path, dim, kind, width, record):
    vertices = np.arange(width * dim, dtype=np.float32).reshape(width, dim)
    indices = np.arange(width, dtype=np.int32)[None, ::-1]
    mesh = jax.jit(lambda mesh: mesh)(kind(vertices, indices))
    assert isinstance(mesh, kind) and isinstance(mesh, Mesh) and mesh.dim == dim
    path = tmp_path / "mesh.obj"
    write_obj(path, mesh)
    *positions, element = path.read_text().splitlines()
    coords = np.array([line.split()[1:] for line in positions], float)
    np.testing.assert_array_equal(coords[:, :dim], vertices)
    if dim == 2:
        assert np.all(coords[:, 2] == 0)
    assert element.split() == [record, *map(str, range(width, 0, -1))]


def test_reject_invalid_mesh_layouts():
    vertices = np.zeros((4, 2))
    with pytest.raises(ValueError, match="4 indices"):
        QuadMesh(vertices, np.zeros((0, 3), int))
    with pytest.raises(TypeError, match="integer"):
        TriMesh(vertices, np.zeros((0, 3)))
    with pytest.raises(ValueError, match="vertices"):
        SegmentMesh(np.zeros((4, 4)), np.zeros((0, 2), int))
