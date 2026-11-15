"""NumPy-only triangle metadata and validated-tree postorder reductions."""
from __future__ import annotations

import numpy as np


def triangle_metadata(e1, e2, *, original_facets=None):
    """Return float32 normals, areas and double areas in input triangle order.

    Packed geometry clamps normal denominators at 1e-30. Indexed boundary
    geometry supplies float64 source facets: exact source degeneracies are
    zeroed before measuring, and every positive length normalizes directly.
    The caller retains responsibility for finite/range validation.
    """
    cross = np.cross(np.asarray(e1, np.float32), np.asarray(e2, np.float32))
    if original_facets is not None:
        original_cross = np.cross(original_facets[:, 1] - original_facets[:, 0],
                                  original_facets[:, 2] - original_facets[:, 0])
        cross[np.all(original_cross == 0.0, axis=1)] = 0.0
    lengths = np.linalg.norm(cross, axis=1)
    area = 0.5 * lengths
    if original_facets is None:
        normal = cross / np.maximum(lengths[:, None], np.float32(1e-30))
    else:
        normal = cross / np.where(lengths[:, None] > 0, lengths[:, None], 1.)
    return normal, area, lengths


def triangle_boundary_features(indices):
    """Undirected triangle edges, including repeated edges of degeneracies."""
    return np.sort(indices[:, [[0, 1], [1, 2], [2, 0]]].reshape(-1, 2), axis=1)


def triangle_mesh_closed(vertices, faces):
    """Packed-layout closure: source-coordinate degeneracy and edge incidence."""
    triangles = vertices[faces].astype(np.float64)
    normals = np.cross(triangles[:, 1] - triangles[:, 0],
                       triangles[:, 2] - triangles[:, 0])
    if np.any(np.all(normals == 0, axis=1)):
        return False
    _, counts = np.unique(triangle_boundary_features(faces), axis=0, return_counts=True)
    return bool(np.all(counts == 2))


def reduce_nodes(primitive, left, right, offset, count, postorder, combine):
    """Reduce leaf-order values on an already validated binary tree.

    Each leaf slice is reduced before its children-first parent. Internal
    reductions receive left then right, preserving float32 summation order.
    Trailing primitive dimensions and empty-tree output shape are retained.
    """
    result = np.empty((len(left),) + primitive.shape[1:], np.float32)
    for node in postorder:
        if count[node]:
            result[node] = combine(primitive[offset[node]:offset[node] + count[node]])
        else:
            result[node] = combine(result[[left[node], right[node]]])
    return result
