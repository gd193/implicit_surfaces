"""NumPy-only triangle metadata and validated-tree postorder reductions."""
from __future__ import annotations

import numpy as np


def triangle_metadata(facets):
    """Measure indexed triangles in float64 before packing boundary metadata."""
    a, b, c = np.asarray(facets, np.float64).transpose(1, 0, 2)
    ab, bc, ca = b - a, c - b, a - c
    # Include the shortest edge to preserve normals when two vertices are close.
    ab2, bc2, ca2 = (np.einsum("ij,ij->i", edge, edge) for edge in (ab, bc, ca))
    use_ca = (ca2 < ab2) & (ca2 < bc2)
    cross = np.cross(ab, bc)
    cross[use_ca] = np.cross(bc[use_ca], ca[use_ca])
    lengths = np.linalg.norm(cross, axis=1)
    area = 0.5 * lengths
    normal = cross / np.where(lengths[:, None] > 0, lengths[:, None], 1.)
    return normal, area, lengths


def triangle_boundary_features(indices):
    """Undirected triangle edges, including repeated edges of degeneracies."""
    return np.sort(indices[:, [[0, 1], [1, 2], [2, 0]]].reshape(-1, 2), axis=1)


def triangle_mesh_closed(vertices, faces):
    """Packed-layout closure: source-coordinate degeneracy and edge incidence."""
    _, _, lengths = triangle_metadata(vertices[faces])
    if np.any(lengths == 0.0):
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
