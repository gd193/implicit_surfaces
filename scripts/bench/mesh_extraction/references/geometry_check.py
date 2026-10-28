"""CPU checks using exact lattice-edge IDs, outside benchmark timing."""

import numpy as np


def warp_lattice_edges(values, threshold=0.0):
    """Exact input-edge identities in Warp's documented scan/vertex order.

    IDs use VTK-m's x-fastest point numbering; this is validation only on the CPU.
    """
    nx, ny, nz = values.shape
    sign = values >= np.float32(threshold)
    crossing = np.zeros(values.shape + (3,), dtype=bool)
    crossing[:-1, :, :, 0] = sign[:-1, :, :] != sign[1:, :, :]
    crossing[:, :-1, :, 1] = sign[:, :-1, :] != sign[:, 1:, :]
    crossing[:, :, :-1, 2] = sign[:, :, :-1] != sign[:, :, 1:]
    selected = np.flatnonzero(crossing)
    point, axis = selected // 3, selected % 3
    x, y, z = point // (ny * nz), (point // nz) % ny, point % nz
    start = (z * ny + y) * nx + x
    return np.stack((start, start + np.asarray([1, nx, nx * ny])[axis]), axis=1)


def compare_with_edges(
    vertices_a,
    faces_a,
    edges_a,
    vertices_b,
    faces_b,
    edges_b,
    *,
    compare_boundary=False,
):
    """Compare exact lattice topology without nearest-position tie ambiguity."""
    va, vb = np.asarray(vertices_a), np.asarray(vertices_b)
    fa, fb = np.asarray(faces_a).reshape(-1, 3), np.asarray(faces_b).reshape(-1, 3)
    ea, eb = (
        np.sort(edges_a, axis=1).astype(np.uint64),
        np.sort(edges_b, axis=1).astype(np.uint64),
    )
    ca = (ea[:, 0] << np.uint64(32)) | ea[:, 1]
    cb = (eb[:, 0] << np.uint64(32)) | eb[:, 1]
    edge_match = bool(len(ca) == len(cb) and np.array_equal(np.sort(ca), np.sort(cb)))
    valid = all(
        np.isfinite(v).all() and (len(f) == 0 or (f.min() >= 0 and f.max() < len(v)))
        for v, f in ((va, fa), (vb, fb))
    )
    result = {
        "vertex_counts": [len(va), len(vb)],
        "triangle_counts": [len(fa), len(fb)],
        "finite_valid_indices": bool(valid),
        "lattice_edge_ids_match": edge_match,
        "comparison_basis": "exact input lattice-edge IDs; no nearest-position mapping",
    }
    if not edge_match or not valid:
        return result
    space = np.sort(ca)
    ma, mb = np.searchsorted(space, ca), np.searchsorted(space, cb)
    aligned_a, aligned_b = np.empty_like(va), np.empty_like(vb)
    aligned_a[ma], aligned_b[mb] = va, vb
    result["max_vertex_distance"] = float(
        np.linalg.norm(aligned_a - aligned_b, axis=1).max(initial=0)
    )

    def canonical(f, oriented):
        if not len(f):
            return f
        if oriented:
            start = np.argmin(f, axis=1)
            f = np.take_along_axis(f, (start[:, None] + np.arange(3)) % 3, axis=1)
        else:
            f = np.sort(f, axis=1)
        return f[np.lexsort(f.T)]

    a, b = ma[fa], mb[fb]
    if compare_boundary:

        def boundary(f):
            edges = np.sort(np.concatenate([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]]), axis=1)
            edges, counts = np.unique(edges, axis=0, return_counts=True)
            return edges[counts == 1]

        result["polygon_boundary_edges_match"] = bool(np.array_equal(boundary(a), boundary(b)))
    result["triangles_match_unoriented"] = bool(
        np.array_equal(canonical(a, False), canonical(b, False))
    )
    result["triangles_match_oriented"] = bool(
        np.array_equal(canonical(a, True), canonical(b, True))
    )
    result["triangles_match_opposite_winding"] = bool(
        np.array_equal(canonical(a, True), canonical(b[:, ::-1], True))
    )
    return result
