"""Adaptive marching cubes for continuous, signed implicit surfaces."""
import numpy as np

from meshExtractor._adaptive import make_adaptive_extractor


def _sample_edges(params, endpoints, fractions):
    start, end = endpoints.astype(float).transpose(1, 0, 2)
    return start + fractions[:, None] * (end - start)


def _mesh_leaves(leaves, positions, links_by_leaf, world):
    vertices, triangles, owners = [], [], []
    vertex_ids = {}

    def vertex_id(point):
        key = tuple(np.asarray(point, np.float32))
        if key not in vertex_ids:
            vertex_ids[key] = len(vertices)
            vertices.append(key)
        return vertex_ids[key]

    for owner, links in enumerate(links_by_leaf):
        while links:
            start = next(iter(links))
            loop = [start]
            edge, _ = links.pop(start)
            while edge != start:
                loop.append(edge)
                edge, _ = links.pop(edge)
            ids = [vertex_id(positions[key]) for key in loop]
            if len(set(ids)) < 3:
                continue
            if len(ids) == 3:
                triangles.append(ids)
                owners.append(owner)
                continue
            if len(ids) == 4:
                a, b, c, d = (np.asarray(vertices[i], float) for i in ids)
                # A consistent diagonal retains all four boundary segments.
                if np.dot(np.cross(b - a, c - a), np.cross(c - a, d - a)) > 0:
                    triangles.extend((ids[:3], (ids[0], ids[2], ids[3])))
                    owners.extend((owner, owner))
                    continue
            center = np.mean([vertices[i] for i in ids], axis=0, dtype=float)
            middle = vertex_id(center)
            fan = [(a, b, middle) for a, b in zip(ids, ids[1:] + ids[:1], strict=True)]
            triangles.extend(fan)
            owners.extend([owner] * len(fan))

    vertices = np.asarray(vertices, np.float32).reshape(-1, 3)
    faces = np.asarray(triangles, np.int32).reshape(-1, 3)
    a, b, c = vertices[faces].astype(float).transpose(1, 0, 2)
    keep = np.any(np.cross(b - a, c - a) != 0, axis=1)
    faces = faces[keep]
    meshed = set(np.asarray(owners, int)[keep])
    collapsed = [leaf[:2] for i, leaf in enumerate(leaves) if i not in meshed]
    used, inverse = np.unique(faces.reshape(-1), return_inverse=True)
    return (vertices[used],
            inverse.reshape(-1, 3).astype(np.int32), collapsed)


def make_marching_cubes(surface, bounds, *, tolerance, max_depth=8):
    """Build an eager extract(params) for a continuous surface in a 3D box.

    Returns an AdaptiveMeshResult with a TriMesh and unresolved boxes.
    tolerance is a world-space, two-sided distance bound within accepted leaves.
    Unresolved leaves contribute their AABB instead of triangles. Point-only
    surfaces need signed_value(params, point); box classifiers are optional.
    """
    return make_adaptive_extractor(surface, bounds, tolerance, max_depth, _sample_edges, _mesh_leaves)
