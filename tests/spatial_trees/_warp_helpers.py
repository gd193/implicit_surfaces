"""Warp fixtures and independent float64 geometry references."""
import jax.numpy as jnp
import numpy as np
import trimesh

from surfaces.base import TriMesh
from surfaces.explicit.accelerators import BVHAccelerator, Config, list_backends
from surfaces.explicit.primitives import TriMeshPrimitives

WARP_BVHS = [n for n in list_backends("bvh", runtime="warp")
             if n in ("bvh.sah.warp", "bvh.snch.warp")]
SAH_WARP = [n for n in WARP_BVHS if n == "bvh.sah.warp"]


def _mesh(vertices, faces):
    vertices = np.asarray(vertices, np.float32)
    faces = np.asarray(faces, np.int32)
    tri = TriMesh(vertices=jnp.asarray(vertices), faces=jnp.asarray(faces))
    return TriMeshPrimitives(tri), vertices, faces


def _sphere(subdivisions=2):
    m = trimesh.creation.icosphere(subdivisions=subdivisions)
    return _mesh(m.vertices, m.faces)


def _closest_triangle(p, tri):
    a, b, c = tri
    ab, ac = b - a, c - a
    cross = np.cross(ab, ac)
    n2 = float(cross @ cross)
    candidates = []
    if n2 > 1e-24:
        proj = p - cross * (float((p - a) @ cross) / n2)
        v0, v1, v2 = ab, ac, proj - a
        d00, d01, d11 = v0 @ v0, v0 @ v1, v1 @ v1
        d20, d21 = v2 @ v0, v2 @ v1
        den = d00 * d11 - d01 * d01
        if abs(float(den)) > 1e-24:
            v = (d11 * d20 - d01 * d21) / den
            w = (d00 * d21 - d01 * d20) / den
            if v >= 0 and w >= 0 and v + w <= 1:
                candidates.append(proj)
    for u, v in ((a, b), (b, c), (c, a)):
        d = v - u
        t = np.clip(float((p - u) @ d) / max(float(d @ d), 1e-30), 0, 1)
        candidates.append(u + t * d)
    return min(candidates, key=lambda q: float((q - p) @ (q - p)))


def _oracle_closest(vertices, faces, p):
    vertices, p = np.asarray(vertices, np.float64), np.asarray(p, np.float64)
    cps = np.stack([_closest_triangle(p, vertices[f]) for f in faces])
    ds = np.linalg.norm(cps - p, axis=1)
    i = int(np.argmin(ds))
    return cps[i], ds[i], i


def _oracle_ray(vertices, faces, o, d, max_t, min_t=0.0):
    """Two-sided ray oracle with closed [min_t, max_t]."""
    hits = []
    for i, f in enumerate(faces):
        a, b, c = vertices[f]
        e1, e2 = b - a, c - a
        h = np.cross(d, e2)
        det = float(e1 @ h)
        if abs(det) < 1e-10:
            continue
        inv = 1.0 / det
        s = o - a
        u = inv * float(s @ h)
        q = np.cross(s, e1)
        v = inv * float(d @ q)
        t = inv * float(e2 @ q)
        if u >= -1e-7 and v >= -1e-7 and u + v <= 1 + 1e-7 and min_t <= t <= max_t:
            hits.append((t, i))
    return (min(hits) if hits else (np.inf, -1))


def _build(name, vertices, faces):
    prims, _, _ = _mesh(vertices, faces)
    return BVHAccelerator.build(prims, name)



def _build_mesh(mesh, backend="bvh.sah.warp", leaf=4):
    return BVHAccelerator.build(_mesh(mesh.vertices, mesh.faces)[0], backend,
                                Config(leaf_size_threshold=leaf))


def _silhouette_edges(vertices, faces):
    vertices = np.asarray(vertices, np.float64)
    triangles = vertices[faces]
    normals = np.cross(triangles[:, 1]-triangles[:, 0], triangles[:, 2]-triangles[:, 0])
    lengths = np.linalg.norm(normals, axis=1)
    normals /= np.maximum(lengths[:, None], 1e-30)
    adjacent = {}
    for i, (a, b, c) in enumerate(faces):
        for u, v in ((a, b), (b, c), (c, a)):
            adjacent.setdefault(tuple(sorted((u, v))), []).append(i)
    edges = []
    for (u, v), fs in adjacent.items():
        if len(fs) <= 2 and np.all(lengths[fs] > 0):
            n0 = normals[fs[0]]
            n1 = normals[fs[1]] if len(fs) == 2 else -n0
            edges.append((vertices[u], vertices[v], n0, n1))
    return np.asarray(edges, np.float64).reshape(-1, 4, 3)


def _oracle_silhouette(vertices, faces, points):
    points = np.asarray(points, np.float64)
    shape = points.shape[:-1]
    a, b, n0, n1 = _silhouette_edges(vertices, faces).transpose(1, 0, 2)
    if not len(a):
        return np.zeros(points.shape), np.full(shape, np.inf)
    view = points.reshape(-1, 1, 3)-a
    edge = b-a
    t = np.clip(np.sum(view*edge, axis=-1)/np.maximum(np.sum(edge*edge, axis=-1), 1e-30), 0, 1)
    cp = a+t[..., None]*edge
    s0, s1 = (np.sum(view*n, axis=-1) for n in (n0, n1))
    valid = ((s0 <= 0) & (s1 >= 0)) | ((s0 >= 0) & (s1 <= 0))
    distances = np.where(valid, np.linalg.norm(points.reshape(-1, 1, 3)-cp, axis=-1), np.inf)
    best = distances.argmin(axis=1)
    rows = np.arange(len(best))
    return cp[rows, best].reshape(points.shape), distances[rows, best].reshape(shape)
