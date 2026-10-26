"""Small, shared geometry fixtures for backend integration tests."""
from __future__ import annotations

from typing import NamedTuple

import numpy as np
import pytest
import trimesh

import jax.numpy as jnp

from surfaces.base import TriMesh
from surfaces.explicit.primitives import TriMeshPrimitives


class TriFixture(NamedTuple):
    tag: str
    primitives: TriMeshPrimitives
    tri_mesh: trimesh.Trimesh
    query_points: jnp.ndarray
    query_ray_origins: jnp.ndarray
    query_ray_dirs: jnp.ndarray
    query_ray_max_t: jnp.ndarray
    query_aabb_lo: jnp.ndarray
    query_aabb_hi: jnp.ndarray


def _random_faces(rng: np.random.Generator,
                                n_vertices: int,
                                n_faces: int) -> np.ndarray:
    out = np.empty((n_faces, 3), dtype=np.int32)
    k = 0
    attempts = 0
    while k < n_faces and attempts < 100 * n_faces:
        candidate = rng.integers(0, n_vertices, size=3)
        if len(set(candidate.tolist())) == 3:
            out[k] = candidate
            k += 1
        attempts += 1
    if k < n_faces:
        raise RuntimeError(
            f"could not draw {n_faces} non-duplicate-vertex faces from "
            f"{n_vertices} vertices")
    return out


def _to_trifixture(tag: str, vertices: np.ndarray, faces: np.ndarray,
                   rng: np.random.Generator,
                   n_queries: int = 24) -> TriFixture:
    tm = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    project_mesh = TriMesh(vertices=jnp.asarray(vertices, dtype=jnp.float32),
                           faces=jnp.asarray(faces, dtype=jnp.int32))
    primitives = TriMeshPrimitives(project_mesh)

    bmin = vertices.min(axis=0) - 0.25
    bmax = vertices.max(axis=0) + 0.25
    qpts = rng.uniform(bmin, bmax, size=(n_queries, 3)).astype(np.float32)

    ray_origins = (rng.uniform(bmin, bmax, size=(n_queries, 3))
                   .astype(np.float32))
    face_idx = rng.integers(0, len(faces), size=n_queries)
    bary_raw = rng.uniform(0.1, 0.8, size=(n_queries, 3)).astype(np.float32)
    bary = bary_raw / bary_raw.sum(axis=1, keepdims=True)
    tri_v = vertices[faces[face_idx]]
    targets = (bary[..., None] * tri_v).sum(axis=1)
    ray_dirs = targets - ray_origins
    norms = np.linalg.norm(ray_dirs, axis=-1, keepdims=True) + 1e-30
    ray_dirs = (ray_dirs / norms).astype(np.float32)
    max_t = np.full((n_queries,), 100.0, dtype=np.float32)

    aabb_centers = rng.uniform(bmin, bmax, size=(n_queries, 3)).astype(np.float32)
    aabb_half = rng.uniform(0.01, 0.5, size=(n_queries, 3)).astype(np.float32)
    aabb_lo = aabb_centers - aabb_half
    aabb_hi = aabb_centers + aabb_half
    far = bmax + 100.0
    aabb_lo[: n_queries // 2] = far
    aabb_hi[: n_queries // 2] = far + 0.1

    return TriFixture(
        tag=tag,
        primitives=primitives,
        tri_mesh=tm,
        query_points=jnp.asarray(qpts),
        query_ray_origins=jnp.asarray(ray_origins),
        query_ray_dirs=jnp.asarray(ray_dirs),
        query_ray_max_t=jnp.asarray(max_t),
        query_aabb_lo=jnp.asarray(aabb_lo),
        query_aabb_hi=jnp.asarray(aabb_hi),
    )


def _gen_tiny_random(rng) -> TriFixture:
    v = rng.uniform(-1.0, 1.0, size=(12, 3)).astype(np.float32)
    f = _random_faces(rng, n_vertices=12, n_faces=8)
    return _to_trifixture("tiny_random_8", v, f, rng, n_queries=16)


def _gen_axis_aligned_grid(rng) -> TriFixture:
    tris = []
    for ix in range(4):
        for iy in range(4):
            for iz in range(4):
                base = np.array([ix, iy, iz], dtype=np.float32)
                tris.append([base, base + [1, 0, 0], base + [0, 1, 0]])
    v = np.array(tris, dtype=np.float32).reshape(-1, 3)
    f = np.arange(v.shape[0], dtype=np.int32).reshape(-1, 3)
    return _to_trifixture("axis_aligned_grid", v, f, rng, n_queries=32)


def _gen_single_cluster(rng) -> TriFixture:
    cluster = rng.normal(scale=0.01, size=(60, 3)).astype(np.float32)
    outliers = rng.uniform(-10, 10, size=(12, 3)).astype(np.float32)
    v = np.concatenate([cluster, outliers], axis=0)
    f = _random_faces(rng, n_vertices=v.shape[0], n_faces=24)
    return _to_trifixture("single_cluster", v, f, rng, n_queries=32)


def _gen_sphere_320(rng) -> TriFixture:
    sphere = trimesh.creation.icosphere(subdivisions=2)
    v = np.asarray(sphere.vertices, dtype=np.float32)
    f = np.asarray(sphere.faces, dtype=np.int32)
    return _to_trifixture("sphere_320", v, f, rng, n_queries=24)


_FIXTURE_SPECS = {
    "tiny_random_8":     (_gen_tiny_random,     1234),
    "axis_aligned_grid": (_gen_axis_aligned_grid, 3456),
    "single_cluster":    (_gen_single_cluster,  5678),
    "sphere_320":        (_gen_sphere_320,       6789),
}


@pytest.fixture(scope="session", params=list(_FIXTURE_SPECS),
                ids=lambda tag: tag)
def tri_fixture(request) -> TriFixture:
    gen, seed = _FIXTURE_SPECS[request.param]
    rng = np.random.default_rng(seed)
    return gen(rng)


@pytest.fixture(scope="session")
def tri_fixture_sphere_320() -> TriFixture:
    rng = np.random.default_rng(6789)
    return _gen_sphere_320(rng)
