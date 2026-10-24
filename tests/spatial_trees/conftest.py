"""Primitive fixtures for spatial-tree tests.

Generates `(primitives, mesh, queries)` triples used across all test
modules. Triangle-based fixtures wrap the project's `TriMeshPrimitives`
on top of a `trimesh.Trimesh` so the trimesh-backed oracle can run
against the same mesh.

Fixture matrix (see `_FIXTURE_SPECS`):
  * tiny_random_8     — sanity
  * random_4k         — matches `extract_mesh.py` scale
  * axis_aligned_grid — many primitives sharing AABB faces
  * degenerate_tris   — zero-area + collinear, sign-loss bug catcher
  * single_cluster    — depth/balance stress
  * sphere_4k         — closed mesh (winding-test compatible)

Each fixture is `pytest.fixture(scope="session")` to avoid rebuilding
expensive trimesh objects across parametrized backend sweeps.
"""
from __future__ import annotations

from typing import NamedTuple

import numpy as np
import pytest
import trimesh

import jax.numpy as jnp

from surfaces.base import TriMesh
from surfaces.explicit.primitives import TriMeshPrimitives


class TriFixture(NamedTuple):
    """One triangle-mesh test fixture: project Primitives + trimesh oracle.

    `tag` is used in pytest IDs; `primitives` and `tri_mesh` are the
    project-side and oracle-side handles for the same geometry.
    `query_points` / `query_rays` / `query_aabbs` are pre-baked random
    query batches so all backends test identical inputs.
    """
    tag: str
    primitives: TriMeshPrimitives
    tri_mesh: trimesh.Trimesh           # external-lib oracle backbone
    query_points: jnp.ndarray           # (M, 3) — for closest_point
    query_ray_origins: jnp.ndarray      # (M, 3)
    query_ray_dirs: jnp.ndarray         # (M, 3) — unit length
    query_ray_max_t: jnp.ndarray        # (M,)
    query_aabb_lo: jnp.ndarray          # (M, 3)
    query_aabb_hi: jnp.ndarray          # (M, 3)


# =====================================================================
# Generators.
# =====================================================================


def _random_nondegenerate_faces(rng: np.random.Generator,
                                n_vertices: int,
                                n_faces: int) -> np.ndarray:
    """Sample `n_faces` triples of distinct vertex indices in [0, n_vertices).

    Pure index-distinctness check — does not exclude geometrically-degenerate
    triangles (collinear, near-zero-area). Those are valid mesh inputs that
    the build pipeline must tolerate; what we filter out is the trivially-
    pathological duplicate-vertex case which produces a zero-area triangle
    by construction and breaks oracle comparisons.
    """
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
                   n_queries: int = 64) -> TriFixture:
    tm = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    project_mesh = TriMesh(vertices=jnp.asarray(vertices, dtype=jnp.float32),
                           faces=jnp.asarray(faces, dtype=jnp.int32))
    primitives = TriMeshPrimitives(project_mesh)

    # Query points: random within an inflated bbox.
    bmin = vertices.min(axis=0) - 0.25
    bmax = vertices.max(axis=0) + 0.25
    qpts = rng.uniform(bmin, bmax, size=(n_queries, 3)).astype(np.float32)

    # Rays: random origins outside the mesh, pointed at random face
    # *interiors* (sampled barycentrically with margin) — never at raw
    # vertices or edges. Pointing rays at vertices generates pathological
    # grazing hits where eps-slack interpretations of M-T disagree across
    # implementations; the fixture's job is to stress build/traversal,
    # not eps-boundary semantics.
    ray_origins = (rng.uniform(bmin, bmax, size=(n_queries, 3))
                   .astype(np.float32))
    face_idx = rng.integers(0, len(faces), size=n_queries)
    # Barycentric weights with a 0.1 margin off every edge.
    bary_raw = rng.uniform(0.1, 0.8, size=(n_queries, 3)).astype(np.float32)
    bary = bary_raw / bary_raw.sum(axis=1, keepdims=True)
    tri_v = vertices[faces[face_idx]]                  # (M, 3, 3)
    targets = (bary[..., None] * tri_v).sum(axis=1)    # (M, 3)
    ray_dirs = targets - ray_origins
    norms = np.linalg.norm(ray_dirs, axis=-1, keepdims=True) + 1e-30
    ray_dirs = (ray_dirs / norms).astype(np.float32)
    max_t = np.full((n_queries,), 100.0, dtype=np.float32)

    # AABBs: random axis-aligned boxes with varied size, half guaranteed
    # to overlap mesh, half guaranteed not.
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
    f = _random_nondegenerate_faces(rng, n_vertices=12, n_faces=8)
    return _to_trifixture("tiny_random_8", v, f, rng, n_queries=16)


def _gen_random_4k(rng) -> TriFixture:
    # Approx scale of `extract_mesh.py`'s output.
    sphere = trimesh.creation.icosphere(subdivisions=5)
    v = np.asarray(sphere.vertices, dtype=np.float32)
    f = np.asarray(sphere.faces, dtype=np.int32)
    # Jitter vertices to break exact symmetry.
    v = v + rng.normal(scale=1e-3, size=v.shape).astype(np.float32)
    return _to_trifixture("random_4k", v, f, rng, n_queries=64)


def _gen_axis_aligned_grid(rng) -> TriFixture:
    # 64 axis-aligned right triangles in a 4x4x4 grid layout — many
    # primitives share AABB face planes with their neighbors.
    tris = []
    for ix in range(4):
        for iy in range(4):
            for iz in range(4):
                base = np.array([ix, iy, iz], dtype=np.float32)
                tris.append([base, base + [1, 0, 0], base + [0, 1, 0]])
    v = np.array(tris, dtype=np.float32).reshape(-1, 3)
    f = np.arange(v.shape[0], dtype=np.int32).reshape(-1, 3)
    return _to_trifixture("axis_aligned_grid", v, f, rng, n_queries=32)


def _gen_degenerate(rng) -> TriFixture:
    # Mix of valid + degenerate tris (zero-area, collinear).
    v = np.array([
        [0, 0, 0], [1, 0, 0], [0, 1, 0],     # valid
        [2, 0, 0], [3, 0, 0], [4, 0, 0],     # collinear
        [5, 0, 0], [5, 0, 0], [5, 0, 0],     # zero-area duplicate
        [6, 0, 0], [7, 1, 0], [6.5, 0.5, 1], # valid
    ], dtype=np.float32)
    f = np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]],
                 dtype=np.int32)
    return _to_trifixture("degenerate_tris", v, f, rng, n_queries=16)


def _gen_single_cluster(rng) -> TriFixture:
    # Many tris clustered tightly + a few far outliers (depth stress).
    cluster = rng.normal(scale=0.01, size=(60, 3)).astype(np.float32)
    outliers = rng.uniform(-10, 10, size=(12, 3)).astype(np.float32)
    v = np.concatenate([cluster, outliers], axis=0)
    f = _random_nondegenerate_faces(rng, n_vertices=v.shape[0], n_faces=24)
    return _to_trifixture("single_cluster", v, f, rng, n_queries=32)


def _gen_sphere_4k(rng) -> TriFixture:
    sphere = trimesh.creation.icosphere(subdivisions=5)
    v = np.asarray(sphere.vertices, dtype=np.float32)
    f = np.asarray(sphere.faces, dtype=np.int32)
    return _to_trifixture("sphere_4k", v, f, rng, n_queries=64)


_FIXTURE_SPECS = {
    "tiny_random_8":     (_gen_tiny_random,     1234),
    "random_4k":         (_gen_random_4k,       2345),
    "axis_aligned_grid": (_gen_axis_aligned_grid, 3456),
    "single_cluster":    (_gen_single_cluster,  5678),
    "sphere_4k":         (_gen_sphere_4k,       6789),
    # `degenerate_tris` (zero-area + collinear) is hand-crafted as
    # intentionally pathological. Oracle libraries disagree on hits at
    # zero-area triangles by definition. Reserved for `_gen_degenerate`
    # callers that want to stress the build pipeline directly; not part
    # of the standard parametrize sweep.
}


# =====================================================================
# Pytest fixtures.
# =====================================================================


@pytest.fixture(scope="session", params=list(_FIXTURE_SPECS),
                ids=lambda tag: tag)
def tri_fixture(request) -> TriFixture:
    """Parametrized triangle-mesh fixture covering every entry of
    `_FIXTURE_SPECS`. Tests using this auto-sweep all geometry classes.
    """
    gen, seed = _FIXTURE_SPECS[request.param]
    rng = np.random.default_rng(seed)
    return gen(rng)


@pytest.fixture(scope="session")
def tri_fixture_sphere_4k() -> TriFixture:
    """Single-mesh fixture for tests that don't need the full sweep
    (e.g. expensive cross-backend pair checks).
    """
    rng = np.random.default_rng(6789)
    return _gen_sphere_4k(rng)
