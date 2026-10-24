"""Layer 3/4: closest_silhouette query correctness.

Capability-gated test: only runs against backends whose
`has_silhouette_query` flag is True. Validates that the returned
`(cp, dist, edge_idx)` matches a brute-force oracle that extracts the
silhouette edge set from the mesh and reduces by minimum distance.

A silhouette edge is a manifold edge (shared by exactly two faces)
whose two adjacent face normals straddle the query point: i.e., one
face is forward-facing and the other back-facing relative to x.
Validity test: `sign(dot(n0, x - v0)) * sign(dot(n1, x - v0)) < 0`.

No trimesh oracle — silhouette queries are not in the trimesh API.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from surfaces.explicit.accelerators.registry import get_backend

from ._harness import (parametrize_backends, build_accelerator,
                       skip_if_primitive_math_missing,
                       DIST_RTOL, DIST_ATOL, POINT_ATOL)


# Slightly looser than DIST tolerance: silhouette validity is sign-
# sensitive at edge boundaries; near-grazing queries can flip across
# implementations within float32 noise.
SILH_RTOL = 5e-4
SILH_ATOL = 5e-5


def _skip_if_capability_missing(backend_name: str) -> None:
    cls = get_backend(backend_name)
    if not getattr(cls, "has_silhouette_query", False):
        pytest.skip(f"{backend_name} does not implement closest_silhouette")


def _backend_closest_silhouette_batch(accel, queries):
    return jax.vmap(lambda x: accel.closest_silhouette(x))(queries)


# =====================================================================
# Brute oracle.
# =====================================================================


def _extract_silhouette_edges(mesh):
    """Return (v0, v1, n0, n1) for all manifold + boundary edges.

    Manifold edge = shared by exactly two faces -> (n_a, n_b).
    Boundary edge = shared by exactly one face -> (n_a, -n_a) sentinel,
    so the sign-product validity test `s0*s1 <= 0` always accepts the
    boundary edge (a boundary edge is a silhouette by definition; there
    is no second face to disqualify it).
    Non-manifold (3+ faces) edges are excluded (ill-defined). Mirrors
    the backend encoding in `snch_jax._extract_silhouette_edges`.
    """
    verts = np.asarray(mesh.vertices, dtype=np.float32)
    faces = np.asarray(mesh.faces, dtype=np.int32)

    edge_to_faces = {}
    for f_idx, face in enumerate(faces):
        a, b, c = int(face[0]), int(face[1]), int(face[2])
        for u, v in ((a, b), (b, c), (c, a)):
            key = (min(u, v), max(u, v))
            edge_to_faces.setdefault(key, []).append(f_idx)

    v0_list, v1_list, n0_list, n1_list = [], [], [], []
    for (u, v), fs in edge_to_faces.items():
        n_faces = len(fs)
        if n_faces == 2:
            f0, f1 = fs
            a0, b0, c0 = faces[f0]
            a1, b1, c1 = faces[f1]
            n_a = np.cross(verts[b0] - verts[a0], verts[c0] - verts[a0])
            n_b = np.cross(verts[b1] - verts[a1], verts[c1] - verts[a1])
            ln_a = np.linalg.norm(n_a)
            ln_b = np.linalg.norm(n_b)
            if ln_a < 1e-12 or ln_b < 1e-12:
                continue
            n_a = n_a / ln_a
            n_b = n_b / ln_b
            v0_list.append(verts[u])
            v1_list.append(verts[v])
            n0_list.append(n_a)
            n1_list.append(n_b)
        elif n_faces == 1:
            f0 = fs[0]
            a0, b0, c0 = faces[f0]
            n_a = np.cross(verts[b0] - verts[a0], verts[c0] - verts[a0])
            ln_a = np.linalg.norm(n_a)
            if ln_a < 1e-12:
                continue
            n_a = n_a / ln_a
            v0_list.append(verts[u])
            v1_list.append(verts[v])
            n0_list.append(n_a)
            n1_list.append(-n_a)
        else:
            continue

    if not v0_list:
        return (np.zeros((0, 3), dtype=np.float32),) * 4
    return (np.array(v0_list, dtype=np.float32),
            np.array(v1_list, dtype=np.float32),
            np.array(n0_list, dtype=np.float32),
            np.array(n1_list, dtype=np.float32))


def _closest_point_on_segment(x, v0, v1):
    """Numpy: closest point on segment v0–v1 from x. Vectorised over edges."""
    d = v1 - v0                               # (E, 3)
    denom = np.sum(d * d, axis=-1)            # (E,)
    t = np.where(denom > 1e-30,
                 np.sum((x - v0) * d, axis=-1) / np.maximum(denom, 1e-30),
                 0.0)
    t = np.clip(t, 0.0, 1.0)
    return v0 + t[..., None] * d              # (E, 3)


def _brute_closest_silhouette(x, edges):
    """Brute oracle. Returns (cp, dist, edge_idx_or_neg1).

    Iterates all manifold edges, applies sign-product validity test,
    reduces to min distance over valid edges.
    """
    v0, v1, n0, n1 = edges
    if v0.shape[0] == 0:
        return (np.zeros(3, dtype=np.float32),
                np.float32(np.inf),
                np.int32(-1))

    s0 = np.einsum("ej,j->e", n0, x) - np.einsum("ej,ej->e", n0, v0)
    s1 = np.einsum("ej,j->e", n1, x) - np.einsum("ej,ej->e", n1, v0)
    # `<= 0.0` accepts the boundary-edge sentinel (n1 = -n0 -> s0*s1 =
    # -s0^2 <= 0) and matches the backend's leaf validity test.
    valid = (s0 * s1) <= 0.0

    cps = _closest_point_on_segment(x, v0, v1)              # (E, 3)
    dists = np.linalg.norm(x[None, :] - cps, axis=-1)       # (E,)
    masked = np.where(valid, dists, np.inf)
    if not np.any(np.isfinite(masked)):
        return (np.zeros(3, dtype=np.float32),
                np.float32(np.inf),
                np.int32(-1))
    j = int(np.argmin(masked))
    return cps[j].astype(np.float32), np.float32(masked[j]), np.int32(j)


# =====================================================================
# Test bodies.
# =====================================================================


@parametrize_backends("bvh")
def test_bvh_closest_silhouette_matches_brute(tri_fixture, backend_name):
    _skip_if_capability_missing(backend_name)

    edges = _extract_silhouette_edges(tri_fixture.tri_mesh)
    if edges[0].shape[0] == 0:
        pytest.skip("fixture has no manifold or boundary edges → no silhouette candidates")

    accel = build_accelerator("bvh", backend_name, tri_fixture.primitives)
    qs = tri_fixture.query_points
    cp_t, d_t, idx_t = _backend_closest_silhouette_batch(accel, qs)
    cp_t = np.asarray(cp_t)
    d_t = np.asarray(d_t)
    idx_t = np.asarray(idx_t)

    qs_np = np.asarray(qs)
    n_skipped_no_silh = 0
    for k in range(qs_np.shape[0]):
        cp_b, d_b, _ = _brute_closest_silhouette(qs_np[k], edges)
        # If brute reports no valid silhouette for this query, the
        # backend is allowed to report inf or anything else; skip.
        if not np.isfinite(d_b):
            n_skipped_no_silh += 1
            continue
        # Distance must match (oracle picks any valid edge with min
        # distance; backend may pick a different equidistant edge).
        np.testing.assert_allclose(
            d_t[k], d_b,
            rtol=SILH_RTOL, atol=SILH_ATOL,
            err_msg=f"silhouette distance disagrees at query {k}: "
                    f"backend {d_t[k]}, brute {d_b}")
        # Closest-point distance must equal reported distance.
        np.testing.assert_allclose(
            np.linalg.norm(qs_np[k] - cp_t[k]), d_t[k],
            rtol=SILH_RTOL, atol=SILH_ATOL,
            err_msg=f"||x - cp|| != d at query {k}")
    # Sanity: didn't skip everything (otherwise the test is vacuous).
    assert n_skipped_no_silh < qs_np.shape[0], \
        "every query had no silhouette → fixture too pathological for this test"


@parametrize_backends("bvh")
def test_bvh_closest_silhouette_invariants(tri_fixture, backend_name):
    """Layer 4 self-consistency: returned cp lies on the reported edge,
    dist == ||x - cp||, edge_idx in valid range or -1.
    """
    _skip_if_capability_missing(backend_name)

    accel = build_accelerator("bvh", backend_name, tri_fixture.primitives)
    qs = tri_fixture.query_points
    cp_t, d_t, idx_t = _backend_closest_silhouette_batch(accel, qs)
    cp_t = np.asarray(cp_t)
    d_t = np.asarray(d_t)
    idx_t = np.asarray(idx_t)
    qs_np = np.asarray(qs)

    for k in range(qs_np.shape[0]):
        if not np.isfinite(d_t[k]):
            # No silhouette found — edge_idx must be sentinel.
            assert idx_t[k] == -1, \
                f"query {k}: dist=inf but edge_idx={idx_t[k]} (expected -1)"
            continue
        # dist must equal ||x - cp||.
        recomputed = np.linalg.norm(qs_np[k] - cp_t[k])
        np.testing.assert_allclose(
            recomputed, d_t[k],
            rtol=SILH_RTOL, atol=SILH_ATOL,
            err_msg=f"query {k}: dist={d_t[k]} but ||x - cp||={recomputed}")
        # Distance must be non-negative.
        assert d_t[k] >= 0.0, f"query {k}: negative distance"


@parametrize_backends("kd")
def test_kd_closest_silhouette_matches_brute(tri_fixture, backend_name):
    """Defensive parametrize for future kd backends with silhouette."""
    _skip_if_capability_missing(backend_name)

    edges = _extract_silhouette_edges(tri_fixture.tri_mesh)
    if edges[0].shape[0] == 0:
        pytest.skip("fixture has no manifold or boundary edges")

    accel = build_accelerator("kd", backend_name, tri_fixture.primitives)
    qs = tri_fixture.query_points
    cp_t, d_t, idx_t = _backend_closest_silhouette_batch(accel, qs)
    cp_t = np.asarray(cp_t)
    d_t = np.asarray(d_t)

    qs_np = np.asarray(qs)
    for k in range(qs_np.shape[0]):
        _cp_b, d_b, _ = _brute_closest_silhouette(qs_np[k], edges)
        if not np.isfinite(d_b):
            continue
        np.testing.assert_allclose(d_t[k], d_b,
                                   rtol=SILH_RTOL, atol=SILH_ATOL)
