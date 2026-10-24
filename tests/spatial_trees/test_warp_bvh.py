"""Warp query checks against independent NumPy oracles."""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import trimesh

from surfaces.base import TriMesh
from surfaces.explicit.accelerators import BVHAccelerator, list_backends
from surfaces.explicit.accelerators.protocol import Config
from surfaces.explicit.primitives import TriMeshPrimitives


WARP_BVHS = [n for n in list_backends("bvh", runtime="warp")
             if n in ("bvh.sah.warp", "bvh.snch.warp")]
SAH_WARP = [n for n in WARP_BVHS if n == "bvh.sah.warp"]
pytestmark = pytest.mark.skipif(not WARP_BVHS,
                                reason="Warp BVH backends are not registered")


def _mesh(vertices, faces):
    vertices = np.asarray(vertices, np.float32)
    faces = np.asarray(faces, np.int32)
    tri = TriMesh(vertices=jnp.asarray(vertices), faces=jnp.asarray(faces))
    return TriMeshPrimitives(tri), vertices, faces


def _sphere(subdivisions=2):
    m = trimesh.creation.icosphere(subdivisions=subdivisions)
    return _mesh(m.vertices, m.faces)


@pytest.mark.parametrize("backend_name", WARP_BVHS, ids=WARP_BVHS)
def test_warp_query_bridge_preserves_cpu_execution(backend_name):
    accel = _build(backend_name, [[0, 0, 0], [1, 0, 0], [0, 1, 0]], [[0, 1, 2]])
    cpu = jax.devices("cpu")[0]
    payload = jax.tree_util.tree_map(lambda x: jax.device_put(x, cpu), accel.payload)
    origin = jax.device_put(np.asarray([.2, .2, 1], np.float32), cpu)
    direction = jax.device_put(np.asarray([0, 0, -1], np.float32), cpu)
    cap = jax.device_put(np.float32(2), cpu)
    outputs = jax.jit(accel.backend.first_hit_with_status)(
        payload, origin, direction, cap)
    assert all(value.device.platform == "cpu" for value in outputs)
    t, face, hit, status = map(np.asarray, outputs)
    assert float(t) == 1 and int(face) == 0 and bool(hit) and int(status) == 1

    cp_outputs = jax.jit(accel.backend.closest_point_with_status)(
        payload, origin)
    assert all(value.device.platform == "cpu" for value in cp_outputs)
    cp, distance, face, status = map(np.asarray, cp_outputs)
    np.testing.assert_allclose(cp, [.2, .2, 0.], rtol=0, atol=2e-6)
    np.testing.assert_allclose(distance, 1., rtol=0, atol=2e-6)
    assert int(face) == 0 and int(status) == 1

    if accel.has_silhouette_query:
        contact = jax.device_put(np.zeros(3, np.float32), cpu)
        outputs = jax.jit(accel.backend.closest_silhouette_with_status)(
            payload, contact, jax.device_put(np.float32(0), cpu))
        assert all(value.device.platform == "cpu" for value in outputs)
        cp, distance, edge, status = map(np.asarray, outputs)
        np.testing.assert_array_equal(cp, [0., 0., 0.])
        assert float(distance) == 0 and int(edge) == 0 and int(status) == 1


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


def _silhouette_edges(vertices, faces):
    edge_faces = {}
    normals = []
    for i, f in enumerate(faces):
        a, b, c = vertices[f]
        n = np.cross(b-a, c-a)
        n /= max(float(np.linalg.norm(n)), 1e-30)
        normals.append(n)
        for u, w in ((f[0], f[1]), (f[1], f[2]), (f[2], f[0])):
            edge_faces.setdefault(tuple(sorted((int(u), int(w)))), []).append(i)
    edges = []
    for (u, w), adj in edge_faces.items():
        if len(adj) == 2:
            edges.append((vertices[u], vertices[w], normals[adj[0]], normals[adj[1]]))
    return edges


def _oracle_silhouette(vertices, faces, point):
    best_cp, best_dist = np.zeros(3, np.float32), np.inf
    for a, b, n0, n1 in _silhouette_edges(vertices, faces):
        s0, s1 = float(n0 @ (point-a)), float(n1 @ (point-a))
        if s0*s1 > 0:
            continue
        d = b-a
        t = np.clip(float((point-a) @ d) / max(float(d @ d), 1e-30), 0, 1)
        cp = a + t*d
        dist = float(np.linalg.norm(point-cp))
        if dist < best_dist:
            best_cp, best_dist = cp, dist
    return best_cp, best_dist


def _build(name, vertices, faces):
    prims, _, _ = _mesh(vertices, faces)
    return BVHAccelerator.build(prims, name)


@pytest.mark.parametrize("backend_name", WARP_BVHS, ids=WARP_BVHS)
def test_warp_bvh_closest_point_matches_numpy_oracle(backend_name):
    prims, v, f = _sphere(2)
    accel = BVHAccelerator.build(prims, backend_name)
    rng = np.random.default_rng(817)
    qs = np.concatenate([rng.uniform(-1.5, 1.5, (29, 3)),
                         np.array([[1, 0, 0], [-1.2, .2, .1]], np.float32)])
    got = jax.jit(jax.vmap(accel.closest_point))(jnp.asarray(qs))
    cp, dist, ids = map(np.asarray, got)
    for k, q in enumerate(qs):
        c, d, _ = _oracle_closest(v, f, q)
        np.testing.assert_allclose(cp[k], c, atol=3e-5, rtol=3e-5)
        np.testing.assert_allclose(dist[k], d, atol=3e-5, rtol=3e-5)
        np.testing.assert_allclose(np.linalg.norm(cp[k] - q), dist[k], atol=3e-5)
        assert 0 <= ids[k] < len(f)


@pytest.mark.parametrize("backend_name", WARP_BVHS, ids=WARP_BVHS)
def test_warp_bvh_closest_point_equal_distance_tie_across_children(backend_name):
    # Equal bounds: face 0 in the later child must win.
    v = np.asarray([[1, 0, 0], [2, 0, 0], [1, 1, 0],
                    [-2, 0, 0], [-1, 0, 0], [-1, 1, 0]], np.float32)
    f = np.asarray([[0, 1, 2], [3, 4, 5]], np.int32)
    accel = BVHAccelerator.build(
        _mesh(v, f)[0], backend_name, Config(leaf_size_threshold=1))
    payload = accel.payload

    left = np.asarray(payload.node_left)
    right = np.asarray(payload.node_right)
    leaf_offset = np.asarray(payload.node_leaf_offset)
    leaf_count = np.asarray(payload.node_leaf_count)
    prim_ids = np.asarray(payload.prim_ids)
    assert left[0] >= 0 and right[0] >= 0
    assert leaf_count[left[0]] == leaf_count[right[0]] == 1
    assert prim_ids[leaf_offset[left[0]]] == 1
    assert prim_ids[leaf_offset[right[0]]] == 0

    q = np.asarray([0., .25, 1.], np.float32)
    query = jax.jit(lambda p, point: accel.backend.closest_point_with_status(
        p, point))
    cp, distance, face_id, status = map(
        np.asarray, query(payload, jnp.asarray(q)))
    assert int(face_id) == 0 and int(status) == 1
    np.testing.assert_allclose(cp, [1., .25, 0.], rtol=0, atol=1e-6)
    np.testing.assert_allclose(distance, np.sqrt(2.), rtol=0, atol=1e-6)


@pytest.mark.parametrize("backend_name", SAH_WARP, ids=SAH_WARP)
def test_warp_bvh_closest_point_degenerate_triangles(backend_name):
    v = np.asarray([[0, 0, 0], [2, 0, 0], [1, 0, 0],
                    [4, 0, 0], [4, 0, 0], [4, 0, 0]], np.float32)
    f = np.asarray([[0, 1, 2], [3, 4, 5]], np.int32)
    accel = _build(backend_name, v, f)
    queries = np.asarray([[.75, 1., 0.], [4.5, 0., 0.]], np.float32)
    cp, distance, face_id, status = map(np.asarray, jax.jit(
        accel.closest_point_with_status)(jnp.asarray(queries)))

    np.testing.assert_allclose(cp, [[.75, 0., 0.], [4., 0., 0.]],
                               rtol=0, atol=1e-6)
    np.testing.assert_allclose(distance, [1., .5], rtol=0, atol=1e-6)
    np.testing.assert_array_equal(face_id, [0, 1])
    np.testing.assert_array_equal(status, [1, 1])


@pytest.mark.parametrize("backend_name", SAH_WARP, ids=SAH_WARP)
def test_warp_bvh_rays_caps_axes_grazes_and_start_contact(backend_name):
    # Regular, collinear and point triangles.
    v = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0],
                  [2, 0, 0], [3, 0, 0], [4, 0, 0],
                  [5, 0, 0], [5, 0, 0], [5, 0, 0]], np.float32)
    f = np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]], np.int32)
    accel = _build(backend_name, v, f)
    origins = np.array([[.2, .2, 1], [1, .5, 0], [.5, .5, 0],
                        [.2, .2, -1], [8, 0, 1]], np.float32)
    dirs = np.array([[0, 0, -1], [-1, 0, 0], [0, 0, 1],
                     [0, 0, 1], [0, 0, 0]], np.float32)
    max_t = np.array([1, 2, 0, .5, 10], np.float32)
    def _first_hit_for(lower):
        return jax.jit(jax.vmap(
            lambda o, d, mt: accel.first_hit(o, d, mt, min_t=lower),
            in_axes=(0, 0, 0)))

    def _any_hit_for(lower):
        return jax.jit(jax.vmap(lambda o, d, mt: accel.backend.any_hit_with_status(
            accel.payload, o, d, mt, min_t=lower), in_axes=(0, 0, 0)))

    calls = [(lower, _first_hit_for(lower)) for lower in (0.0, 1e-4)]
    any_calls = [(lower, _any_hit_for(lower)) for lower in (0.0, 1e-4)]
    for (min_t, call), (_any_lower, any_call) in zip(calls, any_calls):
        ts, ids, hit = map(np.asarray, call(jnp.asarray(origins),
                                             jnp.asarray(dirs), jnp.asarray(max_t)))
        for k in range(len(origins)):
            t, idx = _oracle_ray(v, f, origins[k], dirs[k], max_t[k], min_t)
            assert bool(hit[k]) == np.isfinite(t), (
                f"lower={min_t}, ray={k}, backend t/id/hit="
                f"{ts[k]}/{ids[k]}/{hit[k]}, oracle t/id={t}/{idx}")
            if np.isfinite(t):
                np.testing.assert_allclose(ts[k], t, atol=2e-5, rtol=2e-5)
                assert ids[k] == idx
            else:
                assert ids[k] == -1
                # Misses return max_t with ID -1.
                assert ts[k] == max_t[k]

        any_hit, any_status = map(np.asarray, any_call(
            jnp.asarray(origins), jnp.asarray(dirs), jnp.asarray(max_t)))
        np.testing.assert_array_equal(any_hit, hit)
        valid_ray = ((np.linalg.norm(dirs, axis=1) > 1e-15)
                     & (max_t >= min_t))
        np.testing.assert_array_equal(
            any_status, np.where(valid_ray, np.where(hit, 1, 0), -1))


@pytest.mark.parametrize("backend_name", SAH_WARP, ids=SAH_WARP)
def test_warp_bvh_first_hit_equal_t_face_id_tie_across_children(backend_name):
    # Equal t=1 hits: face 0 lives in the later-visited child.
    centers = np.asarray([4., -4., -3., -2., -1., -1., 1., 2., 3., 4.],
                         np.float32)
    triangles = np.stack([np.column_stack([centers - 8, np.full_like(centers, -2),
                                           np.zeros_like(centers)]),
                          np.column_stack([centers + 8, np.full_like(centers, -2),
                                           np.zeros_like(centers)]),
                          np.column_stack([centers, np.full_like(centers, 4),
                                           np.zeros_like(centers)])], axis=1)
    vertices = triangles.reshape((-1, 3)).astype(np.float32)
    faces = np.arange(len(vertices), dtype=np.int32).reshape((-1, 3))
    accel = BVHAccelerator.build(
        _mesh(vertices, faces)[0], backend_name,
        Config(leaf_size_threshold=1))
    payload = accel.payload

    left = np.asarray(payload.node_left)
    right = np.asarray(payload.node_right)
    leaf_offset = np.asarray(payload.node_leaf_offset)
    leaf_count = np.asarray(payload.node_leaf_count)
    prim_ids = np.asarray(payload.prim_ids)

    def subtree_ids(root):
        ids = set()
        pending = [int(root)]
        while pending:
            node = pending.pop()
            if leaf_count[node] > 0:
                start = int(leaf_offset[node])
                stop = start + int(leaf_count[node])
                ids.update(map(int, prim_ids[start:stop]))
            else:
                pending.extend((int(left[node]), int(right[node])))
        return ids

    assert np.count_nonzero(leaf_count > 0) > 1
    assert left[0] >= 0 and right[0] >= 0
    first_ids = subtree_ids(left[0])
    later_ids = subtree_ids(right[0])
    assert 0 not in first_ids and 0 in later_ids
    assert first_ids and later_ids

    query = jax.jit(lambda p, o, d, cap: accel.backend.first_hit_with_status(
        p, o, d, cap))
    t, face_id, hit, status = query(
        payload, jnp.asarray([0., 0., 1.], jnp.float32),
        jnp.asarray([0., 0., -1.], jnp.float32), jnp.float32(1.))
    assert bool(hit) and int(status) == 1
    np.testing.assert_allclose(t, 1., rtol=0, atol=0)
    assert int(face_id) == 0


@pytest.mark.parametrize("backend_name", SAH_WARP, ids=SAH_WARP)
def test_warp_bvh_ray_preserves_large_int32_source_face_id(backend_name):
    vertices = np.asarray([[0, 0, 0], [1, 0, 0], [0, 1, 0]], np.float32)
    faces = np.asarray([[0, 1, 2]], np.int32)
    accel = _build(backend_name, vertices, faces)
    source_id = np.int32(2**24 + 1)
    payload = accel.payload._replace(
        prim_ids=jnp.asarray([source_id], dtype=jnp.int32))

    query = jax.jit(lambda p, o, d, cap: accel.backend.first_hit_with_status(
        p, o, d, cap))
    t, face_id, hit, status = query(
        payload, jnp.asarray([.2, .2, 1.], jnp.float32),
        jnp.asarray([0., 0., -1.], jnp.float32), jnp.float32(1.))
    assert bool(hit) and int(status) == 1
    np.testing.assert_allclose(t, 1., rtol=0, atol=0)
    assert np.asarray(face_id).dtype == np.int32
    assert int(face_id) == int(source_id)


@pytest.mark.parametrize("backend_name", WARP_BVHS, ids=WARP_BVHS)
@pytest.mark.parametrize("tiny", [np.float32(1e-30),
                                  np.nextafter(np.float32(0), np.float32(1))])
@pytest.mark.parametrize("side", [-1, 1])
def test_warp_bvh_tiny_direction_on_child_box_boundary(backend_name, tiny, side):
    # Exercise 0*inf on an internal child's slab boundary.
    triangles = []
    for i in range(9):
        x = side * 8.0 * i
        triangles.append([[x, 0, 1 + i], [x + side, 0, 1 + i],
                          [x, 1, 1 + i]])
    vertices = np.asarray(triangles, np.float32).reshape((-1, 3))
    faces = np.arange(len(vertices), dtype=np.int32).reshape((-1, 3))
    primitives, _, _ = _mesh(vertices, faces)
    accel = BVHAccelerator.build(
        primitives, backend_name, Config(leaf_size_threshold=1))
    assert np.any(np.asarray(accel.payload.node_leaf_count) == 0)

    origins = jnp.asarray([[0, .25, 0]] * 3, jnp.float32)
    directions = jnp.asarray([[side * tiny, 0, 1]] * 3, jnp.float32)
    lower = jnp.asarray([0, 1, 1.0001], jnp.float32)
    upper = jnp.asarray([jnp.inf, 1, jnp.inf], jnp.float32)
    ts, ids, hits, status = map(np.asarray, jax.jit(
        accel.backend.first_hit_with_status)(
            accel.payload, origins, directions, upper, min_t=lower))
    np.testing.assert_array_equal(hits, [True, True, False])
    np.testing.assert_array_equal(status, [1, 1, 0])
    np.testing.assert_array_equal(ids, [0, 0, -1])
    np.testing.assert_allclose(ts[:2], [1, 1], rtol=0, atol=0)
    assert np.isinf(ts[2])

    any_hit, any_status = map(np.asarray, jax.jit(
        accel.backend.any_hit_with_status)(
            accel.payload, origins, directions, upper, min_t=lower))
    np.testing.assert_array_equal(any_hit, hits)
    np.testing.assert_array_equal(any_status, status)


@pytest.mark.parametrize("backend_name", WARP_BVHS, ids=WARP_BVHS)
def test_warp_bvh_any_hit_matches_first_hit_boolean(backend_name):
    prims, v, f = _sphere(1)
    accel = BVHAccelerator.build(prims, backend_name)
    origins = jnp.asarray([[0., 0., 3.], [3., 3., 3.], [1., 0., 0.]])
    directions = jnp.asarray([[0., 0., -1.], [0., 0., -1.], [0., 1., 0.]])
    caps = jnp.asarray([5., 1., 0.])
    fn = jax.jit(jax.vmap(lambda o, d, cap: accel.any_hit(o, d, cap),
                          in_axes=(0, 0, 0)))
    got = np.asarray(fn(origins, directions, caps), dtype=bool)
    for i in range(len(got)):
        t, _ = _oracle_ray(v, f, np.asarray(origins[i]),
                           np.asarray(directions[i]), float(caps[i]))
        assert got[i] == np.isfinite(t)


@pytest.mark.parametrize("backend_name", WARP_BVHS, ids=WARP_BVHS)
def test_warp_bvh_constant_origin_broadcasts_across_ray_directions(backend_name):
    v = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], np.float32)
    f = np.array([[0, 1, 2]], np.int32)
    accel = _build(backend_name, v, f)
    origin = jnp.asarray([.2, .2, 1.], jnp.float32)  # deliberately unbatched
    directions = jnp.asarray([[0., 0., -1.], [1., 0., 0.], [0., 0., 1.]], jnp.float32)
    caps = jnp.asarray([2., 2., 2.])
    got = jax.jit(jax.vmap(lambda d, cap: accel.first_hit(
        origin, d, cap), in_axes=(0, 0)))(directions, caps)
    ts, ids, hit = map(np.asarray, got)
    want = [_oracle_ray(v, f, np.asarray(origin), np.asarray(d), float(cap))
            for d, cap in zip(np.asarray(directions), np.asarray(caps))]
    np.testing.assert_array_equal(hit, [np.isfinite(t) for t, _ in want])
    for i, (t, idx) in enumerate(want):
        if np.isfinite(t):
            np.testing.assert_allclose(ts[i], t, atol=2e-5)
            assert ids[i] == idx
        else:
            assert ids[i] == -1


@pytest.mark.parametrize("backend_name", WARP_BVHS, ids=WARP_BVHS)
def test_warp_bvh_nested_ray_batches_broadcast_outer_min_t(backend_name):
    v = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], np.float32)
    f = np.array([[0, 1, 2]], np.int32)
    accel = _build(backend_name, v, f)
    origins = jnp.asarray([[.2, .2, 0.], [.2, .2, 1.]], jnp.float32)
    directions = jnp.asarray([[0., 0., 1.], [0., 0., -1.]], jnp.float32)
    max_ts = jnp.asarray([2., 2.], jnp.float32)
    min_ts = jnp.asarray([0., 1e-4], jnp.float32)

    def per_min(min_t):
        return jax.vmap(lambda o, d, hi: accel.first_hit(
            o, d, hi, min_t=min_t), in_axes=(0, 0, 0))(
                origins, directions, max_ts)

    t, idx, hit = map(np.asarray, jax.jit(jax.vmap(per_min))(min_ts))
    np.testing.assert_array_equal(hit, [[True, True], [False, True]])
    np.testing.assert_allclose(t[0], [0., 1.], atol=2e-5)
    assert idx[0, 0] == 0 and idx[0, 1] == 0 and idx[1, 0] == -1 and idx[1, 1] == 0


@pytest.mark.parametrize("backend_name", WARP_BVHS, ids=WARP_BVHS)
def test_warp_payload_reuse_nested_vmap_jit_and_lax_loop(backend_name):
    prims, v, f = _sphere(1)
    accel = BVHAccelerator.build(prims, backend_name)
    qs = jnp.asarray(np.random.default_rng(7).normal(size=(3, 5, 3)).astype(np.float32))
    backend = accel.backend
    qflat = qs.reshape((-1, 3))
    qbank = jnp.concatenate([qflat, jnp.zeros((4, 3), jnp.float32)], axis=0)
    query = jax.jit(jax.vmap(jax.vmap(
        lambda payload, q: backend.closest_point(payload, q),
        in_axes=(None, 0)), in_axes=(None, 0)))
    nested = query(accel.payload, qs)
    assert np.asarray(nested[0]).shape == (3, 5, 3)

    # Reject vmap rules that tile BVH arrays.
    batch = 37
    probe = jax.make_jaxpr(jax.vmap(
        lambda payload, q: backend.closest_point(payload, q),
        in_axes=(None, 0)))(accel.payload, jnp.zeros((batch, 3), jnp.float32))
    source_shapes = [tuple(x.shape) for x in jax.tree_util.tree_leaves(accel.payload)
                     if len(x.shape) >= 1]
    traced_shapes = set()
    jaxpr = probe.jaxpr
    traced_vars = list(jaxpr.constvars) + list(jaxpr.invars) + list(jaxpr.outvars)
    for eqn in jaxpr.eqns:
        traced_vars.extend(eqn.invars)
        traced_vars.extend(eqn.outvars)
    for var in traced_vars:
        aval = getattr(var, "aval", None)
        if aval is not None and hasattr(aval, "shape"):
            traced_shapes.add(tuple(aval.shape))
    tiled_shapes = {(batch,) + shape for shape in source_shapes}
    assert not (traced_shapes & tiled_shapes), (
        "vmap physically tiled one or more BVH payload arrays: "
        f"{sorted(traced_shapes & tiled_shapes)}")

    def loop(payload, points):
        out = jnp.zeros_like(points)
        return jax.lax.fori_loop(
            0, points.shape[0],
            lambda i, acc: acc.at[i].set(backend.closest_point(payload, points[i])[0]),
            out)
    got = np.asarray(jax.jit(loop)(accel.payload, qflat))
    want = np.asarray(nested[0]).reshape((-1, 3))
    np.testing.assert_allclose(got, want, atol=2e-5)
    for n in (1, 7, 19):
        out = jax.jit(jax.vmap(accel.closest_point))(qbank[:n])
        assert np.asarray(out[0]).shape == (n, 3)


@pytest.mark.parametrize("backend_name", [n for n in WARP_BVHS if n == "bvh.snch.warp"],
                         ids=lambda n: n)
def test_warp_snch_silhouette_includes_open_boundary_and_no_candidate(backend_name):
    v = np.array([[0, 0, 0], [2, 0, 0], [0, 2, 0]], np.float32)
    f = np.array([[0, 1, 2]], np.int32)
    accel = _build(backend_name, v, f)
    qs = jnp.asarray([[.5, .5, 1], [4, 4, 1]], jnp.float32)
    cp, d, edge_id = map(np.asarray, jax.jit(jax.vmap(accel.closest_silhouette))(qs))
    for i, q in enumerate(np.asarray(qs)):
        edges = [(v[0], v[1]), (v[1], v[2]), (v[2], v[0])]
        cps = []
        for a, b in edges:
            t = np.clip(np.dot(q-a, b-a) / np.dot(b-a, b-a), 0, 1)
            cps.append(a + t*(b-a))
        ds = [np.linalg.norm(q-x) for x in cps]
        j = int(np.argmin(ds))
        np.testing.assert_allclose(cp[i], cps[j], atol=2e-5)
        np.testing.assert_allclose(d[i], ds[j], atol=2e-5)
        assert edge_id[i] >= 0


@pytest.mark.parametrize("backend_name", [n for n in WARP_BVHS if n == "bvh.snch.warp"],
                         ids=lambda n: n)
def test_warp_snch_closed_mesh_has_no_interior_silhouette(backend_name):
    v = np.array([[1, 1, 1], [-1, -1, 1], [-1, 1, -1], [1, -1, -1]], np.float32)
    f = np.array([[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]], np.int32)
    accel = _build(backend_name, v, f)
    cp, dist, edge_id = accel.closest_silhouette(jnp.zeros((3,), jnp.float32))
    assert np.isinf(float(dist))
    assert int(edge_id) == -1
    assert np.all(np.isfinite(np.asarray(cp)))


@pytest.mark.parametrize("backend_name", [n for n in WARP_BVHS if n == "bvh.snch.warp"],
                         ids=lambda n: n)
def test_warp_snch_silhouette_cap_and_ties_across_children(backend_name):
    v = np.asarray([[1, 0, 0], [2, 0, 0], [1, 1, 0],
                    [-2, 0, 0], [-1, 0, 0], [-1, 1, 0]], np.float32)
    f = np.asarray([[0, 1, 2], [3, 4, 5]], np.int32)
    accel = BVHAccelerator.build(
        _mesh(v, f)[0], backend_name, Config(leaf_size_threshold=1))
    payload = accel.payload
    left = int(payload.node_left[0])
    assert int(payload.prim_ids[payload.node_leaf_offset[left]]) == 1

    points = jnp.asarray([[0., 0., 0.], [0., 0., 0.], [1., 0., 0.]])
    caps = jnp.asarray([1., np.nextafter(np.float32(1), np.float32(0)), 0.])
    query = jax.jit(accel.backend.closest_silhouette_with_status)
    cp, distance, edge, status = map(np.asarray, query(payload, points, caps))
    np.testing.assert_array_equal(status, [1, 0, 1])
    np.testing.assert_array_equal(edge, [0, -1, 0])
    np.testing.assert_array_equal(cp[[0, 2]], [[1., 0., 0.], [1., 0., 0.]])
    np.testing.assert_array_equal(distance[[0, 2]], [1., 0.])
    assert np.isinf(distance[1])


@pytest.mark.parametrize("backend_name", [n for n in WARP_BVHS if n == "bvh.snch.warp"],
                         ids=lambda n: n)
def test_warp_snch_concave_prism_matches_exhaustive_edge_oracle(backend_name):
    # The re-entrant edge catches invalid cone pruning.
    outline = np.array([[-1, -1], [0, -1], [1, -1], [1, 0],
                        [0, 0], [0, 1], [-1, 1]], np.float32)
    n2 = len(outline)
    vertices = np.concatenate([
        np.column_stack([outline, np.full((n2,), -.5)]),
        np.column_stack([outline, np.full((n2,), .5)]),
    ]).astype(np.float32)
    # CCW top triangulation of the two rectangles making the L.
    top = [[0, 1, 4], [0, 4, 5], [0, 5, 6], [1, 2, 3], [1, 3, 4]]
    faces = [x[::-1] for x in top]
    faces += [[i+n2 for i in x] for x in top]
    # Side quads around the CCW outline, split with consistent outward normals.
    for i in range(n2):
        j = (i+1) % n2
        faces.extend([[i, j, j+n2], [i, j+n2, i+n2]])
    faces = np.asarray(faces, np.int32)
    prims, _, _ = _mesh(vertices, faces)
    accel = BVHAccelerator.build(prims, backend_name)
    rng = np.random.default_rng(917)
    qs = rng.uniform([-2., -2., -1.5], [2., 2., 1.5], size=(47, 3)).astype(np.float32)
    got_cp, got_d, got_id = map(np.asarray,
                                jax.jit(jax.vmap(accel.closest_silhouette))(jnp.asarray(qs)))
    for k, q in enumerate(qs):
        want_cp, want_d = _oracle_silhouette(vertices, faces, q)
        np.testing.assert_allclose(got_d[k], want_d, atol=4e-5, rtol=5e-5)
        if np.isfinite(want_d):
            np.testing.assert_allclose(got_cp[k], want_cp, atol=5e-5, rtol=5e-5)
            assert got_id[k] >= 0
        else:
            assert got_id[k] == -1


@pytest.mark.parametrize("backend_name", [n for n in WARP_BVHS if n == "bvh.snch.warp"],
                         ids=lambda n: n)
def test_warp_snch_long_edges_match_exhaustive_edge_oracle(backend_name):
    tm = trimesh.creation.box(extents=[1e4, 2., 1.])
    vertices = np.asarray(tm.vertices, np.float32)
    faces = np.asarray(tm.faces, np.int32)
    prims, _, _ = _mesh(vertices, faces)
    accel = BVHAccelerator.build(prims, backend_name)
    qs = np.asarray([[4999., 3., 2.], [-5001., 2., .5],
                     [0., 1.1, .6], [5000., -2., -1.]], np.float32)
    cp, dist, ids = map(np.asarray,
                        jax.jit(jax.vmap(accel.closest_silhouette))(jnp.asarray(qs)))
    for k, q in enumerate(qs):
        want_cp, want_dist = _oracle_silhouette(vertices, faces, q)
        np.testing.assert_allclose(dist[k], want_dist, atol=3e-4, rtol=3e-5)
        np.testing.assert_allclose(cp[k], want_cp, atol=4e-4, rtol=3e-5)
        assert ids[k] >= 0


@pytest.mark.parametrize("backend_name", [n for n in WARP_BVHS if n == "bvh.snch.warp"],
                         ids=lambda n: n)
def test_warp_snch_bounded_silhouette_on_torus(backend_name):
    tm = trimesh.creation.torus(major_radius=.7, minor_radius=.3,
                               major_sections=24, minor_sections=12)
    prims, vertices, faces = _mesh(tm.vertices, tm.faces)
    accel = BVHAccelerator.build(prims, backend_name, Config(leaf_size_threshold=2))
    qs = np.random.default_rng(647).uniform(-1.2, 1.2, (35, 3)).astype(np.float32)
    caps = np.resize(np.asarray([.15, .75, np.inf], np.float32), len(qs))
    cp, distance, edge, status = map(np.asarray, jax.jit(
        accel.closest_silhouette_with_status)(jnp.asarray(qs), max_distance=jnp.asarray(caps)))
    for k, q in enumerate(qs):
        want_cp, want_distance = _oracle_silhouette(vertices, faces, q)
        if want_distance <= caps[k]:
            assert status[k] == 1 and edge[k] >= 0
            np.testing.assert_allclose(distance[k], want_distance, atol=4e-5, rtol=5e-5)
            np.testing.assert_allclose(cp[k], want_cp, atol=5e-5, rtol=5e-5)
        else:
            assert status[k] == 0 and edge[k] == -1
            assert np.isinf(distance[k])


@pytest.mark.parametrize("scale", [1.e-3, 1., 1.e3])
@pytest.mark.parametrize("backend_name", [n for n in WARP_BVHS if n == "bvh.snch.warp"],
                         ids=lambda n: n)
def test_warp_snch_thin_box_contacts_at_multiple_scales(backend_name, scale):
    tm = trimesh.creation.box(extents=[4., 2., .125])
    rotation = trimesh.transformations.rotation_matrix(.37, [0., 0., 1.])[:3, :3]
    vertices = ((np.asarray(tm.vertices) @ rotation.T) * scale).astype(np.float32)
    faces = np.asarray(tm.faces, np.int32)
    accel = BVHAccelerator.build(
        _mesh(vertices, faces)[0], backend_name, Config(leaf_size_threshold=1))
    qs = np.asarray([[0., 0., .0625], [2., 0., .0625], [0., 1., .0625],
                     [2.0001, 0., .0625], [0., 0., .062501], [0., 0., .062499]])
    qs = ((qs @ rotation.T) * scale).astype(np.float32)
    cp, distance, edge = map(np.asarray, jax.jit(accel.closest_silhouette)(jnp.asarray(qs)))
    for k, q in enumerate(qs):
        _, want_distance = _oracle_silhouette(vertices, faces, q)
        np.testing.assert_allclose(distance[k], want_distance, atol=2e-6 * scale, rtol=5e-5)
        if np.isfinite(want_distance):
            assert edge[k] >= 0
            np.testing.assert_allclose(np.linalg.norm(cp[k] - q), distance[k],
                                       atol=2e-6 * scale, rtol=5e-5)
        else:
            assert edge[k] == -1


@pytest.mark.parametrize("backend_name", [n for n in WARP_BVHS if n == "bvh.snch.warp"],
                         ids=lambda n: n)
def test_warp_snch_silhouette_on_nearly_flat_closed_mesh(backend_name):
    tm = trimesh.creation.icosphere(subdivisions=2)
    vertices = np.asarray(tm.vertices, np.float32) * np.asarray([1., 1., 1.e-4], np.float32)
    faces = np.asarray(tm.faces, np.int32)
    accel = BVHAccelerator.build(
        _mesh(vertices, faces)[0], backend_name, Config(leaf_size_threshold=1))
    payload = jax.tree_util.tree_map(np.asarray, accel.payload)
    for node in np.flatnonzero(payload.node_leaf_count):
        sin_alpha, cos_alpha = payload.node_cone_sin_cos[node]
        if cos_alpha <= 0:
            continue
        start = payload.node_silhouette_offset[node]
        refs = payload.silhouette_refs[start:start + payload.node_silhouette_count[node]]
        normals = np.concatenate([payload.silhouette_n0[refs], payload.silhouette_n1[refs]])
        axis = payload.node_cone_axis[node].astype(np.float64)
        normals = normals.astype(np.float64)
        angles = np.arctan2(np.linalg.norm(np.cross(normals, axis), axis=1), normals @ axis)
        assert np.max(angles) <= np.arctan2(float(sin_alpha), float(cos_alpha)) + 1.e-7
    qs = np.random.default_rng(927).uniform(-1.5, 1.5, (71, 3)).astype(np.float32)
    qs[:, 2] *= 1.e-4
    cp, distance, edge = map(np.asarray, jax.jit(accel.closest_silhouette)(jnp.asarray(qs)))
    for k, q in enumerate(qs):
        _, want_distance = _oracle_silhouette(vertices, faces, q)
        np.testing.assert_allclose(distance[k], want_distance, atol=4e-5, rtol=5e-5)
        if np.isfinite(want_distance):
            assert edge[k] >= 0
            np.testing.assert_allclose(np.linalg.norm(cp[k] - q), distance[k], atol=4e-5)
        else:
            assert edge[k] == -1


@pytest.mark.parametrize("backend_name", [n for n in WARP_BVHS if n == "bvh.snch.warp"],
                         ids=lambda n: n)
def test_warp_snch_star_radius_applies_only_the_requested_cap(backend_name):
    v = np.array([[0, 0, 0], [2, 0, 0], [0, 2, 0]], np.float32)
    f = np.array([[0, 1, 2]], np.int32)
    accel = _build(backend_name, v, f)
    q = jnp.broadcast_to(jnp.asarray([1., 1., 1.]), (5, 3))
    caps = jnp.asarray([10., .2, 0., -1., jnp.nan])
    radius, status = map(np.asarray, jax.jit(jax.vmap(
        accel.star_radius_with_status))(q, caps))
    np.testing.assert_allclose(radius[:3], [1., .2, 0.], atol=2e-5)
    np.testing.assert_array_equal(status, [1, 1, 1, -1, -1])
    assert np.isnan(radius[3:]).all()

    # No silhouette lies within the cap.
    _, bounded_dist, bounded_id, bounded_status = accel.closest_silhouette_with_status(
        q[0], max_distance=jnp.float32(.2))
    assert np.isinf(float(bounded_dist))
    assert int(bounded_id) == -1 and int(bounded_status) == 0

    # Map caps outside point batches; share the tree.
    points = jnp.asarray([[1., 1., 1.], [1., 1., 1.]], jnp.float32)
    outer_caps = jnp.asarray([.2, 10.], jnp.float32)
    def per_cap(cap):
        return jax.vmap(lambda point: accel.star_radius_with_status(point, cap))(points)
    radii, statuses = map(np.asarray, jax.jit(jax.vmap(per_cap))(outer_caps))
    np.testing.assert_allclose(radii, [[.2, .2], [1., 1.]], atol=2e-5)
    assert np.all(statuses == 1)


@pytest.mark.parametrize("backend_name", [n for n in WARP_BVHS if n == "bvh.snch.warp"],
                         ids=lambda n: n)
def test_warp_snch_boundary_sampler_area_pdf_and_null_event(backend_name):
    prims, vertices, faces = _sphere(2)
    accel = BVHAccelerator.build(prims, backend_name)
    rng = np.random.default_rng(136)
    n = 12000
    randoms = rng.random((n, 3), dtype=np.float32)
    points = np.zeros((n, 3), np.float32)
    areas = np.linalg.norm(np.cross(vertices[faces[:, 1]] - vertices[faces[:, 0]],
                                    vertices[faces[:, 2]] - vertices[faces[:, 0]]),
                          axis=1) * .5
    total_area = float(areas.sum())
    sampler = jax.jit(accel.sample_boundary)

    p, normal, pdf, ids, status = map(np.asarray, sampler(
        jnp.asarray(points), jnp.full((n,), jnp.inf, jnp.float32), jnp.asarray(randoms)))
    assert np.all(status == 1)
    assert np.all((ids >= 0) & (ids < len(faces)))
    np.testing.assert_allclose(pdf, 1.0 / total_area, rtol=3e-5, atol=3e-6)
    np.testing.assert_allclose(np.linalg.norm(normal, axis=1), 1., atol=3e-5)
    expected = areas / total_area
    observed = np.bincount(ids.astype(np.int64), minlength=len(faces)) / n
    # Allow sampling noise across 320 faces.
    assert np.max(np.abs(observed - expected)) < .012
    assert np.all(np.isfinite(p))

    # Scalar broadcasting must match repeated point storage.
    small_n = 64
    small_randoms = rng.random((small_n, 3), dtype=np.float32)
    small_radii = np.resize(np.array([np.inf, .99, .99, 0.], np.float32), small_n)
    scalar_point_out = jax.jit(jax.vmap(lambda r, u: accel.sample_boundary(
        jnp.zeros((3,), jnp.float32), r, u), in_axes=(0, 0)))(
            jnp.asarray(small_radii), jnp.asarray(small_randoms))
    repeated_point_out = sampler(
        jnp.zeros((small_n, 3), jnp.float32), jnp.asarray(small_radii),
        jnp.asarray(small_randoms))
    for scalar_leaf, repeated_leaf in zip(scalar_point_out, repeated_point_out):
        np.testing.assert_allclose(scalar_leaf, repeated_leaf, atol=2e-6, rtol=2e-6)

    # Nulls contribute zero to the area estimator.
    radius = .99
    p, normal, pdf, ids, status = map(np.asarray, sampler(
        jnp.zeros((n, 3), jnp.float32), jnp.full((n,), radius, jnp.float32),
        jnp.asarray(randoms)))
    assert np.all(status >= 0)
    accepted = status == 1
    rejected = status == 0
    assert accepted.any() and rejected.any()
    assert np.all(pdf[accepted] > 0) and np.all(ids[accepted] >= 0)
    assert np.all(pdf[rejected] == 0) and np.all(ids[rejected] == -1)
    assert np.all(normal[rejected] == 0.)
    assert np.all(np.linalg.norm(p[accepted], axis=1) <= radius + 2e-5)
    estimated_area = float(np.mean(np.where(accepted, 1. / np.maximum(pdf, 1e-30), 0.)))

    # Independent area-uniform Monte Carlo.
    m = 250000
    rr = np.random.default_rng(2025).random((m, 3))
    face_ids = np.searchsorted(np.cumsum(areas / total_area), rr[:, 0], side="right")
    u = np.sqrt(rr[:, 1])
    bary = np.stack([1-u, u*(1-rr[:, 2]), u*rr[:, 2]], axis=1)
    samples = np.sum(vertices[faces[face_ids]] * bary[..., None], axis=1)
    oracle_area = total_area * float(np.mean(np.linalg.norm(samples, axis=1) <= radius))
    np.testing.assert_allclose(estimated_area, oracle_area, rtol=.08, atol=.015)

    # Infinite radius on an empty tree is a valid null event.
    empty = _build(backend_name, np.zeros((0, 3), np.float32),
                   np.zeros((0, 3), np.int32))
    empty_out = empty.sample_boundary(
        jnp.zeros((1, 3), jnp.float32), jnp.asarray([jnp.inf]),
        jnp.asarray([[.2, .3, .4]], jnp.float32))
    assert int(np.asarray(empty_out[4])[0]) == 0
    assert int(np.asarray(empty_out[3])[0]) == -1
    assert float(np.asarray(empty_out[2])[0]) == 0.


@pytest.mark.parametrize("backend_name", WARP_BVHS, ids=WARP_BVHS)
@pytest.mark.parametrize("empty_tree", [False, True])
def test_warp_bvh_empty_mesh_and_query_batches(backend_name, empty_tree):
    if empty_tree:
        v = np.zeros((0, 3), np.float32)
        f = np.zeros((0, 3), np.int32)
    else:
        v = np.asarray([[0, 0, 0], [1, 0, 0], [0, 1, 0]], np.float32)
        f = np.asarray([[0, 1, 2]], np.int32)
    accel = _build(backend_name, v, f)
    # Empty launches preserve output shapes.
    q = jnp.empty((0, 3), jnp.float32)
    caps = jnp.empty((0,), jnp.float32)
    outputs = [jax.jit(accel.closest_point_with_status)(q),
               jax.jit(accel.first_hit_with_status)(q, q, caps),
               jax.jit(accel.any_hit_with_status)(q, q, caps),
               jax.jit(accel.aabb_overlap_with_status)(q, q)]
    if accel.has_silhouette_query:
        outputs += [jax.jit(accel.closest_silhouette_with_status)(q),
                    jax.jit(accel.sample_boundary)(q, caps, q)]
    jax.block_until_ready(outputs)
    assert all(value.shape[0] == 0
               for result in outputs for value in result)
    if empty_tree:
        cp, distance, idx, status = accel.closest_point_with_status(jnp.zeros(3))
        assert np.isinf(float(distance)) and int(idx) == -1 and int(status) == 0
        assert np.all(np.isfinite(np.asarray(cp)))


@pytest.mark.parametrize("backend_name", WARP_BVHS, ids=WARP_BVHS)
def test_warp_bvh_single_triangle(backend_name):
    v = np.array([[0, 0, 0], [2, 0, 0], [0, 1, 0]], np.float32)
    f = np.array([[0, 1, 2]], np.int32)
    accel = _build(backend_name, v, f)
    queries = jnp.asarray([[.3, .2, 1.], [3., 0., 0.], [-1., 2., 0.]])
    cp, dist, idx = map(np.asarray, jax.jit(jax.vmap(accel.closest_point))(queries))
    for k, q in enumerate(np.asarray(queries)):
        want_cp, want_d, want_id = _oracle_closest(v, f, q)
        np.testing.assert_allclose(cp[k], want_cp, atol=2e-5)
        np.testing.assert_allclose(dist[k], want_d, atol=2e-5)
        assert idx[k] == want_id == 0


@pytest.mark.parametrize("backend_name", WARP_BVHS, ids=WARP_BVHS)
@pytest.mark.parametrize("scale", [1e-4, 1., 1e4])
def test_warp_bvh_scale_robust_triangle_cp_and_ray(backend_name, scale):
    # Check seven orders of scale against analytic answers.
    v = scale * np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], np.float32)
    f = np.array([[0, 1, 2]], np.int32)
    accel = _build(backend_name, v, f)
    q = jnp.asarray(scale * np.array([.2, .3, .4], np.float32))
    cp, dist, idx = accel.closest_point(q)
    np.testing.assert_allclose(cp, scale*np.array([.2, .3, 0]), atol=2e-7*scale, rtol=2e-5)
    np.testing.assert_allclose(dist, .4*scale, atol=2e-7*scale, rtol=2e-5)
    assert int(idx) == 0
    origin = jnp.asarray(scale * np.array([.2, .3, 1], np.float32))
    direction = jnp.asarray([0., 0., -1.], jnp.float32)
    t, prim, hit = accel.first_hit(origin, direction, jnp.float32(2*scale))
    assert bool(hit) and int(prim) == 0
    np.testing.assert_allclose(t, scale, atol=2e-7*scale, rtol=2e-5)


@pytest.mark.parametrize("backend_name", WARP_BVHS, ids=WARP_BVHS)
def test_warp_bvh_long_thin_triangle_uses_stable_projection(backend_name):
    a = np.array([0., 0., 0.], np.float32)
    e1 = np.array([1e4, 0., 0.], np.float32)
    e2 = np.array([1e4, 1e-3, 0.], np.float32)
    v = np.stack([a, a+e1, a+e2])
    f = np.array([[0, 1, 2]], np.int32)
    accel = _build(backend_name, v, f)
    target = a + .2*e1 + .3*e2
    q = jnp.asarray(target + [0., 0., 2.])
    cp, dist, idx = accel.closest_point(q)
    np.testing.assert_allclose(cp, target, atol=2e-4, rtol=2e-6)
    np.testing.assert_allclose(dist, 2., atol=2e-4)
    assert int(idx) == 0


@pytest.mark.parametrize("backend_name", WARP_BVHS, ids=WARP_BVHS)
def test_warp_bvh_closest_point_small_oblique_triangle_from_far_query(backend_name):
    vertices = np.asarray([[0., 0., 0.], [.03, .01, .02],
                           [-.01, .02, .01]], np.float32)
    faces = np.asarray([[0, 1, 2]], np.int32)
    accel = _build(backend_name, vertices, faces)

    v64 = vertices.astype(np.float64)
    a, b, c = v64
    normal = np.cross(b - a, c - a)
    normal /= np.linalg.norm(normal)
    targets = np.stack([a + .25 * (b - a) + .35 * (c - a),
                        .5 * (a + b), c])
    queries = np.asarray(targets + 5. * normal, np.float32)
    oracle = [_oracle_closest(v64, faces, q.astype(np.float64))
              for q in queries]
    expected = np.stack([result[0] for result in oracle])
    expected_distance = np.asarray([result[1] for result in oracle])

    cp, distance, face_id = map(
        np.asarray, jax.jit(jax.vmap(accel.closest_point))(jnp.asarray(queries)))
    np.testing.assert_allclose(cp, expected, rtol=2e-5, atol=2e-5)
    np.testing.assert_allclose(distance, expected_distance, rtol=2e-5, atol=2e-5)
    np.testing.assert_array_equal(face_id, [0, 0, 0])


@pytest.mark.parametrize("backend_name", WARP_BVHS, ids=WARP_BVHS)
@pytest.mark.parametrize("scale", [1e-4, 1., 1e4])
@pytest.mark.parametrize("aspect_ratio", [1., .11, .09, 1e-3, 1e-4, 1e-5])
def test_warp_bvh_closest_point_fastpath_threshold(backend_name, scale,
                                                aspect_ratio):
    vertices = (scale * np.asarray([[0., 0., 0.], [1., 0., 0.],
                                    [.25, aspect_ratio, 0.]], np.float32))
    faces = np.asarray([[0, 1, 2]], np.int32)
    accel = _build(backend_name, vertices, faces)

    centroid = np.mean(vertices.astype(np.float64), axis=0).astype(np.float32)
    queries = np.asarray([
        centroid + np.asarray([0., 0., .4 * scale], np.float32),
        np.asarray([.4 * scale, -.1 * scale, .3 * scale], np.float32),
    ], np.float32)
    expected_cp = np.asarray([[queries[0, 0], queries[0, 1], 0.],
                              [queries[1, 0], 0., 0.]], np.float32)
    expected_distance = np.asarray([
        abs(float(queries[0, 2])),
        np.hypot(float(queries[1, 1]), float(queries[1, 2])),
    ], np.float32)

    cp, distance, face_id = map(
        np.asarray, jax.jit(jax.vmap(accel.closest_point))(jnp.asarray(queries)))
    tolerance = 2e-6 * scale
    np.testing.assert_allclose(cp, expected_cp, rtol=2e-5, atol=tolerance)
    np.testing.assert_allclose(distance, expected_distance,
                               rtol=2e-5, atol=tolerance)
    np.testing.assert_array_equal(face_id, [0, 0])


@pytest.mark.parametrize("backend_name", WARP_BVHS, ids=WARP_BVHS)
@pytest.mark.parametrize("shift", [0, 1, 2])
def test_warp_bvh_closest_point_obtuse_triangle_vertex_order(backend_name, shift):
    vertices = np.asarray([[0., 0., 0.], [1., 0., 0.], [-1., .2, 0.]], np.float32)
    faces = np.roll(np.asarray([[0, 1, 2]], np.int32), shift, axis=1)
    accel = _build(backend_name, vertices, faces)
    query = np.asarray([0., .05, .4], np.float32)
    point, distance, face = map(np.asarray, jax.jit(accel.closest_point)(query))
    np.testing.assert_allclose(point, [0., .05, 0.], rtol=0, atol=2e-6)
    np.testing.assert_allclose(distance, .4, rtol=0, atol=2e-6)
    assert int(face) == 0


@pytest.mark.parametrize("backend_name", WARP_BVHS, ids=WARP_BVHS)
def test_warp_bvh_closest_point_translated_triangle(backend_name):
    vertices = np.asarray([[10000., 10000., 0.], [10001., 10000., 0.],
                           [10000., 10001., 0.]], np.float32)
    accel = _build(backend_name, vertices, [[0, 1, 2]])
    queries = np.asarray([[10000.123, 10000.321, .4],
                          [10000.25, 10000.5, -.3]], np.float32)
    point, distance, face = map(np.asarray, jax.jit(accel.closest_point)(queries))
    expected = queries.copy()
    expected[:, 2] = 0.
    np.testing.assert_allclose(point, expected, rtol=0, atol=2e-6)
    np.testing.assert_allclose(distance, np.abs(queries[:, 2]), rtol=0, atol=2e-6)
    np.testing.assert_array_equal(face, [0, 0])


@pytest.mark.parametrize("backend_name", WARP_BVHS, ids=WARP_BVHS)
def test_warp_bvh_status_distinguishes_invalid_from_empty(backend_name):
    v = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], np.float32)
    f = np.array([[0, 1, 2]], np.int32)
    accel = _build(backend_name, v, f)
    cp_result = accel.backend.closest_point_with_status(
        accel.payload, jnp.asarray([jnp.nan, 0., 0.]))
    assert int(cp_result[3]) == -1
    overflow = accel.backend.closest_point_with_status(
        accel.payload, jnp.asarray([1.e20, 0., 0.], jnp.float32))
    point, distance, face, status = map(np.asarray, overflow)
    np.testing.assert_array_equal(point, np.zeros(3, np.float32))
    assert np.isinf(distance) and int(face) == -1 and int(status) == -1
    ray_result = accel.backend.first_hit_with_status(
        accel.payload, jnp.zeros((3,), jnp.float32),
        jnp.asarray([jnp.nan, 0., -1.]), jnp.float32(2.))
    assert int(ray_result[3]) == -1

    empty = _build(backend_name, np.zeros((0, 3), np.float32),
                   np.zeros((0, 3), np.int32))
    empty_cp = empty.backend.closest_point_with_status(
        empty.payload, jnp.zeros((3,), jnp.float32))
    assert int(empty_cp[3]) == 0
    empty_ray = empty.backend.first_hit_with_status(
        empty.payload, jnp.zeros((3,), jnp.float32),
        jnp.asarray([0., 0., 1.]), jnp.float32(1.))
    assert int(empty_ray[3]) == 0 and not bool(empty_ray[2])

    # Unbounded max_t is supported; an infinite lower bound is invalid.
    hit = accel.backend.first_hit_with_status(
        accel.payload, jnp.asarray([.2, .2, 1.], jnp.float32),
        jnp.asarray([0., 0., -1.], jnp.float32), jnp.float32(jnp.inf))
    assert bool(hit[2]) and int(hit[1]) == 0 and int(hit[3]) == 1
    np.testing.assert_allclose(hit[0], 1., atol=2e-5)
    bad_interval = accel.backend.first_hit_with_status(
        accel.payload, jnp.asarray([.2, .2, 1.], jnp.float32),
        jnp.asarray([0., 0., -1.], jnp.float32), jnp.float32(jnp.inf),
        min_t=jnp.float32(jnp.inf))
    assert int(bad_interval[3]) == -1 and not bool(bad_interval[2])


@pytest.mark.parametrize("backend_name", WARP_BVHS, ids=WARP_BVHS)
def test_warp_bvh_build_config_depth_and_leaf_bounds(backend_name):
    prims, _, _ = _sphere(2)
    with pytest.raises(ValueError):
        BVHAccelerator.build(prims, backend_name, Config(max_stack_depth=0))
    with pytest.raises(ValueError):
        BVHAccelerator.build(prims, backend_name, Config(max_stack_depth=65))
    with pytest.raises(ValueError):
        BVHAccelerator.build(prims, backend_name, Config(leaf_size_threshold=0))
    with pytest.raises(ValueError, match="depth"):
        BVHAccelerator.build(
            prims, backend_name,
            Config(max_stack_depth=1, leaf_size_threshold=1))
    built = BVHAccelerator.build(
        prims, backend_name,
        Config(max_stack_depth=64, leaf_size_threshold=1))
    assert built.payload is not None
