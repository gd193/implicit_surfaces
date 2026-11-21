"""Triangle surfaces exercise bound Sengi queries and preserve source face IDs."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import trimesh

from sengi import bvh
from surfaces.base import TriMesh
from surfaces.explicit.trimesh import TriMeshSurface


pytest.importorskip("warp")


def _index_mesh(mesh, builder):
    from sengi.primitives.triangles import Triangles

    model = TriMesh(
        jnp.asarray(mesh.vertices, jnp.float32), jnp.asarray(mesh.faces, jnp.int32)
    )
    primitive = Triangles()
    source = primitive.indexed(model.vertices, model.faces)
    index = bvh.build(primitive, source, builder=builder)
    boundary = bvh.prepare_boundary(index, source, silhouettes=False)
    return model, index, boundary


@pytest.mark.parametrize("builder", ["sah", "lbvh"])
def test_trimesh_surface_signed_value_and_first_hit(builder):
    mesh, index, boundary = _index_mesh(
        trimesh.creation.box(extents=[2, 2, 2]), builder
    )
    surface = TriMeshSurface(mesh, index, boundary)
    assert surface.index is index
    assert surface.has_box_bounds
    lower = jnp.ones(3, jnp.float32) * 0.1
    assert int(surface.classify_box(None, lower, -lower)) == 0
    points = jnp.array([[0, 0, 0], [2, 0, 0], [1, 0, 0]], jnp.float32)
    np.testing.assert_allclose(jax.jit(surface.signed_value)(None, points), [-1, 1, 0])
    value, radius, _ = surface.empty_sphere(None, points, 4.0)
    np.testing.assert_allclose(value, [-1, 1, 0])
    np.testing.assert_allclose(radius, [1, 1, 0])
    time, face, hit = jax.jit(surface.first_hit)(
        None,
        jnp.array([2.0, 0.25, 0.5]),
        jnp.array([-1.0, 0.0, 0.0]),
        jnp.float32(4.0),
    )
    assert bool(hit) and float(time) == 1.0
    np.testing.assert_array_equal(mesh.vertices[mesh.faces[face], 0], [1.0, 1.0, 1.0])

    for center, expected in (([0, 0, 0], -1), ([2, 0, 0], 1), ([1, 0, 0], 0)):
        center = jnp.array(center, jnp.float32)
        assert int(surface.classify_box(None, center - 0.1, center + 0.1)) == expected
        lower, upper = jax.jit(surface.box_bounds)(None, center - 0.125, center + 0.125)
        assert np.isfinite([lower, upper]).all() and lower <= upper
        # Analytic cube SDF at every corner: inside, outside and crossing boxes.
        offsets = np.array(np.meshgrid(*[[-0.125, 0.125]] * 3)).reshape(3, -1).T
        q = np.abs(np.asarray(center) + offsets) - 1.0
        distances = np.linalg.norm(np.maximum(q, 0.0), axis=1) + np.minimum(
            q.max(axis=1), 0.0
        )
        assert np.all(distances >= float(lower)) and np.all(distances <= float(upper))


@pytest.mark.parametrize("builder", ["sah", "lbvh"])
def test_surface_normals_and_vertex_velocity_use_original_face_ids(builder):
    mesh = trimesh.creation.icosphere(subdivisions=1)
    model, index, boundary = _index_mesh(mesh, builder)
    assert not np.array_equal(index.data.tree.prim_ids, np.arange(len(mesh.faces)))
    weights = np.array([0.2, 0.3, 0.5], np.float32)
    points = jnp.asarray(weights @ mesh.triangles, jnp.float32)
    velocities = (
        np.random.default_rng(81).normal(size=mesh.vertices.shape).astype(np.float32)
    )
    surface = TriMeshSurface(
        model, index, boundary, vertex_velocity_fn=lambda field, ids: field[ids]
    )
    normals = jax.vmap(surface.normal_at, in_axes=(None, 0))(None, points)
    speeds = jax.vmap(surface.surface_velocity_normal, in_axes=(None, 0, 0))(
        jnp.asarray(velocities), points, normals
    )
    np.testing.assert_allclose(normals, mesh.face_normals, atol=2e-6)
    expected = np.sum((weights @ velocities[mesh.faces]) * mesh.face_normals, axis=1)
    np.testing.assert_allclose(speeds, expected, atol=3e-6)
