"""Exact region queries against independent CPU geometry checks."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import trimesh

wp = pytest.importorskip("warp")

from surfaces.explicit.accelerators.bvh.warp import (  # noqa: E402
    DeviceTree, PointsData3, Points, Segments, Triangles, build_boundary,
    build_tree,
)
from surfaces.explicit.accelerators.bvh.backends.warp_queries.queries.region import (  # noqa: E402
    RegionIntersectionPolicy, make_region, make_region_classify,
    point_region_policy, segment_region_policy, triangle_region_policy,
)
from tests.spatial_trees._warp_device import device_payload, run_kernel  # noqa: E402


def test_points_random_count_limit_empty_invalid_and_jit():
    rng = np.random.default_rng(714)
    points = rng.uniform(-2, 2, (97, 3)).astype(np.float32)
    primitive = Points()
    tree, geometry = build_tree(primitive, primitive.data(points))
    lo = rng.uniform(-2, 1, (31, 3)).astype(np.float32)
    hi = lo + rng.uniform(0, 1, (31, 3)).astype(np.float32)
    oracle = np.all((lo[:, None] <= points) & (points <= hi[:, None]), axis=2).sum(axis=1)
    policy = point_region_policy(3)
    count = make_region(primitive, policy, mode="count")
    any_query = make_region(primitive, policy)
    limited = make_region(primitive, policy, mode="count", count_limit=2)
    for query, expected in ((count, oracle), (any_query, oracle > 0),
                            (limited, np.minimum(oracle, 2))):
        result, status = jax.jit(query)(tree, geometry, lo, hi)
        np.testing.assert_array_equal(result, expected)
        np.testing.assert_array_equal(status, 1)
    empty_tree, empty_geometry = build_tree(primitive, primitive.data(np.empty((0, 3), np.float32)))
    result, status = count(empty_tree, empty_geometry, lo[:1], hi[:1])
    np.testing.assert_array_equal(result, 0)
    np.testing.assert_array_equal(status, 1)
    bad_lo = np.array([[np.nan, 0, 0], [2, 0, 0]], np.float32)
    bad_hi = np.array([[1, 1, 1], [1, 1, 1]], np.float32)
    _, status = count(tree, geometry, bad_lo, bad_hi)
    np.testing.assert_array_equal(status, -1)
    with pytest.raises(ValueError, match="count_limit"):
        make_region(primitive, policy, mode="count", count_limit=0)


def test_point2_closed_box_contact():
    primitive = Points(2)
    tree, geometry = build_tree(primitive, primitive.data(np.array([[0, 0], [1, 1]], np.float32)))
    q = make_region(primitive, point_region_policy(2), mode="count")
    result, status = q(tree, geometry,
                       np.array([[0, 0], [.25, .25]], np.float32),
                       np.array([[0, 0], [.75, .75]], np.float32))
    np.testing.assert_array_equal(result, [1, 0])
    np.testing.assert_array_equal(status, 1)


def test_segment_exact_contacts_degenerate_and_scales():
    for scale in (1e-5, 1.0, 1e5):
        primitive = Segments(2)
        a = np.array([[-1, 1], [0, 0], [3, 3]], np.float32) * scale
        b = np.array([[1, -1], [0, 0], [3, 3]], np.float32) * scale
        tree, geometry = build_tree(primitive, primitive.data(a, b))
        q = make_region(primitive, segment_region_policy(2), mode="count")
        lo = np.array([[0, 0], [.5, .5], [3, 3]], np.float32) * scale
        hi = np.array([[0, 0], [.6, .6], [3, 3]], np.float32) * scale
        result, status = q(tree, geometry, lo, hi)
        np.testing.assert_array_equal(result, [2, 0, 1])
        np.testing.assert_array_equal(status, 1)


def test_segment3_closed_faces_and_bbox_false_positive():
    primitive = Segments(3)
    a = np.array([[-1, 1, 1], [0, 0, 0]], np.float32)
    b = np.array([[1, -1, -1], [0, 0, 0]], np.float32)
    tree, geometry = build_tree(primitive, primitive.data(a, b))
    q = make_region(primitive, segment_region_policy(3), mode="count")
    lo = np.array([[0, 0, 0], [.5, .5, .5], [0, 0, 0]], np.float32)
    hi = np.array([[0, 0, 0], [.6, .6, .6], [0, 0, 0]], np.float32)
    result, status = q(tree, geometry, lo, hi)
    np.testing.assert_array_equal(result, [2, 0, 2])
    np.testing.assert_array_equal(status, 1)


def test_triangle_exact_not_just_bounding_box():
    primitive = Triangles()
    vertices = np.array([[0, 0, 0], [2, 0, 0], [0, 2, 0]], np.float32)
    tree, geometry = build_tree(primitive, primitive.indexed(vertices, [[0, 1, 2]]))
    q = make_region(primitive, triangle_region_policy())
    lo = np.array([[1.5, 1.5, -.1], [.2, .2, 0], [2, 0, 0],
                   [0, 0, .1], [0, 0, 0]], np.float32)
    hi = np.array([[1.6, 1.6, .1], [.3, .3, 0], [2, 0, 0],
                   [2, 2, .2], [0, 0, 0]], np.float32)
    result, status = q(tree, geometry, lo, hi)
    np.testing.assert_array_equal(result, [False, True, True, False, True])
    np.testing.assert_array_equal(status, 1)
    # Degenerate triangle is a segment; a box touching that segment counts.
    degenerate = np.array([[0, 0, 0], [2, 0, 0], [1, 0, 0]], np.float32)
    t, g = build_tree(primitive, primitive.indexed(degenerate, [[0, 1, 2]]))
    assert bool(q(t, g, np.array([1, 0, 0], np.float32), np.array([1, 0, 0], np.float32))[0])


@pytest.mark.parametrize("mode", ["any", "count"])
def test_triangle_original_vertices_survive_edge_rounding_and_leaf_packing(mode):
    from surfaces.explicit.accelerators.protocol import Config

    primitive = Triangles()
    offsets = np.random.default_rng(42).permutation(24).astype(np.float32) * 10
    facets = np.tile(np.array([[1e8, 0, 0], [1, 1, 0], [1, 0, 1]], np.float32), (24, 1, 1))
    facets[:, :, 1] += offsets[:, None]
    indices = np.arange(facets.size // 3).reshape(-1, 3)
    tree, geometry = build_tree(primitive, primitive.indexed(facets.reshape(-1, 3), indices),
                                Config(leaf_size_threshold=1))
    assert not np.array_equal(tree.prim_ids, np.arange(len(facets)))
    # Float32 a+e1 loses x=1. Test both the actual vertex and the false x=0
    # reconstruction, for every triangle after a nontrivial leaf permutation.
    vertices = facets[:, 1]
    false_vertices = vertices.copy()
    false_vertices[:, 0] = 0
    boxes = np.concatenate((vertices, false_vertices))
    query = make_region(primitive, triangle_region_policy(), mode=mode)
    result, status = jax.jit(query)(tree, geometry, boxes, boxes)
    np.testing.assert_array_equal(result, np.r_[np.ones(24), np.zeros(24)])
    np.testing.assert_array_equal(status, 1)


def test_external_region_callback_compiles_with_geometry_and_slot():
    from tests.typing.warp_api.real_extension import ExternalArrays, ExternalPoints
    from tests.typing.warp_api.region_valid import external

    primitive = ExternalPoints()
    tree, geometry = build_tree(primitive, ExternalArrays(jnp.array([[0., 0., 0.], [1., 1., 1.]])))
    result, status = jax.jit(external)(tree, geometry, [[0., 0., 0.]], [[0., 0., 0.]])
    np.testing.assert_array_equal(result, [1])
    np.testing.assert_array_equal(status, 1)


@pytest.mark.parametrize("scale,shift", [(1e-30, 0.0), (1e-5, 0.0), (1.0, 0.0), (1e5, 1e6), (1e30, 0.0)])
def test_triangle_scaled_boxes(scale, shift):
    primitive = Triangles()
    vertices = np.array([[0, 0, 0], [2, 0, 0], [0, 2, 0]], np.float32) * scale + shift
    tree, geometry = build_tree(primitive, primitive.indexed(vertices, [[0, 1, 2]]))
    q = make_region(primitive, triangle_region_policy())
    lo = np.array([[.2, .2, -.1], [1.5, 1.5, -.1]], np.float32) * scale + shift
    hi = np.array([[.3, .3, .1], [1.6, 1.6, .1]], np.float32) * scale + shift
    result, status = q(tree, geometry, lo, hi)
    np.testing.assert_array_equal(result, [True, False])
    np.testing.assert_array_equal(status, 1)


def test_closed_surface_inside_outside_intersects():
    primitive = Triangles()
    mesh = trimesh.creation.box(extents=(2, 2, 2))
    tree, geometry, boundary = build_boundary(primitive,
        primitive.indexed(mesh.vertices, mesh.faces), silhouettes=False)
    classify_policy = primitive.boundary_adapter.classification(primitive)
    q = make_region_classify(primitive, triangle_region_policy(), classify_policy)
    lo = np.array([[0, 0, 0], [2, 2, 2], [.9, 0, 0], [1, 0, 0]], np.float32)
    hi = np.array([[.1, .1, .1], [2.1, 2.1, 2.1], [1.1, .1, .1], [1, 0, 0]], np.float32)
    result, status = jax.jit(q)(tree, geometry, boundary, lo, hi)
    np.testing.assert_array_equal(result, [1, -1, 0, 0])
    np.testing.assert_array_equal(status, 1)
    # Intersecting boxes must also reject metadata for an open surface.
    invalid_core = boundary.core._replace(closed=jnp.array([0], jnp.int32))
    invalid_result, invalid_status = q(tree, geometry, invalid_core, lo, hi)
    np.testing.assert_array_equal(invalid_status, -1)
    np.testing.assert_array_equal(invalid_result, 0)
    assert make_region_classify(primitive, triangle_region_policy(), classify_policy,
                               metadata_host_type=type(boundary.core)) is q


@wp.func
def _custom_point(data: PointsData3, slot: int, lo: wp.vec3, hi: wp.vec3) -> bool:
    point = data.a[slot]
    return (lo[0] <= point[0] <= hi[0] and lo[1] <= point[1] <= hi[1] and
            lo[2] <= point[2] <= hi[2] and point[1] == 0.0)


def test_custom_callback_fused_into_one_device_kernel():
    primitive = Points()
    tree, geometry = build_tree(primitive, primitive.data(np.array([[0, 0, 0], [0, 1, 0]], np.float32)))
    query = make_region(primitive, RegionIntersectionPolicy(_custom_point), mode="count")
    fn = query.device
    Geometry = primitive.data_type

    @wp.kernel(module="unique", enable_backward=False)
    def kernel(t: DeviceTree, g: Geometry, result: wp.array(dtype=wp.int32)):
        count, status = wp.static(fn)(wp.vec3(-1, -1, -1), wp.vec3(1, 1, 1), t, g)
        result[0] = count if status == 1 else -1

    device = wp.device_from_jax(jax.devices()[0])
    (output,) = run_kernel(
        kernel, 1, [device_payload(DeviceTree, tree, device),
                    device_payload(Geometry, geometry, device)],
        [((1,), wp.int32)], queries=(query,), device=device,
    )
    np.testing.assert_array_equal(output, [1])
