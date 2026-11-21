"""CPU regression coverage for indexed triangle metadata."""
import numpy as np
import pytest

from sengi.bvh.backends._warp_host_metadata import (
    reduce_nodes, triangle_boundary_features, triangle_mesh_closed, triangle_metadata,
)


def test_triangle_metadata_uses_indexed_vertices():
    facets = np.array([
        [[0, 0, 0], [2, 0, 0], [0, 3, 0]],
        [[0, 0, 0], [1, 1, 1], [2, 2, 2]],
        [[0, 0, 0], [1e-11, 0, 0], [0, 1e-11, 0]],
        [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
        [[-2**25, -2**25, 0], [1, 0, 0], [0, 1, 0]],
    ], np.float32)
    expected_lengths = [6, 0, float(facets[2, 1, 0])**2, 0, 2**26 + 1]
    expected_normals = [[0, 0, 1], [0, 0, 0], [0, 0, 1], [0, 0, 0], [0, 0, 1]]
    for shift in range(3):
        normal, area, lengths = triangle_metadata(np.roll(facets, shift, axis=1))
        np.testing.assert_array_equal(normal, expected_normals)
        np.testing.assert_allclose(lengths, expected_lengths, atol=0, rtol=1e-15)
        np.testing.assert_allclose(area, np.array(expected_lengths) / 2, atol=0, rtol=1e-15)


def test_empty_metadata_and_closure():
    empty = np.empty((0, 3), np.float32)
    normal, area, lengths = triangle_metadata(np.empty((0, 3, 3), np.float32))
    assert normal.shape == (0, 3)
    assert area.shape == lengths.shape == (0,)
    assert triangle_mesh_closed(empty, np.empty((0, 3), np.int32))


def test_closure_open_closed_degenerate_and_tiny():
    vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], np.float32)
    faces = np.array([[0, 2, 1], [0, 1, 3], [1, 2, 3], [2, 0, 3]], np.int32)
    assert triangle_mesh_closed(vertices, faces)
    assert triangle_mesh_closed(vertices * np.float32(1e-20), faces)
    mixed = np.array([[-2**25, -2**25, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], np.float32)
    assert triangle_mesh_closed(mixed, faces)
    assert not triangle_mesh_closed(vertices, faces[:-1])
    vertices[3] = vertices[0]
    assert not triangle_mesh_closed(vertices, faces)
    np.testing.assert_array_equal(triangle_boundary_features(faces[:1]), [[0, 2], [1, 2], [0, 1]])


def test_postorder_reductions_preserve_leaf_order_and_float32_sums():
    # Arbitrary node numbering; leaf slots have an explicit source permutation.
    left = np.array([3, -1, -1, 4, -1])
    right = np.array([1, -1, -1, 2, -1])
    offset = np.array([0, 3, 2, 0, 0])
    count = np.array([0, 1, 1, 0, 2])
    order = [4, 2, 3, 1, 0]
    source = np.array([1., 1e8, -1e8, 3.], np.float32)
    packed = source[[1, 3, 2, 0]]
    nodes = reduce_nodes(packed, left, right, offset, count, order,
                         lambda a: a.sum(axis=0, dtype=np.float32))
    np.testing.assert_array_equal(nodes, [1., 1., -1e8, 0., 1e8])
    intervals = np.stack((packed, packed + np.float32(8)), axis=1)
    bounds = reduce_nodes(intervals, left, right, offset, count, order,
                          lambda a: np.array([a[:, 0].min(), a[:, 1].max()], np.float32))
    np.testing.assert_array_equal(bounds[0], [-1e8, np.float32(1e8) + np.float32(8)])
    np.testing.assert_array_equal(bounds[4], [3., np.float32(1e8) + np.float32(8)])
    assert bounds.dtype == np.float32


def test_empty_node_reduction_does_not_call_combine():
    empty = np.empty(0, np.int32)
    result = reduce_nodes(np.empty((0, 2), np.float32), empty, empty, empty, empty, [],
                          lambda _: pytest.fail("empty tree must not reduce"))
    assert result.shape == (0, 2)
    assert result.dtype == np.float32
