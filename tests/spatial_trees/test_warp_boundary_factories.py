"""Indexed boundary features and metric-independent visibility."""
from dataclasses import replace
from typing import Any

import numpy as np
import pytest

wp = pytest.importorskip("warp")

from scripts.bench.spatial_trees._warp_cases import scene  # noqa: E402

from surfaces.explicit.accelerators.bvh.backends._warp_boundary import (  # noqa: E402
    SilhouetteMeta, build_boundary, resolve_boundary_adapter,
)
from surfaces.explicit.accelerators.bvh.backends._warp_primitives import Segments, Triangles  # noqa: E402
from surfaces.explicit.accelerators.bvh.backends._warp_silhouette import make_silhouette  # noqa: E402
from surfaces.explicit.accelerators.bvh.backends._warp_tree import DeviceTree  # noqa: E402
from surfaces.explicit.accelerators.protocol import Config  # noqa: E402


@wp.func
def _invisible(viewpoint: wp.vec3, tree: DeviceTree, meta: SilhouetteMeta, node: int):
    return False


@wp.func
def _invisible_feature(query: wp.vec2, item: Any):
    return False


@pytest.mark.parametrize("reject", ["node", "feature"])
def test_boundary_custom_visibility_and_optional_metadata(reject):
    primitive = Segments(2)
    tree, geometry, boundary, _ = scene("segments2")
    options = {"visibility": _invisible} if reject == "node" else {"visibility": None, "predicate": _invisible_feature}
    adapter = replace(resolve_boundary_adapter(primitive), **options)
    result = make_silhouette(primitive, adapter=adapter)(tree, geometry, boundary, np.array([3, 0], np.float32))
    assert int(result[2]) == -1 and int(result[3]) == 0
    with pytest.raises(ValueError, match="metadata"):
        make_silhouette(primitive)(tree, geometry, boundary._replace(silhouette=None), np.zeros(2))


@pytest.mark.parametrize("dimension", [2, 3])
def test_boundary_feature_enclosure_and_leaf_order(dimension):
    primitive = Segments(2) if dimension == 2 else Triangles()
    vertices = np.array([[1e8, 0, 0], [-1, 0, 0], [1e8, 1, 0], [1e8, 0, 1]], np.float32)
    indices = np.array([[0, 1], [1, 2]], np.int32) if dimension == 2 else np.array([[0, 1, 2], [0, 3, 1]], np.int32)
    data = primitive.indexed(vertices[:, :2], indices) if dimension == 2 else primitive.data(vertices, indices)
    tree, geometry, boundary = build_boundary(primitive, data, Config(leaf_size_threshold=1))
    assert boundary.core.indices is geometry.indices
    if dimension == 3:
        assert boundary.core.vertices is geometry.vertices
    np.testing.assert_array_equal(boundary.core.indices, indices[np.asarray(tree.prim_ids)])
    meta = boundary.silhouette
    for node in np.flatnonzero(np.asarray(tree.node_left) < 0):
        start, count = int(meta.node_feature_offset[node]), int(meta.node_feature_count[node])
        refs = np.asarray(meta.feature_refs[start:start+count])
        points = np.concatenate([np.asarray(meta.feature_a)[refs], np.asarray(meta.feature_b)[refs]])
        assert np.all(points >= np.asarray(tree.node_lower[node]))
        assert np.all(points <= np.asarray(tree.node_upper[node]))
    collapsed = (primitive.indexed([[0., 0], [0., 0]], [[0, 1]]) if dimension == 2 else
                 primitive.data([[1e8, 1e8, 0], [1, 0, 0], [0, 1, 0]], [[0, 1, 2]]))
    with np.errstate(divide="raise", invalid="raise"):
        with pytest.raises(ValueError, match="non-degenerate"):
            build_boundary(primitive, collapsed)
        _, _, invalid = build_boundary(primitive, collapsed, silhouettes=False)
    np.testing.assert_array_equal(invalid.core.primitive_measure, 0.)
    np.testing.assert_array_equal(invalid.core.closed, 0)
    assert np.isfinite(invalid.core.primitive_normal).all()
    if dimension == 2:
        _, _, tiny = build_boundary(primitive, primitive.indexed([[0., 0], [1e-30, 0]], [[0, 1]]))
        np.testing.assert_allclose(tiny.core.primitive_measure, [1e-30], rtol=1e-6, atol=0)
        np.testing.assert_array_equal(tiny.core.primitive_normal, [[0., -1., 0.]])
