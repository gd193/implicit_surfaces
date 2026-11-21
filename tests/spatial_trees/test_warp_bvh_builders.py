"""SAH and Morton construction feed the same binary Warp traversal."""
import numpy as np
import pytest

pytest.importorskip("warp")

from surfaces.explicit.accelerators.bvh.warp import Points, build_tree  # noqa: E402
from surfaces.explicit.accelerators.protocol import Config  # noqa: E402


def test_lbvh_duplicate_morton_codes_leaf_sizes_and_depth():
    primitive = Points(2)
    # Repeated codes span more than one leaf and share the tree with other codes.
    vertices = np.repeat([[1., 2.], [-3., 2.], [10., 2.]], [11, 5, 7], axis=0)
    source = primitive.data(vertices)
    tree, _ = build_tree(primitive, source, Config(leaf_size_threshold=2), builder="lbvh")
    np.testing.assert_array_equal(np.sort(tree.prim_ids), np.arange(len(vertices)))
    assert int(tree.node_leaf_count.max()) == 2
    np.testing.assert_array_equal(tree.node_lower[0], [-3., 2., 0.])
    np.testing.assert_array_equal(tree.node_upper[0], [10., 2., 0.])
    with pytest.raises(ValueError, match="stack depth"):
        build_tree(primitive, source, Config(max_stack_depth=1, leaf_size_threshold=2), builder="lbvh")


def test_lbvh_retains_float32_morton_quantization_with_overflow_fallback():
    primitive = Points(3)
    config = Config(leaf_size_threshold=1)
    source = primitive.data([[1e-29, 0., 0.], [-1e-29, 0., 0.], [0., 0., 0.]])
    tree, _ = build_tree(primitive, source, config, builder="lbvh")
    # Historically the minimum normalization span put these in the same bucket.
    np.testing.assert_array_equal(tree.prim_ids, [0, 1, 2])
    extreme = np.finfo(np.float32).max
    source = primitive.data([[extreme, 0., 0.], [-extreme, 0., 0.], [0., 0., 0.]])
    tree, _ = build_tree(primitive, source, config, builder="lbvh")
    np.testing.assert_array_equal(tree.prim_ids, [1, 2, 0])
    np.testing.assert_array_equal(tree.node_lower[0], [-extreme, 0., 0.])
    np.testing.assert_array_equal(tree.node_upper[0], [extreme, 0., 0.])


def test_lbvh_empty_and_singleton():
    primitive = Points(2)
    for count in (0, 1):
        tree, _ = build_tree(primitive, primitive.data(np.zeros((count, 2))), builder="lbvh")
        assert tree.node_lower.shape == (count, 3)
        assert tree.node_ray_lower.shape == (count, 4)
        np.testing.assert_array_equal(tree.prim_ids, np.arange(count))
        np.testing.assert_array_equal(tree.node_leaf_count, np.ones(count))
    with pytest.raises(ValueError, match="builder"):
        build_tree(primitive, primitive.data([[0., 0.]]), builder="unknown")
