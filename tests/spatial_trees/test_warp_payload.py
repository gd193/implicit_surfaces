"""Reject unsafe tree payloads before GPU execution."""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from sengi.bvh.backends._warp_layout import _validate_tree


def _tree():
    return SimpleNamespace(
        root=np.array(0, dtype=np.int32), prim_ids=np.array([0, 1], dtype=np.int32),
        node_lower=np.array([[0, 0, 0], [0, 0, 0], [1, 0, 0]], dtype=np.float32),
        node_upper=np.array([[2, 1, 0], [1, 1, 0], [2, 1, 0]], dtype=np.float32),
        node_left=np.array([1, -1, -1], dtype=np.int64),
        node_right=np.array([2, -1, -1], dtype=np.int32),
        node_leaf_offset=np.array([0, 0, 1], dtype=np.int32),
        node_leaf_count=np.array([0, 1, 1], dtype=np.int32),
    )


def test_payload_rejects_cycle_before_postorder_walk():
    tree = _tree()
    tree.node_right[0] = 0
    with pytest.raises(ValueError, match="cycle or shared child"):
        _validate_tree(tree, len(tree.prim_ids))


@pytest.mark.parametrize("field,position,value,error", [
    ("root", (), 1, "root must be node 0"),
    ("prim_ids", 1, 0, "permutation"),
    ("node_left", 0, 1 << 31, "exceeds int32 indexing"),
    ("node_leaf_offset", 2, 0, "overlap"),
    ("node_lower", (1, 1), .5, "leaf bounds must enclose"),
    ("node_upper", (0, 1), .5, "internal bounds must enclose"),
])
def test_build_rejects_invalid_binary_layout(monkeypatch, field, position, value, error):
    pytest.importorskip("warp")
    from sengi.bvh.backends import lbvh_jax
    from sengi.bvh.warp import Points, build_tree

    tree = _tree()
    getattr(tree, field)[position] = value
    monkeypatch.setattr(lbvh_jax, "_build_payload", lambda *args: tree)
    primitive = Points(2)
    with pytest.raises(ValueError, match=error):
        build_tree(primitive, primitive.data([[0., 0.], [2., 0.]]), builder="lbvh")
