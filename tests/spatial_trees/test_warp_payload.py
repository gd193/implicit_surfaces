"""Reject unsafe tree payloads before GPU execution."""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from surfaces.explicit.accelerators.bvh.backends._warp_layout import make_warp_payload


def _tree():
    return SimpleNamespace(
        root=np.int32(0), prim_ids=np.array([0, 1], dtype=np.int32),
        node_lower=np.array([[0, 0, 0], [0, 0, 0], [1, 0, 0]], dtype=np.float32),
        node_upper=np.array([[2, 1, 0], [1, 1, 0], [2, 1, 0]], dtype=np.float32),
        node_left=np.array([1, -1, -1], dtype=np.int32),
        node_right=np.array([2, -1, -1], dtype=np.int32),
        node_leaf_offset=np.array([0, 0, 1], dtype=np.int32),
        node_leaf_count=np.array([0, 1, 1], dtype=np.int32),
    )


def _geometry():
    vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0],
                         [1, 0, 0], [2, 0, 0], [1, 1, 0]], dtype=np.float32)
    faces = np.arange(6, dtype=np.int32).reshape(2, 3)
    return vertices, faces


def test_payload_rejects_cycle_before_postorder_walk():
    tree = _tree()
    tree.node_right[0] = 0
    with pytest.raises(ValueError, match="cycle or shared child"):
        make_warp_payload(tree, *_geometry())


def test_payload_rejects_index_that_would_overflow_packed_child():
    tree = _tree()
    tree.node_left = np.array([1 << 31, -1, -1], dtype=np.int64)
    with pytest.raises(ValueError, match="exceeds int32 indexing"):
        make_warp_payload(tree, *_geometry())
