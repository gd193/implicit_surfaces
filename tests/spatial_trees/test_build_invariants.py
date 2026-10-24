"""Layer-4 (structural): build outputs are well-formed.

Per-backend assertions on the payload arrays produced by `Backend.build`:
  * SoA arrays have correct shapes/dtypes.
  * Every primitive index appears in exactly one leaf.
  * Per-node AABBs satisfy `lower <= upper`.
  * Internal-node AABBs cover their child AABBs.
  * Leaf vs internal flags are consistent (`leaf_count > 0 ↔ leaf`).

Decoupled from query correctness: structural invariants must hold even
when primitive math (closest_point_on_primitive, ...) is unimplemented.
"""
from __future__ import annotations

import numpy as np
import pytest

from ._harness import parametrize_backends, build_accelerator


# =====================================================================
# BVH structural invariants.
# =====================================================================


def _check_bvh_payload(payload, n_primitives: int):
    nl = np.asarray(payload.node_lower)
    nu = np.asarray(payload.node_upper)
    left = np.asarray(payload.node_left)
    right = np.asarray(payload.node_right)
    leaf_off = np.asarray(payload.node_leaf_offset)
    leaf_cnt = np.asarray(payload.node_leaf_count)
    prim_ids = np.asarray(payload.prim_ids)

    assert nl.shape == nu.shape, "node_lower / node_upper shape mismatch"
    assert nl.shape[1] == 3, "node AABB must be 3D"
    n_nodes = nl.shape[0]
    assert left.shape == (n_nodes,)
    assert right.shape == (n_nodes,)
    assert leaf_off.shape == (n_nodes,)
    assert leaf_cnt.shape == (n_nodes,)

    # AABBs lo <= hi on populated nodes (use leaf_count > 0 OR has children
    # as a populated proxy; root may have no children if n_primitives == 1).
    populated = (leaf_cnt > 0) | (left >= 0) | (right >= 0)
    bad_lo_hi = (nl > nu) & populated[:, None]
    assert not bad_lo_hi.any(), (
        f"{int(bad_lo_hi.sum())} populated nodes have lower > upper")

    # leaf_count > 0 ↔ leaf (no children).
    is_leaf = leaf_cnt > 0
    has_children = (left >= 0) | (right >= 0)
    contradictions = is_leaf & has_children
    assert not contradictions.any(), (
        f"{int(contradictions.sum())} nodes are both leaf and internal")

    # Internal-node AABB ⊇ child AABBs (refit invariant).
    for i in range(n_nodes):
        if not has_children[i] or is_leaf[i]:
            continue
        for c in (int(left[i]), int(right[i])):
            if c < 0:
                continue
            assert (nl[i] <= nl[c] + 1e-5).all(), (
                f"node {i} lower {nl[i]} doesn't cover child {c} lower {nl[c]}")
            assert (nu[i] + 1e-5 >= nu[c]).all(), (
                f"node {i} upper {nu[i]} doesn't cover child {c} upper {nu[c]}")

    # prim_ids covers every primitive exactly once.
    leaf_starts = leaf_off[is_leaf]
    leaf_counts = leaf_cnt[is_leaf]
    seen = []
    for s, c in zip(leaf_starts.tolist(), leaf_counts.tolist()):
        seen.extend(prim_ids[s:s + c].tolist())
    assert sorted(seen) == list(range(n_primitives)), (
        f"prim_ids coverage: got {sorted(seen)[:5]}…, "
        f"expected 0..{n_primitives - 1}")


@parametrize_backends("bvh")
def test_bvh_build_invariants(tri_fixture, backend_name):
    accel = build_accelerator("bvh", backend_name, tri_fixture.primitives)
    _check_bvh_payload(accel.payload, int(tri_fixture.primitives.n_primitives))


# =====================================================================
# Kd structural invariants.
# =====================================================================


def _check_kd_payload(payload, n_primitives: int):
    nl = np.asarray(payload.node_lower)
    nu = np.asarray(payload.node_upper)
    axis = np.asarray(payload.node_split_axis)
    left = np.asarray(payload.node_left)
    right = np.asarray(payload.node_right)
    leaf_cnt = np.asarray(payload.node_leaf_count)

    assert nl.shape == nu.shape
    assert nl.shape[1] == 3
    n_nodes = nl.shape[0]
    assert axis.shape == (n_nodes,)
    is_leaf = leaf_cnt > 0
    populated = is_leaf | (left >= 0) | (right >= 0)
    assert not ((nl > nu) & populated[:, None]).any()
    # Split-axis is -1 on leaves, in {0,1,2} on internal nodes.
    bad = ((axis < -1) | (axis > 2))
    assert not (bad & populated).any(), "node_split_axis out of bounds"


@parametrize_backends("kd")
def test_kd_build_invariants(tri_fixture, backend_name):
    accel = build_accelerator("kd", backend_name, tri_fixture.primitives)
    _check_kd_payload(accel.payload, int(tri_fixture.primitives.n_primitives))


# =====================================================================
# Octree structural invariants.
# =====================================================================


def _check_octree_payload(payload, n_primitives: int):
    nl = np.asarray(payload.node_lower)
    nu = np.asarray(payload.node_upper)
    children = np.asarray(payload.node_children)
    is_leaf = np.asarray(payload.node_is_leaf)

    assert nl.shape == nu.shape
    assert children.shape[1] == 8
    populated = is_leaf | (children >= 0).any(axis=1)
    assert not ((nl > nu) & populated[:, None]).any()


@parametrize_backends("octree")
def test_octree_build_invariants(tri_fixture, backend_name):
    accel = build_accelerator("octree", backend_name, tri_fixture.primitives)
    _check_octree_payload(accel.payload,
                          int(tri_fixture.primitives.n_primitives))
