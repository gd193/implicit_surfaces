"""Kd-tree payload — flat SoA arrays describing a split-plane tree.

Layout (N_nodes, N_prim_refs):
  node_split_axis:   (N_nodes,)     0/1/2 for x/y/z, -1 if leaf
  node_split_pos:    (N_nodes,)     plane position along axis
  node_left:         (N_nodes,)     left child index, -1 if leaf
  node_right:        (N_nodes,)     right child index, -1 if leaf
  node_lower:        (N_nodes, 3)   per-node AABB lower
  node_upper:        (N_nodes, 3)   per-node AABB upper
  node_leaf_offset:  (N_nodes,)     start in `prim_ids`, valid if leaf
  node_leaf_count:   (N_nodes,)     count in `prim_ids`, 0 if internal
  prim_ids:          (N_prim_refs,) primitive indices in leaf order
  root:              ()             root node index

Per-node AABBs kept to enable both surface-area-heuristic traversal
ordering and bridging from `SpatialBuildResult` (whose nodes already
carry AABBs).
"""
from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp


class KdPayload(NamedTuple):
    node_split_axis: jnp.ndarray
    node_split_pos: jnp.ndarray
    node_left: jnp.ndarray
    node_right: jnp.ndarray
    node_lower: jnp.ndarray
    node_upper: jnp.ndarray
    node_leaf_offset: jnp.ndarray
    node_leaf_count: jnp.ndarray
    prim_ids: jnp.ndarray
    root: jnp.ndarray
