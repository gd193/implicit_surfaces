"""Octree payload — flat SoA arrays describing an 8-way subdivision tree.

Layout (N_nodes, N_prim_refs):
  node_lower:        (N_nodes, 3)   per-node AABB lower
  node_upper:        (N_nodes, 3)   per-node AABB upper
  node_children:     (N_nodes, 8)   child indices, -1 for absent
  node_is_leaf:      (N_nodes,)     bool
  node_leaf_offset:  (N_nodes,)     start in `prim_ids`, valid if leaf
  node_leaf_count:   (N_nodes,)     count in `prim_ids`, 0 if internal
  prim_ids:          (N_prim_refs,) primitive indices in leaf order
  root:              ()             root node index

Per-node AABBs explicitly stored rather than recomputed from level +
root because non-uniform subdivision (skipping empty octants) breaks
the regular-grid identity. Matches `SpatialTree`'s representation
exactly, so the implicit-tree-reuse bridge is a 1:1 copy.
"""
from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp


class OctreePayload(NamedTuple):
    node_lower: jnp.ndarray
    node_upper: jnp.ndarray
    node_children: jnp.ndarray
    node_is_leaf: jnp.ndarray
    node_leaf_offset: jnp.ndarray
    node_leaf_count: jnp.ndarray
    prim_ids: jnp.ndarray
    root: jnp.ndarray
