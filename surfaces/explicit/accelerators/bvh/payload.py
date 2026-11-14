"""BVH payload — flat SoA arrays describing a binary AABB tree.

The host builders return this JAX-array pytree. Warp construction packs
it into its query payload without changing the host tree's ordering.

Layout (N_nodes total, N_leaves <= N_nodes, N_prim_refs >= n_primitives):
  node_lower:        (N_nodes, 3)   per-node AABB lower bound
  node_upper:        (N_nodes, 3)   per-node AABB upper bound
  node_left:         (N_nodes,)     left child index, -1 if leaf
  node_right:        (N_nodes,)     right child index, -1 if leaf
  node_leaf_offset:  (N_nodes,)     start in `prim_ids`, valid if leaf
  node_leaf_count:   (N_nodes,)     count in `prim_ids`, 0 if internal
  prim_ids:          (N_prim_refs,) primitive indices in leaf order
  root:              ()             root node index (typically 0)

SNCH construction uses a separate NamedTuple with cone and silhouette
fields. Query accelerators use `WarpBVHPayload`, not this build payload.
"""
from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp


class BVHPayload(NamedTuple):
    node_lower: jnp.ndarray
    node_upper: jnp.ndarray
    node_left: jnp.ndarray
    node_right: jnp.ndarray
    node_leaf_offset: jnp.ndarray
    node_leaf_count: jnp.ndarray
    prim_ids: jnp.ndarray
    root: jnp.ndarray
