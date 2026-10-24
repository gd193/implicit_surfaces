"""BVH payload — flat SoA arrays describing a binary AABB tree.

All fields jnp arrays (or NumPy-compatible) so the struct is a clean
pytree. JAX-native traversal traces it through `jit`; FFI/Warp wrappers
pass the same arrays as zero-copy buffers. No Python-side state.

Layout (N_nodes total, N_leaves <= N_nodes, N_prim_refs >= n_primitives):
  node_lower:        (N_nodes, 3)   per-node AABB lower bound
  node_upper:        (N_nodes, 3)   per-node AABB upper bound
  node_left:         (N_nodes,)     left child index, -1 if leaf
  node_right:        (N_nodes,)     right child index, -1 if leaf
  node_leaf_offset:  (N_nodes,)     start in `prim_ids`, valid if leaf
  node_leaf_count:   (N_nodes,)     count in `prim_ids`, 0 if internal
  prim_ids:          (N_prim_refs,) primitive indices in leaf order
  root:              ()             root node index (typically 0)

Concrete backends that need extra fields (e.g. SAH adds split-axis
for ray-traversal ordering) declare a fresh NamedTuple in their own
backend file repeating all `BVHPayload` fields plus the additions —
`typing.NamedTuple` does not support adding fields through inheritance,
so a real subclass is not an option. Set the backend's
`payload_type` to that fresh NamedTuple. Keep the shared `BVHPayload`
as the lowest-common-denominator layout used by accelerator-shell type
hints.
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
