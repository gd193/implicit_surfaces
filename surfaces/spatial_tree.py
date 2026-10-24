"""Spatial partition recorded during implicit subdivision.

A `SpatialTree` is the full node graph produced by an `ImplicitTree`
builder when invoked via `build_with_topology`. It captures every node
visited during subdivision (root + internals + leaves) so that downstream
consumers — chiefly mesh accelerators (`surfaces.explicit.accelerators`)
— can reuse the partition without rebuilding.

Distinct from `meshExtractor.base.Leaves`, which is the surviving
frontier only. `SpatialBuildResult` bundles both plus a frontier→node-id
map so existing leaf-driven extractors keep working unchanged.

Recording is done device-side: the bucketed builder allocates fixed-size
tree arrays up front and writes node records via `.at[...].set` keyed by
a host-side counter that rides the existing per-iter sync. Trees are
returned oversized (no trim); consumers slice with `n_nodes` if needed.
"""
from __future__ import annotations

from typing import NamedTuple, TYPE_CHECKING

import jax.numpy as jnp

if TYPE_CHECKING:
    from meshExtractor.base import Leaves


class SpatialTree(NamedTuple):
    """Static spatial partition. Topology only; payload (triangles, leaf
    quantities, ...) lives in companion arrays maintained by consumers.

    Arrays are sized to an auto-estimated upper bound; only the first
    `n_nodes` entries are valid. `flavor` and `K` are static metadata.
    """
    aabb_lo: jnp.ndarray              # (N_max, 3) float32
    aabb_hi: jnp.ndarray              # (N_max, 3) float32
    parent: jnp.ndarray               # (N_max,)   int32, -1 on root / unused
    children: jnp.ndarray             # (N_max, K) int32, -1 in unused slots
    is_leaf: jnp.ndarray              # (N_max,)   bool
    leaf_classification: jnp.ndarray  # (N_max,)   int8 in {-1, 0, +1}; 0 on internal / UNKNOWN leaves
    n_nodes: int                      # number of populated entries (host scalar)
    flavor: str                       # "octree" | "kd" (BVH not produced by the implicit side)
    K: int                            # max children per node


class SpatialBuildResult(NamedTuple):
    """Full output of `ImplicitTree.build_with_topology`.

    `leaves` matches the legacy `build()` return value verbatim so
    existing extractor code keeps working. `leaf_node_ids` maps each
    surviving frontier entry to its node id in `tree`, enabling mesh
    accelerators to attach triangle payloads to the right nodes.
    """
    tree: SpatialTree
    leaves: "Leaves"  # forward ref to meshExtractor.base.Leaves
    leaf_node_ids: jnp.ndarray  # (N_leaves,) int32


def estimate_max_nodes(max_depth: int, batch_process_size: int, split_factor: int) -> int:
    """Upper bound for preallocated tree arrays.

    Geometric series over depths, capped by `batch_process_size` per level
    after compaction kicks in. Conservative — over-allocates rather than
    truncates. Builders may accept a user override; default uses this.
    """
    # 1 root + per-iter contributions. Each iter adds at most
    # `batch_process_size * split_factor` new nodes after the first
    # bucket-bound iter, summed over `max_depth + 1` iters.
    return 1 + (max_depth + 1) * batch_process_size * split_factor
