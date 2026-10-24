"""Convert an implicit-side `SpatialBuildResult` into an explicit-side
mesh-accelerator payload (flavor-specific).

Inputs:
  result:        SpatialBuildResult from `*Tree.build_with_topology`.
  primitives:    Primitives backend (only its AABBs are needed for
                 the per-leaf primitive list when extending with mesh
                 data; raw topology already lives in `result`).
  tri_to_leaf:   (T,) int — per-triangle index into the leaf id array,
                 produced by `MeshExtractor.extract_with_provenance`.

Outputs:
  flavor-specific payload (`KdPayload` / `OctreePayload`) with primitive
  IDs grouped per leaf. Sorting `tri_to_leaf` and using a segment-sum
  produces a CSR-style layout (`prim_ids`, `node_leaf_offset`,
  `node_leaf_count`).

Implementations are stubs; concrete CSR construction lands together
with the first traversal-only backend that consumes the payload.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import jax.numpy as jnp

from surfaces.explicit.accelerators.kd.payload import KdPayload
from surfaces.explicit.accelerators.octree.payload import OctreePayload
from surfaces.explicit.primitives import Primitives

if TYPE_CHECKING:
    from surfaces.spatial_tree import SpatialBuildResult


def kd_payload_from_spatial_tree(result: "SpatialBuildResult",
                                 primitives: Primitives,
                                 tri_to_leaf: jnp.ndarray) -> KdPayload:
    """Adapt a kd-flavored `SpatialBuildResult` into a `KdPayload`.

    Asserts `result.tree.flavor == "kd"`. Carries split-axis / split-pos
    from the implicit-side topology (which records them at build time)
    into the payload's `node_split_axis` / `node_split_pos` fields.
    """
    raise NotImplementedError(
        "kd_payload_from_spatial_tree lands with first kd traversal-only backend"
    )


def octree_payload_from_spatial_tree(result: "SpatialBuildResult",
                                     primitives: Primitives,
                                     tri_to_leaf: jnp.ndarray
                                     ) -> OctreePayload:
    """Adapt an octree-flavored `SpatialBuildResult` into an
    `OctreePayload`. Topology arrays in `SpatialTree` already match
    octree node layout (children fan-out 8); a 1:1 copy plus CSR
    primitive grouping suffices.
    """
    raise NotImplementedError(
        "octree_payload_from_spatial_tree lands with first octree traversal-only backend"
    )
