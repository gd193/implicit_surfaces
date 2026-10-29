"""Reserved adapters for implicit subdivision topology (not implemented).

When the implicit pipeline already built a `SpatialBuildResult` (via
`KdTree.build_with_topology` or `Octree.build_with_topology`), the
planned explicit-side bridge will reuse it without rebuilding. The
interface takes `SpatialBuildResult + Primitives + tri_to_leaf` and returns a
flavor-specific payload (`KdPayload`, `OctreePayload`, ...). The
resulting payload is then handed to a traversal-only backend via
`Accelerator.from_payload`.

Both exported adapters currently raise ``NotImplementedError``. Standalone
mesh builders are supported today. BVH has no implicit-side counterpart.
"""
from .from_spatial_tree import (  # noqa: F401
    kd_payload_from_spatial_tree,
    octree_payload_from_spatial_tree,
)
