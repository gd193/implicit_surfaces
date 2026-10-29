"""Explicit surface family.

Geometry given as primitives (mesh triangles, point cloud, NURBS, ...).
A `MeshAccelerator` (BVH / kd / octree / grid) backs the universal
`Surface` queries via closest-point and ray-triangle intersection.
"""
from .base import ExplicitSurface  # noqa: F401
