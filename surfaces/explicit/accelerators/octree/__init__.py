"""Octree-flavored mesh accelerator.

Regular 8-way subdivision with standalone mesh builders in ``backends/``.
The reserved ``bridges.from_spatial_tree`` adapter for implicit-tree reuse
is not implemented yet.
"""
from .accelerator import OctreeAccelerator  # noqa: F401
from .payload import OctreePayload  # noqa: F401
from . import backends  # noqa: F401
