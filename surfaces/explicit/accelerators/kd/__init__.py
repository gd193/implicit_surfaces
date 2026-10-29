"""Kd-tree-flavored mesh accelerator.

Top-down split-plane tree with standalone mesh builders in ``backends/``.
The reserved ``bridges.from_spatial_tree`` adapter for implicit-tree reuse
is not implemented yet.
"""
from .accelerator import KdAccelerator  # noqa: F401
from .payload import KdPayload  # noqa: F401
from . import backends  # noqa: F401
