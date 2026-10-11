"""Boundary-condition partitions and data. See docs/boundary_conditions.md."""
from boundaryConditions.base import (MIXED, BoundaryConditions, BoundaryData, BoundaryKind,
                                     BoundaryPartition, BoundaryValues)
from boundaryConditions.data import FunctionBoundaryData
from boundaryConditions.half_space import HalfSpacePartition
from boundaryConditions.trees import TreeArrays, TreePartition, VoxelTreePartition

__all__ = ["MIXED", "BoundaryConditions", "BoundaryData", "BoundaryKind", "BoundaryPartition",
           "BoundaryValues", "FunctionBoundaryData", "HalfSpacePartition", "TreeArrays",
           "TreePartition", "VoxelTreePartition"]
